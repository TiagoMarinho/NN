import { getSum, getHeWeightScale, getRandomFloat } from "./utils/math.js";
import { printHeader, paint, formatStatus, formatList } from "./utils/log.js";
import { padTrailingZeros, formatPercentage } from "./utils/formatting.js";

const CONFIGURATION = {
	BIT_COUNT: 8,
	HIDDEN_LAYERS: [64, 32, 32],
	TOTAL_EPOCHS: 1500,
	SAMPLES_PER_EPOCH: 1500,
	LEARNING_RATE_START: 0.05,
	LEARNING_RATE_MINIMUM: 0.001,
	LOG_FREQUENCY: 50,
};

const activations = {
	sigmoid: {
		calculate: (value) => 1 / (1 + Math.exp(-value)),
		derivative: (output) => output * (1 - output),
	},
	rectifiedLinearUnit: {
		calculate: (value) => Math.max(0, value),
		derivative: (output) => (output > 0 ? 1 : 0),
	},
};

const learningRateSchedulers = {
	cosineAnnealing: (epoch, totalEpochs, startRate, minimumRate) => {
		const range = startRate - minimumRate;
		const progression = epoch / totalEpochs;
		return (
			minimumRate + range * 0.5 * (1 + Math.cos(Math.PI * progression))
		);
	},
};

const losses = {
	meanAbsoluteError: {
		calculate: (target, prediction) => Math.abs(target - prediction),
		derivative: (target, prediction) => (target > prediction ? 1 : -1),
		label: "MAE",
	},
	meanSquaredError: {
		calculate: (target, prediction) => Math.pow(target - prediction, 2),
		derivative: (target, prediction) => target - prediction,
		label: "MSE",
	},
};

class Layer {
	constructor(inputSize, outputSize, activation) {
		this.activation = activation;
		const scale = getHeWeightScale(inputSize);

		this.weights = Array.from({ length: outputSize }, () =>
			Array.from(
				{ length: inputSize },
				() => getRandomFloat(-1, 1) * scale,
			),
		);
		this.biases = Array.from({ length: outputSize }, () => 0.01);

		this.lastInputs = null;
		this.lastOutputs = null;
	}

	forward(inputs) {
		this.lastInputs = inputs;
		this.lastOutputs = this.weights.map((row, i) => {
			const weightedSum = row.reduce(
				(sum, weight, j) => sum + weight * inputs[j],
				this.biases[i],
			);
			return this.activation.calculate(weightedSum);
		});
		return this.lastOutputs;
	}

	backward(errors, learningRate) {
		const gradients = errors.map(
			(error, i) =>
				error * this.activation.derivative(this.lastOutputs[i]),
		);

		const inputErrors = Array(this.lastInputs.length).fill(0);

		for (let i = 0; i < this.weights.length; i++) {
			for (let j = 0; j < this.weights[i].length; j++) {
				inputErrors[j] += this.weights[i][j] * gradients[i];
				this.weights[i][j] +=
					learningRate * gradients[i] * this.lastInputs[j];
			}
			this.biases[i] += learningRate * gradients[i];
		}

		return inputErrors;
	}
}

class NeuralNetwork {
	constructor(inputSize, hiddenLayers, outputSize) {
		const layerSizes = [inputSize, ...hiddenLayers, outputSize];
		this.layers = [];

		for (let i = 0; i < layerSizes.length - 1; i++) {
			const isOutputLayer = i === layerSizes.length - 2;
			const activation = isOutputLayer
				? activations.sigmoid
				: activations.rectifiedLinearUnit;
			this.layers.push(
				new Layer(layerSizes[i], layerSizes[i + 1], activation),
			);
		}
	}

	predict(inputs) {
		return this.layers.reduce(
			(currentData, layer) => layer.forward(currentData),
			inputs,
		);
	}

	optimize(targets, learningRate) {
		const finalLayerOutputs =
			this.layers[this.layers.length - 1].lastOutputs;

		let errors = targets.map((target, i) => {
			return losses.meanSquaredError.derivative(
				target,
				finalLayerOutputs[i],
			);
		});

		for (let i = this.layers.length - 1; i >= 0; i--) {
			errors = this.layers[i].backward(errors, learningRate);
		}
	}
}

const parityLogic = {
	generateInput: (bitCount) =>
		Array.from({ length: bitCount }, () => (Math.random() > 0.5 ? 1 : 0)),

	getExpectedOutput: (input) => {
		const sumOfOnes = getSum(input);
		return [sumOfOnes % 2 === 0 ? 1 : 0];
	},
};

function train(network) {
	printHeader("Training Session");

	for (let epoch = 1; epoch <= CONFIGURATION.TOTAL_EPOCHS; epoch++) {
		let accumulatedAbsoluteError = 0;
		let accumulatedSquaredError = 0;

		const currentLearningRate = learningRateSchedulers.cosineAnnealing(
			epoch,
			CONFIGURATION.TOTAL_EPOCHS,
			CONFIGURATION.LEARNING_RATE_START,
			CONFIGURATION.LEARNING_RATE_MINIMUM,
		);

		for (
			let sample = 0;
			sample < CONFIGURATION.SAMPLES_PER_EPOCH;
			sample++
		) {
			const input = parityLogic.generateInput(CONFIGURATION.BIT_COUNT);
			const target = parityLogic.getExpectedOutput(input);
			const prediction = network.predict(input);

			// We calculate both to see the difference in convergence
			accumulatedAbsoluteError += losses.meanAbsoluteError.calculate(
				target[0],
				prediction[0],
			);
			accumulatedSquaredError += losses.meanSquaredError.calculate(
				target[0],
				prediction[0],
			);

			network.optimize(target, currentLearningRate);
		}

		if (
			epoch % CONFIGURATION.LOG_FREQUENCY === 0 ||
			epoch === CONFIGURATION.TOTAL_EPOCHS
		) {
			const averageMAE = padTrailingZeros(
				accumulatedAbsoluteError / CONFIGURATION.SAMPLES_PER_EPOCH,
				5,
			);
			const averageMSE = padTrailingZeros(
				accumulatedSquaredError / CONFIGURATION.SAMPLES_PER_EPOCH,
				5,
			);
			const rateFormatted = padTrailingZeros(currentLearningRate, 4);

			console.log(
				`${formatList("Epoch", epoch.toString().padStart(4), "cyan")} | ` +
					`${formatList("MAE", averageMAE, "yellow")} | ` +
					`${formatList("MSE", averageMSE, "magenta")} | ` +
					`${formatList("Rate", rateFormatted, "reset")}`,
			);
		}
	}
}

function evaluate(network) {
	printHeader("Final Evaluation");

	let successfulPredictions = 0;
	const testSize = 64;

	for (let i = 0; i < testSize; i++) {
		const input = i
			.toString(2)
			.padStart(CONFIGURATION.BIT_COUNT, "0")
			.split("")
			.map(Number);
		const targets = parityLogic.getExpectedOutput(input);
		const outputs = network.predict(input);

		const neuronResults = outputs.map((raw, index) => {
			const target = targets[index];
			const prediction = raw > 0.5 ? 1 : 0;
			const certainty = Math.abs(raw - 0.5) * 2;

			const rawFormatted = padTrailingZeros(raw, 2);
			const confidenceFormatted = paint(
				formatPercentage(certainty),
				"yellow",
			);

			return {
				isCorrect: prediction === target,
				label: `Target: ${target} | Prediction: ${prediction} | Raw: ${rawFormatted} (${confidenceFormatted})`,
			};
		});

		const isFullyCorrect = neuronResults.every((res) => res.isCorrect);
		if (isFullyCorrect) successfulPredictions++;

		const inputDisplay = formatList(
			"Input",
			`[${input.join("")}]`,
			"reset",
		);
		const statusDisplay = formatStatus(isFullyCorrect);
		const detailsDisplay = neuronResults
			.map((res) => res.label)
			.join(" | ");

		console.log(`${inputDisplay} | ${statusDisplay} | ${detailsDisplay}`);
	}

	const score = `${successfulPredictions} / ${testSize}`;
	const scoreColor = successfulPredictions === testSize ? "green" : "red";
	console.log(`\nFinal Accuracy: ${paint(score, scoreColor)}`);
}

const network = new NeuralNetwork(
	CONFIGURATION.BIT_COUNT,
	CONFIGURATION.HIDDEN_LAYERS,
	1,
);

train(network);
evaluate(network);
