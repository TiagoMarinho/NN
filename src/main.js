import { getSum, getRandomFloat } from "./utils/math.js";
import { getHeWeightScale, getXavierScale } from "./utils/nn.js";
import { printHeader, paint, formatStatus, formatList } from "./utils/log.js";
import { padTrailingZeros, formatPercentage } from "./utils/formatting.js";

const TASKS = {
	parity: (bits) => ({
		name: "Parity",
		generate: () => {
			const input = Array.from({ length: bits }, () =>
				Math.random() > 0.5 ? 1 : 0,
			);
			return { input, target: [getSum(input) % 2 === 0 ? 1 : 0] };
		},
		solve: (input) => [getSum(input) % 2 === 0 ? 1 : 0],
	}),
	majority: (bits) => ({
		name: "Majority",
		generate: () => {
			const input = Array.from({ length: bits }, () =>
				Math.random() > 0.5 ? 1 : 0,
			);
			return { input, target: [getSum(input) > bits / 2 ? 1 : 0] };
		},
		solve: (input) => [getSum(input) > bits / 2 ? 1 : 0],
	}),
	xor: (bits) => ({
		name: "XOR",
		generate: () => {
			const input = Array.from({ length: bits }, () =>
				Math.random() > 0.5 ? 1 : 0,
			);
			return {
				input,
				target: [input.reduce((acc, val) => acc ^ val, 0)],
			};
		},
		solve: (input) => [input.reduce((acc, val) => acc ^ val, 0)],
	}),
	and: (bits) => ({
		name: "AND",
		generate: () => {
			const input = Array.from({ length: bits }, () =>
				Math.random() > 0.5 ? 1 : 0,
			);
			return { input, target: [input.every((v) => v === 1) ? 1 : 0] };
		},
		solve: (input) => [input.every((v) => v === 1) ? 1 : 0],
	}),
};

const schedulers = {
	cosine: (epoch, total, start, min) => {
		const progress = epoch / total;
		return min + (start - min) * 0.5 * (1 + Math.cos(Math.PI * progress));
	},

	stepped: (epoch, total, steps) => {
		const currentStep = steps
			.filter((s) => epoch >= s.epoch)
			.sort((a, b) => b.epoch - a.epoch)[0];
		return currentStep ? currentStep.rate : steps[0].rate;
	},

	linear: (epoch, total, start, min) => {
		return start - (start - min) * (epoch / total);
	},

	constant: (epoch, total, rate) => rate
};

const BITS = 64;
const CONFIGURATION = {
	BITS,
	TASK: TASKS.majority(BITS),
	HIDDEN_LAYERS: [64, 32, 16],
	TOTAL_EPOCHS: 1000,
	SAMPLES_PER_EPOCH: 1000,
	BATCH_SIZE: 1,
	
	SCHEDULER: (e, t) => schedulers.linear(e, t, 0.1, 0.01),

	LOG_FREQUENCY: 10,
};

const activations = {
	sigmoid: {
		calculate: (x) => 1 / (1 + Math.exp(-x)),
		derivative: (y) => y * (1 - y),
	},

	relu: {
		calculate: (x) => Math.max(0, x),
		derivative: (y) => (y > 0 ? 1 : 0),
	},

	leakyRelu: {
		calculate: (x) => (x > 0 ? x : x * 0.01),
		derivative: (y) => (y > 0 ? 1 : 0.01),
	},
};

const losses = {
	mae: {
		calculate: (t, p) => Math.abs(t - p),
		derivative: (t, p) => (t > p ? -1 : 1),
	},

	mse: {
		calculate: (t, p) => Math.pow(t - p, 2),
		derivative: (t, p) => p - t,
	},

	bce: {
		calculate: (t, p) => {
			const eps = 1e-7;
			const clipped = Math.min(1 - eps, Math.max(eps, p));
			return -(t * Math.log(clipped) + (1 - t) * Math.log(1 - clipped));
		},
		derivative: (t, p) => {
			const eps = 1e-7;
			const clipped = Math.min(1 - eps, Math.max(eps, p));
			return (clipped - t) / (clipped * (1 - clipped));
		},
	},
};

class GradientBuffer {
	constructor(outputSize, inputSize) {
		this.outputSize = outputSize;
		this.inputSize = inputSize;
		this.weightGradients = new Float32Array(outputSize * inputSize);
		this.biasGradients = new Float32Array(outputSize);
		this.size = 0;
	}

	add(deltas, inputs) {
		for (let i = 0; i < this.outputSize; i++) {
			const offset = i * this.inputSize;
			const delta = deltas[i];

			for (let j = 0; j < this.inputSize; j++) {
				this.weightGradients[offset + j] += delta * inputs[j];
			}
			this.biasGradients[i] += delta;
		}
		this.size++;
	}

	reset() {
		this.weightGradients.fill(0);
		this.biasGradients.fill(0);
		this.size = 0;
	}
}

class Layer {
	constructor(inputSize, outputSize, activation) {
		this.inputSize = inputSize;
		this.outputSize = outputSize;
		this.activation = activation;

		const scale = 
			activation === activations.sigmoid
				? getXavierScale(inputSize)
				: getHeWeightScale(inputSize);
		this.weights = new Float32Array(outputSize * inputSize).map(
			() => getRandomFloat(-1, 1) * scale,
		);
		this.biases = new Float32Array(outputSize).fill(0.01);

		this.buffer = new GradientBuffer(outputSize, inputSize);
		this.inputCache = null;
		this.outputCache = null;
	}

	forward(inputs) {
		this.inputCache = inputs;
		const outputs = new Float32Array(this.outputSize);

		for (let i = 0; i < this.outputSize; i++) {
			const offset = i * this.inputSize;
			let sum = this.biases[i];

			for (let j = 0; j < this.inputSize; j++) {
				sum += this.weights[offset + j] * inputs[j];
			}
			outputs[i] = this.activation.calculate(sum);
		}

		this.outputCache = outputs;
		return outputs;
	}

	backward(errors) {
		const deltas = new Float32Array(this.outputSize);
		for (let i = 0; i < this.outputSize; i++) {
			deltas[i] =
				errors[i] * this.activation.derivative(this.outputCache[i]);
		}

		const nextErrors = new Float32Array(this.inputSize).fill(0);
		for (let i = 0; i < this.outputSize; i++) {
			const offset = i * this.inputSize;
			const delta = deltas[i];

			for (let j = 0; j < this.inputSize; j++) {
				nextErrors[j] += this.weights[offset + j] * delta;
			}
		}

		this.buffer.add(deltas, this.inputCache);
		return nextErrors;
	}

	update(learningRate) {
		if (this.buffer.size === 0) return;
		const step = learningRate / this.buffer.size;

		for (let i = 0; i < this.weights.length; i++) {
			this.weights[i] -= step * this.buffer.weightGradients[i];
		}
		for (let i = 0; i < this.biases.length; i++) {
			this.biases[i] -= step * this.buffer.biasGradients[i];
		}
		this.buffer.reset();
	}
}

class NeuralNetwork {
	constructor(inputSize, hiddenLayers, outputSize) {
		const sizes = [inputSize, ...hiddenLayers, outputSize];
		this.layers = sizes.slice(0, -1).map((size, i) => {
			const isLast = i === sizes.length - 2;
			return new Layer(
				size,
				sizes[i + 1],
				isLast ? activations.sigmoid : activations.leakyRelu,
			);
		});
	}

	predict(inputs) {
		return this.layers.reduce(
			(current, layer) => layer.forward(current),
			inputs,
		);
	}

	backward(targets, lossType = losses.bce) {
		const outputs = this.layers[this.layers.length - 1].outputCache;
		let errors = targets.map((t, i) => lossType.derivative(t, outputs[i]));
		for (let i = this.layers.length - 1; i >= 0; i--) {
			errors = this.layers[i].backward(errors);
		}
	}

	optimize(learningRate) {
		this.layers.forEach((layer) => layer.update(learningRate));
	}
}

function train(network) {
	const {
		TASK,
		TOTAL_EPOCHS,
		SAMPLES_PER_EPOCH,
		BATCH_SIZE,
		LOG_FREQUENCY,
		SCHEDULER,
	} = CONFIGURATION;
    
	printHeader(`Training: ${TASK.name}`);

	for (let epoch = 1; epoch <= TOTAL_EPOCHS; epoch++) {
		let maeAcc = 0,
			mseAcc = 0;
		
		const rate = SCHEDULER(epoch, TOTAL_EPOCHS);

		for (let epoch = 1; epoch <= SAMPLES_PER_EPOCH; epoch++) {
			const { input, target } = TASK.generate();
			const prediction = network.predict(input);

			maeAcc += losses.mae.calculate(target[0], prediction[0]);
			mseAcc += losses.mse.calculate(target[0], prediction[0]);

			network.backward(target);
			if (epoch % BATCH_SIZE === 0) network.optimize(rate);
		}

		if (epoch % LOG_FREQUENCY === 0 || epoch === TOTAL_EPOCHS) {
			const progress = epoch / TOTAL_EPOCHS;
			console.log(
				`${formatList("Epoch", epoch.toString().padStart(4), "cyan")} (${formatPercentage(progress, 0).padStart(4)}) | ` +
					`${formatList("MAE", padTrailingZeros(maeAcc / SAMPLES_PER_EPOCH, 5), "yellow")} | ` +
					`${formatList("MSE", padTrailingZeros(mseAcc / SAMPLES_PER_EPOCH, 5), "magenta")} | ` +
					`${formatList("Rate", padTrailingZeros(rate, 4), "reset")}`,
			);
		}
	}
}

const testSample = (network, task, binaryString) => {
	const input = binaryString.split("").map(Number);
	const target = task.solve(input)[0];
	const raw = network.predict(input)[0];
	const prediction = raw > 0.5 ? 1 : 0;
	const isCorrect = prediction === target;

	const certainty = paint(
		formatPercentage(Math.abs(raw - 0.5) * 2, 1),
		"yellow",
	);
	const label = formatList("Input", `[${binaryString}]`, "reset");

	console.log(
		`${label} | ${formatStatus(isCorrect)} | ` +
			`Target: ${target} | Predict: ${prediction} | ` +
			`Raw: ${padTrailingZeros(raw, 2)} (${certainty})`,
	);

	return isCorrect ? 1 : 0;
};

const evaluateExhaustively = (network, task, bits, totalPossible) => {
	console.log(
		paint(`> Testing all ${totalPossible} combinations...\n`, "reset"),
	);
	let score = 0;
	for (let i = 0; i < totalPossible; i++) {
		const binaryString = i.toString(2).padStart(bits, "0");
		score += testSample(network, task, binaryString);
	}
	return score;
};

const evaluateRandomly = (network, task, bits, testSize) => {
	console.log(paint(`> Testing ${testSize} random samples...\n`, "reset"));
	let score = 0;
	const seen = new Set();

	while (seen.size < testSize) {
		const input = Array.from({ length: bits }, () =>
			Math.random() > 0.5 ? 1 : 0,
		);
		const binaryString = input.join("");

		if (seen.has(binaryString)) continue;
		seen.add(binaryString);

		score += testSample(network, task, binaryString);
	}
	return score;
};

function evaluate(network) {
	printHeader("Final Evaluation");
	const { TASK, BITS } = CONFIGURATION;

	const totalPossible = Math.pow(2, BITS);
	const limit = 32;
	const isSmallEnough = totalPossible <= limit;

	const score = isSmallEnough
		? evaluateExhaustively(network, TASK, BITS, totalPossible)
		: evaluateRandomly(network, TASK, BITS, limit);

	const testSize = isSmallEnough ? totalPossible : limit;
	const accuracy = formatPercentage(score / testSize, 1);
	const resultColor = score === testSize ? "green" : "red";

	console.log(
		`\nFinal Accuracy: ${paint(`${score} / ${testSize} (${accuracy})`, resultColor)}`,
	);
}

const network = new NeuralNetwork(
	CONFIGURATION.BITS,
	CONFIGURATION.HIDDEN_LAYERS,
	1,
);
train(network);
evaluate(network);
