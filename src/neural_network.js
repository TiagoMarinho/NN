import { getHeWeightScale, getXavierScale } from "./utils/nn.js";
import { getRandomFloat } from "./utils/math.js";

const INITIAL_BIAS = 0.01;
const LEAKY_RELU_SLOPE = 0.01;
const EPSILON = 1e-7;

export const schedulers = {
	cosine: (epoch, total, start, min) => {
		const progress = epoch / total;
		const scale = 0.5 * (1 + Math.cos(Math.PI * progress));
		return min + (start - min) * scale;
	},
	stepped: (epoch, total, steps) => {
		const pastSteps = steps.filter((step) => epoch >= step.epoch);
		const currentStep = pastSteps.sort((a, b) => b.epoch - a.epoch)[0];
		return currentStep ? currentStep.rate : steps[0].rate;
	},
	linear: (epoch, total, start, min) => {
		return start - (start - min) * (epoch / total);
	},
	constant: (epoch, total, rate) => rate,
};

export const activations = {
	linear: {
		calculate: (x) => x,
		derivative: (y) => 1,
	},
	sigmoid: {
		calculate: (x) => 1 / (1 + Math.exp(-x)),
		derivative: (y) => y * (1 - y),
	},
	relu: {
		calculate: (x) => Math.max(0, x),
		derivative: (y) => (y > 0 ? 1 : 0),
	},
	leakyRelu: {
		calculate: (x) => (x > 0 ? x : x * LEAKY_RELU_SLOPE),
		derivative: (y) => (y > 0 ? 1 : LEAKY_RELU_SLOPE),
	},
};

export const losses = {
	mae: {
		calculate: (target, pred) => Math.abs(target - pred),
		derivative: (target, pred) => (target > pred ? 1 : -1),
	},
	mse: {
		calculate: (target, pred) => (target - pred) ** 2,
		derivative: (target, pred) => 2 * (pred - target),
	},
	bce: {
		calculate: (target, pred) => {
			const clipped = Math.min(1 - EPSILON, Math.max(EPSILON, pred));
			return -(target * Math.log(clipped) + (1 - target) * Math.log(1 - clipped));
		},
		derivative: (target, pred) => {
			const clipped = Math.min(1 - EPSILON, Math.max(EPSILON, pred));
			return (clipped - target) / (clipped * (1 - clipped));
		},
	},
};

export const optimizers = {

	sgd: () => ({
		initState: () => null,
		step(weights, biases, weightGrads, biasGrads, batchSize, learningRate, _state) {
			const s = learningRate / batchSize;
			for (let i = 0; i < weights.length; i++) weights[i] -= s * weightGrads[i];
			for (let i = 0; i < biases.length; i++) biases[i] -= s * biasGrads[i];
		},
	}),

	adam: ({ beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 } = {}) => ({
		initState: (weightSize, biasSize) => ({
			t:  0,
			mW: new Float32Array(weightSize), // first moment  (weights)
			vW: new Float32Array(weightSize), // second moment (weights)
			mB: new Float32Array(biasSize),   // first moment  (biases)
			vB: new Float32Array(biasSize),   // second moment (biases)
		}),
		step(weights, biases, weightGrads, biasGrads, batchSize, learningRate, state) {
			state.t++;
			const correctedLR = learningRate * Math.sqrt(1 - beta2 ** state.t) / (1 - beta1 ** state.t);

			for (let i = 0; i < weights.length; i++) {
				const g = weightGrads[i] / batchSize;
				state.mW[i] = beta1 * state.mW[i] + (1 - beta1) * g;
				state.vW[i] = beta2 * state.vW[i] + (1 - beta2) * g * g;
				weights[i] -= correctedLR * state.mW[i] / (Math.sqrt(state.vW[i]) + epsilon);
			}

			for (let i = 0; i < biases.length; i++) {
				const g = biasGrads[i] / batchSize;
				state.mB[i] = beta1 * state.mB[i] + (1 - beta1) * g;
				state.vB[i] = beta2 * state.vB[i] + (1 - beta2) * g * g;
				biases[i] -= correctedLR * state.mB[i] / (Math.sqrt(state.vB[i]) + epsilon);
			}
		},
	}),

};

export class GradientBuffer {
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

export class Layer {
	constructor(inputSize, outputSize, activation, optimizer) {
		this.inputSize = inputSize;
		this.outputSize = outputSize;
		this.activation = activation;
		this.optimizer = optimizer;
		this.weights = new Float32Array(outputSize * inputSize);
		this.biases = new Float32Array(outputSize).fill(INITIAL_BIAS);
		this.buffer = new GradientBuffer(outputSize, inputSize);
		this.optimizerState = optimizer.initState(this.weights.length, this.biases.length);

		const scale = activation === activations.sigmoid ? getXavierScale(inputSize) : getHeWeightScale(inputSize);
		for (let i = 0; i < this.weights.length; i++) {
			this.weights[i] = getRandomFloat(-1, 1) * scale;
		}

		this.inputCache = null;
		this.outputCache = null;
		this.outputs = new Float32Array(outputSize);
		this.deltas = new Float32Array(outputSize);
		this.nextErrors = new Float32Array(inputSize);
	}

	forward(inputs) {
		this.inputCache = inputs;
		for (let i = 0; i < this.outputSize; i++) {
			const offset = i * this.inputSize;
			let sum = this.biases[i];
			for (let j = 0; j < this.inputSize; j++) {
				sum += this.weights[offset + j] * inputs[j];
			}
			this.outputs[i] = this.activation.calculate(sum);
		}
		this.outputCache = this.outputs;
		return this.outputs;
	}

	backward(errors) {
		for (let i = 0; i < this.outputSize; i++) {
			this.deltas[i] = errors[i] * this.activation.derivative(this.outputCache[i]);
		}
		this.nextErrors.fill(0);
		for (let i = 0; i < this.outputSize; i++) {
			const offset = i * this.inputSize;
			const delta = this.deltas[i];
			for (let j = 0; j < this.inputSize; j++) {
				this.nextErrors[j] += this.weights[offset + j] * delta;
			}
		}
		this.buffer.add(this.deltas, this.inputCache);
		return this.nextErrors;
	}

	step(learningRate) {
		if (this.buffer.size === 0) return;
		this.optimizer.step(
			this.weights, this.biases,
			this.buffer.weightGradients, this.buffer.biasGradients,
			this.buffer.size, learningRate,
			this.optimizerState
		);
		this.buffer.reset();
	}
}

export class NeuralNetwork {
	constructor(
		inputSize,
		hiddenLayers,
		outputSize,
		outputActivation = activations.sigmoid,
		optimizer = optimizers.sgd
	) {
		const sizes = [inputSize, ...hiddenLayers, outputSize];
		this.layers = sizes.slice(0, -1).map((size, i) => {
			const isOutput = i === sizes.length - 2;
			return new Layer(size, sizes[i + 1], isOutput ? outputActivation : activations.leakyRelu, optimizer);
		});
		this.errorBuffer = new Float32Array(outputSize);
	}

	predict(inputs) {
		return this.layers.reduce((data, layer) => layer.forward(data), inputs);
	}

	backward(targets, lossType = losses.bce) {
		const outputs = this.layers[this.layers.length - 1].outputCache;
		for (let i = 0; i < this.errorBuffer.length; i++) {
			this.errorBuffer[i] = lossType.derivative(targets[i], outputs[i]);
		}
		let currentErrors = this.errorBuffer;
		for (let i = this.layers.length - 1; i >= 0; i--) {
			currentErrors = this.layers[i].backward(currentErrors);
		}
	}

	optimize(learningRate) {
		this.layers.forEach((layer) => layer.step(learningRate));
	}
}