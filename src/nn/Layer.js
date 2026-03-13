import { activations } from "./activations.js";
import { getHeWeightScale, getXavierScale } from "../utils/nn.js";
import { getRandomFloat } from "../utils/math.js";

const INITIAL_BIAS = 0.01;

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
		return this.outputs;
	}

	backward(errors) {
		for (let i = 0; i < this.outputSize; i++) {
			this.deltas[i] = errors[i] * this.activation.derivative(this.outputs[i]);
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
			this.optimizerState,
		);
		this.buffer.reset();
	}
}