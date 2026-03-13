import { Layer } from "./Layer.js";
import { activations } from "./activations.js";
import { losses } from "./losses.js";
import { optimizers } from "./optimizers.js";

export class NeuralNetwork {
	constructor(
		inputSize,
		hiddenLayers,
		outputSize,
		hiddenActivation = activations.leakyRelu,
		outputActivation = activations.sigmoid,
		optimizer = optimizers.sgd()
	) {
		this.hiddenActivation = hiddenActivation;
		this.outputActivation = outputActivation;
		this.optimizer = optimizer;

		const sizes = [inputSize, ...hiddenLayers, outputSize];
		this.layers = sizes.slice(0, -1).map((size, i) => {
			const isOutput = i === sizes.length - 2;
			const activation = isOutput ? outputActivation : hiddenActivation
			return new Layer(size, sizes[i + 1], activation, optimizer);
		});
		this.errorBuffer = new Float32Array(outputSize);
	}

	predict(inputs) {
		return this.layers.reduce((data, layer) => layer.forward(data), inputs);
	}

	backward(targets, lossType) {
		const outputs = this.layers[this.layers.length - 1].outputs;
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