import { NeuralNetwork } from "./NeuralNetwork.js";
import { activations } from "./activations.js";
import { optimizers } from "./optimizers.js";

const serializeOptimizerState = (state) => {
	if (!state) return null;
	return {
		t:  state.t,
		mW: Array.from(state.mW),
		vW: Array.from(state.vW),
		mB: Array.from(state.mB),
		vB: Array.from(state.vB),
	};
};

const deserializeOptimizerState = (raw) => {
	if (!raw) return null;
	return {
		t:  raw.t,
		mW: new Float32Array(raw.mW),
		vW: new Float32Array(raw.vW),
		mB: new Float32Array(raw.mB),
		vB: new Float32Array(raw.vB),
	};
};

export const toSnapshot = (network, epoch) => ({
	epoch,
	outputActivationName: network.outputActivation.name,
	optimizerName:        network.optimizer.name,
	optimizerParams:      network.optimizer.params,
	layers: network.layers.map((layer) => ({
		inputSize:      layer.inputSize,
		outputSize:     layer.outputSize,
		activationName: layer.activation.name,
		weights:        Array.from(layer.weights),
		biases:         Array.from(layer.biases),
		optimizerState: serializeOptimizerState(layer.optimizerState),
	})),
});

export const fromSnapshot = (snapshot) => {
	const { outputActivationName, optimizerName, optimizerParams, layers } = snapshot;

	const inputSize    = layers[0].inputSize;
	const outputSize   = layers[layers.length - 1].outputSize;
	const hiddenLayers = layers.slice(0, -1).map((l) => l.outputSize);

	const network = new NeuralNetwork(
		inputSize,
		hiddenLayers,
		outputSize,
		activations[outputActivationName],
		optimizers[optimizerName](optimizerParams),
	);

	network.layers.forEach((layer, i) => {
		layer.weights.set(layers[i].weights);
		layer.biases.set(layers[i].biases);
		layer.optimizerState = deserializeOptimizerState(layers[i].optimizerState);
	});

	return network;
};