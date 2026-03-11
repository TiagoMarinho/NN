import { NeuralNetwork, schedulers, losses, activations } from "./neural_network.js";
import { TASKS } from "./tasks.js";
import { train, evaluate } from "./train.js";

const BITS = 16;
const BATCH_SIZE = 4
const CONFIG = {
	BITS,
	TASK: TASKS.parity(BITS),
	HIDDEN_LAYERS: [64, 64],
	TOTAL_EPOCHS: 1000,
	SAMPLES_PER_EPOCH: 1000,
	BATCH_SIZE,
	LOG_FREQUENCY: 100,
	SCHEDULER: (e, t) => schedulers.linear(e, t, 0.05, 0.002),
	LOSS: losses.mse,
	ACTIVATION: activations.sigmoid,
};

const network = new NeuralNetwork(CONFIG.BITS, CONFIG.HIDDEN_LAYERS, CONFIG.TASK.outputSize, CONFIG.ACTIVATION);

train(network, CONFIG);
evaluate(network, CONFIG);