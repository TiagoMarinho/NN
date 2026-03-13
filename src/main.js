import { NeuralNetwork, schedulers, losses, activations, optimizers } from "./nn/index.js";
import { TASKS } from "./training/tasks/index.js";
import { train } from "./training/train.js";
import { evaluate } from "./training/evaluate.js";

const BITS = 16;
const BATCH_SIZE = 4;
const CONFIG = {
	BITS,
	TASK: TASKS.parity(BITS),
	HIDDEN_LAYERS: [32, 16],
	TOTAL_EPOCHS: 1000,
	SAMPLES_PER_EPOCH: 10000,
	BATCH_SIZE,
	LOG_FREQUENCY: 100,
	SCHEDULER: (e, t) => schedulers.linear(e, t, 0.001, 0.0001),
	LOSS: losses.bce,
	ACTIVATION: activations.sigmoid,
	OPTIMIZER: optimizers.adam(),
};

const network = new NeuralNetwork(CONFIG.BITS, CONFIG.HIDDEN_LAYERS, CONFIG.TASK.outputSize, CONFIG.ACTIVATION, CONFIG.OPTIMIZER);

train(network, CONFIG);
evaluate(network, CONFIG);