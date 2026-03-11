import { NeuralNetwork, schedulers } from "./neural_network.js";
import { TASKS } from "./tasks.js";
import { train, evaluate } from "./train.js";

const BITS = 8;
const OUTPUT_SIZE = 1;

const CONFIG = {
	BITS,
	TASK: TASKS.parity(BITS),
	HIDDEN_LAYERS: [32, 16, 16],
	TOTAL_EPOCHS: 500,
	SAMPLES_PER_EPOCH: 1000,
	BATCH_SIZE: 1,
	LOG_FREQUENCY: 100,
	SCHEDULER: (e, t) => schedulers.cosine(e, t, 0.01, 0.001),
};

const network = new NeuralNetwork(CONFIG.BITS, CONFIG.HIDDEN_LAYERS, OUTPUT_SIZE);

train(network, CONFIG);
evaluate(network, CONFIG);