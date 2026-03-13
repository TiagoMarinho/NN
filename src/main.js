import { NeuralNetwork, schedulers, losses, activations, optimizers } from "./nn/index.js";
import { CONFIG } from "./config.js";
import { train } from "./training/train.js";
import { evaluate } from "./training/evaluate.js";

const network = new NeuralNetwork(CONFIG.BITS, CONFIG.HIDDEN_LAYERS, CONFIG.TASK.outputSize, CONFIG.HIDDEN_ACTIVATION, CONFIG.OUTPUT_ACTIVATION, optimizers.adam());

train(network, CONFIG);
evaluate(network, CONFIG);