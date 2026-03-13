import { activations, optimizers, losses, schedulers } from "./nn/index.js";
import { TASKS } from "./training/tasks/index.js";

const BITS = 16;

export const CONFIG = {
	BITS,
	TASK:               TASKS.parity(BITS),
	HIDDEN_LAYERS:      [32, 16],
	TOTAL_EPOCHS:       1000,
	SAMPLES_PER_EPOCH:  10000,
	BATCH_SIZE:         4,
	LOG_FREQUENCY:      100,
	SCHEDULER:          (e, t) => schedulers.linear(e, t, 0.001, 0.0001),
	LOSS:               losses.bce,
	HIDDEN_ACTIVATION:  activations.leakyRelu,
	OUTPUT_ACTIVATION:  activations.sigmoid,
	OPTIMIZER:          optimizers.adam(),
	CHECKPOINT:         { dir: "./checkpoints", frequency: 100 },
};