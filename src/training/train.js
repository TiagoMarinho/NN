import { losses } from "../nn/losses.js";
import { printHeader, printRow } from "../utils/log.js";
import { padTrailingZeros, formatPercentage } from "../utils/formatting.js";
import { saveCheckpoint } from "./checkpoint.js";

const calculateVectorLoss = (targets, preds, lossFn) => {
	let total = 0;
	for (let i = 0; i < targets.length; i++) {
		total += lossFn(targets[i], preds[i]);
	}
	return total / targets.length;
};

export function train(network, config) {
	printHeader(`Training: ${config.TASK.name}`);

	const startEpoch = config.START_EPOCH ?? 1;

	for (let epoch = startEpoch; epoch <= config.TOTAL_EPOCHS; epoch++) {
		const stats = { mae: 0, mse: 0, bce: 0 };
		const rate = config.SCHEDULER(epoch, config.TOTAL_EPOCHS);

		for (let s = 1; s <= config.SAMPLES_PER_EPOCH; s++) {
			const { input, target } = config.TASK.generate();
			const pred = network.predict(input);

			stats.mae += calculateVectorLoss(target, pred, losses.mae.calculate);
			stats.mse += calculateVectorLoss(target, pred, losses.mse.calculate);
			stats.bce += calculateVectorLoss(target, pred, losses.bce.calculate);

			network.backward(target, config.LOSS);
			if (s % config.BATCH_SIZE === 0) network.optimize(rate);
		}

		if (config.CHECKPOINT && epoch % config.CHECKPOINT.frequency === 0) {
			saveCheckpoint(network, epoch, config.CHECKPOINT.dir);
		}

		if (epoch % config.LOG_FREQUENCY === 0 || epoch === config.TOTAL_EPOCHS) {
			const progress = formatPercentage(epoch / config.TOTAL_EPOCHS, 0).padStart(4);
			printRow([
				{ label: "Epoch", value: epoch, color: "cyan",    width: 4, suffix: ` (${progress})` },
				{ label: "MAE",   value: padTrailingZeros(stats.mae / config.SAMPLES_PER_EPOCH, 10), color: "yellow"  },
				{ label: "MSE",   value: padTrailingZeros(stats.mse / config.SAMPLES_PER_EPOCH, 10), color: "magenta" },
				{ label: "BCE",   value: padTrailingZeros(stats.bce / config.SAMPLES_PER_EPOCH, 10), color: "green"   },
				{ label: "Rate",  value: padTrailingZeros(rate, 4),                                  color: "reset"   },
			]);
		}
	}
}