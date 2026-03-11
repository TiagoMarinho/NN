import { losses } from "./neural_network.js";
import { printHeader, paint, formatStatus, printRow } from "./utils/log.js";
import { padTrailingZeros, formatPercentage } from "./utils/formatting.js";

const THRESHOLD = 0.5;
const MAX_SAMPLES = 32;
const PAD_CHAR = "0";

const calculateVectorLoss = (targets, preds, lossFn) => {
	let total = 0;
	for (let i = 0; i < targets.length; i++) {
		total += lossFn(targets[i], preds[i]);
	}
	return total / targets.length;
};

export function train(network, config) {
	printHeader(`Training: ${config.TASK.name}`);

	for (let epoch = 1; epoch <= config.TOTAL_EPOCHS; epoch++) {
		const stats = { mae: 0, mse: 0, bce: 0 };
		const rate = config.SCHEDULER(epoch, config.TOTAL_EPOCHS);

		for (let s = 1; s <= config.SAMPLES_PER_EPOCH; s++) {
			const { input, target } = config.TASK.generate();
			const pred = network.predict(input);

			stats.mae += calculateVectorLoss(target, pred, losses.mae.calculate);
			stats.mse += calculateVectorLoss(target, pred, losses.mse.calculate);
			stats.bce += calculateVectorLoss(target, pred, losses.bce.calculate);

			network.backward(target, config.LOSS);
			if (s % config.BATCH_SIZE === 0) network.optimize(rate, config.BATCH_SIZE);
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

const verify = (network, task, inputString) => {
	const input = inputString.split("").map(Number);
	const target = task.solve(input);
	const pred = network.predict(input);

	const isCorrect = task.evaluate(target, pred);

	const formattedTarget = target.map((v) => padTrailingZeros(v, 6)).join(", ");
	const formattedPred   = Array.from(pred).map((v) => padTrailingZeros(v, 6)).join(", ");

	const meanError = pred.reduce((acc, val, i) => acc + Math.abs(val - target[i]), 0) / pred.length;

	printRow([
		{ label: "Input",   value: `[${inputString}]`,       color: "reset"  },
		{ label: "Status",  value: formatStatus(isCorrect),  color: "reset"  },
		{ label: "Target",  value: `[${formattedTarget}]`,   color: "reset"  },
		{ label: "Predict", value: `[${formattedPred}]`,     color: "reset"  },
		{ label: "Error",   value: formatPercentage(meanError, 1), color: "yellow" },
	]);

	return isCorrect ? 1 : 0;
};

export function evaluate(network, config) {
	printHeader("Final Evaluation");
	const total = Math.pow(2, config.BITS);
	const isExhaustive = total <= MAX_SAMPLES;
	const count = isExhaustive ? total : MAX_SAMPLES;
	let score = 0;

	if (isExhaustive) {
		for (let i = 0; i < total; i++) {
			score += verify(network, config.TASK, i.toString(2).padStart(config.BITS, PAD_CHAR));
		}
	} else {
		const tested = new Set();
		while (tested.size < count) {
			const bits = Array.from({ length: config.BITS }, () => (Math.random() > THRESHOLD ? 1 : 0)).join("");
			if (tested.has(bits)) continue;
			tested.add(bits);
			score += verify(network, config.TASK, bits);
		}
	}

	const accuracy = formatPercentage(score / count, 1);
	const resultColor = score === count ? "green" : "red";
	console.log(`\nFinal Accuracy: ${paint(`${score} / ${count} (${accuracy})`, resultColor)}`);
}