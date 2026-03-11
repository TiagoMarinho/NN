import { losses } from "./neural_network.js";
import { printHeader, paint, formatStatus, formatList } from "./utils/log.js";
import { padTrailingZeros, formatPercentage } from "./utils/formatting.js";

const THRESHOLD = 0.5;
const MAX_SAMPLES = 32;
const BASE_2 = 2;
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

			network.backward(target);
			if (s % config.BATCH_SIZE === 0) network.optimize(rate, config.BATCH_SIZE);
		}

		if (epoch % config.LOG_FREQUENCY === 0 || epoch === config.TOTAL_EPOCHS) {
			const progress = epoch / config.TOTAL_EPOCHS;
			console.log(
				`${formatList("Epoch", epoch.toString().padStart(4), "cyan")} (${formatPercentage(progress, 0).padStart(4)}) | ` +
				`${formatList("MAE", padTrailingZeros(stats.mae / config.SAMPLES_PER_EPOCH, 10), "yellow")} | ` +
				`${formatList("MSE", padTrailingZeros(stats.mse / config.SAMPLES_PER_EPOCH, 10), "magenta")} | ` +
				`${formatList("BCE", padTrailingZeros(stats.bce / config.SAMPLES_PER_EPOCH, 10), "green")} | ` +
				`${formatList("Rate", padTrailingZeros(rate, 4), "reset")}`
			);
		}
	}
}

const verify = (network, task, inputString) => {
	const input = inputString.split("").map(Number);
	const target = task.solve(input);
	const pred = network.predict(input);
	
	const discrete = Array.from(pred, (v) => (v > THRESHOLD ? 1 : 0));
	const isCorrect = target.every((val, i) => val === discrete[i]);

	const inputLabel = formatList("Input", `[${inputString}]`, "reset");
	const details = `Target: [${target.join(",")}] | Predict: [${discrete.join(",")}]`;

	console.log(`${inputLabel} | ${formatStatus(isCorrect)} | ${details}`);

	return isCorrect ? 1 : 0;
};

export function evaluate(network, config) {
	printHeader("Final Evaluation");
	const total = Math.pow(BASE_2, config.BITS);
	const isExhaustive = total <= MAX_SAMPLES;
	const count = isExhaustive ? total : MAX_SAMPLES;
	let score = 0;

	if (isExhaustive) {
		for (let i = 0; i < total; i++) {
			score += verify(network, config.TASK, i.toString(BASE_2).padStart(config.BITS, PAD_CHAR));
		}
	} else {
		const seen = new Set();
		while (seen.size < count) {
			const bits = Array.from({ length: config.BITS }, () => (Math.random() > THRESHOLD ? 1 : 0)).join("");
			if (seen.has(bits)) continue;
			seen.add(bits);
			score += verify(network, config.TASK, bits);
		}
	}

	const accuracy = formatPercentage(score / count, 1);
	console.log(`\nFinal Accuracy: ${paint(`${score} / ${count} (${accuracy})`, score === count ? "green" : "red")}`);
}