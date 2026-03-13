import { printHeader, paint, formatStatus, printRow } from "../utils/log.js";
import { padTrailingZeros, formatPercentage } from "../utils/formatting.js";
import { getRandomBits } from "../utils/math.js";

const MAX_SAMPLES = 32;
const PAD_CHAR    = "0";

const verify = (network, task, inputString) => {
	const input = inputString.split("").map(Number);
	const target = task.solve(input);
	const pred = network.predict(input);

	const isCorrect = task.evaluate(target, pred);
	const formattedTarget = target.map((v) => padTrailingZeros(v, 6)).join(", ");
	const formattedPred = Array.from(pred).map((v) => padTrailingZeros(v, 6)).join(", ");
	const meanError = pred.reduce((acc, val, i) => acc + Math.abs(val - target[i]), 0) / pred.length;

	printRow([
		{ label: "Input",   value: `[${inputString}]`,             color: "reset"  },
		{ label: "Status",  value: formatStatus(isCorrect),        color: "reset"  },
		{ label: "Target",  value: `[${formattedTarget}]`,         color: "reset"  },
		{ label: "Predict", value: `[${formattedPred}]`,           color: "reset"  },
		{ label: "Error",   value: formatPercentage(meanError, 1), color: "yellow" },
	]);

	return isCorrect ? 1 : 0;
};

const exhaustiveInputs = (bits) =>
	Array.from({ length: Math.pow(2, bits) }, (_, i) =>
		i.toString(2).padStart(bits, PAD_CHAR));

const randomInputs = (bits, count) => {
	const seen = new Set();
	while (seen.size < count) {
		seen.add(getRandomBits(bits).join(""));
	}
	return Array.from(seen);
};

const resolveInputs = (bits, count, userInputs) => {
	if (userInputs) return userInputs;
	const total = Math.pow(2, bits);
	return total <= count ? exhaustiveInputs(bits) : randomInputs(bits, count);
};

export function evaluate(network, config, { inputs = null, count = MAX_SAMPLES } = {}) {
	printHeader("Final Evaluation");

	const resolvedInputs = resolveInputs(config.BITS, count, inputs);
	const score = resolvedInputs.reduce((acc, input) => acc + verify(network, config.TASK, input), 0);

	const accuracy = formatPercentage(score / resolvedInputs.length, 1);
	const resultColor = score === resolvedInputs.length ? "green" : "red";
	console.log(`\nFinal Accuracy: ${paint(`${score} / ${resolvedInputs.length} (${accuracy})`, resultColor)}`);
}