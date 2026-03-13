import { printHeader, paint, formatStatus, printRow } from "../utils/log.js";
import { padTrailingZeros, formatPercentage } from "../utils/formatting.js";
import { getRandomBool } from "../utils/math.js";

const MAX_SAMPLES = 32;
const PAD_CHAR = "0";

const verify = (network, task, inputString) => {
	const input = inputString.split("").map(Number);
	const target = task.solve(input);
	const pred = network.predict(input);

	const isCorrect = task.evaluate(target, pred);

	const formattedTarget = target.map((v) => padTrailingZeros(v, 6)).join(", ");
	const formattedPred   = Array.from(pred).map((v) => padTrailingZeros(v, 6)).join(", ");

	const meanError = pred.reduce((acc, val, i) => acc + Math.abs(val - target[i]), 0) / pred.length;

	printRow([
		{ label: "Input",   value: `[${inputString}]`,            color: "reset"  },
		{ label: "Status",  value: formatStatus(isCorrect),       color: "reset"  },
		{ label: "Target",  value: `[${formattedTarget}]`,        color: "reset"  },
		{ label: "Predict", value: `[${formattedPred}]`,          color: "reset"  },
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
			const bits = Array.from({ length: config.BITS }, () => (getRandomBool() ? 1 : 0)).join("");
			if (tested.has(bits)) continue;
			tested.add(bits);
			score += verify(network, config.TASK, bits);
		}
	}

	const accuracy = formatPercentage(score / count, 1);
	const resultColor = score === count ? "green" : "red";
	console.log(`\nFinal Accuracy: ${paint(`${score} / ${count} (${accuracy})`, resultColor)}`);
}