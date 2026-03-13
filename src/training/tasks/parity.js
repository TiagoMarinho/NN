import { getRandomBits, getSum } from "../../utils/math.js";

export const parity = (bits) => ({
	name: "Parity",
	outputSize: 1,
	generate: () => {
		const input = getRandomBits(bits);
		const target = [getSum(input) % 2 === 0 ? 1 : 0];
		return { input, target };
	},
	solve: (input) => [getSum(input) % 2 === 0 ? 1 : 0],
	evaluate: (target, pred) => target[0] === Math.round(pred[0]),
	formatPrediction: (pred) => `[${pred.map((v) => v.toFixed(2)).join(", ")}]`,
})