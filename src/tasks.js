import { getSum } from "./utils/math.js";

const RANDOM_BIT_THRESHOLD = 0.5;

const generateBits = (length) => Array.from({ length }, () => (Math.random() > RANDOM_BIT_THRESHOLD ? 1 : 0));

export const TASKS = {
	parity: (bits) => ({
		name: "Parity",
		generate: () => {
			const input = generateBits(bits);
			return { input, target: [getSum(input) % 2 === 0 ? 1 : 0] };
		},
		solve: (input) => [getSum(input) % 2 === 0 ? 1 : 0],
	}),
	majority: (bits) => ({
		name: "Majority",
		generate: () => {
			const input = generateBits(bits);
			return { input, target: [getSum(input) > bits / 2 ? 1 : 0] };
		},
		solve: (input) => [getSum(input) > bits / 2 ? 1 : 0],
	}),
	xor: (bits) => ({
		name: "XOR",
		generate: () => {
			const input = generateBits(bits);
			return { input, target: [input.reduce((acc, val) => acc ^ val, 0)] };
		},
		solve: (input) => [input.reduce((acc, val) => acc ^ val, 0)],
	}),
	and: (bits) => ({
		name: "AND",
		generate: () => {
			const input = generateBits(bits);
			return { input, target: [input.every((v) => v === 1) ? 1 : 0] };
		},
		solve: (input) => [input.every((v) => v === 1) ? 1 : 0],
	}),
};