import { getSum } from "./utils/math.js";

const generateBits = (length) =>
	Array.from({ length }, () => (Math.random() > 0.5 ? 1 : 0));

export const TASKS = {
	parity: (bits) => ({
		name: "Parity",
		outputSize: 1,
		generate: () => {
			const input = generateBits(bits);
			const target = [getSum(input) % 2 === 0 ? 1 : 0];
			return { input, target };
		},
		solve: (input) => [getSum(input) % 2 === 0 ? 1 : 0],
	}),

	majority: (bits) => ({
		name: "Majority",
		outputSize: 1,
		generate: () => {
			const input = generateBits(bits);
			const target = [getSum(input) > bits / 2 ? 1 : 0];
			return { input, target };
		},
		solve: (input) => [getSum(input) > bits / 2 ? 1 : 0],
	}),

	xor: (bits) => ({
		name: "XOR",
		outputSize: 1,
		generate: () => {
			const input = generateBits(bits);
			const target = [input.reduce((acc, val) => acc ^ val, 0)];
			return { input, target };
		},
		solve: (input) => [input.reduce((acc, val) => acc ^ val, 0)],
	}),

	and: (bits) => ({
		name: "AND",
		outputSize: 1,
		generate: () => {
			const input = generateBits(bits);
			const target = [input.every((v) => v === 1) ? 1 : 0];
			return { input, target };
		},
		solve: (input) => [input.every((v) => v === 1) ? 1 : 0],
	}),

	rangeAnalysis: (bits) => ({
		name: "Range Analysis",
		outputSize: 2,
		generate: () => {
			const input = generateBits(bits);
			const sum = getSum(input);
			const target = [
				sum % 2 === 0 ? 1 : 0,
				sum > bits / 2 ? 1 : 0,
			];
			return { input, target };
		},
		solve: (input) => {
			const sum = getSum(input);
			return [
				sum % 2 === 0 ? 1 : 0,
				sum > bits / 2 ? 1 : 0,
			];
		},
	}),
};