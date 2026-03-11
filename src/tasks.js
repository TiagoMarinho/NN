import { getSum } from "./utils/math.js";

const generateBits = (length) =>
	Array.from({ length }, () => (Math.random() > 0.5 ? 1 : 0));

function solveBitGeometry(input, bits) {
    let maxOnes = 0, maxZeros = 0, transitions = 0;
    let curOnes = 0, curZeros = 0;

    for (let i = 0; i < input.length; i++) {
        if (input[i] === 1) {
            curOnes++;
            curZeros = 0;
        } else {
            curZeros++;
            curOnes = 0;
        }
        if (curOnes > maxOnes) maxOnes = curOnes;
        if (curZeros > maxZeros) maxZeros = curZeros;
        if (i > 0 && input[i] !== input[i - 1]) transitions++;
    }

    return [
        maxOnes / bits,       // longest run of 1s, normalized [0, 1]
        maxZeros / bits,      // longest run of 0s, normalized [0, 1]
        transitions / (bits - 1), // transition density, normalized [0, 1]
    ];
}

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
		evaluate: (target, pred) => target[0] === Math.round(pred[0]),
		formatPrediction: (pred) => `[${pred.map((v) => v.toFixed(2)).join(", ")}]`,
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
		evaluate: (target, pred) => target[0] === Math.round(pred[0]),
		formatPrediction: (pred) => `[${pred.map((v) => v.toFixed(2)).join(", ")}]`,
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
		evaluate: (target, pred) => target[0] === Math.round(pred[0]),
		formatPrediction: (pred) => `[${pred.map((v) => v.toFixed(2)).join(", ")}]`,
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
		evaluate: (target, pred) => target[0] === Math.round(pred[0]),
		formatPrediction: (pred) => `[${pred.map((v) => v.toFixed(2)).join(", ")}]`,
	}),

	normalizedCount: (bits) => ({
		name: "Normalized Count",
		outputSize: 1,
		generate: () => {
			const input = generateBits(bits);
			const target = [getSum(input) / bits];
			return { input, target };
		},
		solve: (input) => [getSum(input) / input.length],
		evaluate: (target, pred) => {
			const THRESHOLD = 0.01
			return Math.abs(target[0] - pred[0]) < THRESHOLD
		},
		formatPrediction: (pred) => `[${pred.map((v) => v.toFixed(2)).join(", ")}]`,
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
				sum > input.length / 2 ? 1 : 0,
			];
		},
		evaluate: (target, pred) => {
			return target.every((v, i) => Math.round(pred[i]) === v);
		},
		formatPrediction: (pred) => `[${pred.map((v) => v.toFixed(2)).join(", ")}]`,
	}),
	alternatingRatio: (bits) => ({
		name: "Alternating Ratio",
		outputSize: 1,
		generate: () => {
			const input = generateBits(bits);
			let alternations = 0;
			for (let i = 1; i < input.length; i++) {
				if (input[i] !== input[i - 1]) alternations++;
			}
			const ratio = alternations / (bits - 1);
			const target = [ratio > 0.6 ? 1 : 0]; // target 1 if alternating enough
			return { input, target };
		},
		solve: (input) => {
			let alternations = 0;
			for (let i = 1; i < input.length; i++) {
				if (input[i] !== input[i - 1]) alternations++;
			}
			const ratio = alternations / (input.length - 1);
			return [ratio > 0.6 ? 1 : 0];
		},
		evaluate: (target, pred) => target[0] === Math.round(pred[0]),
		formatPrediction: (pred) => `[${pred.map((v) => v.toFixed(2)).join(", ")}]`,
	}),
	patternSpectrum: (bits) => ({
		name: "Pattern Spectrum",
		outputSize: 3,
		generate: () => {
			const input = generateBits(bits);
			const countPairs = (pattern) =>
				input.reduce((acc, val, i) => {
					if (i < input.length - 1 && input[i] === pattern[0] && input[i+1] === pattern[1]) acc++;
					return acc;
				}, 0);
	
			const p01 = Math.sqrt(countPairs([0, 1]) / (bits - 1));
			const p10 = Math.sqrt(countPairs([1, 0]) / (bits - 1));
			const p11 = Math.sqrt(countPairs([1, 1]) / (bits - 1));
	
			const target = [p01, p10, p11];
			return { input, target };
		},
		solve: (input) => {
			const countPairs = (pattern) =>
				input.reduce((acc, val, i) => {
					if (i < input.length - 1 && input[i] === pattern[0] && input[i+1] === pattern[1]) acc++;
					return acc;
				}, 0);
	
			const p01 = Math.sqrt(countPairs([0, 1]) / (input.length - 1));
			const p10 = Math.sqrt(countPairs([1, 0]) / (input.length - 1));
			const p11 = Math.sqrt(countPairs([1, 1]) / (input.length - 1));
	
			return [p01, p10, p11];
		},
		evaluate: (target, pred) => {
			const THRESHOLD = 0.01;
			return target.every((v, i) => Math.abs(v - pred[i]) < THRESHOLD);
		},
		formatPrediction: (pred) => `[${pred.map((v) => v.toFixed(2)).join(", ")}]`,
	}),
	bitGeometry: (bits) => ({
		name: "Bit Geometry",
		outputSize: 3,
		generate: () => {
			const input = generateBits(bits);
			const target = solveBitGeometry(input, bits);
			return { input, target };
		},
		solve: (input) => solveBitGeometry(input, input.length),
		evaluate: (target, pred) => {
			const THRESHOLD = 0.01;
			return target.every((v, i) => Math.abs(v - Math.min(1, Math.max(0, pred[i]))) < THRESHOLD)
		},
		formatPrediction: (pred) => `[${pred.map((v) => v.toFixed(2)).join(", ")}]`,
	}),
};