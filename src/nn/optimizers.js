export const optimizers = {

	sgd: () => ({
		name: "sgd",
		params: {},
		initState: () => null,
		step(weights, biases, weightGrads, biasGrads, batchSize, learningRate, _state) {
			const s = learningRate / batchSize;
			for (let i = 0; i < weights.length; i++) weights[i] -= s * weightGrads[i];
			for (let i = 0; i < biases.length; i++) biases[i] -= s * biasGrads[i];
		},
	}),

	adam: ({ beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 } = {}) => ({
		name: "adam",
		params: { beta1, beta2, epsilon },
		initState: (weightSize, biasSize) => ({
			t:  0,
			mW: new Float32Array(weightSize),
			vW: new Float32Array(weightSize),
			mB: new Float32Array(biasSize),
			vB: new Float32Array(biasSize),
		}),
		step(weights, biases, weightGrads, biasGrads, batchSize, learningRate, state) {
			state.t++;
			const correctedLR = learningRate * Math.sqrt(1 - beta2 ** state.t) / (1 - beta1 ** state.t);

			for (let i = 0; i < weights.length; i++) {
				const g = weightGrads[i] / batchSize;
				state.mW[i] = beta1 * state.mW[i] + (1 - beta1) * g;
				state.vW[i] = beta2 * state.vW[i] + (1 - beta2) * g * g;
				weights[i] -= correctedLR * state.mW[i] / (Math.sqrt(state.vW[i]) + epsilon);
			}

			for (let i = 0; i < biases.length; i++) {
				const g = biasGrads[i] / batchSize;
				state.mB[i] = beta1 * state.mB[i] + (1 - beta1) * g;
				state.vB[i] = beta2 * state.vB[i] + (1 - beta2) * g * g;
				biases[i] -= correctedLR * state.mB[i] / (Math.sqrt(state.vB[i]) + epsilon);
			}
		},
	}),

};