const LEAKY_RELU_SLOPE = 0.01;

export const activations = {
	linear: {
		calculate: (x) => x,
		derivative: (y) => 1,
	},
	sigmoid: {
		calculate: (x) => 1 / (1 + Math.exp(-x)),
		derivative: (y) => y * (1 - y),
	},
	relu: {
		calculate: (x) => Math.max(0, x),
		derivative: (y) => (y > 0 ? 1 : 0),
	},
	leakyRelu: {
		calculate: (x) => (x > 0 ? x : x * LEAKY_RELU_SLOPE),
		derivative: (y) => (y > 0 ? 1 : LEAKY_RELU_SLOPE),
	},
};