const EPSILON = 1e-7;

export const losses = {
	mae: {
		calculate: (target, pred) => Math.abs(target - pred),
		derivative: (target, pred) => (pred > target ? 1 : -1),
	},
	mse: {
		calculate: (target, pred) => (target - pred) ** 2,
		derivative: (target, pred) => 2 * (pred - target),
	},
	bce: {
		calculate: (target, pred) => {
			const clipped = Math.min(1 - EPSILON, Math.max(EPSILON, pred));
			return -(target * Math.log(clipped) + (1 - target) * Math.log(1 - clipped));
		},
		derivative: (target, pred) => {
			const clipped = Math.min(1 - EPSILON, Math.max(EPSILON, pred));
			return (clipped - target) / (clipped * (1 - clipped));
		},
	},
};