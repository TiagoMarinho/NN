export const schedulers = {
	cosine: (epoch, total, start, min) => {
		const progress = epoch / total;
		const scale = 0.5 * (1 + Math.cos(Math.PI * progress));
		return min + (start - min) * scale;
	},
	stepped: (epoch, total, steps) => {
		const pastSteps = steps.filter((step) => epoch >= step.epoch);
		const currentStep = pastSteps.sort((a, b) => b.epoch - a.epoch)[0];
		return currentStep ? currentStep.rate : steps[0].rate;
	},
	linear: (epoch, total, start, min) => {
		return start - (start - min) * (epoch / total);
	},
	constant: (epoch, total, rate) => rate,
};