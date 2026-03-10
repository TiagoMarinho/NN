export const padTrailingZeros = (value, decimals = 2) => {
	return value.toFixed(decimals);
};

export const formatPercentage = (decimal, decimals = 1) => {
	return (decimal * 100).toFixed(decimals) + "%";
};

export const padText = (text, length, char = " ") => {
	return text.toString().padEnd(length, char);
};
