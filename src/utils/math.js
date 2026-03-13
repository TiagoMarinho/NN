export const getRandomFloat = (minimum, maximum) => Math.random() * (maximum - minimum) + minimum;
export const getRandomInt = (minimum, maximum) => Math.floor(Math.random() * (maximum - minimum + 1)) + minimum;
export const getRandomBool = (threshold = 0.5) => Math.random() < threshold;
export const getSum = (numbers) => numbers.reduce((total, n) => total + n, 0);
export const clamp = (value, min, max) => Math.min(Math.max(value, min), max);
export const lerp = (start, end, interpolation) => start + (end - start) * interpolation;
export const round = (value, decimals = 2) => {
    const factor = 10 ** decimals;
    return Math.round(value * factor) / factor;
};
export const getRandomBits = (length) => Array.from({ length }, () => getRandomBool() ? 1 : 0);