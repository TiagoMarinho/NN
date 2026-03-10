const COLORS = {
    RESET: "\x1b[0m",
    BRIGHT: "\x1b[1m",
    MAGENTA: "\x1b[35m",
    GREEN: "\x1b[32m",
    RED: "\x1b[31m",
    CYAN: "\x1b[36m",
    YELLOW: "\x1b[33m"
};

export const paint = (text, colorName) => {
    const color = COLORS[colorName.toUpperCase()] || COLORS.RESET;
    return `${color}${text}${COLORS.RESET}`;
};

export const printHeader = (text) => {
    console.log(`\n${COLORS.BRIGHT}${COLORS.MAGENTA}--- ${text.toUpperCase()} ---${COLORS.RESET}`);
};

export const formatStatus = (isSuccess) => {
    return isSuccess ? paint("SUCCESS", "green") : paint("FAILURE", "red");
};

export const formatList = (label, value, color) => {
    return `${paint(label, "cyan")}: ${paint(value, color)}`;
};