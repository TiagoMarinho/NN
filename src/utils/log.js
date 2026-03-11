const COLORS = {
	RESET: "\x1b[0m",
	BRIGHT: "\x1b[1m",
	MAGENTA: "\x1b[35m",
	GREEN: "\x1b[32m",
	RED: "\x1b[31m",
	CYAN: "\x1b[36m",
	YELLOW: "\x1b[33m",
};

export const paint = (text, colorName) => {
	const color = COLORS[colorName.toUpperCase()] || COLORS.RESET;
	return `${color}${text}${COLORS.RESET}`;
};

export const printHeader = (text) => {
	console.log(
		`\n${COLORS.BRIGHT}${COLORS.MAGENTA}--- ${text.toUpperCase()} ---${COLORS.RESET}`,
	);
};

export const formatStatus = (isSuccess) => {
	return isSuccess ? paint("SUCCESS", "green") : paint("FAILURE", "red");
};

export const printRow = (columns) => {
	const parts = columns.map(
		({
			label,
			value,
			color = "reset",
			width,
			prefix = "",
			suffix = "",
		}) => {
			const padded = width
				? value.toString().padStart(width)
				: value.toString();
			const colored = paint(padded, color);
			const labelStr = paint(label, "cyan");
			return `${labelStr}: ${prefix}${colored}${suffix}`;
		},
	);
	console.log(parts.join(" | "));
};

export const printTable = (columns, rows) => {
	const widths = columns.map((col) => {
		const valueWidths = rows.map(
			(row) => String(row[col.key] ?? "").length,
		);
		return Math.max(col.label.length, ...valueWidths, col.width ?? 0);
	});

	const header = columns
		.map((col, i) => paint(col.label.padEnd(widths[i]), "cyan"))
		.join(" | ");
	console.log(header);
	console.log(widths.map((w) => "─".repeat(w)).join("─┼─"));

	for (const row of rows) {
		const cells = columns.map((col, i) => {
			const raw = String(row[col.key] ?? "");
			const padded =
				col.align === "right"
					? raw.padStart(widths[i])
					: raw.padEnd(widths[i]);
			return paint(padded, col.color ?? "reset");
		});
		console.log(cells.join(" | "));
	}
};
