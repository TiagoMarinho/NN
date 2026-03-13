export const parseArgs = (argv) => {
	const args = {};
	for (let i = 2; i < argv.length; i++) {
		if (!argv[i].startsWith("--")) continue;
		const key  = argv[i].slice(2);
		const next = argv[i + 1];
		const hasValue = next && !next.startsWith("--");
		args[key] = hasValue ? (i++, next) : true;
	}
	return args;
};