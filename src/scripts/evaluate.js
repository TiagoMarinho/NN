import { evaluate } from "../training/evaluate.js";
import { loadCheckpoint, loadLatestCheckpoint } from "../training/checkpoint.js";
import { CONFIG } from "../config.js";
import { parseArgs } from "../utils/args.js";
import { loadSnapshot } from "../io/serialize.js";

const args = parseArgs(process.argv);

if (!("checkpoint" in args)) throw new Error("Usage: evaluate.js --checkpoint [path]");

const { network } = typeof args.checkpoint === "string"
	? loadCheckpoint(args.checkpoint)
	: loadLatestCheckpoint(CONFIG.CHECKPOINT.dir);

const inputs = args.inputs
	? JSON.parse(loadSnapshot(args.inputs))
	: args.input
		? [args.input]
		: null;

const count = args.count ? parseInt(args.count) : undefined;

evaluate(network, CONFIG, { inputs, count });