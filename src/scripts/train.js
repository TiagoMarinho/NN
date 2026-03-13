import { NeuralNetwork } from "../nn/index.js";
import { train } from "../training/train.js";
import { loadCheckpoint, loadLatestCheckpoint } from "../training/checkpoint.js";
import { CONFIG } from "../config.js";
import { parseArgs } from "../utils/args.js";
import { loadSnapshot } from "../io/serialize.js";

const args = parseArgs(process.argv);

const resolveConfig = () => {
	const fileOverrides = args.config ? loadSnapshot(args.config) : {};
	const cliOverrides  = {
		...(args.epochs && { TOTAL_EPOCHS: parseInt(args.epochs)      }),
		...(args["batch-size"] && { BATCH_SIZE:parseInt(args["batch-size"]) }),
		...(args.samples && { SAMPLES_PER_EPOCH: parseInt(args.samples)     }),
	};
	return { ...CONFIG, ...fileOverrides, ...cliOverrides };
};

const fromScratch = (config) => ({
	network: new NeuralNetwork(config.BITS, config.HIDDEN_LAYERS, config.TASK.outputSize, config.HIDDEN_ACTIVATION, config.OUTPUT_ACTIVATION, config.OPTIMIZER),
	startEpoch: 1,
});

const fromCheckpoint = ({ network, epoch }) => ({ network, startEpoch: epoch + 1 });

const resolveStartingPoint = (config) => {
	if (!("checkpoint" in args)) return fromScratch(config);
	return fromCheckpoint(
		typeof args.checkpoint === "string"
			? loadCheckpoint(args.checkpoint)
			: loadLatestCheckpoint(config.CHECKPOINT.dir)
	);
};

const config = resolveConfig();
const { network, startEpoch } = resolveStartingPoint(config);

train(network, { ...config, START_EPOCH: startEpoch });