import path from "path";
import fs from "fs";

const CHECKPOINT_PREFIX = "checkpoint_epoch_";

const toFilename = (epoch) => `${CHECKPOINT_PREFIX}${String(epoch).padStart(6, "0")}.json`;

export const checkpointPath = (dir, epoch) => path.join(dir, toFilename(epoch));

export const latestCheckpointPath = (dir) => {
	const files = fs.readdirSync(dir).filter((f) => f.startsWith(CHECKPOINT_PREFIX)).sort();
	if (files.length === 0) throw new Error(`No checkpoints found in: ${dir}`);
	return path.join(dir, files[files.length - 1]);
};