import path from "path";
import { toSnapshot, fromSnapshot } from "../nn/snapshot.js";
import { saveSnapshot, loadSnapshot } from "../io/serialize.js";

const toFilename = (epoch) => `checkpoint_epoch_${String(epoch).padStart(6, "0")}.json`;

export const saveCheckpoint = (network, epoch, dir) =>
	saveSnapshot(toSnapshot(network, epoch), path.join(dir, toFilename(epoch)));

export const loadCheckpoint = (filePath) => {
	const snapshot = loadSnapshot(filePath);
	return { network: fromSnapshot(snapshot), epoch: snapshot.epoch };
};