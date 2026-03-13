import { checkpointPath, latestCheckpointPath } from "../io/checkpointStore.js";
import { saveSnapshot, loadSnapshot } from "../io/serialize.js";
import { toSnapshot, fromSnapshot } from "../nn/snapshot.js";

export const saveCheckpoint = (network, epoch, dir) =>
	saveSnapshot(toSnapshot(network, epoch), checkpointPath(dir, epoch));

export const loadCheckpoint = (filePath) => {
	const snapshot = loadSnapshot(filePath);
	return { network: fromSnapshot(snapshot), epoch: snapshot.epoch };
};

export const loadLatestCheckpoint = (dir) =>
	loadCheckpoint(latestCheckpointPath(dir));