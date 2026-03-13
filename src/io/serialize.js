import fs from "fs";
import path from "path";

export const saveSnapshot = (snapshot, filePath) => {
	fs.mkdirSync(path.dirname(filePath), { recursive: true });
	fs.writeFileSync(filePath, JSON.stringify(snapshot, null, 2), "utf8");
};

export const loadSnapshot = (filePath) =>
	JSON.parse(fs.readFileSync(filePath, "utf8"));