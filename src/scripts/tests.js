import { TASKS } from "../training/tasks/index.js";
import { schedulers, activations, losses, GradientBuffer, Layer, NeuralNetwork, optimizers } from "../nn/index.js";
import { paint, printHeader, formatStatus, printRow, printTable } from "../utils/log.js";
import { padTrailingZeros } from "../utils/formatting.js";

const ERROR_TOLERANCE = 1e-5;
const LOSS_LOGGING_PRECISION = 6;

const createAssertion = (isConditionMet, errorMessage) => {
	if (!isConditionMet) throw new Error(errorMessage);
};

const assertStrictEqual = (actualValue, expectedValue, context) =>
	createAssertion(actualValue === expectedValue,
		`[StrictEqualMismatch] ${context} | Expected: ${expectedValue} | Actual: ${actualValue}`);

const assertFloatingPointEqual = (actualValue, expectedValue, context) =>
	createAssertion(Math.abs(actualValue - expectedValue) <= ERROR_TOLERANCE,
		`[FloatingPointMismatch] ${context} | Expected: ${expectedValue} | Actual: ${actualValue}`);

const assertLessThan = (actualValue, maximumThreshold, context) =>
	createAssertion(actualValue < maximumThreshold,
		`[LessThanMismatch] ${context} | Expected less than: ${maximumThreshold} | Actual: ${actualValue}`);

const testSchedulers = () => {
	const TOTAL_EPOCHS = 100;
	const RATE_HIGH = 0.1;
	const RATE_LOW = 0.01;
	const SCHEDULE_STEPS = [{ epoch: 0, rate: 0.1 }, { epoch: 50, rate: 0.05 }];

	assertFloatingPointEqual(schedulers.linear(0, TOTAL_EPOCHS, RATE_HIGH, RATE_LOW), RATE_HIGH, "Linear Start");
	assertFloatingPointEqual(schedulers.linear(TOTAL_EPOCHS, TOTAL_EPOCHS, RATE_HIGH, RATE_LOW), RATE_LOW, "Linear End");
	assertFloatingPointEqual(schedulers.constant(50, TOTAL_EPOCHS, 0.05), 0.05, "Constant");
	assertFloatingPointEqual(schedulers.stepped(25, TOTAL_EPOCHS, SCHEDULE_STEPS), 0.1, "Stepped Early");
	assertFloatingPointEqual(schedulers.stepped(75, TOTAL_EPOCHS, SCHEDULE_STEPS), 0.05, "Stepped Late");
	assertFloatingPointEqual(schedulers.cosine(50, TOTAL_EPOCHS, RATE_HIGH, RATE_LOW), 0.055, "Cosine Midpoint");
};

const testActivations = () => {
	const RELU_LEAK_SLOPE = 0.01;

	assertFloatingPointEqual(activations.sigmoid.calculate(0), 0.5, "Sigmoid Calculate");
	assertFloatingPointEqual(activations.relu.calculate(-5), 0, "ReLU Inactive");
	assertFloatingPointEqual(activations.relu.calculate(5), 5, "ReLU Active");
	assertFloatingPointEqual(activations.relu.derivative(5), 1, "ReLU Deriv Active");
	assertFloatingPointEqual(activations.leakyRelu.calculate(-5), -5 * RELU_LEAK_SLOPE, "Leaky Calculate");
};

const testLosses = () => {
	assertFloatingPointEqual(losses.mse.calculate(1, 0.5), 0.25, "MSE Calculate");
	assertFloatingPointEqual(losses.bce.calculate(1, 0.5), 0.693147, "BCE Calculate");
	assertFloatingPointEqual(losses.mse.derivative(1, 0.5), -1.0, "MSE Derivative");
};

const testGradientBuffer = () => {
	const buffer = new GradientBuffer(2, 3);
	buffer.add([0.1, 0.2], [1, 2, 3]);

	assertStrictEqual(buffer.size, 1, "Buffer Size Post-Add");
	assertFloatingPointEqual(buffer.biasGradients[0], 0.1, "Bias Grad 0");
	assertFloatingPointEqual(buffer.weightGradients[5], 0.6, "Weight Grad 5");

	buffer.reset();
	assertStrictEqual(buffer.size, 0, "Buffer Size Post-Reset");
	assertFloatingPointEqual(buffer.biasGradients[0], 0, "Bias Cleaned");
};

const testLayer = () => {
	const layer = new Layer(2, 2, activations.relu, optimizers.sgd());
	const outputs = layer.forward([1, 1]);

	assertStrictEqual(outputs.length, 2, "Forward Output Size");

	const backErrors = layer.backward([0.1, 0.1]);
	assertStrictEqual(backErrors.length, 2, "Backward Error Size");
	assertStrictEqual(layer.buffer.size, 1, "Buffer Tracking");

	layer.step(0.1);
	assertStrictEqual(layer.buffer.size, 0, "Buffer Update Reset");
};

const testNeuralNetwork = () => {
	const network = new NeuralNetwork(2, [4], 1);
	const result = network.predict([1, 0]);

	assertStrictEqual(result.length, 1, "Network Output Size");

	network.backward([1]);
	network.optimize(0.1);
};

const calculateNetworkLoss = (network, dataset) =>
	dataset.reduce((acc, { input, target }) =>
		acc + losses.mse.calculate(target[0], network.predict(input)[0]), 0) / dataset.length;

const regressionResults = [];

const testTaskRegression = (taskName, taskBuilder) => {
	const BITS = 4;
	const EPOCHS = 100;
	const IMPROVEMENT_REQUIRED = 0.9;

	const task = taskBuilder(BITS);
	const network = new NeuralNetwork(BITS, [16], task.outputSize);

	const combinations = Array.from({ length: Math.pow(2, BITS) }, (_, i) => {
		const input = i.toString(2).padStart(BITS, "0").split("").map(Number);
		return { input, target: task.solve(input) };
	});

	const initialLoss = calculateNetworkLoss(network, combinations);

	for (let i = 0; i < EPOCHS; i++) {
		combinations.forEach(({ input, target }) => {
			network.predict(input);
			network.backward(target);
			network.optimize(0.1);
		});
	}

	const finalLoss = calculateNetworkLoss(network, combinations);
	const threshold = initialLoss * IMPROVEMENT_REQUIRED;

	regressionResults.push({
		task:   taskName,
		start:  padTrailingZeros(initialLoss, LOSS_LOGGING_PRECISION),
		end:    padTrailingZeros(finalLoss, LOSS_LOGGING_PRECISION),
		status: formatStatus(finalLoss < threshold),
	});

	assertLessThan(finalLoss, threshold, `${taskName} Convergence`);
};

const runSuite = (label, action) => {
	try {
		action();
		console.log(`${paint("PASS", "green")} ${label}`);
	} catch (e) {
		console.log(`${paint("FAIL", "red")} ${label}`);
		console.error(paint(e.message, "red"));
		process.exitCode = 1;
	}
};

printHeader("Unit Tests: Components");
runSuite("Schedulers",         testSchedulers);
runSuite("Activations",        testActivations);
runSuite("Losses",             testLosses);
runSuite("Gradient Buffer",    testGradientBuffer);
runSuite("Layer Logic",        testLayer);
runSuite("Network Structural", testNeuralNetwork);

printHeader("Integration Tests: Task Regression");
runSuite("Parity Convergence", () => testTaskRegression("Parity", TASKS.parity));

printTable(
	[
		{ label: "Task",   key: "task",   color: "reset",  align: "left"  },
		{ label: "Start",  key: "start",  color: "yellow", align: "right" },
		{ label: "End",    key: "end",    color: "green",  align: "right" },
		{ label: "Status", key: "status", color: "reset",  align: "left"  },
	],
	regressionResults,
);