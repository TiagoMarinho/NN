import { TASKS } from "./tasks.js";
import {
	schedulers,
	activations,
	losses,
	GradientBuffer,
	Layer,
	NeuralNetwork,
} from "./neural_network.js";

import {
	paint,
	printHeader,
	formatStatus,
	printRow,
	printTable,
} from "./utils/log.js";

import { padTrailingZeros } from "./utils/formatting.js";

const ERROR_TOLERANCE = 1e-5;
const LOSS_LOGGING_PRECISION = 6;

const createAssertion = (isConditionMet, errorMessage) => {
	if (!isConditionMet) {
		throw new Error(errorMessage);
	}
};

const assertStrictEqual = (actualValue, expectedValue, context) => {
	const isConditionMet = actualValue === expectedValue;
	const errorMessage = `[StrictEqualMismatch] ${context} | Expected: ${expectedValue} | Actual: ${actualValue}`;
	createAssertion(isConditionMet, errorMessage);
};

const assertFloatingPointEqual = (actualValue, expectedValue, context) => {
	const absoluteDifference = Math.abs(actualValue - expectedValue);
	const isConditionMet = absoluteDifference <= ERROR_TOLERANCE;
	const errorMessage = `[FloatingPointMismatch] ${context} | Expected: ${expectedValue} | Actual: ${actualValue}`;
	createAssertion(isConditionMet, errorMessage);
};

const assertLessThan = (actualValue, maximumThreshold, context) => {
	const isConditionMet = actualValue < maximumThreshold;
	const errorMessage = `[LessThanMismatch] ${context} | Expected less than: ${maximumThreshold} | Actual: ${actualValue}`;
	createAssertion(isConditionMet, errorMessage);
};

const testSchedulers = () => {
	const EPOCH_START = 0;
	const EPOCH_QUARTER = 25;
	const EPOCH_HALF = 50;
	const EPOCH_THREE_QUARTERS = 75;
	const TOTAL_EPOCHS = 100;

	const RATE_HIGH = 0.1;
	const RATE_LOW = 0.01;
	const RATE_CONSTANT = 0.05;
	const RATE_COSINE_MID = 0.055;

	const STEP_ONE_EPOCH = 0;
	const STEP_ONE_RATE = 0.1;
	const STEP_TWO_EPOCH = 50;
	const STEP_TWO_RATE = 0.05;

	const SCHEDULE_STEPS = [
		{ epoch: STEP_ONE_EPOCH, rate: STEP_ONE_RATE },
		{ epoch: STEP_TWO_EPOCH, rate: STEP_TWO_RATE },
	];

	assertFloatingPointEqual(
		schedulers.linear(EPOCH_START, TOTAL_EPOCHS, RATE_HIGH, RATE_LOW),
		RATE_HIGH,
		"Linear Start",
	);
	assertFloatingPointEqual(
		schedulers.linear(TOTAL_EPOCHS, TOTAL_EPOCHS, RATE_HIGH, RATE_LOW),
		RATE_LOW,
		"Linear End",
	);
	assertFloatingPointEqual(
		schedulers.constant(EPOCH_HALF, TOTAL_EPOCHS, RATE_CONSTANT),
		RATE_CONSTANT,
		"Constant",
	);
	assertFloatingPointEqual(
		schedulers.stepped(EPOCH_QUARTER, TOTAL_EPOCHS, SCHEDULE_STEPS),
		STEP_ONE_RATE,
		"Stepped Early",
	);
	assertFloatingPointEqual(
		schedulers.stepped(EPOCH_THREE_QUARTERS, TOTAL_EPOCHS, SCHEDULE_STEPS),
		STEP_TWO_RATE,
		"Stepped Late",
	);
	assertFloatingPointEqual(
		schedulers.cosine(EPOCH_HALF, TOTAL_EPOCHS, RATE_HIGH, RATE_LOW),
		RATE_COSINE_MID,
		"Cosine Midpoint",
	);
};

const testActivations = () => {
	const INPUT_ZERO = 0;
	const INPUT_POSITIVE = 5;
	const INPUT_NEGATIVE = -5;

	const SIGMOID_ZERO_OUTPUT = 0.5;
	const RELU_LEAK_SLOPE = 0.01;
	const RELU_DERIVATIVE_ACTIVE = 1;

	assertFloatingPointEqual(
		activations.sigmoid.calculate(INPUT_ZERO),
		SIGMOID_ZERO_OUTPUT,
		"Sigmoid Calculate",
	);
	assertFloatingPointEqual(
		activations.relu.calculate(INPUT_NEGATIVE),
		0,
		"ReLU Inactive",
	);
	assertFloatingPointEqual(
		activations.relu.calculate(INPUT_POSITIVE),
		INPUT_POSITIVE,
		"ReLU Active",
	);
	assertFloatingPointEqual(
		activations.relu.derivative(INPUT_POSITIVE),
		RELU_DERIVATIVE_ACTIVE,
		"ReLU Deriv Active",
	);
	assertFloatingPointEqual(
		activations.leakyRelu.calculate(INPUT_NEGATIVE),
		INPUT_NEGATIVE * RELU_LEAK_SLOPE,
		"Leaky Calculate",
	);
};

const testLosses = () => {
	const TARGET = 1;
	const PREDICTION = 0.5;
	const EXPECTED_MSE = 0.25;
	const EXPECTED_BCE = 0.693147;
	const EXPECTED_MSE_DERIVATIVE = -0.5;

	assertFloatingPointEqual(
		losses.mse.calculate(TARGET, PREDICTION),
		EXPECTED_MSE,
		"MSE Calculate",
	);
	assertFloatingPointEqual(
		losses.bce.calculate(TARGET, PREDICTION),
		EXPECTED_BCE,
		"BCE Calculate",
	);
	assertFloatingPointEqual(
		losses.mse.derivative(TARGET, PREDICTION),
		EXPECTED_MSE_DERIVATIVE,
		"MSE Derivative",
	);
};

const testGradientBuffer = () => {
	const OUTPUTS = 2;
	const INPUTS = 3;
	const DELTAS = [0.1, 0.2];
	const FEATURES = [1, 2, 3];

	const buffer = new GradientBuffer(OUTPUTS, INPUTS);
	buffer.add(DELTAS, FEATURES);

	assertStrictEqual(buffer.size, 1, "Buffer Size Post-Add");
	assertFloatingPointEqual(buffer.biasGradients[0], 0.1, "Bias Grad 0");
	assertFloatingPointEqual(buffer.weightGradients[5], 0.6, "Weight Grad 5");

	buffer.reset();
	assertStrictEqual(buffer.size, 0, "Buffer Size Post-Reset");
	assertFloatingPointEqual(buffer.biasGradients[0], 0, "Bias Cleaned");
};

const testLayer = () => {
	const INPUT_DIM = 2;
	const OUTPUT_DIM = 2;
	const INPUT_DATA = [1, 1];
	const ERROR_DATA = [0.1, 0.1];
	const RATE = 0.1;

	const layer = new Layer(INPUT_DIM, OUTPUT_DIM, activations.relu);
	const outputs = layer.forward(INPUT_DATA);

	assertStrictEqual(outputs.length, OUTPUT_DIM, "Forward Output Size");

	const backErrors = layer.backward(ERROR_DATA);
	assertStrictEqual(backErrors.length, INPUT_DIM, "Backward Error Size");
	assertStrictEqual(layer.buffer.size, 1, "Buffer Tracking");

	layer.update(RATE);
	assertStrictEqual(layer.buffer.size, 0, "Buffer Update Reset");
};

const testNeuralNetwork = () => {
	const INPUTS = 2;
	const HIDDEN = [4];
	const OUTPUTS = 1;

	const network = new NeuralNetwork(INPUTS, HIDDEN, OUTPUTS);
	const result = network.predict([1, 0]);

	assertStrictEqual(result.length, OUTPUTS, "Network Output Size");

	network.backward([1]);
	network.optimize(0.1, 1);
};

const calculateNetworkLoss = (network, dataset) => {
	const total = dataset.reduce((acc, { input, target }) => {
		const pred = network.predict(input)[0];
		return acc + losses.mse.calculate(target[0], pred);
	}, 0);
	return total / dataset.length;
};

const regressionResults = [];

const testTaskRegression = (taskName, taskBuilder) => {
	const BITS = 4;
	const EPOCHS = 100;
	const RATE = 0.2;
	const IMPROVEMENT_REQUIRED = 0.9;

	const task = taskBuilder(BITS);
	const network = new NeuralNetwork(BITS, [16], 1);

	const combinations = Array.from({ length: Math.pow(2, BITS) }, (_, i) => {
		const input = i.toString(2).padStart(BITS, "0").split("").map(Number);
		return { input, target: task.solve(input) };
	});

	const initialLoss = calculateNetworkLoss(network, combinations);

	for (let i = 0; i < EPOCHS; i++) {
		combinations.forEach(({ input, target }) => {
			network.predict(input);
			network.backward(target);
			network.optimize(RATE, 1);
		});
	}

	const finalLoss = calculateNetworkLoss(network, combinations);
	const threshold = initialLoss * IMPROVEMENT_REQUIRED;
	const hasConverged = finalLoss < threshold;

	regressionResults.push({
		task: taskName,
		start: padTrailingZeros(initialLoss, LOSS_LOGGING_PRECISION),
		end: padTrailingZeros(finalLoss, LOSS_LOGGING_PRECISION),
		status: formatStatus(hasConverged),
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

const startTesting = () => {
	printHeader("Unit Tests: Components");
	runSuite("Schedulers", testSchedulers);
	runSuite("Activations", testActivations);
	runSuite("Losses", testLosses);
	runSuite("Gradient Buffer", testGradientBuffer);
	runSuite("Layer Logic", testLayer);
	runSuite("Network Structural", testNeuralNetwork);

	printHeader("Integration Tests: Task Regression");
	runSuite("Parity Convergence", () =>
		testTaskRegression("Parity", TASKS.parity),
	);
	runSuite("Majority Convergence", () =>
		testTaskRegression("Majority", TASKS.majority),
	);
	runSuite("XOR Convergence", () => testTaskRegression("XOR", TASKS.xor));
	runSuite("AND Convergence", () => testTaskRegression("AND", TASKS.and));

	printTable(
		[
			{ label: "Task", key: "task", color: "reset", align: "left" },
			{ label: "Start", key: "start", color: "yellow", align: "right" },
			{ label: "End", key: "end", color: "green", align: "right" },
			{ label: "Status", key: "status", color: "reset", align: "left" },
		],
		regressionResults,
	);
};

startTesting();
