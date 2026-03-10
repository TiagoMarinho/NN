import { getSum, getHeWeightScale, getRandomFloat } from "./utils/math.js";
import { printHeader, paint, formatStatus, formatList } from "./utils/log.js";
import { padTrailingZeros, formatPercentage } from "./utils/formatting.js";

const CONFIGURATION = {
    BIT_COUNT: 8,
    HIDDEN_LAYERS: [32, 16],
    TOTAL_EPOCHS: 1000,
    SAMPLES_PER_EPOCH: 1000,
    BATCH_SIZE: 1,
    LEARNING_RATE_START: 0.05,
    LEARNING_RATE_MINIMUM: 0.001,
    LOG_FREQUENCY: 50,
};

const activations = {
    sigmoid: {
        calculate: (x) => 1 / (1 + Math.exp(-x)),
        derivative: (y) => y * (1 - y),
    },
    relu: {
        calculate: (x) => Math.max(0, x),
        derivative: (y) => (y > 0 ? 1 : 0),
    },
};

const losses = {
    mae: {
        calculate: (t, p) => Math.abs(t - p),
        derivative: (t, p) => (t > p ? 1 : -1),
    },
    mse: {
        calculate: (t, p) => Math.pow(t - p, 2),
        derivative: (t, p) => t - p,
    },
};

class GradientBuffer {
    constructor(outputSize, inputSize) {
        this.weightGradients = Array.from({ length: outputSize }, () => Array(inputSize).fill(0));
        this.biasGradients = Array(outputSize).fill(0);
        this.size = 0;
    }

    add(gradients, inputs) {
        for (let i = 0; i < this.weightGradients.length; i++) {
            for (let j = 0; j < this.weightGradients[i].length; j++) {
                this.weightGradients[i][j] += gradients[i] * inputs[j];
            }
            this.biasGradients[i] += gradients[i];
        }
        this.size++;
    }

    reset() {
        this.weightGradients.forEach(row => row.fill(0));
        this.biasGradients.fill(0);
        this.size = 0;
    }
}

class Layer {
    constructor(inputSize, outputSize, activation) {
        this.activation = activation;
        this.weights = this.#initializeWeights(inputSize, outputSize);
        this.biases = Array(outputSize).fill(0.01);
        this.buffer = new GradientBuffer(outputSize, inputSize);
        
        this.inputCache = null;
        this.outputCache = null;
    }

    #initializeWeights(inputSize, outputSize) {
        const scale = getHeWeightScale(inputSize);
        return Array.from({ length: outputSize }, () =>
            Array.from({ length: inputSize }, () => getRandomFloat(-1, 1) * scale)
        );
    }

    forward(inputs) {
        this.inputCache = inputs;
        this.outputCache = this.weights.map((row, i) => {
            const sum = row.reduce((acc, w, j) => acc + w * inputs[j], this.biases[i]);
            return this.activation.calculate(sum);
        });
        return this.outputCache;
    }

    backward(errors) {
        const deltas = errors.map((err, i) => err * this.activation.derivative(this.outputCache[i]));
        const nextErrors = Array(this.inputCache.length).fill(0);

        for (let i = 0; i < this.weights.length; i++) {
            for (let j = 0; j < this.weights[i].length; j++) {
                nextErrors[j] += this.weights[i][j] * deltas[i];
            }
        }

        this.buffer.add(deltas, this.inputCache);
        return nextErrors;
    }

    update(learningRate) {
        if (this.buffer.size === 0) return;

        const step = learningRate / this.buffer.size;

        for (let i = 0; i < this.weights.length; i++) {
            for (let j = 0; j < this.weights[i].length; j++) {
                this.weights[i][j] += step * this.buffer.weightGradients[i][j];
            }
            this.biases[i] += step * this.buffer.biasGradients[i];
        }
        this.buffer.reset();
    }
}

class NeuralNetwork {
    constructor(inputSize, hiddenLayers, outputSize) {
        const sizes = [inputSize, ...hiddenLayers, outputSize];
        this.layers = sizes.slice(0, -1).map((size, i) => {
            const isLast = i === sizes.length - 2;
            const activation = isLast ? activations.sigmoid : activations.relu;
            return new Layer(size, sizes[i + 1], activation);
        });
    }

    predict(inputs) {
        return this.layers.reduce((current, layer) => layer.forward(current), inputs);
    }

    backward(targets, lossType = losses.mse) {
        const outputs = this.layers[this.layers.length - 1].outputCache;
        let errors = targets.map((t, i) => lossType.derivative(t, outputs[i]));

        for (let i = this.layers.length - 1; i >= 0; i--) {
            errors = this.layers[i].backward(errors);
        }
    }

    optimize(learningRate) {
        this.layers.forEach(layer => layer.update(learningRate));
    }
}

const parityTask = {
    generateInput: (bits) => Array.from({ length: bits }, () => (Math.random() > 0.5 ? 1 : 0)),
    getOutput: (input) => [getSum(input) % 2 === 0 ? 1 : 0]
};

function train(network) {
    printHeader("Training Session");

    for (let epoch = 1; epoch <= CONFIGURATION.TOTAL_EPOCHS; epoch++) {
        let maeAccumulator = 0;
        let mseAccumulator = 0;

        const learningRate = (function calculateRate() {
            const range = CONFIGURATION.LEARNING_RATE_START - CONFIGURATION.LEARNING_RATE_MINIMUM;
            const progress = epoch / CONFIGURATION.TOTAL_EPOCHS;
            return CONFIGURATION.LEARNING_RATE_MINIMUM + range * 0.5 * (1 + Math.cos(Math.PI * progress));
        })();

        for (let s = 1; s <= CONFIGURATION.SAMPLES_PER_EPOCH; s++) {
            const input = parityTask.generateInput(CONFIGURATION.BIT_COUNT);
            const target = parityTask.getOutput(input);
            const prediction = network.predict(input);

            maeAccumulator += losses.mae.calculate(target[0], prediction[0]);
            mseAccumulator += losses.mse.calculate(target[0], prediction[0]);

            network.backward(target);

            if (s % CONFIGURATION.BATCH_SIZE === 0) {
                network.optimize(learningRate);
            }
        }

        if (epoch % CONFIGURATION.LOG_FREQUENCY === 0 || epoch === CONFIGURATION.TOTAL_EPOCHS) {
            const logData = {
                epoch: epoch.toString().padStart(4),
                progress: formatPercentage(epoch / CONFIGURATION.TOTAL_EPOCHS, 0).padStart(4),
                mae: padTrailingZeros(maeAccumulator / CONFIGURATION.SAMPLES_PER_EPOCH, 5),
                mse: padTrailingZeros(mseAccumulator / CONFIGURATION.SAMPLES_PER_EPOCH, 5),
                rate: padTrailingZeros(learningRate, 4)
            };

            console.log(
                `${formatList("Epoch", logData.epoch, "cyan")} (${logData.progress}) | ` +
                `${formatList("MAE", logData.mae, "yellow")} | ` +
                `${formatList("MSE", logData.mse, "magenta")} | ` +
                `${formatList("Rate", logData.rate, "reset")}`
            );
        }
    }
}

function evaluate(network) {
    printHeader("Final Evaluation");
    let score = 0;
    const size = 16;

    for (let i = 0; i < size; i++) {
        const input = i.toString(2).padStart(CONFIGURATION.BIT_COUNT, "0").split("").map(Number);
        const target = parityTask.getOutput(input)[0];
        const raw = network.predict(input)[0];
        const prediction = raw > 0.5 ? 1 : 0;

        if (prediction === target) score++;

        const certainty = paint(formatPercentage(Math.abs(raw - 0.5) * 2), "yellow");
        const details = `Target: ${target} | Prediction: ${prediction} | Raw: ${padTrailingZeros(raw, 2)} (${certainty})`;
        
        console.log(`${formatList("Input", `[${input.join("")}]`, "reset")} | ${formatStatus(prediction === target)} | ${details}`);
    }

    console.log(`\nFinal Accuracy: ${paint(`${score} / ${size}`, score === size ? "green" : "red")}`);
}

const network = new NeuralNetwork(CONFIGURATION.BIT_COUNT, CONFIGURATION.HIDDEN_LAYERS, 1);
train(network);
evaluate(network);