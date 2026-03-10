import { getSum, getHeWeightScale, getRandomFloat } from "./utils/math.js";
import { printHeader, paint, formatStatus, formatList } from "./utils/log.js";
import { padTrailingZeros, formatPercentage } from "./utils/formatting.js";

const TASKS = {
    parity: (bits) => ({
        name: "Parity",
        generate: () => {
            const input = Array.from({ length: bits }, () => Math.random() > 0.5 ? 1 : 0);
            return { input, target: [getSum(input) % 2 === 0 ? 1 : 0] };
        },
        solve: (input) => [getSum(input) % 2 === 0 ? 1 : 0]
    }),
    majority: (bits) => ({
        name: "Majority",
        generate: () => {
            const input = Array.from({ length: bits }, () => Math.random() > 0.5 ? 1 : 0);
            return { input, target: [getSum(input) > bits / 2 ? 1 : 0] };
        },
        solve: (input) => [getSum(input) > bits / 2 ? 1 : 0]
    }),
    xor: (bits) => ({
        name: "XOR",
        generate: () => {
            const input = Array.from({ length: bits }, () => Math.random() > 0.5 ? 1 : 0);
            return { input, target: [input.reduce((acc, val) => acc ^ val, 0)] };
        },
        solve: (input) => [input.reduce((acc, val) => acc ^ val, 0)]
    }),
    and: (bits) => ({
        name: "AND",
        generate: () => {
            const input = Array.from({ length: bits }, () => Math.random() > 0.5 ? 1 : 0);
            return { input, target: [input.every(v => v === 1) ? 1 : 0] };
        },
        solve: (input) => [input.every(v => v === 1) ? 1 : 0]
    })
};

const BITS = 4;
const CONFIGURATION = {
    BITS,
    TASK: TASKS.xor(BITS),
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
    mae: { calculate: (t, p) => Math.abs(t - p), derivative: (t, p) => (t > p ? 1 : -1) },
    mse: { calculate: (t, p) => Math.pow(t - p, 2), derivative: (t, p) => t - p }
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
        const scale = getHeWeightScale(inputSize);
        this.weights = Array.from({ length: outputSize }, () =>
            Array.from({ length: inputSize }, () => getRandomFloat(-1, 1) * scale)
        );
        this.biases = Array(outputSize).fill(0.01);
        this.buffer = new GradientBuffer(outputSize, inputSize);
        this.inputCache = null;
        this.outputCache = null;
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
            return new Layer(size, sizes[i + 1], isLast ? activations.sigmoid : activations.relu);
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

function train(network) {
    const { TASK, TOTAL_EPOCHS, SAMPLES_PER_EPOCH, BATCH_SIZE, LEARNING_RATE_START, LEARNING_RATE_MINIMUM, LOG_FREQUENCY } = CONFIGURATION;
    printHeader(`Training: ${TASK.name}`);

    for (let epoch = 1; epoch <= TOTAL_EPOCHS; epoch++) {
        let maeAcc = 0, mseAcc = 0;
        const progress = epoch / TOTAL_EPOCHS;
        const rate = LEARNING_RATE_MINIMUM + (LEARNING_RATE_START - LEARNING_RATE_MINIMUM) * 0.5 * (1 + Math.cos(Math.PI * progress));

        for (let s = 1; s <= SAMPLES_PER_EPOCH; s++) {
            const { input, target } = TASK.generate();
            const prediction = network.predict(input);

            maeAcc += losses.mae.calculate(target[0], prediction[0]);
            mseAcc += losses.mse.calculate(target[0], prediction[0]);

            network.backward(target);
            if (s % BATCH_SIZE === 0) network.optimize(rate);
        }

        if (epoch % LOG_FREQUENCY === 0 || epoch === TOTAL_EPOCHS) {
            console.log(
                `${formatList("Epoch", epoch.toString().padStart(4), "cyan")} (${formatPercentage(progress, 0).padStart(4)}) | ` +
                `${formatList("MAE", padTrailingZeros(maeAcc / SAMPLES_PER_EPOCH, 5), "yellow")} | ` +
                `${formatList("MSE", padTrailingZeros(mseAcc / SAMPLES_PER_EPOCH, 5), "magenta")} | ` +
                `${formatList("Rate", padTrailingZeros(rate, 4), "reset")}`
            );
        }
    }
}

function evaluate(network) {
    printHeader("Final Evaluation");
    const { TASK, BITS } = CONFIGURATION;
    
    const totalPossible = Math.pow(2, BITS);
    const testSize = Math.min(totalPossible, 16); 
    
    let score = 0;

    for (let i = 0; i < testSize; i++) {
        const binaryString = i.toString(2).padStart(BITS, "0");
        const input = binaryString.split("").map(Number);
        
        const target = TASK.solve(input)[0];
        const raw = network.predict(input)[0];
        const prediction = raw > 0.5 ? 1 : 0;

        if (prediction === target) score++;
        
        const certainty = paint(formatPercentage(Math.abs(raw - 0.5) * 2, 1), "yellow");
        const inputLabel = formatList("Input", `[${binaryString}]`, "reset");
        
        console.log(
            `${inputLabel} | ${formatStatus(prediction === target)} | ` +
            `Target: ${target} | Predict: ${prediction} | ` +
            `Raw: ${padTrailingZeros(raw, 2)} (${certainty})`
        );
    }

    const finalScore = paint(`${score} / ${testSize}`, score === testSize ? "green" : "red");
    console.log(`\nFinal Accuracy: ${finalScore}`);
}

const network = new NeuralNetwork(CONFIGURATION.BITS, CONFIGURATION.HIDDEN_LAYERS, 1);
train(network);
evaluate(network);