/**
 * network.ts
 * ==========
 *
 * This file is a from-scratch TypeScript re-implementation of the exact
 * same math used to train the neural network in the companion Python
 * project (`mnist-neural-network`). There is no machine-learning
 * library here either -- just plain arrays and arithmetic, running
 * entirely in your browser.
 *
 * Why re-implement it at all, instead of just calling an API?
 * --------------------------------------------------------------
 * The network only needed to *learn* once -- that happened offline, in
 * Python, over many passes through 50,000 training images. Once
 * learning is done, all the network's "knowledge" boils down to a
 * fairly small list of numbers: the weights and biases described in the
 * training project's README. Using that trained network afterwards (to
 * make a single prediction) is just a handful of basic arithmetic
 * operations -- cheap enough to run instantly on any device.
 *
 * That means this web app doesn't need a backend server at all: the
 * trained numbers are bundled into the page (`public/model/trained_network.json`),
 * and every prediction happens right there in your browser, instantly
 * and privately -- your uploaded image is never sent anywhere.
 */

/** The exact shape the trained model is saved in by `Network.save()` in Python. */
export interface TrainedModel {
  /** Number of neurons in each layer, e.g. [784, 100, 100, 100, 10]. */
  sizes: number[];
  /** One weight matrix per layer-to-layer connection. */
  weights: number[][][];
  /** One bias column per layer (excluding the input layer). */
  biases: number[][][];
}

/**
 * The sigmoid ("S-shaped") activation function: squashes any real number
 * into the range (0, 1). Every neuron in this network uses it to decide
 * how strongly to "fire" in response to its inputs. See the Python
 * project's `src/activations.py` for a fuller explanation -- the formula
 * here is identical, just written in TypeScript.
 */
function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

/**
 * A from-scratch, trained-weights-only neural network, matching the
 * architecture and math of the Python `Network` class exactly. This
 * class cannot learn (there's no training code here, on purpose) -- it
 * can only make predictions using weights and biases that were already
 * learned offline.
 */
export class NeuralNetwork {
  private readonly weights: number[][][];
  private readonly biases: number[][][];

  constructor(model: TrainedModel) {
    this.weights = model.weights;
    this.biases = model.biases;
  }

  /**
   * Fetch the trained model file and build a ready-to-use network from
   * it. This is the normal way to get a `NeuralNetwork` instance in the
   * app: `const net = await NeuralNetwork.loadFromUrl("/model/trained_network.json")`.
   */
  static async loadFromUrl(url: string): Promise<NeuralNetwork> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load trained model from ${url}`);
    }
    const model: TrainedModel = await response.json();
    return new NeuralNetwork(model);
  }

  /**
   * Run one 784-pixel image through the whole network and return its
   * raw output: 10 numbers between 0 and 1, one per digit (0-9), where a
   * larger number means the network is more confident the image shows
   * that digit.
   *
   * This mirrors `Network.feedforward` in the Python project line for
   * line, just written without NumPy's matrix shortcuts. Each layer:
   *
   * 1. Computes every neuron's "weighted sum": multiply every input
   *    coming into a neuron by that connection's learned weight, add
   *    them all up, then add the neuron's own learned bias.
   * 2. Squashes that weighted sum through `sigmoid` to get the neuron's
   *    final output.
   * 3. Passes that whole layer's outputs forward as the next layer's
   *    inputs.
   *
   * @param pixels A flat array of 784 numbers, each between 0 (black)
   *   and 1 (white) -- one per pixel of a 28x28 image, read left-to-right,
   *   top-to-bottom, exactly like the images the network trained on.
   */
  predict(pixels: number[]): number[] {
    // `activation` starts as the input image, and is replaced with each
    // layer's output as we move forward through the network -- exactly
    // like the Python version's `a` variable in `feedforward`.
    let activation: number[] = pixels;

    for (let layer = 0; layer < this.weights.length; layer++) {
      const layerWeights = this.weights[layer]; // shape: [neurons in this layer][neurons in previous layer]
      const layerBiases = this.biases[layer]; // shape: [neurons in this layer][1]
      const nextActivation: number[] = new Array(layerWeights.length);

      // For every neuron in this layer...
      for (let neuron = 0; neuron < layerWeights.length; neuron++) {
        let weightedSum = layerBiases[neuron][0];
        const neuronWeights = layerWeights[neuron];

        // ...add up "previous layer's output times the connecting weight"
        // for every incoming connection. This loop is the plain-TypeScript
        // equivalent of NumPy's `np.dot(w, a)` matrix multiplication in
        // the Python version -- NumPy just does this same sum internally,
        // in optimized, compiled code instead of a visible loop.
        for (let incoming = 0; incoming < neuronWeights.length; incoming++) {
          weightedSum += neuronWeights[incoming] * activation[incoming];
        }

        nextActivation[neuron] = sigmoid(weightedSum);
      }

      activation = nextActivation;
    }

    return activation;
  }

  /**
   * A convenience wrapper around `predict` that also figures out the
   * single most likely digit and how confident the network is in it,
   * plus a full breakdown for every digit (handy for a confidence-bar
   * chart in the UI).
   */
  classify(pixels: number[]): {
    digit: number;
    confidence: number;
    probabilities: number[];
  } {
    const outputs = this.predict(pixels);

    // The raw sigmoid outputs don't necessarily add up to 1 (they
    // weren't designed to represent a strict probability distribution --
    // each output neuron independently estimates "how much does this
    // look like my digit?"). To show the user friendly percentages that
    // add up to 100%, we normalize the outputs: divide each one by the
    // total of all ten, so they're proportional to each other but now
    // sum to exactly 1.
    const total = outputs.reduce((sum, value) => sum + value, 0);
    const probabilities = total > 0 ? outputs.map((v) => v / total) : outputs;

    let digit = 0;
    for (let i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > probabilities[digit]) {
        digit = i;
      }
    }

    return { digit, confidence: probabilities[digit], probabilities };
  }
}
