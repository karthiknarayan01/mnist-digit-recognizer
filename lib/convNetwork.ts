/**
 * convNetwork.ts
 * ==============
 *
 * A from-scratch TypeScript re-implementation of the **convolutional**
 * network's forward pass (as opposed to `network.ts`, which implements
 * the original plain, fully-connected network's forward pass).
 *
 * Read `network.ts`'s file header first if you haven't -- the short
 * version is: a trained network's "knowledge" is just a list of
 * numbers, and *using* an already-trained network (making one
 * prediction) is cheap enough to redo from scratch, right here in the
 * browser, with no server and no ML framework involved.
 *
 * This file mirrors the Python training project's `src/conv_layers.py`
 * and `src/conv_network.py` layer by layer. One simplification here
 * compared to the Python version: the Python training code processes a
 * whole *batch* of images at once (as 4D arrays), because that's what
 * makes training fast. This file only ever needs to classify *one*
 * image at a time (whatever you just uploaded or drew), so every layer
 * below works on a single 3D image (channels x height x width) with
 * plain nested loops -- easier to read, and still instant at this
 * scale (a few hundred thousand additions is nothing for a browser).
 */

/** A single-image tensor: `data[channel][row][col]`. */
type Tensor3D = number[][][];

/** The JSON shape written by `ConvNetwork.save()` in Python. */
interface ConvLayerData {
  type: string;
  W?: number[][][][] | number[][];
  b?: number[];
}

export interface ConvModel {
  architecture: "cnn";
  num_filters: number;
  filter_size: number;
  hidden_units: number;
  layers: ConvLayerData[];
}

/**
 * A convolutional layer's forward pass: slide every filter in `W`
 * across the input, producing one output "feature map" per filter.
 *
 * This is the plain, literal version of the sliding-window idea
 * explained in this project's docs -- for every filter, every output
 * position, and every pixel inside that filter's window, multiply and
 * add. The Python training code computes the exact same sum a smarter,
 * vectorized way (see `conv_layers.py`'s `Conv2D.forward`); the answer
 * is identical either way, just arrived at with ordinary loops here
 * instead of NumPy's batched array tricks, since we only ever need to
 * do this once per prediction rather than thousands of times per
 * second during training.
 */
function conv2d(x: Tensor3D, W: number[][][][], b: number[]): Tensor3D {
  const inChannels = x.length;
  const inH = x[0].length;
  const inW = x[0][0].length;
  const outChannels = W.length;
  const filterH = W[0][0].length;
  const filterW = W[0][0][0].length;
  const outH = inH - filterH + 1;
  const outW = inW - filterW + 1;

  const out: Tensor3D = Array.from({ length: outChannels }, () =>
    Array.from({ length: outH }, () => new Array(outW).fill(0))
  );

  for (let oc = 0; oc < outChannels; oc++) {
    for (let i = 0; i < outH; i++) {
      for (let j = 0; j < outW; j++) {
        let sum = b[oc];
        for (let ic = 0; ic < inChannels; ic++) {
          for (let di = 0; di < filterH; di++) {
            for (let dj = 0; dj < filterW; dj++) {
              sum += x[ic][i + di][j + dj] * W[oc][ic][di][dj];
            }
          }
        }
        out[oc][i][j] = sum;
      }
    }
  }
  return out;
}

/** ReLU: replace every negative number with zero, leave the rest untouched. */
function relu3d(x: Tensor3D): Tensor3D {
  return x.map((channel) => channel.map((row) => row.map((v) => Math.max(0, v))));
}

/**
 * Max pooling: shrink every channel by keeping only the largest value
 * in each `poolSize` x `poolSize` block.
 */
function maxPool2d(x: Tensor3D, poolSize = 2): Tensor3D {
  return x.map((channel) => {
    const outH = Math.floor(channel.length / poolSize);
    const outW = Math.floor(channel[0].length / poolSize);
    const pooled: number[][] = [];
    for (let i = 0; i < outH; i++) {
      const row: number[] = [];
      for (let j = 0; j < outW; j++) {
        let best = -Infinity;
        for (let di = 0; di < poolSize; di++) {
          for (let dj = 0; dj < poolSize; dj++) {
            best = Math.max(best, channel[i * poolSize + di][j * poolSize + dj]);
          }
        }
        row.push(best);
      }
      pooled.push(row);
    }
    return pooled;
  });
}

/**
 * Flatten a (channels, height, width) tensor into one flat list of
 * numbers, in the same channel-then-row-then-column order NumPy's
 * `reshape` uses in the Python training code -- this order has to
 * match exactly, since the next (Dense) layer's weights were learned
 * assuming this specific ordering of flattened pixels.
 */
function flatten(x: Tensor3D): number[] {
  const flat: number[] = [];
  for (const channel of x) {
    for (const row of channel) {
      for (const value of row) {
        flat.push(value);
      }
    }
  }
  return flat;
}

/** A fully-connected layer's forward pass -- identical idea to `network.ts`'s `predict`. */
function dense(x: number[], W: number[][], b: number[]): number[] {
  const out = new Array(W.length).fill(0);
  for (let o = 0; o < W.length; o++) {
    let sum = b[o];
    const weights = W[o];
    for (let i = 0; i < weights.length; i++) {
      sum += weights[i] * x[i];
    }
    out[o] = sum;
  }
  return out;
}

function relu1d(x: number[]): number[] {
  return x.map((v) => Math.max(0, v));
}

/**
 * Softmax: turn raw output scores into probabilities that sum to 1.
 * See `conv_layers.py`'s `softmax` for the full explanation of why a
 * CNN's output layer uses this instead of the dense network's
 * independent sigmoid neurons.
 */
function softmax(logits: number[]): number[] {
  const maxLogit = Math.max(...logits);
  const expScores = logits.map((v) => Math.exp(v - maxLogit));
  const total = expScores.reduce((sum, v) => sum + v, 0);
  return expScores.map((v) => v / total);
}

/**
 * The convolutional network: loads a trained model (conv layer, pooling
 * layer, and two fully-connected layers) and runs a single image
 * through it to produce a prediction.
 */
export class ConvNeuralNetwork {
  constructor(private readonly model: ConvModel) {}

  static async loadFromUrl(url: string): Promise<ConvNeuralNetwork> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load trained CNN model from ${url}`);
    }
    const model: ConvModel = await response.json();
    return new ConvNeuralNetwork(model);
  }

  /**
   * Run one 28x28 image through every layer in turn, in the exact order
   * they were trained in, and return the network's 10 raw output
   * scores ("logits") -- call `classify` instead if you want
   * normalized probabilities and the top guess.
   *
   * @param pixels A flat array of 784 numbers (0 = black, 1 = white),
   *   the same input format `network.ts`'s `NeuralNetwork.predict` uses.
   */
  private forwardLogits(pixels: number[]): number[] {
    // Reshape the flat 784-pixel list back into a single-channel
    // 28x28 image, matching the (1, 28, 28) shape the network trained on.
    let tensor: Tensor3D = [
      Array.from({ length: 28 }, (_, row) => pixels.slice(row * 28, row * 28 + 28)),
    ];
    let flat: number[] | null = null;

    for (const layer of this.model.layers) {
      switch (layer.type) {
        case "Conv2D":
          tensor = conv2d(tensor, layer.W as number[][][][], layer.b as number[]);
          break;
        case "ReLU":
          if (flat === null) {
            tensor = relu3d(tensor);
          } else {
            flat = relu1d(flat);
          }
          break;
        case "MaxPool2D":
          tensor = maxPool2d(tensor);
          break;
        case "Flatten":
          flat = flatten(tensor);
          break;
        case "Dense":
          flat = dense(flat as number[], layer.W as number[][], layer.b as number[]);
          break;
        default:
          throw new Error(`Unknown layer type in saved model: ${layer.type}`);
      }
    }

    if (flat === null) {
      throw new Error("Model finished without producing a flat output vector.");
    }
    return flat;
  }

  classify(pixels: number[]): {
    digit: number;
    confidence: number;
    probabilities: number[];
  } {
    const logits = this.forwardLogits(pixels);
    const probabilities = softmax(logits);

    let digit = 0;
    for (let i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > probabilities[digit]) {
        digit = i;
      }
    }

    return { digit, confidence: probabilities[digit], probabilities };
  }
}
