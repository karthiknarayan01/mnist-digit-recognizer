# mnist-digit-recognizer

A web app that recognizes handwritten digits using a neural network
trained **entirely from scratch** — no ML framework anywhere in this
repo, not even in the browser. The network itself (its forward pass) is
re-implemented from first principles in plain TypeScript, and the
weights it uses were trained in a companion project,
[**mnist-neural-network**](https://github.com/karthiknarayan01/mnist-neural-network),
which trains the same network using nothing but NumPy.

**🔗 Try it live: <!-- LIVE_URL_PLACEHOLDER -->**

<!-- SCREENSHOT_PLACEHOLDER -->

## How it works

Everything happens client-side, in your browser — there's no backend
server making predictions, and your image never leaves your device.

1. **You provide a digit.** Either upload a photo/screenshot of a
   handwritten digit, or draw one directly on the canvas.
2. **Preprocessing** ([`lib/preprocess.ts`](lib/preprocess.ts)) converts
   whatever you provided into the exact style of image the network
   learned from: grayscale, inverted if needed so the digit is light on
   a dark background, cropped to the digit itself, and centered in a
   28×28 pixel grid — mirroring how the original MNIST dataset itself
   was built.
3. **Prediction** ([`lib/network.ts`](lib/network.ts)) runs those 784
   pixel values through a TypeScript re-implementation of the trained
   network's forward pass — the same math as the Python training
   project's `feedforward`, just written without NumPy's matrix
   shortcuts — and returns a confidence score for every digit 0–9.
4. **The UI** shows the network's top guess, its confidence, and a full
   breakdown across all 10 digits.

The trained weights live in
[`public/model/trained_network.json`](public/model/trained_network.json),
exported directly from the Python project — nothing in this repo trains
anything; it only *uses* a network that was already trained elsewhere.

## The network, in brief

```
784 inputs  →  100 neurons  →  100 neurons  →  100 neurons  →  10 outputs
 (pixels)       (hidden 1)       (hidden 2)       (hidden 3)     (digits 0-9)
```

Sigmoid neurons, trained with cross-entropy cost, L2 regularization, and
mini-batch SGD. Full details, benchmarks, and the training code itself
are in the [mnist-neural-network](https://github.com/karthiknarayan01/mnist-neural-network)
repo.

## Running it locally

```bash
git clone https://github.com/karthiknarayan01/mnist-digit-recognizer.git
cd mnist-digit-recognizer
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Tech stack

- [Next.js](https://nextjs.org/) (App Router) + TypeScript
- [Tailwind CSS](https://tailwindcss.com/) for styling
- No ML framework, no backend — the neural network's forward pass is
  ~50 lines of plain TypeScript, entirely client-side.
- Deployed on [Vercel](https://vercel.com/).

## Project structure

```
mnist-digit-recognizer/
├── app/
│   ├── page.tsx         # the app's single page
│   ├── layout.tsx
│   └── globals.css
├── components/
│   ├── UploadZone.tsx   # drag-and-drop / click-to-browse image upload
│   ├── DrawPad.tsx       # draw a digit with mouse/touch
│   └── PredictionPanel.tsx  # displays the prediction + confidence breakdown
├── lib/
│   ├── preprocess.ts    # turns any uploaded image into network-ready pixels
│   └── network.ts       # the trained network's forward pass, in TypeScript
└── public/model/
    └── trained_network.json  # the trained weights, exported from Python
```

## License

[MIT](LICENSE)
