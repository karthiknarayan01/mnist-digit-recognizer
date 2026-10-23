# mnist-digit-recognizer

A web app that recognizes handwritten digits using a neural network
trained **entirely from scratch** — no ML framework anywhere, not even
in the browser. The network's forward pass is re-implemented from first
principles in plain TypeScript; the weights come from a companion
project, [**mnist-neural-network**](https://github.com/karthiknarayan01/mnist-neural-network),
which trains the same network using nothing but NumPy.

**🔗 Try it live: [mnist-digit-recognizer-coral.vercel.app](https://mnist-digit-recognizer-coral.vercel.app)**

![Screenshot of the app correctly identifying an uploaded "9" with 100.0% confidence](docs/screenshot.png)

## How it works

Everything happens client-side — no backend, your image never leaves
your device.

1. **You provide a digit** — upload a photo, or draw one on the canvas.
2. **Preprocessing** ([`lib/preprocess.ts`](lib/preprocess.ts)) converts
   it into MNIST's style: grayscale, cropped to the digit, scaled to
   28×28, and centered by **center of mass** rather than bounding box —
   this mattered more than expected, fixing it took confidence from
   flaky sub-90% guesses to consistent 99%+.
3. **Prediction** ([`lib/convNetwork.ts`](lib/convNetwork.ts)) runs that
   image through a TypeScript port of the trained network's forward
   pass and returns a confidence score per digit.
4. **The UI** shows the top guess, its confidence, and the full
   breakdown across all 10 digits.

The trained weights live in
[`public/model/trained_cnn.json`](public/model/trained_cnn.json),
exported directly from the Python project — nothing here trains
anything, it only uses a network trained elsewhere.

## The network

```
28×28 image → Conv(20, 5×5) → ReLU → Pool → Conv(40, 5×5) → ReLU → Pool
           → Dense(640→100) → ReLU → Dense(100→10) → softmax
```

A small CNN: two rounds of "scan for patterns, then shrink," then
fully-connected layers that decide the digit. Trained with softmax
cross-entropy, L2 regularization, and rotation-based data augmentation.
**99.35% test accuracy** — the best of five configurations tried on the
training side (architecture and augmentation each helped, going deeper
helped again); see the training repo for the full comparison and how
the architecture was chosen.

This repo also ships [`lib/network.ts`](lib/network.ts), a from-scratch
plain fully-connected network (98.47% on its own), kept for comparison.
Curious why the CNN does better? The training repo links a genuinely
kid-friendly PDF explaining it with pictures.

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
- No ML framework, no backend — the forward pass is plain TypeScript,
  entirely client-side
- Deployed on [Vercel](https://vercel.com/)

## Project structure

```
mnist-digit-recognizer/
├── app/                  # page.tsx, layout.tsx, globals.css
├── components/           # UploadZone, DrawPad, PredictionPanel
├── lib/
│   ├── preprocess.ts     # turns an uploaded image into network-ready pixels
│   ├── convNetwork.ts    # the CNN's forward pass, in TypeScript (used by the app)
│   └── network.ts        # the plain dense network's forward pass (comparison only)
└── public/model/         # trained_cnn.json (used), trained_network.json (comparison)
```

## License

[MIT](LICENSE)
