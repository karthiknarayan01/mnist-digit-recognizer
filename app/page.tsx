"use client";

/**
 * page.tsx
 * ========
 *
 * The app's single page. It wires together the pieces defined
 * elsewhere in this project:
 *
 * - `UploadZone` / `DrawPad` — get an image of a digit from the user.
 * - `lib/preprocess.ts` — turn that image into the 784 numbers the
 *   network expects.
 * - `lib/network.ts` — run those numbers through the trained network
 *   and get a prediction back.
 * - `PredictionPanel` — show the result beautifully.
 *
 * Everything here runs client-side, in the browser. There's no server
 * involved in making a prediction: the trained model is a static file
 * (`public/model/trained_network.json`) shipped alongside the page, and
 * `NeuralNetwork.loadFromUrl` fetches it once when the page loads.
 */

import { useEffect, useRef, useState } from "react";
import UploadZone from "@/components/UploadZone";
import DrawPad from "@/components/DrawPad";
import PredictionPanel from "@/components/PredictionPanel";
import { NeuralNetwork } from "@/lib/network";
import { preprocessImageFile } from "@/lib/preprocess";

type Mode = "upload" | "draw";

type Result = {
  digit: number;
  confidence: number;
  probabilities: number[];
  previewDataUrl: string;
};

export default function Home() {
  const [mode, setMode] = useState<Mode>("upload");
  const [result, setResult] = useState<Result | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const networkRef = useRef<NeuralNetwork | null>(null);
  const [modelReady, setModelReady] = useState(false);

  useEffect(() => {
    NeuralNetwork.loadFromUrl("/model/trained_network.json")
      .then((net) => {
        networkRef.current = net;
        setModelReady(true);
      })
      .catch(() => setError("Could not load the trained model. Please refresh the page."));
  }, []);

  const handleFile = async (file: File) => {
    setIsBusy(true);
    setError(null);
    setResult(null);
    try {
      const { pixels, previewDataUrl } = await preprocessImageFile(file);
      const net = networkRef.current;
      if (!net) throw new Error("Model not loaded yet.");
      const { digit, confidence, probabilities } = net.classify(pixels);
      setResult({ digit, confidence, probabilities, previewDataUrl });
    } catch {
      setError("Something went wrong reading that image. Try a different one?");
    } finally {
      setIsBusy(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center px-4 py-16 sm:py-24">
      <div className="ambient-glow" />

      <div className="flex w-full max-w-xl flex-col items-center gap-3 text-center">
        <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-medium text-white/50">
          784 → 100 → 100 → 100 → 10 · pure NumPy · zero ML frameworks
        </span>
        <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
          A neural network, built from scratch, reading your handwriting
        </h1>
        <p className="max-w-md text-sm text-white/50">
          Upload a photo of a handwritten digit, or draw one yourself. A
          sigmoid network trained entirely with hand-written NumPy code —
          no PyTorch, no TensorFlow — will tell you what it sees, instantly,
          right here in your browser.
        </p>
      </div>

      <div className="mt-10 flex w-full max-w-xl flex-col items-center gap-6">
        <div className="flex rounded-full border border-white/10 bg-white/[0.03] p-1">
          {(["upload", "draw"] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => {
                setMode(m);
                setResult(null);
                setError(null);
              }}
              className={`rounded-full px-5 py-1.5 text-sm font-medium capitalize transition ${
                mode === m
                  ? "bg-white text-black shadow"
                  : "text-white/50 hover:text-white/80"
              }`}
            >
              {m === "upload" ? "Upload image" : "Draw it"}
            </button>
          ))}
        </div>

        <div className="w-full">
          {mode === "upload" ? (
            <UploadZone onFileSelected={handleFile} disabled={!modelReady || isBusy} />
          ) : (
            <DrawPad onSubmit={handleFile} disabled={!modelReady || isBusy} />
          )}
        </div>

        {!modelReady && !error && (
          <p className="text-xs text-white/40">Loading trained model…</p>
        )}
        {isBusy && <p className="text-xs text-white/40">Thinking…</p>}
        {error && <p className="text-xs text-red-400">{error}</p>}

        {result && (
          <PredictionPanel
            digit={result.digit}
            confidence={result.confidence}
            probabilities={result.probabilities}
            previewDataUrl={result.previewDataUrl}
          />
        )}
      </div>

      <footer className="mt-20 flex flex-col items-center gap-1 text-xs text-white/30">
        <p>
          Trained on the classic 70,000-image MNIST dataset — 50k training,
          10k validation, 10k test.
        </p>
        <p>
          Source on{" "}
          <a
            href="https://github.com/karthiknarayan01/mnist-digit-recognizer"
            className="underline hover:text-white/60"
          >
            GitHub
          </a>{" "}
          · training code at{" "}
          <a
            href="https://github.com/karthiknarayan01/mnist-neural-network"
            className="underline hover:text-white/60"
          >
            mnist-neural-network
          </a>
        </p>
      </footer>
    </main>
  );
}
