"use client";

/**
 * PredictionPanel.tsx
 * ====================
 *
 * Displays the result of a prediction: the big headline digit, a
 * side-by-side look at exactly what the network "saw" after
 * preprocessing, and a confidence bar for all 10 possible digits so you
 * can see not just the network's top guess, but how sure (or unsure) it
 * was, and what it considered as runner-up guesses.
 */

interface PredictionPanelProps {
  digit: number;
  confidence: number;
  probabilities: number[];
  previewDataUrl: string;
}

export default function PredictionPanel({
  digit,
  confidence,
  probabilities,
  previewDataUrl,
}: PredictionPanelProps) {
  const sorted = probabilities
    .map((p, d) => ({ digit: d, probability: p }))
    .sort((a, b) => b.probability - a.probability);

  return (
    <div className="flex w-full flex-col gap-6 rounded-2xl border border-white/10 bg-white/[0.03] p-6 sm:p-8">
      <div className="flex flex-col items-center gap-6 sm:flex-row sm:items-stretch sm:justify-center">
        <div className="flex flex-col items-center gap-2">
          <span className="text-xs uppercase tracking-wider text-white/40">
            What the network saw
          </span>
          <img
            src={previewDataUrl}
            alt="28x28 preprocessed digit"
            className="h-28 w-28 rounded-lg border border-white/15 [image-rendering:pixelated]"
          />
        </div>

        <div className="flex flex-1 flex-col items-center justify-center gap-1">
          <span className="text-xs uppercase tracking-wider text-white/40">
            Prediction
          </span>
          <span className="bg-gradient-to-br from-violet-300 via-fuchsia-300 to-white bg-clip-text text-8xl font-bold leading-none text-transparent">
            {digit}
          </span>
          <span className="text-sm text-white/60">
            {(confidence * 100).toFixed(1)}% confident
          </span>
        </div>
      </div>

      <div className="flex flex-col gap-2">
        <span className="text-xs uppercase tracking-wider text-white/40">
          Full breakdown
        </span>
        {sorted.map(({ digit: d, probability }) => (
          <div key={d} className="flex items-center gap-3">
            <span
              className={`w-4 text-sm font-semibold ${
                d === digit ? "text-violet-300" : "text-white/50"
              }`}
            >
              {d}
            </span>
            <div className="h-2 flex-1 overflow-hidden rounded-full bg-white/[0.06]">
              <div
                className={`h-full rounded-full ${
                  d === digit
                    ? "bg-gradient-to-r from-violet-500 to-fuchsia-500"
                    : "bg-white/20"
                }`}
                style={{ width: `${Math.max(probability * 100, 1)}%` }}
              />
            </div>
            <span className="w-12 text-right text-xs text-white/40">
              {(probability * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
