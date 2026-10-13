"use client";

/**
 * DrawPad.tsx
 * ===========
 *
 * An alternative to uploading a file: draw a digit directly in the
 * browser with your mouse or finger. Internally, this just paints onto
 * an HTML `<canvas>` (a pixel grid the browser lets us draw on), and
 * when you click "Predict", it converts that canvas into a File so the
 * exact same preprocessing and prediction pipeline used for uploads can
 * handle it too -- the rest of the app doesn't need to know whether a
 * digit came from a file or from your mouse.
 */

import { useEffect, useRef, useState } from "react";

interface DrawPadProps {
  onSubmit: (file: File) => void;
  /** Called whenever the canvas is cleared or a new stroke begins, so the
   *  parent can discard any prediction left over from a previous drawing. */
  onClear: () => void;
  disabled?: boolean;
}

export default function DrawPad({ onSubmit, onClear, disabled }: DrawPadProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDrawing = useRef(false);
  const lastPoint = useRef<{ x: number; y: number } | null>(null);
  const [hasDrawn, setHasDrawn] = useState(false);

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setHasDrawn(false);
    onClear();
  };

  useEffect(() => {
    clearCanvas();
  }, []);

  const getPoint = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * canvas.width,
      y: ((e.clientY - rect.top) / rect.height) * canvas.height,
    };
  };

  const draw = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDrawing.current || disabled) return;
    const ctx = canvasRef.current!.getContext("2d")!;
    const point = getPoint(e);
    ctx.strokeStyle = "white";
    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    const from = lastPoint.current ?? point;
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
    lastPoint.current = point;
    setHasDrawn(true);
  };

  return (
    <div className="flex flex-col items-center gap-4">
      <canvas
        ref={canvasRef}
        width={280}
        height={280}
        className={`touch-none rounded-2xl border-2 border-white/15 bg-black shadow-inner ${
          disabled ? "cursor-not-allowed opacity-50" : "cursor-crosshair"
        }`}
        onPointerDown={(e) => {
          if (disabled) return;
          // Starting a brand new stroke means the old prediction (if any)
          // no longer describes what's on the canvas -- clear it right
          // away rather than leaving a stale, now-wrong answer on screen.
          if (!isDrawing.current) onClear();
          isDrawing.current = true;
          lastPoint.current = null;
          draw(e);
        }}
        onPointerMove={draw}
        onPointerUp={() => {
          isDrawing.current = false;
          lastPoint.current = null;
        }}
        onPointerLeave={() => {
          isDrawing.current = false;
          lastPoint.current = null;
        }}
      />
      <div className="flex gap-3">
        <button
          onClick={clearCanvas}
          disabled={disabled}
          className="rounded-full border border-white/15 px-5 py-2 text-sm font-medium text-white/70 transition hover:bg-white/5 disabled:opacity-40"
        >
          Clear
        </button>
        <button
          onClick={() => {
            const canvas = canvasRef.current;
            if (!canvas || !hasDrawn) return;
            canvas.toBlob((blob) => {
              if (blob) onSubmit(new File([blob], "drawing.png", { type: "image/png" }));
            });
          }}
          disabled={disabled || !hasDrawn}
          className="rounded-full bg-gradient-to-r from-violet-500 to-fuchsia-500 px-6 py-2 text-sm font-semibold text-white shadow-lg shadow-violet-500/30 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Predict this
        </button>
      </div>
    </div>
  );
}
