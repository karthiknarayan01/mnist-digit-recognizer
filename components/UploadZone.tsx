"use client";

/**
 * UploadZone.tsx
 * ==============
 *
 * The drag-and-drop / click-to-browse box where a user picks an image of
 * a handwritten digit. It doesn't know anything about neural networks --
 * its only job is "get a File object from the user, however they choose
 * to provide it", show them a preview of what they picked, and only
 * hand that file to the parent component once they press "Predict" --
 * mirroring how `DrawPad` works, so both ways of providing a digit feel
 * the same.
 */

import { useCallback, useEffect, useRef, useState } from "react";

interface UploadZoneProps {
  onSubmit: (file: File) => void;
  /** Called whenever the picked image is cleared, so the parent can
   *  discard any prediction left over from a previous image. */
  onClear: () => void;
  disabled?: boolean;
}

export default function UploadZone({ onSubmit, onClear, disabled }: UploadZoneProps) {
  const [isDraggingOver, setIsDraggingOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Clean up the preview's object URL whenever it changes or the
  // component goes away, so the browser can free that memory.
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0];
      if (file && file.type.startsWith("image/")) {
        setSelectedFile(file);
        setPreviewUrl(URL.createObjectURL(file));
      }
    },
    []
  );

  const clearSelection = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    onClear();
  };

  if (selectedFile && previewUrl) {
    return (
      <div className="flex flex-col items-center gap-4">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={previewUrl}
          alt="The image you picked"
          className="h-64 w-64 rounded-2xl border-2 border-white/15 bg-black object-contain shadow-inner"
        />
        <div className="flex gap-3">
          <button
            onClick={clearSelection}
            disabled={disabled}
            className="rounded-full border border-white/15 px-5 py-2 text-sm font-medium text-white/70 transition hover:bg-white/5 disabled:opacity-40"
          >
            Clear
          </button>
          <button
            onClick={() => onSubmit(selectedFile)}
            disabled={disabled}
            className="rounded-full bg-gradient-to-r from-violet-500 to-fuchsia-500 px-6 py-2 text-sm font-semibold text-white shadow-lg shadow-violet-500/30 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Predict this
          </button>
        </div>
      </div>
    );
  }

  return (
    <div
      role="button"
      tabIndex={0}
      aria-disabled={disabled}
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(e) => {
        if (!disabled && (e.key === "Enter" || e.key === " ")) {
          inputRef.current?.click();
        }
      }}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setIsDraggingOver(true);
      }}
      onDragLeave={() => setIsDraggingOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDraggingOver(false);
        if (!disabled) handleFiles(e.dataTransfer.files);
      }}
      className={`group relative flex h-64 w-full cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed p-6 text-center transition-all duration-200
        ${
          isDraggingOver
            ? "border-violet-400 bg-violet-500/10 scale-[1.01]"
            : "border-white/15 bg-white/[0.03] hover:border-white/30 hover:bg-white/[0.06]"
        }
        ${disabled ? "cursor-not-allowed opacity-50" : ""}
      `}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        disabled={disabled}
        onChange={(e) => handleFiles(e.target.files)}
      />
      <div className="flex h-14 w-14 items-center justify-center rounded-full bg-gradient-to-br from-violet-500 to-fuchsia-500 text-2xl shadow-lg shadow-violet-500/30 transition-transform group-hover:scale-110">
        ✍️
      </div>
      <p className="text-sm font-medium text-white/90">
        Drop a picture here, or click to choose one
      </p>
      <p className="text-xs text-white/40">
        A photo of a single number you&apos;ve written works best.
      </p>
    </div>
  );
}
