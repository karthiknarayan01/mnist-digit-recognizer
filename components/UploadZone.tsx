"use client";

/**
 * UploadZone.tsx
 * ==============
 *
 * The drag-and-drop / click-to-browse box where a user picks an image of
 * a handwritten digit. It doesn't know anything about neural networks --
 * its only job is "get a File object from the user, however they choose
 * to provide it", and hand that file to the parent component.
 */

import { useCallback, useRef, useState } from "react";

interface UploadZoneProps {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

export default function UploadZone({ onFileSelected, disabled }: UploadZoneProps) {
  const [isDraggingOver, setIsDraggingOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback(
    (files: FileList | null) => {
      const file = files?.[0];
      if (file && file.type.startsWith("image/")) {
        onFileSelected(file);
      }
    },
    [onFileSelected]
  );

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
        Drop an image here, or click to browse
      </p>
      <p className="text-xs text-white/40">
        A photo or screenshot of a single handwritten digit works best — PNG, JPG, or WEBP.
      </p>
    </div>
  );
}
