/**
 * preprocess.ts
 * =============
 *
 * The neural network was trained exclusively on MNIST's very particular
 * style of image: a single handwritten digit, in white, perfectly
 * centered, on a plain black background, squeezed into a tiny 28x28
 * pixel grid. A photo or screenshot you upload looks nothing like that
 * -- it might be a large photo, a dark digit on a light background, off
 * to one side of the frame, surrounded by empty space.
 *
 * This file's job is to transform whatever image you upload into
 * something that *looks*, to the network, as close as possible to the
 * kind of image it learned from. Skipping this step and just shrinking
 * your photo straight to 28x28 would badly confuse the network -- most
 * of the accuracy the app appears to have comes from getting this
 * preprocessing right.
 *
 * The steps, in order, are:
 *
 * 1. Draw your image onto an invisible "canvas" (a pixel grid the
 *    browser lets us draw on and inspect) at its original size.
 * 2. Convert it to grayscale (brightness only, no color) -- the network
 *    only ever learned about brightness.
 * 3. Figure out whether the digit is drawn dark-on-light or
 *    light-on-dark, and flip it if needed so it matches MNIST's
 *    white-digit-on-black-background convention.
 * 4. Find the digit's bounding box (the smallest rectangle containing
 *    all its "ink") and crop to just that box, discarding the mostly-
 *    empty margins around it.
 * 5. Shrink that cropped digit down to fit a 28x28 grid, then center it
 *    precisely by its **center of mass** (not just its bounding box) --
 *    the same way the original MNIST dataset itself was built. The
 *    network turns out to be quite sensitive to exactly where the ink
 *    sits in the frame, so matching this step closely matters a lot for
 *    real-world accuracy.
 * 6. Flatten the final 28x28 grid of brightness values (0 = black,
 *    1 = white) into the single list of 784 numbers the network expects.
 */

export interface PreprocessResult {
  /** 784 numbers (0-1), the exact input the neural network expects. */
  pixels: number[];
  /** A 28x28 preview of exactly what the network "sees", for the UI to display. */
  previewDataUrl: string;
}

const TARGET_SIZE = 28; // the network's input grid is 28x28, matching MNIST
const DIGIT_BOX_SIZE = 20; // MNIST digits are scaled to fit a 20x20 box, leaving a small margin

/** Load an uploaded file into an HTMLImageElement we can draw from. */
function loadImage(file: File): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Could not read that image file."));
    };
    img.src = url;
  });
}

export async function preprocessImageFile(file: File): Promise<PreprocessResult> {
  const img = await loadImage(file);

  // --- Step 1 & 2: draw the image, read it back as grayscale brightness values ---
  const sourceCanvas = document.createElement("canvas");
  sourceCanvas.width = img.naturalWidth;
  sourceCanvas.height = img.naturalHeight;
  const sourceCtx = sourceCanvas.getContext("2d")!;
  sourceCtx.drawImage(img, 0, 0);
  const { data, width, height } = sourceCtx.getImageData(
    0,
    0,
    sourceCanvas.width,
    sourceCanvas.height
  );

  // `data` is a flat list of red, green, blue, alpha values, four numbers
  // per pixel. We combine red/green/blue into a single "brightness"
  // number per pixel using the standard luminance formula (human eyes
  // are more sensitive to green than red or blue, hence the different
  // weights), and ignore transparency by treating it as white background.
  const brightness = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    const alpha = data[i * 4 + 3] / 255;
    // Blend with a white background wherever the pixel is partly
    // transparent, then convert to 0 (black) - 255 (white) brightness.
    const gray = (0.299 * r + 0.587 * g + 0.114 * b) * alpha + 255 * (1 - alpha);
    brightness[i] = gray;
  }

  // --- Step 3: decide dark-on-light vs. light-on-dark, and normalize to
  // "white digit on black background" like MNIST ---
  let sum = 0;
  for (let i = 0; i < brightness.length; i++) sum += brightness[i];
  const averageBrightness = sum / brightness.length;

  // If most of the image is bright (a light background, the common case
  // for a photographed page or a screenshot), the digit itself must be
  // the darker minority -- so we invert brightness values, turning dark
  // ink into high "ink intensity" values, matching the training data.
  const isLightBackground = averageBrightness > 127;
  const ink = new Float32Array(brightness.length);
  for (let i = 0; i < brightness.length; i++) {
    const value = isLightBackground ? 255 - brightness[i] : brightness[i];
    ink[i] = value / 255; // normalize to 0-1
  }

  // --- Step 4: find the bounding box of the digit's "ink" ---
  // Anything reasonably above background noise counts as part of the digit.
  const INK_THRESHOLD = 0.15;
  let minX = width,
    minY = height,
    maxX = -1,
    maxY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (ink[y * width + x] > INK_THRESHOLD) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  // If nothing was drawn/uploaded that looks like ink, fall back to using
  // the whole image rather than crashing.
  const hasInk = maxX >= minX && maxY >= minY;
  const boxX = hasInk ? minX : 0;
  const boxY = hasInk ? minY : 0;
  const boxWidth = hasInk ? maxX - minX + 1 : width;
  const boxHeight = hasInk ? maxY - minY + 1 : height;

  // --- Step 5: scale the cropped digit into a 20x20 box, centered in a
  // 28x28 canvas (matching how MNIST itself was constructed) ---
  const scale = DIGIT_BOX_SIZE / Math.max(boxWidth, boxHeight);
  const scaledWidth = Math.max(1, Math.round(boxWidth * scale));
  const scaledHeight = Math.max(1, Math.round(boxHeight * scale));
  const offsetX = Math.floor((TARGET_SIZE - scaledWidth) / 2);
  const offsetY = Math.floor((TARGET_SIZE - scaledHeight) / 2);

  // Put the cropped region onto its own small canvas first, so the
  // browser's built-in image scaling (smoothed, good quality) can resize
  // it for us instead of us resampling pixels by hand.
  const croppedCanvas = document.createElement("canvas");
  croppedCanvas.width = boxWidth;
  croppedCanvas.height = boxHeight;
  const croppedCtx = croppedCanvas.getContext("2d")!;
  const croppedImageData = croppedCtx.createImageData(boxWidth, boxHeight);
  for (let y = 0; y < boxHeight; y++) {
    for (let x = 0; x < boxWidth; x++) {
      const srcIndex = (boxY + y) * width + (boxX + x);
      const value = Math.round(ink[srcIndex] * 255);
      const destIndex = (y * boxWidth + x) * 4;
      croppedImageData.data[destIndex] = value;
      croppedImageData.data[destIndex + 1] = value;
      croppedImageData.data[destIndex + 2] = value;
      croppedImageData.data[destIndex + 3] = 255;
    }
  }
  croppedCtx.putImageData(croppedImageData, 0, 0);

  const finalCanvas = document.createElement("canvas");
  finalCanvas.width = TARGET_SIZE;
  finalCanvas.height = TARGET_SIZE;
  const finalCtx = finalCanvas.getContext("2d")!;

  const paint = (x: number, y: number) => {
    finalCtx.fillStyle = "black";
    finalCtx.fillRect(0, 0, TARGET_SIZE, TARGET_SIZE);
    finalCtx.imageSmoothingEnabled = true;
    finalCtx.drawImage(croppedCanvas, x, y, scaledWidth, scaledHeight);
  };

  // A first draft placement, centered purely on the bounding box.
  paint(offsetX, offsetY);

  // --- A second, more precise centering pass: center of mass ---
  // The real MNIST dataset wasn't centered by bounding box at all -- it
  // was centered by *center of mass*, treating each pixel's brightness
  // as a tiny weight and balancing the whole image like a mobile, so the
  // "weight" of the ink is evenly distributed around the middle. A thick
  // digit like "2" or "5" can have noticeably more ink toward its bottom
  // half, which pulls its bounding-box center away from its true visual
  // center -- so this extra pass measurably improves how closely our
  // input matches what the network actually trained on.
  const draft = finalCtx.getImageData(0, 0, TARGET_SIZE, TARGET_SIZE).data;
  let massX = 0;
  let massY = 0;
  let totalMass = 0;
  for (let y = 0; y < TARGET_SIZE; y++) {
    for (let x = 0; x < TARGET_SIZE; x++) {
      const weight = draft[(y * TARGET_SIZE + x) * 4] / 255;
      massX += x * weight;
      massY += y * weight;
      totalMass += weight;
    }
  }

  if (totalMass > 0) {
    const centerOfMassX = massX / totalMass;
    const centerOfMassY = massY / totalMass;
    const shiftX = Math.round(TARGET_SIZE / 2 - centerOfMassX);
    const shiftY = Math.round(TARGET_SIZE / 2 - centerOfMassY);
    paint(offsetX + shiftX, offsetY + shiftY);
  }

  // --- Step 6: flatten the final 28x28 grid into 784 numbers ---
  const finalData = finalCtx.getImageData(0, 0, TARGET_SIZE, TARGET_SIZE).data;
  const pixels: number[] = new Array(TARGET_SIZE * TARGET_SIZE);
  for (let i = 0; i < TARGET_SIZE * TARGET_SIZE; i++) {
    pixels[i] = finalData[i * 4] / 255; // red channel == grayscale value here
  }

  return {
    pixels,
    previewDataUrl: finalCanvas.toDataURL(),
  };
}
