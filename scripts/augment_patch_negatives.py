#!/usr/bin/env python3
"""
augment_patch_negatives.py — generate color/brightness-shifted variants of
existing patch negative images to bridge a confirmed domain gap: Attain's
patches were photographed on darker Iranian urban asphalt, but a failing
real-world test case (Rahmaniya, UAE) has lighter, sandier desert asphalt.

This does NOT need new data collection — it synthetically extends the
patch images you already have rights to use (Attain, CC BY 4.0) toward a
lighter/warmer color range, without inventing new road content.

Usage:
  python augment_patch_negatives.py --source attain_negatives/priority --out attain_negatives/priority_augmented --variants 3
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

VALID_EXTS = {".jpg", ".jpeg", ".png"}


def lighten_and_warm(img: np.ndarray, brightness_delta: int, warmth_delta: int) -> np.ndarray:
    """Shift toward lighter/sandier asphalt tones: increase brightness (V in HSV),
    reduce saturation slightly (sandy asphalt looks less saturated than dark wet
    asphalt), and nudge the color balance warmer (more red/yellow, less blue)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] - 15, 0, 255)  # slightly desaturate
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + brightness_delta, 0, 255)  # brighten
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.int16)

    # warm the color balance: boost red channel, reduce blue channel slightly
    result[:, :, 2] = np.clip(result[:, :, 2] + warmth_delta, 0, 255)  # R channel (BGR order)
    result[:, :, 0] = np.clip(result[:, :, 0] - warmth_delta // 2, 0, 255)  # B channel
    return result.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path,
                     help="folder of existing patch negative images to augment")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--variants", type=int, default=3,
                     help="how many color-shifted variants to generate per source image")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    images = [p for p in args.source.iterdir() if p.suffix.lower() in VALID_EXTS]
    print(f"Augmenting {len(images)} source images x {args.variants} variants each...")

    total_written = 0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for v in range(args.variants):
            # randomize within a range that targets "lighter/sandier" without
            # being unrealistic — not a uniform fixed shift every time
            brightness_delta = random.randint(25, 55)
            warmth_delta = random.randint(3, 10)
            variant = lighten_and_warm(img, brightness_delta, warmth_delta)

            out_name = f"aug{v}_{img_path.name}"
            cv2.imwrite(str(args.out / out_name), variant)
            total_written += 1

    print(f"Wrote {total_written} augmented images to {args.out}")
    print(f"\nNext: python add_background_images.py --source {args.out} "
          f"--dataset ../datasets/potholes --independent --dry-run")
    print("These are synthetic variants of images you already used — check the dry-run "
          "cap doesn't overshoot your existing negative fraction (you already have 317 "
          "negatives in; this adds more, so recheck --max-fraction-of-positives math).")


if __name__ == "__main__":
    main()
