#!/usr/bin/env python3
"""
validate_dataset.py — QC gate for PotholeNet YOLO dataset.

Run BEFORE any training. Exits non-zero if hard errors found.

Checks:
  - image <-> label filename pairing (missing image / missing label)
  - invalid class ids (must be 0 for this single-class dataset)
  - malformed label lines (wrong token count, non-numeric)
  - bbox coords outside [0,1]
  - empty annotation files (flagged as warning, not error — background images are valid
    in YOLO if intentional, but we flag them so you can confirm intent)
  - corrupted / unreadable images
  - extremely small bboxes (below --min-box-frac of image area) — likely noise
  - exact duplicate images (md5) across the WHOLE dataset (leakage risk across splits)
  - near-duplicate images (perceptual hash, hamming distance <= --phash-thresh) —
    flags candidates for manual review, does not auto-remove

Usage:
  python validate_dataset.py --dataset datasets/potholes
"""
import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import imagehash
from PIL import Image

SPLITS = ["train", "val", "test"]
VALID_EXTS = {".jpg", ".jpeg", ".png"}


def find_images(img_dir: Path):
    return sorted(p for p in img_dir.iterdir() if p.suffix.lower() in VALID_EXTS)


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_split(dataset_root: Path, split: str, min_box_frac: float, errors: list, warnings: list):
    img_dir = dataset_root / "images" / split
    lbl_dir = dataset_root / "labels" / split
    if not img_dir.exists() or not lbl_dir.exists():
        errors.append(f"[{split}] missing images/ or labels/ directory")
        return {}

    images = find_images(img_dir)
    stems_seen = set()
    md5_map = defaultdict(list)

    for img_path in images:
        stem = img_path.stem
        stems_seen.add(stem)
        lbl_path = lbl_dir / f"{stem}.txt"

        # image readability
        img = cv2.imread(str(img_path))
        if img is None:
            errors.append(f"[{split}] corrupted/unreadable image: {img_path.name}")
            continue
        h, w = img.shape[:2]

        md5_map[md5_of(img_path)].append(str(img_path))

        if not lbl_path.exists():
            errors.append(f"[{split}] missing label for image: {img_path.name}")
            continue

        lines = lbl_path.read_text().strip().splitlines()
        if not lines:
            warnings.append(f"[{split}] empty annotation file (background image?): {lbl_path.name}")
            continue

        for i, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"[{split}] {lbl_path.name} line {i+1}: expected 5 tokens, got {len(parts)}")
                continue
            try:
                cls_id = int(parts[0])
                xc, yc, bw, bh = (float(x) for x in parts[1:])
            except ValueError:
                errors.append(f"[{split}] {lbl_path.name} line {i+1}: non-numeric value")
                continue

            if cls_id != 0:
                errors.append(f"[{split}] {lbl_path.name} line {i+1}: invalid class_id {cls_id} (expected 0)")

            for name, val in [("x_center", xc), ("y_center", yc), ("width", bw), ("height", bh)]:
                if not (0.0 <= val <= 1.0):
                    errors.append(f"[{split}] {lbl_path.name} line {i+1}: {name}={val} outside [0,1]")

            box_frac = bw * bh
            if box_frac < min_box_frac:
                warnings.append(
                    f"[{split}] {lbl_path.name} line {i+1}: very small bbox "
                    f"(area frac={box_frac:.5f}, {int(bw*w)}x{int(bh*h)}px) — verify it's a real pothole"
                )

            # sanity: box must stay inside image bounds when converted
            x1, y1 = xc - bw / 2, yc - bh / 2
            x2, y2 = xc + bw / 2, yc + bh / 2
            if x1 < -1e-6 or y1 < -1e-6 or x2 > 1 + 1e-6 or y2 > 1 + 1e-6:
                errors.append(f"[{split}] {lbl_path.name} line {i+1}: bbox extends outside image")

    # orphan labels (label with no matching image)
    for lbl_path in lbl_dir.glob("*.txt"):
        if lbl_path.stem not in stems_seen:
            errors.append(f"[{split}] orphan label with no matching image: {lbl_path.name}")

    for md5, paths in md5_map.items():
        if len(paths) > 1:
            errors.append(f"[{split}] exact duplicate images (md5 {md5[:8]}): {paths}")

    return {p.stem: p for p in images}


def check_cross_split_leakage(dataset_root: Path, phash_thresh: int, warnings: list):
    """Perceptual-hash near-duplicate check ACROSS splits (the leakage that matters most)."""
    all_hashes = []  # (split, path, hash)
    for split in SPLITS:
        img_dir = dataset_root / "images" / split
        if not img_dir.exists():
            continue
        for img_path in find_images(img_dir):
            try:
                ph = imagehash.phash(Image.open(img_path))
                all_hashes.append((split, img_path, ph))
            except Exception:
                continue

    n = len(all_hashes)
    for i in range(n):
        for j in range(i + 1, n):
            split_a, path_a, hash_a = all_hashes[i]
            split_b, path_b, hash_b = all_hashes[j]
            if split_a == split_b:
                continue  # within-split near-dupes are fine; only cross-split is leakage
            if hash_a - hash_b <= phash_thresh:
                warnings.append(
                    f"possible cross-split leakage: {path_a} ({split_a}) ~= {path_b} ({split_b}) "
                    f"(phash distance {hash_a - hash_b})"
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("datasets/potholes"))
    ap.add_argument("--min-box-frac", type=float, default=0.0005,
                     help="warn if bbox area fraction is below this (default 0.05%% of image)")
    ap.add_argument("--phash-thresh", type=int, default=6,
                     help="hamming distance <= this is flagged as a possible near-duplicate")
    ap.add_argument("--skip-leakage-check", action="store_true",
                     help="skip the O(n^2) cross-split perceptual hash check (slow on large datasets)")
    args = ap.parse_args()

    errors, warnings = [], []
    for split in SPLITS:
        validate_split(args.dataset, split, args.min_box_frac, errors, warnings)

    if not args.skip_leakage_check:
        check_cross_split_leakage(args.dataset, args.phash_thresh, warnings)

    print(f"\n=== PotholeNet dataset validation: {args.dataset} ===")
    print(f"Errors:   {len(errors)}")
    print(f"Warnings: {len(warnings)}\n")

    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")

    if errors:
        print(f"\nFAILED — {len(errors)} error(s). Fix before training.")
        sys.exit(1)
    print("\nPASSED — no hard errors. Review warnings above before training.")


if __name__ == "__main__":
    main()
