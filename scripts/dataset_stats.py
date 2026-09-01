#!/usr/bin/env python3
"""
dataset_stats.py — quick architecture/composition report for the PotholeNet dataset.

Usage:
  python dataset_stats.py --dataset ../datasets/potholes
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2

SPLITS = ["train", "val", "test"]
VALID_EXTS = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    args = ap.parse_args()

    print(f"=== PotholeNet dataset report: {args.dataset} ===\n")

    total_images, total_labels, total_boxes = 0, 0, 0
    resolutions = Counter()
    boxes_per_split = defaultdict(int)
    empty_labels_per_split = defaultdict(int)
    ext_counts = Counter()

    for split in SPLITS:
        img_dir = args.dataset / "images" / split
        lbl_dir = args.dataset / "labels" / split
        if not img_dir.exists():
            print(f"[{split}] MISSING images dir\n")
            continue

        images = [p for p in img_dir.iterdir() if p.suffix.lower() in VALID_EXTS]
        labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        n_boxes = 0
        n_empty = 0
        for lbl_path in labels:
            lines = lbl_path.read_text().strip().splitlines()
            if not lines:
                n_empty += 1
            n_boxes += len(lines)

        # sample resolutions (first 100 images — full scan is slow on large sets)
        for img_path in images[:100]:
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                resolutions[f"{w}x{h}"] += 1

        for img_path in images:
            ext_counts[img_path.suffix.lower()] += 1

        print(f"[{split}]")
        print(f"  images: {len(images)}")
        print(f"  labels: {len(labels)}  (missing: {len(images) - len(labels)})")
        print(f"  total bboxes: {n_boxes}")
        print(f"  avg boxes/image: {n_boxes/len(images):.2f}" if images else "  avg boxes/image: n/a")
        print(f"  empty (background) label files: {n_empty}")
        print()

        total_images += len(images)
        total_labels += len(labels)
        total_boxes += n_boxes
        boxes_per_split[split] = n_boxes
        empty_labels_per_split[split] = n_empty

    print("=== Totals ===")
    print(f"Total images: {total_images}")
    print(f"Total labels: {total_labels}")
    print(f"Total bboxes (pothole instances): {total_boxes}")
    if total_images:
        train_n = boxes_per_split.get("train", 0)
        val_n = boxes_per_split.get("val", 0)
        test_n = boxes_per_split.get("test", 0)
        print(f"\nSplit ratio (by image count):")
        for split in SPLITS:
            img_dir = args.dataset / "images" / split
            n = len([p for p in img_dir.iterdir() if p.suffix.lower() in VALID_EXTS]) if img_dir.exists() else 0
            print(f"  {split}: {n} ({n/total_images*100:.1f}%)")

    print(f"\nFile extensions in use: {dict(ext_counts)}")
    print(f"Sample resolutions (first 100 imgs/split): {dict(resolutions)}")
    if len(resolutions) > 1:
        print("NOTE: multiple resolutions present — expected when merging sources; "
              "YOLO will letterbox/resize to --imgsz at train time, this is not an error.")


if __name__ == "__main__":
    main()
