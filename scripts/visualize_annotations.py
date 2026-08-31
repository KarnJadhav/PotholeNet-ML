#!/usr/bin/env python3
"""
visualize_annotations.py — draw YOLO bboxes on random samples for manual QC.

Usage:
  python visualize_annotations.py --dataset datasets/potholes --split train --n 40 --out qc_preview
"""
import argparse
import random
from pathlib import Path

import cv2

VALID_EXTS = {".jpg", ".jpeg", ".png"}


def draw_boxes(img_path: Path, lbl_path: Path, out_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    h, w = img.shape[:2]

    if lbl_path.exists():
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            _, xc, yc, bw, bh = parts
            xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, "pothole", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(img, "NO LABEL FILE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("datasets/potholes"))
    ap.add_argument("--split", choices=["train", "val", "test"], default="train")
    ap.add_argument("--n", type=int, default=30, help="number of random samples to render")
    ap.add_argument("--out", type=Path, default=Path("qc_preview"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    img_dir = args.dataset / "images" / args.split
    lbl_dir = args.dataset / "labels" / args.split
    args.out.mkdir(parents=True, exist_ok=True)

    images = [p for p in img_dir.iterdir() if p.suffix.lower() in VALID_EXTS]
    sample = random.sample(images, min(args.n, len(images)))

    ok = 0
    for img_path in sample:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        out_path = args.out / f"qc_{img_path.name}"
        if draw_boxes(img_path, lbl_path, out_path):
            ok += 1

    print(f"Rendered {ok}/{len(sample)} QC preview images to {args.out}/")
    print("Manually eyeball these before trusting the dataset for training.")


if __name__ == "__main__":
    main()
