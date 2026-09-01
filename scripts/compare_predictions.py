#!/usr/bin/env python3
"""
compare_predictions.py — draws GROUND TRUTH (green) vs PREDICTED (red) boxes on
the same image, side by side, for manual false-positive/false-negative review.
This is the actual "eyeball it" step the spec requires before trusting mAP numbers.

Usage:
  python compare_predictions.py --weights runs/potholenet_yolo11n_v13/weights/best.pt \
    --dataset ../datasets/potholes --split val --n 25 --conf 0.35
"""
import argparse
import random
from pathlib import Path

import cv2
from ultralytics import YOLO

VALID_EXTS = {".jpg", ".jpeg", ".png"}


def draw_gt(img, lbl_path, w, h):
    if not lbl_path.exists():
        return
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, xc, yc, bw, bh = parts
        xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
        x1, y1 = int((xc - bw / 2) * w), int((yc - bh / 2) * h)
        x2, y2 = int((xc + bw / 2) * w), int((yc + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)  # green = ground truth


def draw_pred(img, boxes):
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
        conf = float(box.conf[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red = prediction
        cv2.putText(img, f"{conf:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--out", type=Path, default=Path("prediction_review"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model = YOLO(str(args.weights))
    img_dir = args.dataset / "images" / args.split
    lbl_dir = args.dataset / "labels" / args.split
    args.out.mkdir(parents=True, exist_ok=True)

    images = [p for p in img_dir.iterdir() if p.suffix.lower() in VALID_EXTS]
    random.seed(args.seed)
    sample = random.sample(images, min(args.n, len(images)))

    flagged_fn, flagged_fp = [], []  # images worth extra attention

    for img_path in sample:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        lbl_path = lbl_dir / f"{img_path.stem}.txt"

        n_gt = len(lbl_path.read_text().strip().splitlines()) if lbl_path.exists() else 0

        results = model.predict(str(img_path), conf=args.conf, verbose=False)[0]
        n_pred = len(results.boxes)

        draw_gt(img, lbl_path, w, h)
        draw_pred(img, results.boxes)

        cv2.putText(img, f"GT={n_gt} PRED={n_pred}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

        tag = ""
        if n_pred < n_gt:
            tag = "_LIKELY_FN"
            flagged_fn.append(img_path.name)
        elif n_pred > n_gt:
            tag = "_LIKELY_FP"
            flagged_fp.append(img_path.name)

        cv2.imwrite(str(args.out / f"{img_path.stem}{tag}.jpg"), img)

    print(f"Wrote {len(sample)} comparison images to {args.out}/")
    print("Green = ground truth, Red = prediction (with confidence).")
    print(f"\nImages where prediction count < GT count (possible missed potholes): {len(flagged_fn)}")
    for f in flagged_fn:
        print(f"  {f}")
    print(f"\nImages where prediction count > GT count (possible false positives): {len(flagged_fp)}")
    for f in flagged_fp:
        print(f"  {f}")
    print("\nNote: count mismatch is a rough proxy, not exact — a box could be in the "
          "wrong place even when counts match. Open the actual images to confirm.")


if __name__ == "__main__":
    main()
