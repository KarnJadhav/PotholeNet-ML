#!/usr/bin/env python3
"""
inference.py — run the trained detector on a single image or a directory, CLI-side.
(The FastAPI service in app/ wraps the same logic for the running microservice.)

Usage:
  python inference.py --weights models/potholenet_best.pt --source path/to/image.jpg
  python inference.py --weights models/potholenet_best.pt --source path/to/dir --conf 0.4
"""
import argparse
import json
import time
from pathlib import Path

from ultralytics import YOLO

VALID_EXTS = {".jpg", ".jpeg", ".png"}


def run_one(model: YOLO, img_path: Path, conf: float):
    t0 = time.perf_counter()
    results = model.predict(str(img_path), conf=conf, verbose=False)[0]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        detections.append({
            "class_id": int(box.cls[0]),
            "class_name": "pothole",
            "confidence": round(float(box.conf[0]), 4),
            "bbox": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)},
        })

    return {
        "success": True,
        "image": str(img_path),
        "detections": detections,
        "inference_time_ms": round(elapsed_ms, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--source", required=True, type=Path, help="image file or directory")
    ap.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args()

    model = YOLO(str(args.weights))

    targets = [args.source] if args.source.is_file() else \
        sorted(p for p in args.source.iterdir() if p.suffix.lower() in VALID_EXTS)

    for img_path in targets:
        print(json.dumps(run_one(model, img_path, args.conf), indent=2))


if __name__ == "__main__":
    main()
