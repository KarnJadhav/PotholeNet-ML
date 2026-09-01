#!/usr/bin/env python3
"""
train.py — fine-tune pretrained YOLO on the PotholeNet dataset.

NEVER trains from scratch: starts from an Ultralytics pretrained checkpoint
(yolo11n.pt / yolo11s.pt), downloaded automatically by ultralytics on first use.

Usage:
  python train.py --model yolo11n.pt --data datasets/potholes/data.yaml --epochs 100 --imgsz 640
  python train.py --model yolo11s.pt --data datasets/potholes/data.yaml --epochs 100 --imgsz 640 --name potholenet_yolo11s
"""
import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


def dataset_counts(data_yaml: Path):
    import yaml
    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg["path"])
    counts = {}
    for split, rel in [("train", cfg["train"]), ("val", cfg["val"]), ("test", cfg.get("test"))]:
        if not rel:
            continue
        img_dir = root / rel
        counts[split] = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png"))) + \
            len(list(img_dir.glob("*.jpeg")))
    return counts


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown (not a git repo or git unavailable)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt", help="pretrained checkpoint to fine-tune from")
    ap.add_argument("--data", type=Path, default=Path("datasets/potholes/data.yaml"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1, help="-1 = ultralytics auto-batch by GPU memory")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default=None, help="defaults to potholenet_<model-stem>")
    ap.add_argument("--dataset-version", default="unspecified",
                     help="tag your dataset snapshot (e.g. git tag or date) for reproducibility")
    args = ap.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA GPU detected — training on CPU will be slow. "
              "Consider Google Colab / Kaggle for the actual training run.")

    run_name = args.name or f"potholenet_{Path(args.model).stem}"

    model = YOLO(args.model)  # loads pretrained weights — fine-tuning, not from scratch
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=run_name,
        device=device,
    )

    # Don't assume the save dir matches args.name — Ultralytics auto-increments
    # (potholenet_yolo11n_v1 -> _v12, _v13, ...) if a prior/interrupted run already
    # used that name. Always read back the ACTUAL dir it used.
    run_dir = Path(model.trainer.save_dir)
    counts = dataset_counts(args.data)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_checkpoint": args.model,
        "dataset_yaml": str(args.data),
        "dataset_version_tag": args.dataset_version,
        "dataset_counts": counts,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "git_commit": git_commit(),
        "run_dir": str(run_dir),
        "best_weights": str(run_dir / "weights" / "best.pt"),
        "last_weights": str(run_dir / "weights" / "last.pt"),
    }

    meta_path = run_dir / "run_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"\nRun metadata written to {meta_path}")
    print(f"best.pt -> {metadata['best_weights']}")
    print(f"last.pt -> {metadata['last_weights']}")
    print("Use best.pt for deployment. Do NOT deploy until evaluate.py has run on the untouched test set.")


if __name__ == "__main__":
    main()
