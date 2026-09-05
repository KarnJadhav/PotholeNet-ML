#!/usr/bin/env python3
"""
export_onnx.py — export a trained YOLO11 checkpoint to ONNX for torch-free,
lightweight CPU inference (Render, Cloud Run, or anywhere RAM is constrained).

This uses Ultralytics' own exporter (correct, well-tested) — the export step
itself still needs torch installed (run this on your training machine), but
the RESULTING .onnx file + onnxruntime is all the deployed service needs.

Usage:
  python export_onnx.py --weights runs/potholenet_yolo11n_v13/weights/best.pt
  python export_onnx.py --weights runs/potholenet_yolo11n_v13/weights/best.pt \
    --upload-repo Karn81/PotholeNet-YOLO11n
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--upload-repo", default=None,
                     help="if set, upload the resulting .onnx to this HF Hub model repo "
                          "(same repo as your .pt is fine — just a second file in it)")
    args = ap.parse_args()

    model = YOLO(str(args.weights))
    onnx_path = model.export(format="onnx", imgsz=args.imgsz, simplify=True, opset=12)
    onnx_path = Path(onnx_path)
    print(f"Exported: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    if args.upload_repo:
        from huggingface_hub import upload_file
        upload_file(
            path_or_fileobj=str(onnx_path),
            path_in_repo=onnx_path.name,
            repo_id=args.upload_repo,
        )
        print(f"Uploaded to https://huggingface.co/{args.upload_repo}/blob/main/{onnx_path.name}")

    print("\nVerify correctness before trusting this for deployment:")
    print(f"  python verify_onnx_export.py --pt-weights {args.weights} "
          f"--onnx-weights {onnx_path} --image <path/to/test/image.jpg>")


if __name__ == "__main__":
    main()
