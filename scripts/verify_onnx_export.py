#!/usr/bin/env python3
"""
verify_onnx_export.py — sanity-check the torch-free ONNX inference path
against the original .pt model's output. Run this BEFORE deploying the
ONNX-based service — subtle letterbox or NMS bugs produce plausible-looking
but wrong boxes that won't show up unless you directly compare.

IMPORTANT: --iou must match on both sides for a fair comparison. Ultralytics'
own model.predict() defaults to IoU 0.7 if not passed explicitly, while
inference_onnx.py's ONNXPotholeDetector defaults to IoU 0.45 (the value
actually used in production, via app/main.py). Comparing PyTorch-at-0.7
against ONNX-at-0.45 produces a false "mismatch" — different NMS strictness,
not a decode bug.

Usage:
  python verify_onnx_export.py \
    --pt-weights runs/potholenet_yolo11n_v2_negatives2/weights/best.pt \
    --onnx-weights runs/potholenet_yolo11n_v2_negatives2/weights/best.onnx \
    --image ../datasets/potholes/images/val/pothole_10.jpg
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # allow `from app.inference_onnx import ...`

from app.inference_onnx import ONNXPotholeDetector
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt-weights", required=True, type=Path)
    ap.add_argument("--onnx-weights", required=True, type=Path)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45,
                     help="NMS IoU threshold — must match on both sides for a fair comparison. "
                          "0.45 matches inference_onnx.py's default (used in production). "
                          "Ultralytics' predict() defaults to 0.7 if not set explicitly, which "
                          "causes a false mismatch here if left unpinned.")
    args = ap.parse_args()

    print(f"=== PyTorch (.pt) model === (conf={args.conf}, iou={args.iou})")
    pt_model = YOLO(str(args.pt_weights))
    pt_results = pt_model.predict(str(args.image), conf=args.conf, iou=args.iou, verbose=False)[0]
    pt_boxes = []
    for box in pt_results.boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        conf = float(box.conf[0])
        pt_boxes.append((conf, x1, y1, x2, y2))
        print(f"  conf={conf:.4f}  bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})")

    print(f"\n=== ONNX Runtime model === (conf={args.conf}, iou={args.iou})")
    onnx_model = ONNXPotholeDetector(str(args.onnx_weights))
    onnx_dets = onnx_model.predict(str(args.image), conf_threshold=args.conf, iou_threshold=args.iou)
    onnx_boxes = []
    for det in onnx_dets:
        onnx_boxes.append((det.confidence, det.x1, det.y1, det.x2, det.y2))
        print(f"  conf={det.confidence:.4f}  bbox=({det.x1:.1f}, {det.y1:.1f}, {det.x2:.1f}, {det.y2:.1f})")

    print(f"\nPyTorch detections: {len(pt_boxes)}  |  ONNX detections: {len(onnx_boxes)}")

    if len(pt_boxes) != len(onnx_boxes):
        print("\n⚠ MISMATCH — different detection counts even with matched IoU/conf thresholds.")
        print("This now points to an actual decode/letterbox discrepancy, not a threshold mismatch.")
        print("Check: imgsz consistency, output tensor shape assumptions in inference_onnx.py.")
        return

    pt_sorted = sorted(pt_boxes, key=lambda x: -x[0])
    onnx_sorted = sorted(onnx_boxes, key=lambda x: -x[0])

    max_box_diff = 0.0
    max_relative_diff = 0.0
    max_conf_diff = 0.0
    for (pc, px1, py1, px2, py2), (oc, ox1, oy1, ox2, oy2) in zip(pt_sorted, onnx_sorted):
        conf_diff = abs(pc - oc)
        box_diff = max(abs(px1 - ox1), abs(py1 - oy1), abs(px2 - ox2), abs(py2 - oy2))
        box_w, box_h = max(px2 - px1, 1.0), max(py2 - py1, 1.0)
        relative_diff = box_diff / min(box_w, box_h)
        max_conf_diff = max(max_conf_diff, conf_diff)
        max_box_diff = max(max_box_diff, box_diff)
        max_relative_diff = max(max_relative_diff, relative_diff)

    print(f"\nMax confidence difference: {max_conf_diff:.4f}")
    print(f"Max bbox coordinate difference: {max_box_diff:.2f} px ({max_relative_diff*100:.2f}% of box size)")

    if max_relative_diff > 0.03 or max_conf_diff > 0.05:
        print("\n⚠ Differences larger than expected floating-point/export noise.")
        print("Do not trust the ONNX path for deployment until this is resolved —")
        print("check the letterbox math and output-tensor layout assumptions in inference_onnx.py.")
    else:
        print("\n✓ ONNX output matches PyTorch closely enough to trust for deployment.")


if __name__ == "__main__":
    main()