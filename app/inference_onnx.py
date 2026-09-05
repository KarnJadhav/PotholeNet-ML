"""
inference_onnx.py — torch-free inference for the fine-tuned YOLO11 detector.

Uses onnxruntime directly (not the Ultralytics Python API, which still imports
torch under the hood even for ONNX models — using it would defeat the point of
this file, which exists specifically to avoid a torch dependency in the
deployed container).

Implements, by hand, the three steps Ultralytics normally does for you:
  1. Letterbox preprocessing (resize + pad, preserving aspect ratio)
  2. Raw output decoding (YOLO11's head outputs [1, 4+nc, num_anchors] —
     box coords + per-class score, no separate objectness term)
  3. Non-Maximum Suppression via cv2.dnn.NMSBoxes (pure OpenCV, no torch)
  4. Rescaling boxes from the padded/resized model input back to the
     ORIGINAL image's pixel coordinates

Correctness of steps 1 and 4 has to match exactly, or you'll get
plausible-looking but subtly wrong bounding boxes. Verify against the
original .pt model's output before trusting this — see verify_onnx_export.py.
"""
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class ONNXDetection:
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class ONNXPotholeDetector:
    def __init__(self, onnx_path: str, imgsz: int = 640):
        self.imgsz = imgsz
        # CPU execution provider only — this class exists for CPU deployment.
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def _letterbox(self, img: np.ndarray):
        """Resize + pad to a square imgsz x imgsz, preserving aspect ratio.
        Returns the padded image plus the scale factor and padding offsets
        needed to map predicted boxes back to the ORIGINAL image size."""
        h, w = img.shape[:2]
        scale = min(self.imgsz / h, self.imgsz / w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h, pad_w = self.imgsz - new_h, self.imgsz - new_w
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2

        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return padded, scale, left, top

    def _preprocess(self, img: np.ndarray):
        padded, scale, pad_x, pad_y = self._letterbox(img)
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None, ...]  # HWC -> CHW -> NCHW
        return np.ascontiguousarray(blob), scale, pad_x, pad_y

    def predict(self, image_path: str, conf_threshold: float = 0.35,
                iou_threshold: float = 0.45) -> list[ONNXDetection]:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"could not read image: {image_path}")
        orig_h, orig_w = img.shape[:2]

        blob, scale, pad_x, pad_y = self._preprocess(img)
        outputs = self.session.run(None, {self.input_name: blob})[0]  # [1, 4+nc, num_anchors]

        # transpose to [num_anchors, 4+nc] for easier row-wise filtering
        preds = outputs[0].transpose(1, 0)
        boxes_cxcywh = preds[:, :4]
        # single-class model: column 4 IS the confidence directly (no separate
        # objectness term in YOLO11's head, unlike YOLOv5's format)
        scores = preds[:, 4]

        keep_mask = scores >= conf_threshold
        boxes_cxcywh = boxes_cxcywh[keep_mask]
        scores = scores[keep_mask]

        if len(scores) == 0:
            return []

        # cxcywh (in imgsz-space) -> xyxy (still in imgsz-space)
        cx, cy, bw, bh = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        w = bw
        h = bh

        nms_boxes = np.stack([x1, y1, w, h], axis=1).tolist()
        nms_scores = scores.tolist()
        keep_indices = cv2.dnn.NMSBoxes(nms_boxes, nms_scores, conf_threshold, iou_threshold)
        if len(keep_indices) == 0:
            return []
        keep_indices = np.array(keep_indices).flatten()

        detections = []
        for i in keep_indices:
            bx1 = x1[i]
            by1 = y1[i]
            bx2 = bx1 + w[i]
            by2 = by1 + h[i]

            # undo letterbox: subtract padding, divide by scale, to get back
            # to ORIGINAL image pixel coordinates
            ox1 = (bx1 - pad_x) / scale
            oy1 = (by1 - pad_y) / scale
            ox2 = (bx2 - pad_x) / scale
            oy2 = (by2 - pad_y) / scale

            # clip to original image bounds
            ox1 = max(0.0, min(ox1, orig_w))
            oy1 = max(0.0, min(oy1, orig_h))
            ox2 = max(0.0, min(ox2, orig_w))
            oy2 = max(0.0, min(oy2, orig_h))

            detections.append(ONNXDetection(
                class_id=0, confidence=float(scores[i]),
                x1=ox1, y1=oy1, x2=ox2, y2=oy2,
            ))

        return detections
