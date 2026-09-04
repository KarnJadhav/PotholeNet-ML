"""
severity.py — rule-based severity ESTIMATION, deliberately separate from the detector.

The YOLO model returns class + confidence + bbox only. Confidence is NOT severity.
This module turns detector output into a severity label using only information that's
actually available from a single uncalibrated RGB image + bbox geometry:

  - bbox size relative to frame (proxy for "how much of the visible road it covers")
  - how many potholes are in the same frame (road-condition proxy)
  - vertical position in frame (closer to camera => lower in frame, roughly => larger
    real-world size for the same pixel area — a weak proxy, not a distance measurement)

It does NOT and cannot claim real-world depth, diameter, or volume without camera
calibration or a depth signal (stereo, LiDAR, known object of reference, etc.). Every
output includes "estimate_only": true and a plain-language caveat for that reason.

If you later add calibration (known camera height/angle, or a depth model), swap the
scoring function but keep the same output contract so the API layer doesn't change.
"""
from dataclasses import dataclass
from enum import Enum


class SeverityLabel(str, Enum):
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


@dataclass
class SeverityEstimate:
    label: SeverityLabel
    score: float  # 0-1, higher = more severe, NOT a physical measurement
    reasons: list[str]
    estimate_only: bool = True
    caveat: str = (
        "Severity is a heuristic estimate from bounding-box geometry and frame context "
        "only. It is not a physical measurement of depth, diameter, or volume — this "
        "system has no camera calibration or depth sensing. Do not use this as the sole "
        "basis for prioritizing repairs; use it to help triage for human review."
    )


# Tunable thresholds — recalibrate against real Indian-road field data once you have
# enough labeled examples to compare estimated severity against on-the-ground assessment.
AREA_FRAC_THRESHOLDS = {"minor": 0.01, "moderate": 0.04}  # bbox area / image area
MULTI_POTHOLE_BONUS_PER_EXTRA = 0.08
LOWER_FRAME_BONUS_MAX = 0.15  # boxes lower in frame (closer to camera) nudge severity up


def _area_frac(box: Detection, img_w: float, img_h: float) -> float:
    w = max(0.0, box.x2 - box.x1)
    h = max(0.0, box.y2 - box.y1)
    return (w * h) / (img_w * img_h)


def _vertical_position_frac(box: Detection, img_h: float) -> float:
    """0 = top of frame (far away), 1 = bottom of frame (close to camera)."""
    center_y = (box.y1 + box.y2) / 2
    return min(1.0, max(0.0, center_y / img_h))


def estimate_severity(
    detections: list[Detection],
    image_width: float,
    image_height: float,
) -> list[SeverityEstimate]:
    """One estimate per detection, but informed by how many potholes share the frame."""
    if not detections:
        return []

    multi_bonus = MULTI_POTHOLE_BONUS_PER_EXTRA * (len(detections) - 1)
    results = []

    for det in detections:
        area_frac = _area_frac(det, image_width, image_height)
        vpos = _vertical_position_frac(det, image_height)
        proximity_bonus = LOWER_FRAME_BONUS_MAX * vpos

        score = min(1.0, area_frac * 10 + multi_bonus + proximity_bonus)

        reasons = [f"bbox covers {area_frac*100:.2f}% of frame area"]
        if len(detections) > 1:
            reasons.append(f"{len(detections)} potholes detected in same frame (+{multi_bonus:.2f})")
        if proximity_bonus > 0.01:
            reasons.append(f"positioned low in frame, likely closer to camera (+{proximity_bonus:.2f})")

        if area_frac < AREA_FRAC_THRESHOLDS["minor"] and score < 0.3:
            label = SeverityLabel.MINOR
        elif area_frac < AREA_FRAC_THRESHOLDS["moderate"] and score < 0.6:
            label = SeverityLabel.MODERATE
        else:
            label = SeverityLabel.SEVERE

        results.append(SeverityEstimate(label=label, score=round(score, 3), reasons=reasons))

    return results


if __name__ == "__main__":
    # quick smoke test
    dets = [
        Detection(x1=120, y1=300, x2=450, y2=470, confidence=0.94),
        Detection(x1=500, y1=100, x2=540, y2=130, confidence=0.61),
    ]
    for est in estimate_severity(dets, image_width=640, image_height=480):
        print(est)
