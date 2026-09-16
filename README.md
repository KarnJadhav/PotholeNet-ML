# PotholeNet-ML

A fine-tuned YOLO11 pothole detection microservice with a decoupled rule-based severity estimator, served via FastAPI and designed to sit behind a Node.js backend.

[![Status](https://img.shields.io/badge/status-baseline%20trained-yellow)]()
[![Model](https://img.shields.io/badge/model-YOLO11n-blue)]()
[![Python](https://img.shields.io/badge/python-3.11-blue)]()

---

## Table of Contents

- [Project Status](#project-status)
- [Architecture](#architecture)
- [Repository Layout](#repository-layout)
- [Environment Setup](#environment-setup)
- [Dataset](#dataset)
  - [Sources](#sources)
  - [Preparation Pipeline](#preparation-pipeline)
  - [Validation Gate](#validation-gate)
  - [Known Dataset Limitations](#known-dataset-limitations)
- [Training](#training)
- [Evaluation](#evaluation)
- [Baseline Results](#baseline-results)
- [Error Analysis Workflow](#error-analysis-workflow)
- [Severity Estimation](#severity-estimation)
- [FastAPI Service](#fastapi-service)
- [Node.js Integration](#nodejs-integration)
- [Configuration Reference](#configuration-reference)
- [Production Readiness](#production-readiness)
- [Design Principles](#design-principles)
- [Roadmap](#roadmap)
- [Contributing / Reproducing](#contributing--reproducing)

---

## Project Status

| Component | Status |
|---|---|
| Dataset merge pipeline (`prepare_dataset.py`) | ✅ Built, source-tagged + content-deduped before split |
| Dataset validation (`validate_dataset.py`) | ✅ Passing — 0 errors |
| Duplicate/leakage remediation (`dedupe_dataset.py`, `inspect_dedupe_clusters.py`) | ✅ Built and used to fix a real cross-split leakage bug (see [Known Dataset Limitations](#known-dataset-limitations)) |
| Annotation visual QC (`visualize_annotations.py`) | ✅ Used before training |
| YOLO11n baseline training | ✅ Complete — see [Baseline Results](#baseline-results) |
| YOLO11s / YOLO11m comparison | ⏳ Not yet run |
| Manual false-positive / false-negative review | 🔶 In progress (`compare_predictions.py`) |
| Test-set evaluation (one-time, post-decision) | ⏳ Not yet run — still iterating on val |
| FastAPI service (`/predict`, `/health`) | ✅ Built, model loads once at startup |
| Severity module | ✅ Built, rule-based, thresholds are placeholders |
| Node.js integration | ✅ Example route provided, not yet wired into a production backend |
| Rate limiting | ❌ Not implemented |
| Hard inference timeout | ❌ Logged only, not enforced |

**Current baseline: YOLO11n, val mAP50 = 0.771, mAP50-95 = 0.544.** See [Baseline Results](#baseline-results) for the full breakdown and what's still unverified.

---

## Architecture

```
                     ┌──────────────────┐
                     │     Frontend     │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  Node.js Backend │
                     │  auth / upload   │
                     │  rate limiting   │
                     └────────┬─────────┘
                              │ HTTP / multipart
                              ▼
                ┌────────────────────────────┐
                │      PotholeNet-ML          │
                │      FastAPI Service        │
                │                              │
                │  POST /predict               │
                │       │                       │
                │       ▼                       │
                │  YOLO11 Detector              │
                │       │                       │
                │       ▼                       │
                │  Severity Estimator           │
                │  (separate, rule-based)       │
                └──────────────┬───────────────┘
                               │
                               ▼
                        Detection JSON
```

The frontend never calls the ML service directly — the Node backend is the only intended caller. This keeps model versioning, auth, and rate limiting out of the ML service's concerns.

---

## Repository Layout

```
PotholeNet-ML/
├── backend-integration-example.js   # Express route: POST /api/detection/image
└── ml-service/
    ├── .env.example
    ├── .gitignore                   # dataset images/labels, weights, review dirs excluded
    ├── requirements.txt
    ├── README.md
    │
    ├── app/
    │   ├── __init__.py
    │   ├── main.py                  # FastAPI: /predict, /health
    │   └── severity.py              # rule-based severity estimator (decoupled)
    │
    ├── datasets/potholes/
    │   ├── data.yaml
    │   ├── images/{train,val,test}/ # not tracked in git — see .gitignore
    │   └── labels/{train,val,test}/ # not tracked in git — see .gitignore
    │
    ├── models/                      # deployment copy of best.pt goes here
    ├── runs/                        # training run outputs — not tracked in git
    │
    └── scripts/
        ├── prepare_dataset.py       # merge RDD2022 + Pothole-600 + custom → YOLO format
        ├── validate_dataset.py      # hard QC gate
        ├── dedupe_dataset.py        # fix duplicate/leakage found by validator
        ├── inspect_dedupe_clusters.py  # visual sanity-check before deleting anything
        ├── dataset_stats.py         # per-split counts, bbox stats, resolutions
        ├── visualize_annotations.py # render GT boxes for manual QC
        ├── train.py                 # fine-tune YOLO11n/s/m, logs full repro metadata
        ├── evaluate.py              # precision/recall/mAP/latency on val or test
        ├── inference.py             # CLI single-image / directory inference
        └── compare_predictions.py   # GT vs. prediction overlay for FP/FN review
```

Weight files (`*.pt`), the dataset's actual images/labels, and generated review folders (`qc_preview/`, `dedupe_review/`, `prediction_review/`, `runs/`) are gitignored. This repo tracks code and configuration, not binaries — reproduce the dataset and model locally using the scripts below.

---

## Environment Setup

Development environment this project was built and trained on:

```
Host OS:      Windows 11
Runtime:      WSL2, Ubuntu 24.04.4 LTS
Python:       3.11.16 (conda env `potholenet`)
GPU:          NVIDIA GeForce RTX 5060, 8 GB VRAM
PyTorch:      2.11.0+cu128
Ultralytics:  8.3.0
```

**GPU note:** RTX 50-series (Blackwell, `sm_120`) requires a PyTorch build with CUDA 12.8 (`cu128`) support. A `cu121`-era install will detect the GPU but is not actually compatible and will warn or fail — verify with:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output should show a `+cu128` (or newer) build.

### Install

```bash
cd ml-service
conda create -n potholenet python=3.11
conda activate potholenet
python -m pip install -r requirements.txt
```

If installing PyTorch separately for a specific CUDA version:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

---

## Dataset

### Sources

| Source | Notes |
|---|---|
| **RDD2022** | Road Damage Dataset. Only the `D40` (pothole) class is extracted; other damage classes are dropped. Annotations converted from Pascal VOC/XML to YOLO format. |
| **Pothole-600** | ~600 images with Asian/Indian-relevant road scenes. Verify license before redistribution. |
| **Custom Indian-road images** | Collected separately — dashcam, smartphone, field images. Intended to cover daylight/low-light, wet roads, shadows, multiple/partial potholes, varied camera heights. |

Before merging any source: check its license, verify class definitions, and check for duplicates against the other sources — see [Known Dataset Limitations](#known-dataset-limitations) for what happens when this step is skipped.

### Preparation Pipeline

```bash
python scripts/prepare_dataset.py \
  --rdd2022-images /path/to/rdd/images \
  --rdd2022-annots /path/to/rdd/annotations \
  --pothole600-images /path/to/pothole600/images \
  --pothole600-labels /path/to/pothole600/labels \
  --custom-images /path/to/custom/images \
  --custom-labels /path/to/custom/labels \
  --out datasets/potholes
```

This script:
- Converts RDD2022 VOC XML → YOLO format, keeping only `D40` boxes
- Forces all source class IDs to `0` (single-class detector)
- Runs a **content-based dedup pass before splitting** (md5 exact-match + phash near-duplicate clustering) — this exists specifically because merging multiple public pothole datasets tends to pull in the same underlying images under different filenames/formats
- Groups images by sequence ID so a video/burst never splits across train/val/test
- Writes into `datasets/potholes/images|labels/<split>/` with source-tagged filenames

### Validation Gate

**Mandatory before every training run:**

```bash
python scripts/validate_dataset.py --dataset datasets/potholes
```

Checks: image/label pairing, invalid class IDs, malformed rows, out-of-bounds bboxes, empty label files, exact duplicates (within-split), and cross-split near-duplicate leakage (perceptual hash).

If it reports errors, save the output and investigate before fixing anything:

```bash
python scripts/validate_dataset.py --dataset datasets/potholes > validation.txt 2>&1
grep "^ERROR" validation.txt | cut -d' ' -f2 | sort | uniq -c
```

**Do not blindly delete flagged images.** Near-duplicate (phash) flags in particular can be false positives — pothole photos naturally share asphalt texture and camera angle, which perceptual hashing can mistake for duplication. Use `inspect_dedupe_clusters.py` to visually confirm before deleting:

```bash
python scripts/inspect_dedupe_clusters.py --dataset datasets/potholes --phash-thresh 6 --sample 15
```

This prints cluster-size distribution (large clusters suggest false-positive chaining, not real dupes) and writes side-by-side montage images to `dedupe_review/` for manual confirmation. Once confirmed, resolve:

```bash
# exact-only (safe, no false-positive risk) — removes byte-identical images,
# and consolidates any that span multiple splits (real leakage)
python scripts/dedupe_dataset.py --dataset datasets/potholes --exact-only --dry-run
python scripts/dedupe_dataset.py --dataset datasets/potholes --exact-only
```

### Known Dataset Limitations

Documenting these honestly rather than hiding them:

1. **Split ratio drift.** After removing exact-duplicate/cross-split-leaked images, the split moved from the target 70/20/10 to roughly **56/27/16** (896 train / 435 val / 257 test). This happened because duplicate resolution prioritizes keeping test and val intact (they're smaller and more expensive to reconstruct correctly) and strips duplicates from train first. **Train is thinner than intended** — expanding it with genuinely new images (not more of the same duplicated sources) is a priority before the next training iteration.

2. **Zero background (no-pothole) images.** Every image in the current dataset has at least one labeled pothole. The original spec called for negative examples — road patches, shadows, cracks, manholes that should *not* be classified as potholes — and there currently aren't any. This likely means false-positive rates on clean/ambiguous road surfaces are **untested**, not necessarily good.

3. **Resolution mismatch.** Source images range from ~140px to 2000px on a side, most well below the `imgsz=640` training resolution. Small source images get upsampled, which can blur exactly the fine detail needed for small/distant pothole detection. Combined with a meaningful number of already-small bounding boxes in the label set, small-object recall is a known risk area — see [Baseline Results](#baseline-results).

4. **Sequence metadata isn't recoverable for the currently merged dataset.** The images currently in `datasets/potholes/` were assembled from a pre-existing organized dump rather than run end-to-end through `prepare_dataset.py`, so filenames don't carry the sequence tags the split-leakage prevention relies on. Future additions to this dataset should go through `prepare_dataset.py` from raw sources so sequence-aware splitting applies correctly.

---

## Training

Fine-tuning only — this project never trains a detector from scratch.

```bash
python scripts/train.py \
  --model yolo11n.pt \
  --data datasets/potholes/data.yaml \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --name potholenet_yolo11n_v1
```

- `--model` accepts `yolo11n.pt`, `yolo11s.pt`, or `yolo11m.pt` — pretrained weights download automatically via Ultralytics on first use.
- `--batch 8` is a safe explicit starting point for an 8 GB card; Ultralytics auto-batch (`--batch -1`) is also available but a fixed value is more predictable on unfamiliar hardware.
- YOLO11m is meaningfully heavier than n/s — don't assume it fits without monitoring VRAM.

**Reproducibility:** `train.py` writes a `run_metadata.json` alongside the weights, capturing dataset counts, hyperparameters, GPU, and package versions. It reads the actual save directory back from `model.trainer.save_dir` rather than assuming it matches `--name` — Ultralytics silently auto-increments the run folder (`_v1` → `_v13`, etc.) if a prior/interrupted run already used that name, and earlier versions of this script wrote metadata to the wrong (non-existent) path when that happened.

**Interrupted training:** Ultralytics only writes `weights/last.pt` after an epoch fully completes (train + validation). If training is interrupted before epoch 1 finishes, there is nothing to resume from — just restart. If a checkpoint exists:

```python
from ultralytics import YOLO
model = YOLO("runs/<run_name>/weights/last.pt")
model.train(resume=True)
```

---

## Evaluation

```bash
# iterate against validation while tuning
python scripts/evaluate.py --weights runs/<run_name>/weights/best.pt --data datasets/potholes/data.yaml --split val

# ONE time, once val-based iteration is finished
python scripts/evaluate.py --weights runs/<run_name>/weights/best.pt --data datasets/potholes/data.yaml --split test
```

Reports precision, recall, mAP50, mAP50-95, average inference latency, and FPS estimate. **mAP is not sufficient sign-off** — it does not tell you *where* the model fails. Manual review is required before treating a model as usable.

---

## Baseline Results

**YOLO11n, 100 epochs, imgsz=640, batch=8, RTX 5060 8GB, ~20 minutes wall-clock.**

| Metric | Validation |
|---|---|
| Precision | 0.767 |
| Recall | 0.682 |
| mAP50 | 0.771 |
| mAP50-95 | 0.544 |
| Avg. inference latency | ~45 ms |
| FPS estimate | ~22 |

`best.pt` was selected from an earlier epoch (val mAP50 peaked around epoch 60–82 at ~0.76) rather than epoch 100, where mAP50 had drifted slightly lower (~0.749) — Ultralytics' checkpoint selection handled this correctly.

**This number is not a deployment sign-off.** Per the [Known Dataset Limitations](#known-dataset-limitations) above — zero background images in training data — false-positive behavior on shadows, cracks, manholes, and puddles is not represented in this metric at all. Manual review via `compare_predictions.py` is in progress; see [Error Analysis Workflow](#error-analysis-workflow).

YOLO11s and YOLO11m comparisons have not yet been run on this dataset.

---

## Error Analysis Workflow

mAP alone doesn't tell you what's actually going wrong. After training:

```bash
python scripts/compare_predictions.py \
  --weights runs/<run_name>/weights/best.pt \
  --dataset datasets/potholes \
  --split val \
  --n 25 --conf 0.35
```

This draws **ground truth in green, predictions in red** on the same image, and heuristically tags output filenames `_LIKELY_FN` (predicted fewer boxes than labeled — possible miss) or `_LIKELY_FP` (predicted more — possible false alarm) based on box count mismatch. Count mismatch is a rough proxy, not exact — a matched count can still hide a misplaced box — so open the images.

Pay particular attention to the failure modes called out in the original spec:

- **False positives on:** road patches, shadows, manholes, cracks, puddles, vehicle shadows, pavement markings
- **False negatives on:** small/distant potholes, partially occluded potholes, poor lighting, wet roads, unusual shapes

Error-driven improvement loop:

```
inference.py on real field images
        ↓
collect FP / FN examples
        ↓
correct or add annotations
        ↓
re-run validate_dataset.py
        ↓
retrain
        ↓
re-evaluate on val (not test)
```

Never repeatedly evaluate against the test set during this loop — that reintroduces the same overfitting risk the sequence-aware split was designed to prevent, just at the model-selection level instead of the data level.

---

## Severity Estimation

`app/severity.py` is deliberately decoupled from the detector. The YOLO model answers "where is the pothole"; severity answers "how serious does it look" — and does so **without ever claiming to measure physical depth, diameter, or volume**, since a single uncalibrated RGB image and bounding box cannot support that claim.

Inputs used:
- Bounding-box area as a fraction of frame area
- Number of potholes detected in the same frame
- Vertical position in frame (weak proxy for proximity to camera)

Output shape:

```json
{
  "label": "moderate",
  "score": 0.41,
  "reasons": [
    "bbox covers 2.10% of frame area",
    "positioned low in frame, likely closer to camera (+0.09)"
  ],
  "estimate_only": true,
  "caveat": "Severity is a heuristic estimate ... not a physical measurement ..."
}
```

Thresholds (`AREA_FRAC_THRESHOLDS`, proximity/multi-pothole bonuses) are **placeholders** — they have not been calibrated against real field data or human severity judgments, and shouldn't be treated as tuned until that calibration happens. Doing that calibration now, before the detector itself has been through error-driven improvement, would mostly be guessing against a moving target.

---

## FastAPI Service

```bash
cp .env.example .env   # edit ML_MODEL_PATH etc.
cp runs/<run_name>/weights/best.pt models/potholenet_best.pt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The model loads once at startup via FastAPI's `lifespan` context — not per-request.

### `GET /health`

```json
{"status": "ok", "model_loaded": true, "model_version": "potholenet_best", "load_error": null}
```

### `POST /predict` (multipart, field name `file`)

```bash
curl -X POST http://localhost:8000/predict -F "file=@road.jpg"
```

```json
{
  "success": true,
  "detections": [
    {
      "class_id": 0, "class_name": "pothole", "confidence": 0.94,
      "bbox": {"x1": 120, "y1": 80, "x2": 450, "y2": 310},
      "severity": {
        "label": "moderate", "score": 0.41,
        "reasons": ["bbox covers 2.10% of frame area"],
        "estimate_only": true,
        "caveat": "Severity is a heuristic estimate ..."
      }
    }
  ],
  "inference_time_ms": 42.1,
  "model_version": "potholenet_best"
}
```

Uploaded files are written to `/tmp` for the duration of inference only and deleted immediately after (`finally: tmp_path.unlink()`), regardless of success or failure.

---

## Node.js Integration

`backend-integration-example.js` provides a drop-in Express route (`POST /api/detection/image`) that validates the upload, forwards it to the ML service, and handles the ML service's error states (503 model not loaded, 504 timeout, 502 unreachable). The frontend calls the Node backend; the Node backend calls the ML service. The frontend never calls the ML service directly.

---

## Configuration Reference

`.env` (copy from `.env.example`):

| Variable | Purpose | Default |
|---|---|---|
| `ML_MODEL_PATH` | Path to the deployed weights file | `models/potholenet_best.pt` |
| `ML_CONFIDENCE_THRESHOLD` | Minimum confidence to report a detection | `0.35` |
| `ML_MAX_IMAGE_SIZE_MB` | Reject uploads larger than this | `10` |
| `ML_TIMEOUT_MS` | Logged if inference exceeds this — not yet a hard cutoff | `30000` |
| `ML_ALLOWED_ORIGINS` | CORS allow-list | `http://localhost:5000` |
| `ML_LOG_LEVEL` | Logging verbosity | `INFO` |

Never commit `.env` — only `.env.example` should be tracked.

---

## Production Readiness

### API service

- [x] Request size limits, file type validation
- [x] Model loaded once at startup, version reported in responses
- [x] Structured logging, health check
- [x] Temp file cleanup after every request (success or failure)
- [x] CORS restricted to configured origins
- [ ] **Rate limiting** — not implemented. Recommend `slowapi` in front of `/predict`, or enforce at the Node layer since it's the only allowed caller
- [ ] **Hard inference timeout** — currently logged only (`ML_TIMEOUT_MS`), does not actually cancel a slow inference. Add `asyncio.wait_for` or a worker-based cutoff if this matters for your latency SLA

### Dataset

- [x] Sources identified, provenance tracked via filename tagging
- [x] Converted to unified YOLO format, single class
- [x] Duplicate/leakage remediation completed for current merge
- [ ] License verification for RDD2022 and Pothole-600 redistribution — confirm before any commercial deployment
- [ ] Background/negative examples — currently absent, see [Known Dataset Limitations](#known-dataset-limitations)
- [ ] Train set expansion — current 896 images is thin after dedup

### Model

- [x] YOLO11n baseline trained and evaluated on validation
- [ ] Manual FP/FN review completed and acted on
- [ ] YOLO11s / YOLO11m comparison
- [ ] Test-set evaluation (one-time, after the above)
- [ ] Final model frozen for deployment

**A model is not production-ready because training completed successfully.** Deployment should require: a clean `validate_dataset.py` pass, manual annotation review, a single clean test-set evaluation, manual false-positive/negative review, and inference checks against real Indian-road field images the model has not seen during training or validation.

---

## Design Principles

- **Fine-tune, never train from scratch.** All training starts from pretrained YOLO11 weights.
- **One detector class.** `0 = pothole`. Severity levels are not detector classes.
- **Detection and severity are separate systems.** The detector answers "where"; severity answers "how bad, as an estimate." Neither should leak into the other's responsibility.
- **Validate before training, every time.** A dataset with unresolved validation errors should never be used to train.
- **Sequence-aware splitting.** Frame/video-derived data must group by source sequence when splitting, not by individual image.
- **Test set stays untouched during iteration.** Only evaluated once, after model/dataset decisions are otherwise finalized.
- **Every training run is reproducible.** Model, dataset counts, software versions, GPU, hyperparameters, and results are all recorded automatically.

---

## Roadmap

1. Manual FP/FN review of the current baseline (`compare_predictions.py`) — in progress
2. Expand train split with genuinely new (non-duplicate) Indian-road images
3. Add background/negative examples (clean road, shadows, cracks, manholes) with no pothole label
4. Re-run baseline training with expanded dataset
5. Train and compare YOLO11s (and YOLO11m if VRAM/latency budget allows)
6. One-time test-set evaluation on the selected model
7. Calibrate severity thresholds against real field data once detector quality is settled
8. Implement rate limiting and hard inference timeout
9. Wire into production Node.js backend with auth/authorization

---

## Contributing / Reproducing

This repo intentionally does not track dataset images/labels or trained weights (see `.gitignore`). To reproduce:

1. Obtain RDD2022 and Pothole-600 yourself, verify their licenses for your use case
2. Collect and label your own Indian-road images
3. Run `prepare_dataset.py` → `validate_dataset.py` → fix any errors → `visualize_annotations.py`
4. Run `train.py`, then `evaluate.py`, then `compare_predictions.py` for manual review
5. Iterate per [Error Analysis Workflow](#error-analysis-workflow) before considering deployment

Issues and PRs that improve dataset quality, add real background examples, or extend the FP/FN review tooling are the most valuable contributions at this stage — the codebase itself is functionally complete for a first iteration; the data is the bottleneck.
