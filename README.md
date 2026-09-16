<!-- # PotholeNet-ML

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

Issues and PRs that improve dataset quality, add real background examples, or extend the FP/FN review tooling are the most valuable contributions at this stage — the codebase itself is functionally complete for a first iteration; the data is the bottleneck. -->
# PotholeNet-ML

A fine-tuned YOLO11 pothole detection microservice with a decoupled rule-based severity estimator, deployable as either a GPU-backed PyTorch service or a lightweight, torch-free ONNX Runtime service, sitting behind a Node.js backend.

[![Status](https://img.shields.io/badge/status-deployed-brightgreen)]()
[![Model](https://img.shields.io/badge/model-YOLO11n-blue)]()
[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![License](https://img.shields.io/badge/code%20license-MIT-lightgrey)]()

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
  - [Negative Example Injection](#negative-example-injection)
  - [Known Dataset Limitations](#known-dataset-limitations)
- [Training](#training)
- [Evaluation](#evaluation)
- [Experiment Results](#experiment-results)
- [Known Limitation: Patch/Repair False Positive](#known-limitation-patchrepair-false-positive)
- [Error Analysis Workflow](#error-analysis-workflow)
- [Severity Estimation](#severity-estimation)
- [ONNX Export & Deployment](#onnx-export--deployment)
- [FastAPI Service](#fastapi-service)
- [Node.js Integration](#nodejs-integration)
- [Configuration Reference](#configuration-reference)
- [Production Readiness](#production-readiness)
- [Design Principles](#design-principles)
- [Roadmap](#roadmap)
- [Contributing / Reproducing](#contributing--reproducing)
- [License & Dataset Attribution](#license--dataset-attribution)

---

## Project Status

| Component | Status |
|---|---|
| Dataset merge pipeline (`prepare_dataset.py`) | ✅ Source-tagged, content-deduped before split |
| Dataset validation (`validate_dataset.py`) | ✅ Passing — 0 errors |
| Duplicate/leakage remediation (`dedupe_dataset.py`, `inspect_dedupe_clusters.py`) | ✅ Fixed a real cross-split leakage bug affecting ~1,000+ images (two source dumps of the same underlying images merged under different filenames) |
| Negative/background examples | ✅ Added — 317 images from the Attain dataset (faded-marking + patch classes), see [Negative Example Injection](#negative-example-injection) |
| YOLO11n baseline training | ✅ Complete, 3 experiments run — see [Experiment Results](#experiment-results) |
| YOLO11s / YOLO11m comparison | ⏳ Not yet run |
| Manual false-positive / false-negative review | ✅ Done against real-world (non-dataset) images — found and partially resolved a real false-positive pattern, see [Known Limitation](#known-limitation-patchrepair-false-positive) |
| Test-set evaluation (one-time, post-decision) | ⏳ Not yet run — still iterating on val |
| ONNX export pipeline | ✅ Built, verified against PyTorch output, deployed |
| FastAPI service (`/predict`, `/health`) | ✅ Deployed live on Render |
| Severity module | ✅ Built, rule-based, thresholds are placeholders |
| Node.js integration | ⏳ Example route provided (`backend-integration-example.js`), **not yet wired into the actual backend codebase** |
| Rate limiting | ❌ Not implemented |
| Hard inference timeout | ❌ Logged only, not enforced |
| License compliance tracking | ✅ See [License & Dataset Attribution](#license--dataset-attribution) — some sources still unconfirmed |

**Deployed model: YOLO11n, val precision 0.772 / recall 0.659 / mAP50 0.754 / mAP50-95 0.491.** This is a deliberate rollback from a later experiment that scored better on paper but performed worse on real-world test images — see [Experiment Results](#experiment-results) for why aggregate metrics were overridden by a manual regression test.

---

## Architecture
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
            │   PotholeNet-ML FastAPI     │
            │   (deployed on Render)      │
            │                              │
            │  POST /predict               │
            │       │                       │
            │       ▼                       │
            │  ML_BACKEND=onnx:             │
            │  ONNX Runtime (CPU, torch-    │
            │  free) — app/inference_onnx.py│
            │       │                       │
            │       ▼                       │
            │  Severity Estimator           │
            │  (separate, rule-based)       │
            └──────────────┬───────────────┘
                           │
                           ▼
                    Detection JSON
                           ▲
                           │ weights pulled at startup
                           │
            ┌──────────────────────────┐
            │   Hugging Face Hub        │
            │   Karn81/PotholeNet-YOLO11n│
            │   (best.pt, best.onnx)     │
            └──────────────────────────┘
            
Two inference backends exist behind one `ML_BACKEND` env switch:

- **`ML_BACKEND=torch`** — full PyTorch + Ultralytics, used for local GPU development, training, and evaluation. Requires CUDA-compatible PyTorch.
- **`ML_BACKEND=onnx`** — hand-written, torch-free inference (`app/inference_onnx.py`) using ONNX Runtime directly, not the Ultralytics wrapper (which still pulls in torch even for ONNX models). Used for the deployed Render service, where RAM is constrained.

The frontend never calls the ML service directly — the Node backend is the only intended caller.

---

## Repository Layout
PotholeNet-ML/
├── LICENSE.md # MIT (code only — see License & Dataset Attribution)
├── DATASETS.md # per-source license/compliance tracker
├── backend-integration-example.js # Express route: POST /api/detection/image
└── ml-service/
├── .env.example
├── .gitignore # dataset images/labels, weights, review dirs, attain_negatives/ excluded
├── requirements.txt # full PyTorch + Ultralytics stack (local GPU dev)
├── requirements-onnx.txt # lightweight, torch-free stack (deployment)
├── Dockerfile # Render-ready, ML_BACKEND=onnx, binds to $PORT
├── README.md
│
├── app/
│ ├── init.py
│ ├── main.py # FastAPI: /predict, /health, ML_BACKEND switch
│ ├── inference_onnx.py # torch-free ONNX Runtime inference (letterbox, decode, NMS by hand)
│ └── severity.py # rule-based severity estimator (decoupled)
│
├── datasets/potholes/
│ ├── data.yaml
│ ├── images/{train,val,test}/ # not tracked in git — see .gitignore
│ └── labels/{train,val,test}/ # not tracked in git — see .gitignore
│
├── models/ # deployment copy of best.pt goes here (local dev only)
├── runs/ # training run outputs — not tracked in git
│
└── scripts/
├── prepare_dataset.py # merge RDD2022 + Pothole-600 + custom → YOLO format
├── validate_dataset.py # hard QC gate
├── dedupe_dataset.py # fix duplicate/leakage found by validator (--exact-only mode)
├── inspect_dedupe_clusters.py # visual cluster sanity-check before deleting anything
├── dataset_stats.py # per-split counts, bbox stats, resolutions
├── visualize_annotations.py # render GT boxes for manual QC
├── filter_attain_negatives.py # extract pothole-free images from the Attain dataset
├── add_background_images.py # inject negative examples into an already-split dataset
├── augment_patch_negatives.py # color/brightness augmentation for domain-gap negatives
├── train.py # fine-tune YOLO11n/s/m, logs full repro metadata
├── evaluate.py # precision/recall/mAP/latency on val or test
├── inference.py # CLI single-image / directory inference (PyTorch)
├── compare_predictions.py # GT vs. prediction overlay for FP/FN review
├── export_onnx.py # export .pt → .onnx, optional HF Hub upload
└── verify_onnx_export.py # confirm ONNX output matches PyTorch before trusting deployment

Weight files (`*.pt`, `*.onnx`), the dataset's actual images/labels, and generated review folders (`qc_preview/`, `dedupe_review/`, `prediction_review/`, `attain_negatives/`, `runs/`) are gitignored. This repo tracks code and configuration only — weights live on Hugging Face Hub, the dataset is reproducible from public/licensed sources via the scripts here.

---

## Environment Setup

Development environment this project was built and trained on:
Host OS: Windows 11
Runtime: WSL2, Ubuntu 24.04.4 LTS
Python: 3.11.16 (conda env potholenet)
GPU: NVIDIA GeForce RTX 5060, 8 GB VRAM
PyTorch: 2.9.0+cu128 (local training/eval)
Ultralytics: 8.3.0
Deployment: Render (CPU-only, 0.1 vCPU free tier), ONNX Runtime 1.29.0

**GPU note:** RTX 50-series (Blackwell, `sm_120`) requires a PyTorch build with CUDA 12.8 (`cu128`) support. A `cu121`-era install will detect the GPU but is not actually compatible and will warn or fail — verify with:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Local install (full stack, GPU dev)

```bash
cd ml-service
conda create -n potholenet python=3.11
conda activate potholenet
python -m pip install -r requirements.txt
```

### Deployment install (lightweight, torch-free)

```bash
pip install -r requirements-onnx.txt
```

This is what `Dockerfile` uses for the Render deployment — no PyTorch, no Ultralytics, just `onnxruntime` + `opencv-python-headless` + FastAPI.

---

## Dataset

### Sources

| Source | License | Notes |
|---|---|---|
| **RDD2022** | ⚠️ Unconfirmed — see `DATASETS.md` | Only the `D40` (pothole) class extracted; other damage classes dropped. VOC/XML → YOLO conversion. |
| **Pothole-600** | ❌ Never verified | ~600 images, Asian/Indian-relevant road scenes. |
| **Attain** (Mendeley, 2025) | ✅ CC BY 4.0 | Used specifically for negative examples — faded-marking and patch/utility-cut classes. Attribution required, see `DATASETS.md`. |
| **Custom Indian-road images** | N/A (own data) | Collected separately — dashcam, smartphone, field images. |

Full compliance tracking, including which licenses are still unconfirmed, lives in `DATASETS.md` at the repo root — don't treat a source as cleared for commercial use until it shows ✅ there.

### Preparation Pipeline

```bash
python scripts/prepare_dataset.py \
  --rdd2022-images /path/to/rdd/images \
  --rdd2022-annots /path/to/rdd/annotations \
  --pothole600-images /path/to/pothole600/images \
  --pothole600-labels /path/to/pothole600/labels \
  --custom-images /path/to/custom/images \
  --custom-labels /path/to/custom/labels \
  --out datasets/potholes \
  --rdd-keep-negatives-frac 0.15
```

This script:
- Converts RDD2022 VOC XML → YOLO format, keeping only `D40` boxes
- **`--rdd-keep-negatives-frac`**: RDD2022 images with no `D40` box are real negatives for a single-class detector (they may have other damage classes, or be clean road) — this flag keeps a configurable fraction of them as background examples instead of discarding them for free
- Runs a **content-based dedup pass before splitting** (md5 exact-match + phash near-duplicate clustering, transitively unioned) — this exists specifically because merging multiple public pothole datasets tends to pull in the same underlying images under different filenames/formats, confirmed as a real issue in this project (see [Known Dataset Limitations](#known-dataset-limitations))
- Groups images by sequence ID so a video/burst never splits across train/val/test
- Writes into `datasets/potholes/images|labels/<split>/` with source-tagged filenames

### Validation Gate

**Mandatory before every training run:**

```bash
python scripts/validate_dataset.py --dataset datasets/potholes
```

Checks: image/label pairing, invalid class IDs, malformed rows, out-of-bounds bboxes, empty label files, exact duplicates (within-split), and cross-split near-duplicate leakage (perceptual hash).

**Do not blindly delete flagged images.** Near-duplicate (phash) flags can be false positives — pothole photos naturally share asphalt texture and camera angle, which perceptual hashing can mistake for duplication. This project hit that exact false-positive pattern once (see below) and confirmed it via `inspect_dedupe_clusters.py` before deciding not to delete those images.

```bash
python scripts/inspect_dedupe_clusters.py --dataset datasets/potholes --phash-thresh 6 --sample 15
```

Prints cluster-size distribution and writes side-by-side montages to `dedupe_review/`. Once confirmed:

```bash
python scripts/dedupe_dataset.py --dataset datasets/potholes --exact-only --dry-run
python scripts/dedupe_dataset.py --dataset datasets/potholes --exact-only
```

**Real-world finding:** an early merge of this dataset had ~1,054 exact-duplicate images (two source dumps of the same underlying pothole photos, saved under different filenames/formats — `pothole_N.jpg` vs `potholesN.png`), including several spanning across train/val/test splits (genuine leakage). `--exact-only` mode resolves this without the false-positive risk of the near-duplicate (phash) check, which flagged ~970 additional "duplicate" clusters that turned out to be visually-similar-but-genuinely-different pothole photos, not duplicates — confirmed via manual review before deciding **not** to delete them.

### Negative Example Injection

The original dataset had **zero background (no-pothole) images** — every training example had at least one labeled pothole. This was flagged as a real risk, and confirmed as a real problem: manual testing on real-world (non-dataset) photos found the model false-positiving on painted road markings and patched/repaired asphalt.

Fix pipeline, built and used in this project:

```bash
# 1. Filter a source dataset for pothole-free images, split by category
python scripts/filter_attain_negatives.py --attain-root /path/to/Attain --out attain_negatives

# 2. Inject into the existing (already-split) dataset, preserving the current train/val/test ratio
python scripts/add_background_images.py --source attain_negatives/priority --dataset datasets/potholes --independent

# 3. Re-validate and visually spot-check before trusting
python scripts/validate_dataset.py --dataset datasets/potholes
python scripts/visualize_annotations.py --dataset datasets/potholes --split train --n 40
```

`add_background_images.py` caps negatives at a configurable fraction of the current dataset size (default 20%) to avoid tanking recall, and supports `--independent` mode for pre-shuffled research datasets where images are independent stills rather than consecutive video frames (the default sequence-grouping heuristic, which assumes filenames like `video003_042.jpg`, can badly misfire on such data — confirmed in this project when it initially dumped 317 negatives entirely into one split before this flag was added).

**Result:** 317 Attain images injected (179 train / 87 val / 51 test), split proportionally to the existing 56/27/16 ratio. This measurably fixed the false-positive on painted road markings. It did **not** fully fix the patch/repair false-positive — see [Known Limitation](#known-limitation-patchrepair-false-positive) for the full investigation, including a follow-up augmentation attempt that was tried and rolled back.

### Known Dataset Limitations

1. **Split ratio drift.** After removing exact-duplicate/cross-split-leaked images and injecting negatives, the split sits at roughly **56/27/16** (1,075 train / 521 val / 308 test) rather than the target 70/20/10. This happened because duplicate resolution prioritizes keeping test and val intact and strips duplicates from train first. Train is thinner than intended relative to val/test.

2. **Sequence metadata isn't recoverable for the original positive images.** The core pothole dataset (pre-negatives) was assembled from a pre-existing organized dump rather than run end-to-end through `prepare_dataset.py`, so filenames don't carry sequence tags. Future additions to this specific set should go through `prepare_dataset.py` from raw sources so sequence-aware splitting applies correctly. (This does not apply to the Attain negatives, which were correctly identified as independent stills and split with `--independent`.)

3. **Resolution mismatch.** Source images range from ~140px to 2000px+ on a side, most below the `imgsz=640` training resolution. Small-object recall (small/distant potholes) is a known risk area.

4. **An abandoned experiment's data is still physically present in the dataset.** A color/brightness-augmented batch of 152 patch-negative images was added while testing a fix for the patch false-positive (see [Known Limitation](#known-limitation-patchrepair-false-positive)). That experiment's resulting model was rejected and rolled back, but **the 152 augmented images themselves were never removed from `datasets/potholes/`**. Anyone retraining from the current dataset state will include them, producing a dataset different from what the currently-deployed model (`potholenet_yolo11n_v2_negatives2`) was actually trained on. If exact reproducibility of the deployed model matters, remove files matching `bg__aug*` from `datasets/potholes/images|labels/` before retraining, or retrain from a dataset snapshot taken before that experiment.

5. **At least one confirmed cross-split near-duplicate remains.** `validate_dataset.py` flags `pothole_842.jpg` (val, since removed) and similar phash-distance-0 pairs — same underlying photo saved through different compression pipelines, missed by exact-md5 dedup. At least one such pair was found and removed; others may remain (check current `validate_dataset.py` output for `phash distance 0` warnings specifically, which indicate near-certain duplicates rather than the more common false-positive-prone higher-distance warnings).

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

**Reproducibility:** `train.py` writes `run_metadata.json` alongside the weights and reads the actual save directory back from `model.trainer.save_dir` rather than assuming it matches `--name` — Ultralytics silently auto-increments the run folder if a prior/interrupted run already used that name.

**Interrupted training:** Ultralytics only writes `weights/last.pt` after an epoch fully completes. If training is interrupted before epoch 1 finishes, there is nothing to resume from — just restart.

**Do not edit dataset files while training is running.** Confirmed to crash mid-run in this project (`FileNotFoundError` in a DataLoader worker) when a duplicate image was deleted concurrently with an active training process — Ultralytics caches the file list once at startup and doesn't detect deletions until it tries to load that specific file.

---

## Evaluation

```bash
python scripts/evaluate.py --weights runs/<run_name>/weights/best.pt --data datasets/potholes/data.yaml --split val
python scripts/evaluate.py --weights runs/<run_name>/weights/best.pt --data datasets/potholes/data.yaml --split test  # once, after decisions are final
```

**mAP is not sufficient sign-off.** This project directly confirmed why: an experiment that scored *better* on every val metric performed *worse* on real-world test images (see below). Aggregate metrics measure fit to the val split, which itself shares a source distribution with train — they do not measure generalization to genuinely novel images.

---

## Experiment Results

Three training runs, same recipe (YOLO11n, 100 epochs, imgsz=640, batch=8, RTX 5060), different datasets:

| Run | Dataset change | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| `v13` (baseline) | Original positives only, 1,588 images | 0.767 | 0.682 | 0.771 | 0.544 |
| `v2_negatives` (**deployed**) | + 317 Attain negatives | 0.772 | 0.659 | 0.754 | 0.491 |
| `v3_patch_augmented` (rejected) | + 152 color-augmented patch negatives | **0.781** | **0.677** | **0.759** | **0.492** |

`v3_patch_augmented` has the best numbers in this table. **It was not deployed.** A 4-image manual regression test against real-world (non-dataset) photos — 2 previously-failing (a road-marking false positive, a patch/repair false positive) and 2 previously-passing (genuine clean roads) — showed:

| Test image | `v2_negatives` | `v3_patch_augmented` |
|---|---|---|
| Road marking (previously false-positived) | ✅ Fixed | ❌ **Regressed** — false positive returned at 72% confidence |
| Patch/repair (previously false-positived) | ⚠️ Still fires, 41% confidence | ❌ **Worse** — confidence rose to 76% |
| Clean road #1 | ✅ Correct | ✅ Correct |
| Clean road #2 | ✅ Correct | ✅ Correct |

`v3_patch_augmented` was rolled back despite better aggregate metrics. `v2_negatives` is the deployed model. This is the concrete evidence behind the "mAP is not sufficient" principle stated throughout this README — trust a targeted regression test on real, independent images over aggregate validation metrics when they disagree.

---

## Known Limitation: Patch/Repair False Positive

**Status: unresolved, documented, not silently hidden by deployment behavior.**

The deployed model false-positives on light-colored desert-road patch/repair textures. Confirmed on a real UAE road photo (dashcam-style, "Rahmaniya" test case):

- **PyTorch (`v2_negatives`), matched conf/iou settings:** 41.07% confidence — a genuine detection, above the 35% threshold.
- **Deployed ONNX export, same weights, same settings:** 0 detections — but **this is not a confirmed fix.** Independently measured PyTorch↔ONNX conversion noise on this model is ~2-5% confidence drift on typical images; 41.07% is only ~6 points above the 35% threshold, well within that noise band. The suppression is most likely an artifact of normal export precision loss landing on a borderline case, not evidence the model learned to reject this pattern.
- **Do not treat "no detection" on the current Render deployment as resolved.** A different export run, a different `onnxruntime` version, or the same model redeployed could plausibly push this back above threshold.

**Root cause, confirmed via visual comparison:** the Attain dataset's patch examples were photographed on darker Iranian urban asphalt, at a different camera height/angle than the failing test case's UAE desert asphalt. The model appears to have learned "Attain-style patches aren't potholes" without that generalizing to this visually different patch style — a real domain gap, not a data-volume problem (187 patch-tagged source images were available pre-cap, not a small number).

**Attempted fixes and results:**
1. Injecting Attain's patch/faded-marking negatives directly — fixed the marking false-positive, did not fix this one.
2. Color/brightness augmentation of the existing patch images (bridging Attain's darker asphalt toward this lighter tone) — net **regression**: broke the marking fix, made this case worse. Rolled back (see [Experiment Results](#experiment-results)).

**Recommended next steps, not yet attempted:** source patch/repair negatives from a dataset with better regional/visual match (a UAE- or Gulf-region-specific road dataset was searched for during this investigation but none was found with usable pothole/patch/negative labeling — `EMT`, a UAE dashcam dataset, was found but has zero road-surface annotations and can't be trusted as a confirmed-negative source). Real field data from actual deployment usage is likely the most reliable path forward.

---

## Error Analysis Workflow

```bash
python scripts/compare_predictions.py \
  --weights runs/<run_name>/weights/best.pt \
  --dataset datasets/potholes \
  --split val \
  --n 25 --conf 0.35
```

Draws **ground truth in green, predictions in red**, tags filenames `_LIKELY_FN` / `_LIKELY_FP` based on box-count mismatch (a rough proxy, not exact).

For testing generalization, val-split review isn't enough on its own — this project's real findings came from testing against genuinely independent, non-dataset images (stock/news photos of real roads), which is what surfaced both the marking and patch false-positives that val-split metrics alone did not reveal.

Error-driven improvement loop:
inference.py / verify_onnx_export.py on real, non-dataset images
↓
collect FP / FN examples
↓
correct annotations OR add targeted negative examples
↓
re-run validate_dataset.py
↓
retrain
↓
re-evaluate on val AND re-run the same real-world regression images
↓
compare aggregate metrics AND manual regression results before deciding

Never repeatedly evaluate against the test set during this loop.

---

## Severity Estimation

`app/severity.py` is deliberately decoupled from the detector — it never claims to measure physical depth, diameter, or volume, since a single uncalibrated RGB image and bounding box cannot support that claim.

Inputs: bounding-box area fraction, pothole count in-frame, vertical position (weak proximity proxy).

```json
{
  "label": "moderate",
  "score": 0.41,
  "reasons": ["bbox covers 2.10% of frame area", "positioned low in frame, likely closer to camera (+0.09)"],
  "estimate_only": true,
  "caveat": "Severity is a heuristic estimate ... not a physical measurement ..."
}
```

Thresholds are **placeholders**, uncalibrated against real field data or human severity judgments. Calibrating them now, before the detector's own known false-positive issue is resolved, would be tuning against a moving target.

---

## ONNX Export & Deployment

The deployed service runs **without PyTorch or Ultralytics installed at all** — `app/inference_onnx.py` implements letterbox preprocessing, output decoding, and NMS by hand using `onnxruntime` + `opencv` directly, since the Ultralytics Python API pulls in torch even when loading an ONNX model.

### Export and verify (mandatory before trusting any new export)

```bash
python scripts/export_onnx.py --weights runs/<run_name>/weights/best.pt --upload-repo Karn81/PotholeNet-YOLO11n
python scripts/verify_onnx_export.py --pt-weights runs/<run_name>/weights/best.pt --onnx-weights runs/<run_name>/weights/best.onnx --image <test_image.jpg>
```

`verify_onnx_export.py` compares detection count, confidence, and box coordinates between the PyTorch and ONNX outputs at **matched confidence and IoU thresholds** — Ultralytics' `predict()` defaults to IoU 0.7 if not passed explicitly, while `inference_onnx.py`'s default is 0.45 (the production value); comparing mismatched thresholds produces a false "different detection count" failure, which happened once during this project's own development before being fixed.

Difference tolerance is **relative to each box's own size** (3% of the smaller box dimension), not a flat pixel count — a flat threshold falsely flagged large images (a 1280×720 test image showed a "failing" 6.33px difference that was actually a passing 1.45% relative difference) before this was corrected.

### Weight hosting

Weights are hosted on **Hugging Face Hub** (`Karn81/PotholeNet-YOLO11n`), not committed to this repo. `app/main.py` downloads the configured file (`ML_HF_MODEL_FILE`, e.g. `best.onnx`) from `ML_HF_MODEL_REPO` at startup via `huggingface_hub.hf_hub_download`, with a local-file fallback (`ML_MODEL_PATH`) if `ML_HF_MODEL_REPO` is unset — so local dev works without any HF Hub dependency.

### Deployment target: Render

`Dockerfile` builds a `python:3.11-slim` image with only `requirements-onnx.txt` installed, binds to Render's injected `$PORT`, and sets `ML_BACKEND=onnx`. Render's free tier is CPU-only (0.1 vCPU) — this was a deliberate choice after confirming the full PyTorch + Ultralytics stack (600MB-1GB+ typical footprint) would not reliably fit Render's free-tier RAM, versus the torch-free ONNX path's much lighter footprint.

**Measured latency on Render's free tier:** 1.5-5.6 seconds per image in this project's testing — genuinely CPU-bound, not a cold-start artifact (confirmed by repeated calls showing consistent multi-second latency, not a one-time spike). Acceptable for an async "upload and wait a moment" flow; not suitable for real-time/live-camera use without a paid tier with more CPU, or a different host.

**Startup warmup:** both backends run one dummy inference during the FastAPI `lifespan` startup, before serving real traffic — CUDA context initialization (torch backend) and ONNX Runtime graph optimization (onnx backend) both carry a one-time first-inference cost (observed up to ~2.7 seconds locally on GPU) that this pays upfront instead of on a real user's first request.

---

## FastAPI Service

```bash
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### `GET /health`
```json
{"status": "ok", "model_loaded": true, "model_version": "best", "load_error": null}
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
      "class_id": 0, "class_name": "pothole", "confidence": 0.85,
      "bbox": {"x1": 136.1, "y1": 312.1, "x2": 1123.1, "y2": 750.0},
      "severity": {
        "label": "moderate", "score": 0.41,
        "reasons": ["bbox covers 2.10% of frame area"],
        "estimate_only": true,
        "caveat": "Severity is a heuristic estimate ..."
      }
    }
  ],
  "inference_time_ms": 131.8,
  "model_version": "best"
}
```

Uploaded files are written to `/tmp` for the duration of inference only and deleted immediately after, regardless of success or failure.

---

## Node.js Integration

`backend-integration-example.js` provides a drop-in Express route (`POST /api/detection/image`). **This is still just an example file — it has not yet been wired into a real backend's `detection` module.** The frontend calls the Node backend; the Node backend calls the ML service; the frontend never calls the ML service directly.

---

## Configuration Reference

| Variable | Purpose | Default |
|---|---|---|
| `ML_BACKEND` | `torch` (local GPU dev) or `onnx` (lightweight deployment) | `torch` |
| `ML_MODEL_PATH` | Local weights fallback if `ML_HF_MODEL_REPO` unset | `models/potholenet_best.pt` |
| `ML_HF_MODEL_REPO` | Hugging Face Hub repo to pull weights from | `Karn81/PotholeNet-YOLO11n` |
| `ML_HF_MODEL_FILE` | Filename within that repo | `best.pt` (torch) / `best.onnx` (onnx) |
| `ML_CONFIDENCE_THRESHOLD` | Minimum confidence to report a detection | `0.35` |
| `ML_MAX_IMAGE_SIZE_MB` | Reject uploads larger than this | `10` |
| `ML_TIMEOUT_MS` | Logged if inference exceeds this — not yet a hard cutoff | `30000` |
| `ML_ALLOWED_ORIGINS` | CORS allow-list | `http://localhost:5000` |
| `ML_LOG_LEVEL` | Logging verbosity | `INFO` |

Never commit `.env` — only `.env.example` should be tracked.

---

## Production Readiness

### API service
- [x] Request size limits, file type validation, model-loaded-once, health check, temp-file cleanup, CORS
- [x] Torch-free lightweight deployment path (ONNX Runtime)
- [ ] **Rate limiting** — not implemented
- [ ] **Hard inference timeout** — logged only

### Dataset
- [x] Negative/background examples added (partially resolves prior gap)
- [ ] RDD2022 and Pothole-600 license verification — see `DATASETS.md`
- [ ] Train set expansion — 1,075 images is still thin after dedup

### Model
- [x] Baseline trained, 3 experiments compared, real-world regression testing performed
- [x] One known false-positive pattern fixed (road markings)
- [ ] One known false-positive pattern unresolved (patch/repair on certain asphalt tones) — see [Known Limitation](#known-limitation-patchrepair-false-positive)
- [ ] YOLO11s / YOLO11m comparison
- [ ] Test-set evaluation (one-time, after remaining decisions)

### Backend
- [ ] Real Node.js `detection` module integration (currently only an example file)
- [ ] Auth/authorization on the upload endpoint

**A model is not production-ready because training completed successfully — or because it scored well on validation metrics.** This project directly demonstrated both: the deployed model was chosen over one with better validation numbers, based on real-world regression testing. Deployment readiness requires clean validation, manual review against genuinely independent images, and honest documentation of what's still unresolved — not just a passing metric.

---

## Design Principles

- **Fine-tune, never train from scratch.**
- **One detector class.** Severity is a separate concern, never a detector class.
- **Detection and severity are separate systems.**
- **Validate before training, every time.**
- **Sequence-aware splitting** for video/frame-derived data; **independent-image splitting** for pre-shuffled research datasets — using the wrong one for the data's actual structure caused a real bug in this project.
- **Test set stays untouched during iteration.**
- **Aggregate metrics do not override real-world regression testing** — proven necessary, not just theoretical, in this project's experiment history.
- **Every training run is reproducible** — with the one documented exception in [Known Dataset Limitations](#known-dataset-limitations) #4.

---

## Roadmap

1. ✅ ~~Manual FP/FN review against real-world images~~ — done, found 2 real issues
2. ✅ ~~Add background/negative examples~~ — done, fixed 1 of 2 issues
3. Source better-matched patch/repair negatives (regional match) or gather real field data
4. Clean up the physically-present-but-unused augmented dataset files (see Known Limitation #4)
5. Expand train split with genuinely new, non-duplicate images
6. Train and compare YOLO11s / YOLO11m
7. One-time test-set evaluation on the final selected model
8. Calibrate severity thresholds against real field data
9. Implement rate limiting and hard inference timeout
10. Wire the real Node.js `detection` module integration
11. Resolve remaining dataset license verifications (RDD2022, Pothole-600)

---

## Contributing / Reproducing

This repo does not track dataset images/labels or trained weights. To reproduce:

1. Obtain RDD2022, Pothole-600, and Attain yourself — verify licenses per `DATASETS.md`
2. Run `prepare_dataset.py` → `validate_dataset.py` → fix any errors → `visualize_annotations.py`
3. Run `filter_attain_negatives.py` → `add_background_images.py` for negative examples
4. Run `train.py` → `evaluate.py` → `compare_predictions.py`
5. **Test against real-world, non-dataset images before trusting aggregate metrics** — this is not optional based on this project's own history
6. Export via `export_onnx.py` → `verify_onnx_export.py` before any deployment

---

## License & Dataset Attribution

- **Code**: MIT, see `LICENSE.md` at the repo root.
- **Datasets and trained weights are NOT covered by the MIT license** — see `LICENSE.md`'s scope section and `DATASETS.md` for the full per-source compliance tracker, including which licenses remain unconfirmed.
- **Attain dataset attribution** (CC BY 4.0, used for negative examples): Rezaeimanesh et al., *Data in Brief*, 2025. Any redistribution of this dataset or derived training data must include this attribution.