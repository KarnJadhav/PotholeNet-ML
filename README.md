# PotholeNet-ML

An ML microservice for detecting potholes in road images using a fine-tuned YOLO11 object detector, with a separate rule-based severity estimation module and a FastAPI inference service.

The service is designed to be called by a Node.js backend rather than directly by the frontend.

---

## 1. Project Status

### Current status

* ✅ YOLO11 fine-tuning pipeline
* ✅ Dataset preparation/normalization pipeline
* ✅ Dataset validation and leakage checks
* ✅ Annotation visualization
* ✅ Evaluation pipeline
* ✅ CLI inference
* ✅ FastAPI inference service
* ✅ Node.js integration example
* ✅ Separate severity-estimation module
* ⚠️ Dataset validation currently reports errors that must be fixed
* ⏳ First detector training run
* ⏳ Production rate limiting
* ⏳ Hard inference-timeout enforcement

### Current dataset validation result

The current dataset contains approximately **2,642 images**, but the validator currently reports:

```text
Errors:   504
Warnings: 1338
```

**Training must not begin until all validation errors are resolved.**

Warnings such as very small bounding boxes do not automatically mean the annotations are invalid. They require visual inspection.

---

# 2. Architecture

```text
                         ┌─────────────────────┐
                         │      Frontend       │
                         │  Web / Mobile App   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    Node.js Backend  │
                         │                     │
                         │ Auth / API / Upload │
                         └──────────┬──────────┘
                                    │
                             HTTP / multipart
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │       PotholeNet-ML          │
                    │        FastAPI Service       │
                    │                              │
                    │  POST /predict               │
                    │       │                      │
                    │       ▼                      │
                    │   YOLO11 Detector            │
                    │       │                      │
                    │       ▼                      │
                    │   Pothole Detections         │
                    │       │                      │
                    │       ▼                      │
                    │   Severity Estimator          │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                            Detection JSON
```

The frontend should **not call the ML service directly**.

The Node.js backend acts as the application-facing API and communicates with the ML service internally.

---

# 3. System Environment

Current development environment:

```text
Host OS:       Windows 11
Runtime:       WSL2
Linux:         Ubuntu 24.04.4 LTS
Architecture:  x86_64 / AMD64

Python:        3.11.16
Conda:         25.11.1

GPU:           NVIDIA GeForce RTX 5060
VRAM:          8 GB

PyTorch:       2.11.0+cu128
CUDA:          12.8
Ultralytics:   8.3.0
```

Architecture:

```text
Windows 11
    ↓
WSL2
    ↓
Ubuntu 24.04
    ↓
Conda / Python 3.11
    ↓
PyTorch + CUDA
    ↓
NVIDIA RTX 5060 8 GB
```

Verify the architecture:

```bash
uname -m
```

Expected:

```text
x86_64
```

Verify the GPU:

```bash
nvidia-smi
```

Verify PyTorch CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

# 4. Project Structure

```text
PotholeNet-ML/
│
├── .claude/
│   └── settings.json
│
├── backend-integration-example.js
│
└── ml-service/
    │
    ├── .env.example
    ├── requirements.txt
    ├── README.md
    │
    ├── app/
    │   ├── __init__.py
    │   ├── main.py
    │   └── severity.py
    │
    ├── datasets/
    │   └── potholes/
    │       ├── data.yaml
    │       ├── images/
    │       │   ├── train/
    │       │   ├── val/
    │       │   └── test/
    │       │
    │       └── labels/
    │           ├── train/
    │           ├── val/
    │           └── test/
    │
    ├── models/
    │   └── potholenet_best.pt
    │
    ├── runs/
    │   └── ...
    │
    └── scripts/
        ├── prepare_dataset.py
        ├── validate_dataset.py
        ├── visualize_annotations.py
        ├── train.py
        ├── evaluate.py
        └── inference.py
```

---

# 5. ML Approach

PotholeNet uses a **single-class object detector**.

```text
Class 0 = pothole
```

The detector is based on a pretrained YOLO11 model.

Initial model candidates:

```text
YOLO11n
YOLO11s
YOLO11m
```

The current hardware target is:

```text
NVIDIA RTX 5060
8 GB VRAM
```

The project uses **fine-tuning only**.

It does **not** train YOLO from scratch.

---

# 6. Dataset Sources

The intended dataset combines three sources:

### 6.1 RDD2022

Use the relevant pothole/road-damage class:

```text
D40
```

RDD2022 annotations are converted from Pascal VOC/XML to YOLO format.

Only the required pothole class is retained.

---

### 6.2 Pothole-600

Pothole-600 provides additional pothole examples, including road scenes relevant to Asian/Indian environments.

Before using it:

* verify the license
* preserve source provenance
* check for duplicate images
* normalize annotations to YOLO format
* force the pothole class to class `0`

---

### 6.3 Custom Indian Road Dataset

Custom Indian-road images should be collected and labeled separately.

Recommended sources include:

* dashcam footage
* smartphone road images
* manually collected field images
* representative Indian road conditions

Custom images should contain realistic variation:

* daylight
* cloudy weather
* shadows
* wet roads
* different road surfaces
* urban roads
* rural roads
* highways
* different camera heights
* near and far potholes
* multiple potholes
* partially visible potholes

---

# 7. Dataset Preparation

The dataset preparation script combines the three sources.

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

The resulting structure should be:

```text
datasets/potholes/
│
├── data.yaml
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
└── labels/
    ├── train/
    ├── val/
    └── test/
```

---

# 8. Dataset Splitting

The dataset should be split approximately:

```text
Train: 70%
Validation: 20%
Test: 10%
```

The important requirement is that the split is **sequence-aware**.

Adjacent frames from the same video/burst must never be distributed across train, validation and test.

Bad:

```text
video_001_frame_001 → train
video_001_frame_002 → val
video_001_frame_003 → test
```

Good:

```text
video_001 → train
video_002 → validation
video_003 → test
```

This prevents data leakage and gives a more realistic estimate of generalization.

---

# 9. Dataset Validation

Dataset validation is a mandatory gate before training.

Run:

```bash
python scripts/validate_dataset.py --dataset datasets/potholes
```

The validator checks:

* image/label pairing
* missing labels
* missing images
* invalid class IDs
* malformed annotation rows
* bounding-box coordinates
* out-of-bounds coordinates
* empty label files
* extremely small bounding boxes
* exact duplicate images
* near-duplicate images
* cross-split leakage

---

# 10. Current Validation Problem

The current dataset validation result is:

```text
Errors:   504
Warnings: 1338
```

Example warnings:

```text
very small bbox
area frac=0.00028
53x14px
```

Small bounding boxes are **warnings**, not automatically invalid annotations.

A small pothole can be legitimate when it is far away from the camera.

However, each warning should be visually inspected.

The critical issue is the:

```text
504 ERRORS
```

Training should not start until these are understood and fixed.

---

# 11. Investigating Validation Errors

Save the complete validation output:

```bash
python scripts/validate_dataset.py \
  --dataset datasets/potholes 2>&1 | tee validation.txt
```

Show only errors:

```bash
grep '^ERROR' validation.txt
```

Summarize error types:

```bash
grep '^ERROR' validation.txt | \
  cut -d: -f1-2 | \
  sort | \
  uniq -c | \
  sort -nr
```

Do **not** simply delete the affected annotations.

First determine whether the problem is:

* bad source annotation
* conversion bug
* filename mismatch
* class mapping problem
* invalid YOLO coordinates
* duplicate data
* dataset split leakage
* genuinely bad annotation

Fix the underlying problem and rerun validation.

Target:

```text
Errors:   0
```

Warnings can remain only when they have been reviewed and determined to be legitimate.

---

# 12. Visual Annotation QC

After validation:

```bash
python scripts/visualize_annotations.py \
  --dataset datasets/potholes
```

Manually inspect the generated samples.

Check:

* bounding box surrounds the pothole
* box is not shifted
* box is not excessively large
* pothole is actually visible
* annotation is not another road object
* multiple potholes are correctly labeled
* tiny objects are genuine potholes
* no obvious labeling mistakes exist

Automated validation cannot replace visual QC.

---

# 13. Dataset Quality Gate

The required pipeline is:

```text
Raw datasets
      ↓
prepare_dataset.py
      ↓
validate_dataset.py
      ↓
0 validation errors
      ↓
visualize_annotations.py
      ↓
manual annotation QC
      ↓
TRAINING
```

Never skip the validation step.

---

# 14. YOLO Dataset Configuration

`data.yaml` should contain a single class:

```yaml
path: /path/to/datasets/potholes

train: images/train
val: images/val
test: images/test

names:
  0: pothole
```

The actual `path` should be configured for the environment where training runs.

---

# 15. Training

Once the dataset passes validation and visual QC, start with a reproducible baseline.

The training script uses pretrained YOLO11 weights.

Example:

```bash
python scripts/train.py
```

The training script should:

* load pretrained weights
* never train from scratch
* record dataset statistics
* record Python/package versions
* record GPU information
* record hyperparameters
* record the training configuration
* save model checkpoints
* save reproducibility metadata

---

# 16. RTX 5060 8 GB Considerations

The RTX 5060 has 8 GB VRAM.

YOLO11m is more demanding than YOLO11n/s.

Therefore, do not assume a large batch size will fit.

Recommended approach:

```text
1. Get dataset clean
2. Start YOLO11m conservatively
3. Monitor VRAM
4. Reduce batch size if necessary
5. Establish baseline
6. Compare against YOLO11s/n
```

The first objective is not maximum performance.

The first objective is a **reproducible baseline**.

---

# 17. Training Strategy

Recommended experiment sequence:

```text
Experiment 1
YOLO11n
↓
baseline

Experiment 2
YOLO11s
↓
compare

Experiment 3
YOLO11m
↓
compare if VRAM/performance justify it
```

Compare:

* precision
* recall
* mAP50
* mAP50-95
* inference latency
* FPS
* GPU memory
* false positives
* false negatives

Choose the model based on the actual deployment requirement rather than model size alone.

---

# 18. Evaluation

After training:

```bash
python scripts/evaluate.py
```

Evaluate validation data first.

Use the test set only after model/data decisions are finished.

Recommended workflow:

```text
Train
  ↓
Validation
  ↓
Error analysis
  ↓
Dataset/model improvements
  ↓
Final model
  ↓
Test evaluation
```

Do not repeatedly tune against the test set.

---

# 19. Error Analysis

After the first training run, inspect:

### False positives

Examples where the detector incorrectly predicts:

```text
pothole
```

Possible causes:

* road patches
* shadows
* manholes
* cracks
* puddles
* debris
* pavement markings

### False negatives

Actual potholes that were missed.

Investigate whether they are:

* very small
* far away
* partially occluded
* poorly illuminated
* unusual shapes
* wet
* severe but visually ambiguous

Error analysis should drive the next dataset/training iteration.

---

# 20. Inference

Single image:

```bash
python scripts/inference.py \
  --source /path/to/image.jpg
```

Directory:

```bash
python scripts/inference.py \
  --source /path/to/images/
```

The detector outputs pothole detections with bounding boxes and confidence values.

---

# 21. Severity Module

Severity is intentionally separate from the detector.

Current implementation is **rule-based**.

It uses signals such as:

* bounding-box area fraction
* number of potholes in the frame
* vertical position
* weak proximity-related image cues

It does **not** claim to measure:

* physical pothole depth
* actual diameter
* real-world dimensions
* road damage volume

The output is explicitly an estimate.

Example conceptual response:

```json
{
  "severity": {
    "label": "moderate",
    "score": 0.64,
    "estimate_only": true,
    "reasons": [
      "large detected pothole",
      "multiple potholes detected"
    ]
  }
}
```

Current thresholds are placeholders.

They should be recalibrated using real field data and human severity judgments.

---

# 22. FastAPI Service

Start the service:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The model is loaded once during application startup.

It is not loaded for every request.

---

# 23. API Endpoints

## Health

```http
GET /health
```

Used to determine whether the ML service is available and whether the model has loaded correctly.

---

## Prediction

```http
POST /predict
```

Accepts an image upload.

Conceptual response:

```json
{
  "detections": [
    {
      "bbox": [
        100,
        150,
        400,
        350
      ],
      "confidence": 0.91,
      "class_id": 0,
      "class_name": "pothole",
      "severity": {
        "label": "moderate",
        "score": 0.64,
        "estimate_only": true
      }
    }
  ],
  "inference_time_ms": 18.4
}
```

---

# 24. API Security

Current service protections include:

* image size limits
* content/type validation
* temporary-file cleanup
* structured logging
* health endpoint
* model version reporting
* CORS configuration
* no filesystem-path leakage
* no credential leakage

Still required for production:

```text
❌ Rate limiting
❌ Hard inference timeout
```

Rate limiting can be implemented with:

```text
slowapi
```

or enforced at the Node.js backend/API gateway.

---

# 25. Node.js Integration

The intended architecture is:

```text
Client
  ↓
Node.js API
  ↓
POST /predict
  ↓
FastAPI ML Service
  ↓
YOLO11
  ↓
Detection + Severity
  ↓
Node.js
  ↓
Client
```

The provided:

```text
backend-integration-example.js
```

shows how the Node.js backend can forward an uploaded image to the ML service.

The Node layer should handle:

* authentication
* authorization
* rate limiting
* request validation
* ML service timeout
* retry policy where appropriate
* API response formatting

---

# 26. Environment Configuration

Copy:

```bash
cp .env.example .env
```

Example:

```env
ML_MODEL_PATH=models/potholenet_best.pt
ML_CONFIDENCE_THRESHOLD=0.35
ML_MAX_IMAGE_SIZE_MB=10
ML_TIMEOUT_MS=30000
ML_ALLOWED_ORIGINS=http://localhost:5000
```

Do not commit secrets to Git.

---

# 27. Dependency Installation

Create/activate the environment:

```bash
conda activate potholenet
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

If individual dependencies are required:

```bash
python -m pip install \
  fastapi \
  uvicorn \
  pydantic \
  python-multipart \
  python-dotenv \
  opencv-python-headless \
  pillow \
  numpy \
  pandas \
  imagehash
```

Verify Ultralytics:

```bash
python -c "import ultralytics; print(ultralytics.__version__)"
```

Expected:

```text
8.3.0
```

---

# 28. Complete Development Workflow

The complete workflow is:

```text
                    RAW DATA
                       │
          ┌────────────┼────────────┐
          │            │            │
       RDD2022    Pothole-600   Indian shots
          │            │            │
          └────────────┼────────────┘
                       ↓
             prepare_dataset.py
                       ↓
              Normalized YOLO
                       ↓
             validate_dataset.py
                       ↓
                 0 ERRORS
                       ↓
          visualize_annotations.py
                       ↓
                Manual QC
                       ↓
                 YOLO11 baseline
                       ↓
                  evaluate.py
                       ↓
               Error analysis
                       ↓
             Dataset improvements
                       ↓
             YOLO11n/s/m comparison
                       ↓
              Select final detector
                       ↓
                 FastAPI service
                       ↓
                 Node.js backend
                       ↓
                  Application
```

---

# 29. Current Immediate Task

The current blocker is **dataset validation**.

Current result:

```text
Errors:   504
Warnings: 1338
```

Therefore:

```text
DO NOT TRAIN YET.
```

First save and inspect the errors:

```bash
python scripts/validate_dataset.py \
  --dataset datasets/potholes 2>&1 | tee validation.txt
```

Then:

```bash
grep '^ERROR' validation.txt
```

Summarize them:

```bash
grep '^ERROR' validation.txt | \
  cut -d: -f1-2 | \
  sort | \
  uniq -c | \
  sort -nr
```

Fix the underlying dataset/conversion problems.

Then rerun:

```bash
python scripts/validate_dataset.py \
  --dataset datasets/potholes
```

Target:

```text
Errors:   0
```

---

# 30. Recommended Next Milestones

### Milestone 1 — Dataset

```text
RDD2022
+
Pothole-600
+
Indian custom data
        ↓
prepare_dataset.py
        ↓
0 validation errors
```

### Milestone 2 — Visual QC

```text
visualize_annotations.py
        ↓
manual inspection
        ↓
correct labels
```

### Milestone 3 — Baseline

```text
YOLO11n
        ↓
train
        ↓
evaluate
```

### Milestone 4 — Model Comparison

```text
YOLO11n
   vs
YOLO11s
   vs
YOLO11m
```

Compare accuracy, latency and VRAM.

### Milestone 5 — Error Analysis

Identify:

```text
false positives
false negatives
small-object failures
difficult lighting
road-surface confusion
```

### Milestone 6 — Production

Complete:

```text
rate limiting
hard inference timeout
logging/monitoring
model versioning
Node integration
```

---

# 31. Design Principles

PotholeNet-ML follows these principles:

### Fine-tune, don't train from scratch

Use pretrained YOLO11 weights.

### One detector class

```text
0 = pothole
```

Keep the initial detector simple.

### Separate severity from detection

Detection answers:

```text
"Where is the pothole?"
```

Severity answers:

```text
"How serious does it appear?"
```

These should remain separate systems.

### Prevent data leakage

Sequence-aware splitting is mandatory for frame/video-derived data.

### Validate before training

A dataset with annotation errors should not be used for model training.

### Test only after model decisions

The test set should remain untouched during iterative development.

### Reproducibility

Every training run should record:

* model
* dataset
* dataset counts
* software versions
* GPU
* configuration
* hyperparameters
* source revision
* training results

---

# 32. Production Readiness Checklist

## Dataset

* [ ] RDD2022 license verified
* [ ] Pothole-600 license verified
* [ ] Custom dataset provenance recorded
* [ ] Labels converted to YOLO format
* [ ] Class mapping verified
* [ ] Sequence-aware split verified
* [ ] No missing image/label pairs
* [ ] No invalid class IDs
* [ ] No invalid bounding boxes
* [ ] Duplicate checks completed
* [ ] Near-duplicate leakage checked
* [ ] Visual QC completed
* [ ] Validator reports 0 errors

## Model

* [ ] YOLO11n baseline trained
* [ ] YOLO11s comparison completed
* [ ] YOLO11m evaluated if appropriate for VRAM
* [ ] Validation metrics recorded
* [ ] Test evaluation completed only after final decisions
* [ ] False positives analyzed
* [ ] False negatives analyzed
* [ ] Final model selected

## Severity

* [ ] Detector-independent severity module
* [ ] Severity thresholds calibrated
* [ ] Human severity labels collected
* [ ] Estimate-only disclaimer retained
* [ ] No unsupported depth/dimension claims

## API

* [x] FastAPI service
* [x] `/health`
* [x] `/predict`
* [x] Model loaded once
* [x] Request-size limits
* [x] File validation
* [x] Temporary-file cleanup
* [x] Structured logging
* [ ] Rate limiting
* [ ] Hard inference timeout
* [ ] Production monitoring

## Backend

* [x] Node.js integration example
* [ ] Integrate with existing backend
* [ ] Authentication/authorization
* [ ] Rate limiting
* [ ] ML timeout handling
* [ ] Error handling
* [ ] Production logging

---

# 33. Final Target

The finished system should look like:

```text
Indian Road Image
       │
       ▼
Node.js Backend
       │
       ▼
PotholeNet-ML
       │
       ├───────────────┐
       ▼               ▼
   YOLO11 Detector   Severity
       │               │
       └───────┬───────┘
               ▼
        Detection Result
               │
               ▼
         Node.js API
               │
               ▼
            Frontend
```

The immediate objective is **not additional feature development**.

The immediate objective is:

```text
504 validation errors
        ↓
FIX DATASET
        ↓
0 validation errors
        ↓
VISUAL QC
        ↓
YOLO11 BASELINE
        ↓
EVALUATE
        ↓
ITERATE
```

Once the first clean training/evaluation run exists, model performance—not additional scaffolding—should determine the next engineering decision.
