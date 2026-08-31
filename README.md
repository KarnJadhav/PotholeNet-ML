# PotholeNet 2.0 — ML Service

Fine-tuned YOLO11 pothole detector + FastAPI microservice. Fine-tuning only — never trained from scratch.

## Layout

```
ml-service/
  datasets/potholes/{images,labels}/{train,val,test}/  data.yaml
  scripts/   validate_dataset.py  visualize_annotations.py  train.py  evaluate.py  inference.py
  app/       main.py (FastAPI: POST /predict, GET /health)
  models/    trained weights (best.pt) go here for the running service
  runs/      training run outputs (weights, metrics, run_metadata.json)
```

## 1. Environment setup

```bash
cd ml-service
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For CUDA: install the matching `torch`/`torchvision` build from
https://pytorch.org/get-started/locally/ **before** `pip install -r requirements.txt`,
or pip will pull CPU-only wheels.

## 2. Dataset preparation

Do not blindly merge sources. Before combining RDD2022 (D40 class), the open pothole
dataset, and the custom Indian-road PotholeNet set:

1. Check each source's license (RDD2022 and most open pothole sets are research-licensed —
   confirm terms allow your intended use, including any commercial deployment).
2. Normalize all annotations to YOLO format: `class_id x_center y_center width height`,
   normalized 0–1, single class `0: pothole`.
3. Dedupe within and across sources (`validate_dataset.py` catches exact + near-duplicates).
4. Split 70/10/20 (or per data.yaml) by **source video/sequence/burst**, not by individual
   frame — adjacent frames from the same clip must land in the same split, or you leak
   information into val/test and your metrics will lie to you.
5. Place files under `datasets/potholes/images/<split>/` and `labels/<split>/` with
   matching stems (`road_001.jpg` ↔ `road_001.txt`).
6. Fill in `datasets/potholes/data.yaml` — set `path` to the absolute path on the
   training machine.

Then, **before any training**:

```bash
python scripts/validate_dataset.py --dataset datasets/potholes
python scripts/visualize_annotations.py --dataset datasets/potholes --split train --n 40
```

Fix every ERROR from validate_dataset.py. Manually eyeball the rendered boxes in
`qc_preview/`. Do not proceed to training on unverified annotations.

## 3. Baseline training (YOLO11n)

```bash
python scripts/train.py \
  --model yolo11n.pt \
  --data datasets/potholes/data.yaml \
  --epochs 100 --imgsz 640 \
  --name potholenet_yolo11n \
  --dataset-version v1-2026-08-31
```

`--batch` defaults to Ultralytics auto-batch (fits your GPU memory); pass an explicit
value to override. Falls back to CPU automatically if no CUDA GPU is found — expect this
to be slow; use Colab/Kaggle for the real run if you don't have local GPU access.

Outputs: `runs/potholenet_yolo11n/weights/{best,last}.pt` plus `run_metadata.json`
(model/dataset versions, counts, hyperparams, hardware, package versions — for
reproducibility).

## 4. Evaluation

```bash
# iterate against val while tuning
python scripts/evaluate.py --weights runs/potholenet_yolo11n/weights/best.pt --split val

# ONE final run against the untouched test set, once you're done iterating
python scripts/evaluate.py --weights runs/potholenet_yolo11n/weights/best.pt --split test
```

Reports precision, recall, mAP50, mAP50-95, latency, FPS estimate. **mAP is not a
sign-off** — manually review false positives/negatives, especially: road patches,
shadows, cracks, manholes, puddles, dark road areas, vehicle shadows, small/distant
potholes, multi-pothole scenes, partial occlusion, night images.

## 5. YOLO11n vs YOLO11s comparison

Repeat step 3 with `--model yolo11s.pt --name potholenet_yolo11s` on the **identical**
dataset/split, then compare both `eval_report_test.json` files on mAP50, mAP50-95,
recall, precision, latency, and file size. Pick based on accuracy/latency tradeoff for
your deployment target (mobile-facing detection likely favors 11n unless 11s's accuracy
gain clearly justifies the latency cost).

## 6. Error-driven improvement loop

1. `scripts/inference.py` on real PotholeNet field images (not val/test).
2. Manually collect false positives / false negatives.
3. Add the hard examples to the training set with corrected annotations.
4. Re-run `validate_dataset.py`.
5. Retrain.
6. Re-evaluate on val only, until satisfied — then a single fresh test-set evaluation.

Never repeatedly evaluate-and-tune against the test set — that's the same leakage
problem as sequence leakage, just slower to notice.

## 7. Severity — NOT part of the detector

YOLO returns class + confidence + bbox only. Confidence is not severity. A separate
rule-based severity module (bbox size relative to frame, count of potholes in-frame,
road context) produces a labeled *estimate* — it does not claim physical
depth/dimensions from monocular RGB unless the system has real calibration or depth
input. Build this as its own module, not inside the detector or the FastAPI response
schema shown below.

## 8. Running the FastAPI service

```bash
cp .env.example .env   # edit ML_MODEL_PATH etc.
cp runs/potholenet_yolo11n/weights/best.pt models/potholenet_best.pt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

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
    {"class_id": 0, "class_name": "pothole", "confidence": 0.94,
     "bbox": {"x1": 120, "y1": 80, "x2": 450, "y2": 310}}
  ],
  "inference_time_ms": 42.1,
  "model_version": "potholenet_best"
}
```

Model loads once at startup (FastAPI `lifespan`), not per-request. Uploaded files are
written to `/tmp` only for the duration of inference and deleted immediately after
(`finally: tmp_path.unlink()`) — nothing persists on the ML service's filesystem.

## 9. Node.js integration

See `../backend-integration-example.js` for a drop-in `POST /api/detection/image`
Express route: validates the upload, forwards it to the ML service, and returns
`{ success, detections, inference_time_ms }` to the frontend. The React frontend never
calls the ML service directly — everything routes through the Node backend.

## 10. Production checklist (not yet fully implemented — track before deploy)

- [x] Request size limit (`ML_MAX_IMAGE_SIZE_MB`, enforced in `/predict`)
- [x] File type validation (`ALLOWED_CONTENT_TYPES`)
- [x] Structured logging with request IDs
- [x] Health check with model-load status
- [x] Model version reporting in responses
- [x] Configurable confidence threshold
- [x] CORS restricted to configured origins
- [x] Temp file cleanup after every request
- [ ] Rate limiting — not yet added; recommend `slowapi` in front of `/predict`, or
      enforce it at the Node backend layer since that's the only allowed caller
- [ ] Inference timeout enforcement is currently logged-only (`TIMEOUT_MS`), not a hard
      cutoff — add `asyncio.wait_for` or a worker-based timeout if hard cancellation
      matters for your latency SLA

## Do not deploy on training-completion alone

A model is not production-ready because `train.py` exits cleanly. Deployment should
require: passing `validate_dataset.py`, manual annotation review, a clean test-set
evaluation (run once), manual false-positive/negative review, and inference checks
against real Indian-road field images the model hasn't seen.
