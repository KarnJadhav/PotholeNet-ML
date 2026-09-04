"""
PotholeNet ML service — FastAPI wrapper around the fine-tuned YOLO detector.

Loads the model ONCE at startup. The Node.js backend is the only intended caller;
the React frontend must never hit this service directly.
"""
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from .severity import Detection as SeverityInput, estimate_severity

load_dotenv()

HF_MODEL_REPO = os.getenv("ML_HF_MODEL_REPO", "")  # e.g. "username/PotholeNet-YOLO11n"
HF_MODEL_FILE = os.getenv("ML_HF_MODEL_FILE", "best.pt")
MODEL_PATH = os.getenv("ML_MODEL_PATH", "models/potholenet_best.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("ML_CONFIDENCE_THRESHOLD", "0.35"))
MAX_IMAGE_SIZE_MB = float(os.getenv("ML_MAX_IMAGE_SIZE_MB", "10"))
TIMEOUT_MS = int(os.getenv("ML_TIMEOUT_MS", "30000"))
ALLOWED_ORIGINS = os.getenv("ML_ALLOWED_ORIGINS", "http://localhost:5000").split(",")
LOG_LEVEL = os.getenv("ML_LOG_LEVEL", "INFO")
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("potholenet-ml")

# Mutable app-state container populated at startup
model_state = {"model": None, "loaded": False, "load_error": None, "version": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from ultralytics import YOLO
    try:
        if HF_MODEL_REPO:
            # Pull from Hugging Face Hub — this is how weights reach a fresh deploy
            # (e.g. HF Spaces, Cloud Run, Render) without committing .pt files to git.
            weights_path = Path(hf_hub_download(HF_MODEL_REPO, HF_MODEL_FILE))
            logger.info(f"Downloaded weights from HF Hub: {HF_MODEL_REPO}/{HF_MODEL_FILE}")
        else:
            weights_path = Path(MODEL_PATH)
            if not weights_path.exists():
                raise FileNotFoundError(
                    f"model weights not found at {weights_path}, and ML_HF_MODEL_REPO "
                    f"is not set — nothing to download either"
                )
        model_state["model"] = YOLO(str(weights_path))
        model_state["loaded"] = True
        model_state["version"] = weights_path.stem
        logger.info(f"Model loaded from {weights_path}")
    except Exception as exc:  # noqa: BLE001 — surface any load failure via /health
        model_state["loaded"] = False
        model_state["load_error"] = str(exc)
        logger.error(f"Model failed to load: {exc}")
    yield
    model_state.clear()


app = FastAPI(title="PotholeNet ML Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None
    load_error: str | None


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class SeverityOut(BaseModel):
    label: str
    score: float
    reasons: list[str]
    estimate_only: bool
    caveat: str


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox
    severity: SeverityOut | None = None


class PredictResponse(BaseModel):
    success: bool
    detections: list[Detection]
    inference_time_ms: float
    model_version: str | None = None


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if model_state["loaded"] else "degraded",
        model_loaded=model_state["loaded"],
        model_version=model_state["version"],
        load_error=model_state["load_error"],
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not model_state["loaded"]:
        raise HTTPException(status_code=503, detail="model not loaded — check /health")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail=f"unsupported content type: {file.content_type}")

    raw = await file.read()
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > MAX_IMAGE_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"image exceeds {MAX_IMAGE_SIZE_MB}MB limit")

    request_id = str(uuid.uuid4())[:8]
    tmp_path = Path(f"/tmp/potholenet_{request_id}_{file.filename}")
    try:
        tmp_path.write_bytes(raw)
        try:
            Image.open(tmp_path).verify()
        except UnidentifiedImageError:
            raise HTTPException(status_code=400, detail="file is not a valid image")

        t0 = time.perf_counter()
        results = model_state["model"].predict(
            str(tmp_path), conf=CONFIDENCE_THRESHOLD, verbose=False
        )[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if elapsed_ms > TIMEOUT_MS:
            logger.warning(f"[{request_id}] inference exceeded configured timeout "
                            f"({elapsed_ms:.0f}ms > {TIMEOUT_MS}ms)")

        img_w, img_h = Image.open(tmp_path).size

        raw_boxes = [
            (int(box.cls[0]), round(float(box.conf[0]), 4), [float(v) for v in box.xyxy[0]])
            for box in results.boxes
        ]
        severity_inputs = [
            SeverityInput(x1=b[0], y1=b[1], x2=b[2], y2=b[3], confidence=conf)
            for _, conf, b in raw_boxes
        ]
        severities = estimate_severity(severity_inputs, image_width=img_w, image_height=img_h)

        detections = []
        for (cls_id, conf, (x1, y1, x2, y2)), sev in zip(raw_boxes, severities):
            detections.append(Detection(
                class_id=cls_id,
                class_name="pothole",
                confidence=conf,
                bbox=BBox(x1=round(x1, 1), y1=round(y1, 1), x2=round(x2, 1), y2=round(y2, 1)),
                severity=SeverityOut(
                    label=sev.label.value, score=sev.score, reasons=sev.reasons,
                    estimate_only=sev.estimate_only, caveat=sev.caveat,
                ),
            ))

        logger.info(f"[{request_id}] {len(detections)} detection(s) in {elapsed_ms:.1f}ms")

        return PredictResponse(
            success=True,
            detections=detections,
            inference_time_ms=round(elapsed_ms, 2),
            model_version=model_state["version"],
        )
    finally:
        tmp_path.unlink(missing_ok=True)  # never leave uploaded images on disk
