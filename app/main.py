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
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

load_dotenv()

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
        weights_path = Path(MODEL_PATH)
        if not weights_path.exists():
            raise FileNotFoundError(f"model weights not found at {weights_path}")
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


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox


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

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            detections.append(Detection(
                class_id=int(box.cls[0]),
                class_name="pothole",
                confidence=round(float(box.conf[0]), 4),
                bbox=BBox(x1=round(x1, 1), y1=round(y1, 1), x2=round(x2, 1), y2=round(y2, 1)),
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
