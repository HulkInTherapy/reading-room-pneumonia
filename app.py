"""Standalone Reading Room. Inference uses only the downloaded portable model."""

import base64
import importlib.util
import io
import json
import threading
import time
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from scripts.download_models import verify_models

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
MODELS = ROOT / "models"
app = FastAPI(title="Reading Room pneumonia research demo")
app.mount("/assets", StaticFiles(directory=ROOT / "web/assets"), name="assets")
manifest = json.loads((SAMPLES / "manifest.json").read_text())
lock = threading.Lock()
module = None
verified = None


def model_ready():
    global verified
    if verified is None:
        verified = not verify_models()
    return verified


def inference_module():
    global module
    if module is None:
        spec = importlib.util.spec_from_file_location(
            "reading_room_model", MODELS / "inference.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def png(image):
    out = io.BytesIO()
    image.save(out, format="PNG")
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


@app.get("/")
def index():
    return FileResponse(ROOT / "web/index.html")


@app.get("/api/status")
def status():
    return {
        "ready": model_ready(),
        "model": "D: lung crop + ResNet18 / DeiT-Tiny probability blend",
        "setup_hint": "Run setup_demo.py to install and verify the public model release.",
    }


@app.get("/api/samples")
def samples():
    return [
        {"id": r["id"], "name": r["name"], "image": "/api/sample/" + r["id"]}
        for r in manifest
    ]


@app.get("/api/sample/{sample_id}")
def sample(sample_id: str):
    if not any(r["id"] == sample_id for r in manifest):
        raise HTTPException(404, "Unknown sample")
    return FileResponse(SAMPLES / sample_id / "full.png")


@app.get("/api/results")
def results():
    return json.loads((SAMPLES / "benchmark_results.json").read_text())


@app.post("/api/predict")
@app.post("/api/experiment/predict")
async def predict(
    file: Annotated[UploadFile | None, File()] = None,
    sample_id: Annotated[str | None, Form()] = None,
):
    if not model_ready():
        raise HTTPException(
            503,
            "Model files are missing or incomplete. Run setup_demo.py, then restart the demo.",
        )
    if not lock.acquire(blocking=False):
        raise HTTPException(409, "An image is already processing. Please wait.")
    try:
        started = time.time()
        truth = None
        source = "upload"
        if file is not None:
            raw = await file.read(12 * 1024**2 + 1)
            if len(raw) > 12 * 1024**2:
                raise HTTPException(413, "Choose an image smaller than 12 MB.")
            try:
                image = Image.open(io.BytesIO(raw))
                if image.format not in ["JPEG", "PNG"]:
                    raise HTTPException(415, "Choose a PNG or JPEG image.")
                if image.width * image.height > 25_000_000:
                    raise HTTPException(
                        413, "Choose an image with fewer than 25 million pixels."
                    )
                image.load()
            except Image.DecompressionBombError:
                raise HTTPException(413, "This image is too large.")
            except (UnidentifiedImageError, OSError, ValueError):
                raise HTTPException(
                    400, "This file is not a readable PNG or JPEG image."
                )
            full, crop, mask, meta = inference_module().segment(image)
        else:
            selected = next((r for r in manifest if r["id"] == sample_id), None)
            if selected is None:
                raise HTTPException(400, "Select a sample or upload a PNG/JPEG image.")
            full = Image.open(SAMPLES / sample_id / "full.png").convert("RGB")
            crop = Image.open(SAMPLES / sample_id / "crop.png").convert("RGB")
            mask = Image.open(SAMPLES / sample_id / "mask.png").convert("L")
            meta = selected["crop_metadata"]
            truth = selected["label"]
            source = "prepared sample"
        preprocessing = time.time() - started
        classification_started = time.time()
        output = inference_module().classify(full, crop)
        classifier = time.time() - classification_started
        response = dict(
            output,
            prediction=output["conditions"]["D"]["prediction"],
            pneumonia_score=output["conditions"]["D"]["pneumonia_score"],
            full_image=png(full),
            crop_image=png(crop),
            mask_image=png(mask),
            crop_metadata=meta,
            known_test_label=truth,
            source=source,
            preprocessing_seconds=preprocessing,
            classifier_seconds=classifier,
            seconds=time.time() - started,
        )
        return response
    finally:
        lock.release()
