from __future__ import annotations

"""Temporary A/B evaluation service -- NOT part of the shipped app.

Wraps the YOLOS Apache-2.0 plate-detector candidate (nickmuchi/yolos-small-
finetuned-license-plate-detection) behind the same "POST a JPEG, get boxes
back" shape ocr_service/ already uses for OCR -- so app/shadow_compare.py
can send it the same frame the production (Ultralytics, AGPL-3.0) detector
just ran on, and log how the two compare, without loading a second heavy
model into app_live's own process. Runs on its own GPU (see the systemd
unit template) for the same reason ocr_service does: isolating it means a
crash or slow response here can never affect the live camera pipeline.

This whole directory exists only to gather evidence for the AGPL-vs-Apache
detector decision (see the licensing discussion this was built for) --
delete it, delete app/shadow_compare.py, and unset SHADOW_DETECTOR_URL once
that decision is made. It is not meant to become a permanent second
detector running alongside the real one.
"""

import io
import logging
import threading

import torch
from fastapi import FastAPI, HTTPException, Request
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForObjectDetection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_ID = "nickmuchi/yolos-small-finetuned-license-plate-detection"
# Matches the post_process_object_detection threshold used during the
# initial benchmark -- kept low-ish on purpose so the log captures
# borderline shadow detections too (including the false-positive kind),
# rather than only the confident ones.
_SCORE_THRESHOLD = 0.3

app = FastAPI(title="Shadow plate detector (YOLOS, Apache-2.0) -- temporary A/B evaluation only")

_device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Loading %s on %s", MODEL_ID, _device)
_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
_model = AutoModelForObjectDetection.from_pretrained(MODEL_ID).to(_device).eval()
# One model instance, called from FastAPI's threadpool -- a lock serializes
# inference calls the same way ocr_service's does, since app/shadow_compare.py
# only ever sends one sample at a time per camera anyway (see its
# sample_interval throttling), so contention here should be rare.
_lock = threading.Lock()


@app.post("/detect")
async def detect(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty body")
    try:
        image = Image.open(io.BytesIO(body)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="couldn't decode image")

    with _lock:
        inputs = _processor(images=image, return_tensors="pt").to(_device)
        with torch.no_grad():
            outputs = _model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = _processor.post_process_object_detection(
            outputs, threshold=_SCORE_THRESHOLD, target_sizes=target_sizes
        )[0]

    boxes = [
        {"box": [round(x) for x in box.tolist()], "score": round(score.item(), 3)}
        for score, box in zip(results["scores"], results["boxes"])
    ]
    return {"boxes": boxes}


@app.get("/health")
def health():
    return {"ok": True, "device": _device}
