from __future__ import annotations

import logging
import threading

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request

from app.ocr import PlateReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ALPR OCR service (GPU)")

# One PaddleOCR instance for the life of this process -- this service exists
# specifically so it can be the only thing on GPU-enabled paddlepaddle in the
# whole deployment (see README.md's Known limitations on why paddlepaddle-gpu
# can't share a venv with a CUDA build of torch). Launch this under
# CUDA_VISIBLE_DEVICES pointed at whichever GPU the detector isn't using
# (see ocr_service.service) -- "cuda:0" below then resolves to that GPU, not
# necessarily physical GPU 0.
_reader = PlateReader(device="cuda:0")
# PaddleOCR inference isn't guaranteed safe to call concurrently on one
# instance -- every camera's OCR worker thread ends up calling into this
# same process, so serialize actual inference the same way
# camera_worker.py's in-process shared reader does.
_lock = threading.Lock()


@app.post("/read")
async def read(request: Request):
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty body")
    arr = np.frombuffer(body, dtype=np.uint8)
    crop = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if crop is None:
        raise HTTPException(status_code=400, detail="couldn't decode image")
    with _lock:
        text, confidence = _reader.read(crop)
    return {"text": text, "confidence": confidence}


@app.get("/health")
def health():
    return {"ok": True}
