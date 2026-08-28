from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app import registered_plates_db
from app.config import settings

from .pipeline import LiveOnlyPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_event_queue: queue.Queue = queue.Queue()
_ws_clients: list[WebSocket] = []
_pipeline: LiveOnlyPipeline | None = None
_stop_event = threading.Event()
# In-memory only, capped, gone on restart -- this is the only "history" this
# version has; nothing is ever written to disk.
_recent: deque = deque(maxlen=50)


def _processing_loop():
    global _pipeline
    _pipeline = LiveOnlyPipeline()
    logger.info(
        "ALPR live-only pipeline started (camera source=%r) -- no DB, no JSON export, no saved crops",
        settings.camera_source,
    )
    while not _stop_event.is_set():
        try:
            for event in _pipeline.step():
                _recent.append(event)
                _event_queue.put(event)
        except Exception:
            logger.exception("Error in pipeline step")
            time.sleep(0.5)
        time.sleep(0.03)


async def _broadcast_events():
    while True:
        try:
            event = _event_queue.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.1)
            continue
        payload = json.dumps(event)
        stale = []
        for ws in _ws_clients:
            try:
                await ws.send_text(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            _ws_clients.remove(ws)


@asynccontextmanager
async def lifespan(app: FastAPI):
    registered_plates_db.init_db()
    thread = threading.Thread(target=_processing_loop, daemon=True)
    thread.start()
    broadcaster = asyncio.create_task(_broadcast_events())
    yield
    _stop_event.set()
    broadcaster.cancel()
    if _pipeline:
        _pipeline.stop()


app = FastAPI(title="Thai ALPR (live-only, no storage)", lifespan=lifespan)

# Reuses the existing dashboard assets in place rather than duplicating them.
_static_dir = Path(__file__).resolve().parent.parent / "app" / "static"
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


def _mjpeg_generator():
    while True:
        if _pipeline is None:
            time.sleep(0.1)
            continue
        frame_bytes = _pipeline.latest_jpeg()
        if frame_bytes is None:
            time.sleep(0.1)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        time.sleep(0.05)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        _mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.websocket("/ws/detections")
async def ws_detections(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


@app.get("/api/detections")
def api_detections(limit: int = 50):
    # Served from the in-memory deque above -- there is no database in this
    # version, so a server restart clears this too.
    return list(_recent)[-limit:][::-1]


@app.get("/", response_class=HTMLResponse)
def index():
    return (_static_dir / "index.html").read_text(encoding="utf-8")


@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    return (_static_dir / "admin.html").read_text(encoding="utf-8")


@app.get("/api/registered-plates")
def api_list_registered_plates():
    return registered_plates_db.list_registered_plates()


@app.post("/api/registered-plates")
def api_add_registered_plate(payload: dict = Body(...)):
    plate_text = (payload.get("plate_text") or "").strip()
    if not plate_text:
        raise HTTPException(status_code=400, detail="plate_text is required")
    registered_plates_db.add_registered_plate(plate_text, payload.get("label"))
    return {"ok": True}


@app.delete("/api/registered-plates/{plate_text}")
def api_remove_registered_plate(plate_text: str):
    registered_plates_db.remove_registered_plate(plate_text)
    return {"ok": True}
