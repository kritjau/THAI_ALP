from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import db
from .config import settings
from .pipeline import ALPRPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_event_queue: queue.Queue = queue.Queue()
_ws_clients: list[WebSocket] = []
_pipeline: ALPRPipeline | None = None
_stop_event = threading.Event()


def _with_image_url(item: dict) -> dict:
    """Adds a browser-usable `image` URL alongside the stored `image_path`,
    resolved to a fixed /captures/<filename> URL regardless of what
    CAPTURES_DIR is actually named on disk."""
    image_path = item.get("image_path")
    item["image"] = f"/captures/{Path(image_path).name}" if image_path else None
    return item


def _processing_loop():
    global _pipeline
    _pipeline = ALPRPipeline()
    logger.info(
        "ALPR pipeline started (%d camera(s): %s)",
        len(_pipeline.cameras), [c.name for c in _pipeline.cameras],
    )
    while not _stop_event.is_set():
        try:
            for event in _pipeline.step():
                _event_queue.put(_with_image_url(event))
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
    db.init_db()
    thread = threading.Thread(target=_processing_loop, daemon=True)
    thread.start()
    broadcaster = asyncio.create_task(_broadcast_events())
    yield
    _stop_event.set()
    broadcaster.cancel()
    if _pipeline:
        _pipeline.stop()


app = FastAPI(title="Thai ALPR", lifespan=lifespan)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

Path(settings.captures_dir).mkdir(parents=True, exist_ok=True)
app.mount("/captures", StaticFiles(directory=settings.captures_dir), name="captures")


async def _mjpeg_generator(camera_id: str | None):
    # async, not a plain generator: Starlette runs a sync generator's each-
    # next() call in its (limited-size) request threadpool, and a
    # StreamingResponse holds that generator open for the connection's whole
    # lifetime -- so every open video stream would permanently pin one
    # threadpool worker, and enough concurrent/stale connections (dashboard
    # reconnects, multiple tabs) starve that pool outright, hanging even
    # unrelated endpoints. An async generator yields control back to the
    # event loop on every `await` instead.
    while True:
        if _pipeline is None:
            await asyncio.sleep(0.1)
            continue
        frame_bytes = _pipeline.latest_jpeg(camera_id)
        if frame_bytes is None:
            await asyncio.sleep(0.1)
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        await asyncio.sleep(0.05)


@app.get("/video_feed")
def video_feed():
    """Kept as the first/primary camera for backward compatibility (existing
    bookmarks, a single-camera .env) -- /video_feed/{camera_id} is what a
    multi-camera dashboard actually uses."""
    return StreamingResponse(
        _mjpeg_generator(None), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/video_feed/{camera_id}")
def video_feed_by_camera(camera_id: str):
    return StreamingResponse(
        _mjpeg_generator(camera_id), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/cameras")
def api_cameras():
    return _pipeline.camera_list() if _pipeline else []


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
    return [_with_image_url(item) for item in db.recent_detections(limit)]


@app.get("/", response_class=HTMLResponse)
def index():
    return (static_dir / "index.html").read_text(encoding="utf-8")
