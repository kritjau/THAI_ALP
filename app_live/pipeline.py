from __future__ import annotations

import base64
import os
import queue
import time

import cv2

from app import registered_plates_db
from app.camera_worker import CameraWorker
from app.config import settings
from app.gate import open_gate

# ByteTrack can still fragment a car into a new track id after an occlusion or
# a gap between processed frames, which would otherwise show up as a second
# "new" detection for the same physical car. This cooldown dedupes by the OCR
# text itself (per camera -- the same plate on two different cameras is two
# genuine sightings, not a fragmented track), independent of track id, so a
# re-appearing plate within this window is treated as a continuation, not a
# new sighting. Local to app_live (not a shared app/ setting) -- read straight
# from the environment so .env doesn't need an app/config.py field for it.
_PLATE_COOLDOWN_SECONDS = float(os.environ.get("PLATE_COOLDOWN_SECONDS", 45))


def _as_data_uri(crop) -> str | None:
    """Encodes the plate crop as an inline base64 image so the dashboard can
    show what was actually captured next to the OCR'd text -- entirely in
    memory (part of the same event payload already sent over the WebSocket),
    never written to disk, consistent with this pipeline never persisting
    anything."""
    if crop is None or crop.size == 0:
        return None
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


class LiveOnlyPipeline:
    """Runs one CameraWorker per configured camera (Settings.camera_configs()),
    same detect -> track -> OCR flow as app/pipeline.py (see
    app/camera_worker.py), but nothing is ever written to disk: no saved
    crop, no database row, no JSON export. A plate's text only exists in
    memory for as long as its track is alive, so nothing here needs a
    data-retention policy. Also checks each read against the registered-
    plate whitelist and triggers the gate -- see app/registered_plates_db.py
    and app/gate.py."""

    def __init__(self):
        self._new_events: queue.Queue = queue.Queue()
        # (camera_id, plate_text) -> last time it was seen. Only ever touched
        # by the OCR worker threads below (one per camera, but each key is
        # only ever written by its own camera's thread), so no lock needed.
        self._recent_plates: dict[tuple[str, str], float] = {}
        self.cameras = [
            CameraWorker(cfg["id"], cfg["name"], cfg["source"], self._on_new_read)
            for cfg in settings.camera_configs()
        ]
        self._cameras_by_id = {cam.camera_id: cam for cam in self.cameras}

    def step(self) -> list[dict]:
        for cam in self.cameras:
            cam.step()
        return self._drain_events()

    def _on_new_read(self, worker, track_id, track, box, crop, text, ocr_conf, color, was_logged_before):
        now = time.time()
        # Only a track's *first* successful read is a "new sighting" --
        # later re-OCRs of the same track are refinements and should
        # always go through so the dashboard row keeps updating.
        is_new_sighting = not was_logged_before
        cooldown_key = (worker.camera_id, text)
        last_seen = self._recent_plates.get(cooldown_key)
        is_recent_duplicate = (
            is_new_sighting and last_seen is not None and (now - last_seen) < _PLATE_COOLDOWN_SECONDS
        )
        self._recent_plates[cooldown_key] = now
        self._prune_recent_plates()

        registered = registered_plates_db.is_registered_plate(text)
        if registered and not track.gate_opened:
            open_gate(text)
            track.gate_opened = True

        if not is_recent_duplicate:
            self._new_events.put(
                {
                    "id": track_id,
                    "camera_id": worker.camera_id,
                    "camera_name": worker.name,
                    "plate_text": text,
                    "confidence": ocr_conf,
                    "color": color,
                    "bbox": list(box),
                    "timestamp": now,
                    "image": _as_data_uri(crop),
                    "registered": registered,
                }
            )

    def _drain_events(self) -> list[dict]:
        events = []
        while True:
            try:
                events.append(self._new_events.get_nowait())
            except queue.Empty:
                return events

    def _prune_recent_plates(self):
        cutoff = time.time() - _PLATE_COOLDOWN_SECONDS
        stale = [key for key, seen in self._recent_plates.items() if seen < cutoff]
        for key in stale:
            del self._recent_plates[key]

    def camera_list(self) -> list[dict]:
        return [{"id": cam.camera_id, "name": cam.name} for cam in self.cameras]

    def latest_jpeg(self, camera_id: str | None = None) -> bytes | None:
        cam = self._resolve_camera(camera_id)
        return cam.latest_jpeg() if cam else None

    def _resolve_camera(self, camera_id: str | None) -> CameraWorker | None:
        if camera_id is None:
            return self.cameras[0] if self.cameras else None
        return self._cameras_by_id.get(camera_id)

    def stop(self):
        for cam in self.cameras:
            cam.stop()
