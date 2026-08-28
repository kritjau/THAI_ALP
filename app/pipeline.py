from __future__ import annotations

import queue
import time
from pathlib import Path

import cv2

from . import db
from .camera_worker import CameraWorker
from .config import settings
from .json_export import JsonExporter


class ALPRPipeline:
    """Runs one CameraWorker per configured camera (Settings.camera_configs())
    and handles what's specific to this app: logging every read to SQLite,
    a saved crop, and the JSON export -- see app/camera_worker.py for the
    detect/track/OCR logic shared with app_live/pipeline.py."""

    def __init__(self):
        Path(settings.captures_dir).mkdir(parents=True, exist_ok=True)
        self.json_exporter = JsonExporter()
        self._new_events: queue.Queue = queue.Queue()
        self.cameras = [
            CameraWorker(cfg["id"], cfg["name"], cfg["source"], self._on_new_read)
            for cfg in settings.camera_configs()
        ]
        self._cameras_by_id = {cam.camera_id: cam for cam in self.cameras}

    def step(self) -> list[dict]:
        for cam in self.cameras:
            cam.step()
        self.json_exporter.maybe_flush()
        return self._drain_events()

    def _on_new_read(self, worker, _track_id, track, box, crop, text, ocr_conf, color, was_logged_before):
        image_path = self._save_crop(crop, text)
        if was_logged_before:
            db.update_detection(
                track.db_id, text, ocr_conf, box, image_path,
                color=color, camera_id=worker.camera_id, camera_name=worker.name,
            )
            self._delete_crop(track.image_path)
        else:
            track.db_id = db.insert_detection(
                text, ocr_conf, box, image_path,
                color=color, camera_id=worker.camera_id, camera_name=worker.name,
            )
        track.image_path = image_path
        self.json_exporter.record(
            text, ocr_conf, image_path,
            color=color,
        )
        self._new_events.put(
            {
                "id": track.db_id,
                "camera_id": worker.camera_id,
                "camera_name": worker.name,
                "plate_text": text,
                "confidence": ocr_conf,
                "color": color,
                "bbox": list(box),
                "timestamp": time.time(),
                "image_path": image_path,
            }
        )

    def _drain_events(self) -> list[dict]:
        events = []
        while True:
            try:
                events.append(self._new_events.get_nowait())
            except queue.Empty:
                return events

    def _save_crop(self, crop, text) -> str | None:
        if crop.size == 0:
            return None
        safe_text = "".join(c for c in text if c.isalnum()) or "plate"
        filename = f"{int(time.time() * 1000)}_{safe_text}.jpg"
        path = Path(settings.captures_dir) / filename
        cv2.imwrite(str(path), crop)
        return str(path)

    @staticmethod
    def _delete_crop(image_path: str | None):
        if image_path:
            Path(image_path).unlink(missing_ok=True)

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
