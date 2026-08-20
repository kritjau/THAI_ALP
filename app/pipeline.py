from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2

from . import db
from .camera import CameraStream
from .config import settings
from .detector import PlateDetector
from .json_export import JsonExporter
from .ocr import PlateReader, looks_like_thai_plate
from .tracker import PlateTracker

logger = logging.getLogger(__name__)


class ALPRPipeline:
    def __init__(self):
        self.camera = CameraStream(settings.camera_source_value()).start()
        self.detector = PlateDetector()
        self.reader = PlateReader()
        self.tracker = PlateTracker(
            ttl_seconds=settings.track_ttl_seconds,
            reocr_seconds=settings.track_reocr_seconds,
        )
        self.json_exporter = JsonExporter()
        self._latest_annotated = None
        self._frame_count = 0
        self._visible_tracks: list = []
        Path(settings.captures_dir).mkdir(parents=True, exist_ok=True)

    def step(self) -> list[dict]:
        """Grab one frame; run detection+OCR only every Nth frame so a CPU-bound
        pipeline still keeps the live view smooth."""
        frame = self.camera.read()
        if frame is None:
            return []

        self._frame_count += 1
        if self._frame_count % settings.process_every_n_frames == 0:
            events = self._process(frame)
        else:
            self._draw(frame)
            events = []

        self.json_exporter.maybe_flush()
        return events

    def _process(self, frame) -> list[dict]:
        events = []
        boxes = self.detector.detect(frame)
        current_tracks = []
        for (x1, y1, x2, y2, _det_conf, track_id) in boxes:
            if track_id is None:
                continue  # tracker hasn't confirmed an id for this box yet
            track = self.tracker.get(track_id, (x1, y1, x2, y2))
            current_tracks.append(track)
            if not self.tracker.needs_ocr(track):
                continue

            crop = frame[max(0, y1):y2, max(0, x1):x2]
            text, ocr_conf = self.reader.read(crop)
            track.last_ocr = time.time()

            # Discard noise (motion blur, false-positive detections without Thai
            # script) and don't overwrite an already-logged plate with a worse read
            # -- driving footage re-OCRs the same persisting track every few seconds.
            if not text or ocr_conf < settings.min_log_confidence or not looks_like_thai_plate(text):
                continue
            if track.logged and ocr_conf <= track.confidence:
                continue

            track.plate_text = text
            track.confidence = ocr_conf
            image_path = self._save_crop(crop, text)
            if track.logged:
                db.update_detection(track.db_id, text, ocr_conf, (x1, y1, x2, y2), image_path)
                self._delete_crop(track.image_path)
            else:
                track.db_id = db.insert_detection(text, ocr_conf, (x1, y1, x2, y2), image_path)
                track.logged = True
            track.image_path = image_path
            self.json_exporter.record(text, ocr_conf, image_path)
            events.append(
                {
                    "id": track.db_id,
                    "plate_text": text,
                    "confidence": ocr_conf,
                    "bbox": [x1, y1, x2, y2],
                    "timestamp": time.time(),
                    "image_path": image_path,
                }
            )

        self._visible_tracks = current_tracks
        self._draw(frame)
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

    def _draw(self, frame):
        # OpenCV can't render Thai glyphs, so the overlay only shows a box and
        # confidence; the full Thai plate text is shown in the web log panel.
        annotated = frame.copy()
        for t in self._visible_tracks:
            x1, y1, x2, y2 = t.box[:4]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = f"{t.confidence * 100:.0f}%" if t.plate_text else "..."
            cv2.putText(
                annotated, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2,
            )
        self._latest_annotated = annotated

    def latest_jpeg(self) -> bytes | None:
        frame = self._latest_annotated if self._latest_annotated is not None else self.camera.read()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality])
        return buf.tobytes() if ok else None

    def stop(self):
        self.camera.stop()
