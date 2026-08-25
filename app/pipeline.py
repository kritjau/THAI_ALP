from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path

import cv2

from . import db
from .camera import CameraStream
from .color import classify_color
from .config import settings
from .detector import PlateDetector
from .json_export import JsonExporter
from .ocr import PlateReader, looks_like_thai_plate
from .tracker import PlateTracker
from .vehicle_detector import VehicleDetector

logger = logging.getLogger(__name__)


class ALPRPipeline:
    def __init__(self):
        self.camera = CameraStream(settings.camera_source_value()).start()
        self.detector = PlateDetector()
        self.vehicle_detector = VehicleDetector()
        self.reader = PlateReader()
        self.tracker = PlateTracker(
            ttl_seconds=settings.track_ttl_seconds,
            reocr_seconds=settings.track_reocr_seconds,
        )
        self.json_exporter = JsonExporter()
        self._latest_annotated = None
        self._latest_jpeg = None
        self._frame_count = 0
        self._visible_tracks: list = []
        Path(settings.captures_dir).mkdir(parents=True, exist_ok=True)

        # OCR (CPU) + vehicle-color detection are the slow part of a "new
        # sighting" -- running them inline used to make the visible frame
        # wait on however long that took. They now run on a dedicated
        # background worker instead, so the capture/detect/draw loop below
        # never blocks on them; a bounded queue plus this in-flight set caps
        # the backlog and stops the same track being queued twice.
        self._ocr_queue: queue.Queue = queue.Queue(maxsize=4)
        self._in_flight: set[int] = set()
        self._in_flight_lock = threading.Lock()
        self._new_events: queue.Queue = queue.Queue()
        threading.Thread(target=self._ocr_worker, daemon=True).start()

    def step(self) -> list[dict]:
        """Grab one frame, run (fast, GPU) detection at most every Nth frame,
        and always draw immediately -- OCR for any plate that needs it is
        handed off to the background worker rather than run inline."""
        frame = self.camera.read()
        if frame is None:
            return []

        self._frame_count += 1
        if self._frame_count % settings.process_every_n_frames == 0:
            self._detect_and_dispatch(frame)
        self._draw(frame)

        self.json_exporter.maybe_flush()
        return self._drain_events()

    def _detect_and_dispatch(self, frame):
        boxes = self.detector.detect(frame)
        current_tracks = []
        for (x1, y1, x2, y2, _det_conf, track_id) in boxes:
            if track_id is None:
                continue  # tracker hasn't confirmed an id for this box yet
            track = self.tracker.get(track_id, (x1, y1, x2, y2))
            current_tracks.append(track)
            if not self.tracker.needs_ocr(track):
                continue

            with self._in_flight_lock:
                if track_id in self._in_flight:
                    continue  # already queued/being read, don't duplicate
                self._in_flight.add(track_id)

            crop = frame[max(0, y1):y2, max(0, x1):x2].copy()
            try:
                self._ocr_queue.put_nowait((track_id, track, crop, (x1, y1, x2, y2), frame))
            except queue.Full:
                # Backlogged -- drop for now, needs_ocr() stays true so this
                # track is simply retried on a later processed frame.
                with self._in_flight_lock:
                    self._in_flight.discard(track_id)

        self._visible_tracks = current_tracks

    def _ocr_worker(self):
        while True:
            track_id, track, crop, box, frame = self._ocr_queue.get()
            try:
                self._read_and_log(track, crop, box, frame)
            except Exception:
                logger.exception("OCR worker failed for track %s", track_id)
            finally:
                with self._in_flight_lock:
                    self._in_flight.discard(track_id)

    def _read_and_log(self, track, crop, box, frame):
        x1, y1, x2, y2 = box
        text, ocr_conf = self.reader.read(crop)
        track.last_ocr = time.time()

        # Discard noise (motion blur, false-positive detections without Thai
        # script) and don't overwrite an already-logged plate with a worse read
        # -- driving footage re-OCRs the same persisting track every few seconds.
        if not text or ocr_conf < settings.min_log_confidence or not looks_like_thai_plate(text):
            return
        if track.logged and ocr_conf <= track.confidence:
            return

        vehicle_boxes = self.vehicle_detector.detect(frame)
        vehicle_box = VehicleDetector.find_containing(vehicle_boxes, box)
        vehicle_crop = None
        plate_in_vehicle = None
        if vehicle_box:
            vx1, vy1, vx2, vy2 = vehicle_box
            vehicle_crop = frame[vy1:vy2, vx1:vx2]
            plate_in_vehicle = (x1 - vx1, y1 - vy1, x2 - vx1, y2 - vy1)

        track.plate_text = text
        track.confidence = ocr_conf
        track.color = classify_color(vehicle_crop, exclude_box=plate_in_vehicle)
        image_path = self._save_crop(crop, text)
        if track.logged:
            db.update_detection(
                track.db_id, text, ocr_conf, box, image_path,
                color=track.color,
            )
            self._delete_crop(track.image_path)
        else:
            track.db_id = db.insert_detection(
                text, ocr_conf, box, image_path,
                color=track.color,
            )
            track.logged = True
        track.image_path = image_path
        self.json_exporter.record(
            text, ocr_conf, image_path,
            color=track.color,
        )
        self._new_events.put(
            {
                "id": track.db_id,
                "plate_text": text,
                "confidence": ocr_conf,
                "color": track.color,
                "bbox": [x1, y1, x2, y2],
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
        # Encoded once here rather than per MJPEG client per poll -- the frame
        # only actually changes once per processed camera frame, not every time
        # a viewer's stream generator wakes up (or once per viewer). Downscaled
        # first: boxes are drawn at full resolution so they stay pixel-accurate,
        # but a wide-FOV CCTV frame is far bigger than a dashboard needs, and
        # encoding/transferring/decoding it at full size is real, avoidable cost.
        ok, buf = cv2.imencode(
            ".jpg", self._resize_for_stream(annotated), [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality]
        )
        self._latest_jpeg = buf.tobytes() if ok else None

    @staticmethod
    def _resize_for_stream(frame):
        h, w = frame.shape[:2]
        if w <= settings.stream_max_width:
            return frame
        scale = settings.stream_max_width / w
        return cv2.resize(frame, (settings.stream_max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

    def latest_jpeg(self) -> bytes | None:
        if self._latest_jpeg is not None:
            return self._latest_jpeg
        frame = self.camera.read()
        if frame is None:
            return None
        ok, buf = cv2.imencode(
            ".jpg", self._resize_for_stream(frame), [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality]
        )
        return buf.tobytes() if ok else None

    def stop(self):
        self.camera.stop()
