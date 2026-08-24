from __future__ import annotations

import logging
import os
import queue
import threading
import time

import cv2

from app.camera import CameraStream
from app.color import classify_vehicle_color
from app.config import settings
from app.detector import PlateDetector
from app.ocr import PlateReader, looks_like_thai_plate
from app.tracker import PlateTracker
from app.vehicle_detector import VehicleDetector

logger = logging.getLogger(__name__)

# ByteTrack can still fragment a car into a new track id after an occlusion or
# a gap between processed frames, which would otherwise show up as a second
# "new" detection for the same physical car. This cooldown dedupes by the OCR
# text itself, independent of track id, so a re-appearing plate within this
# window is treated as a continuation, not a new sighting. Local to app_live
# (not a shared app/ setting) -- read straight from the environment so .env
# doesn't need an app/config.py field for it.
_PLATE_COOLDOWN_SECONDS = float(os.environ.get("PLATE_COOLDOWN_SECONDS", 45))


class LiveOnlyPipeline:
    """Same detect -> track -> OCR flow as app/pipeline.py, but nothing is ever
    written to disk: no saved crop, no database row, no JSON export. A plate's
    text only exists in memory for as long as its track is alive, so nothing
    here needs a data-retention policy."""

    def __init__(self):
        self.camera = CameraStream(settings.camera_source_value()).start()
        self.detector = PlateDetector()
        self.vehicle_detector = VehicleDetector()
        self.reader = PlateReader()
        self.tracker = PlateTracker(
            ttl_seconds=settings.track_ttl_seconds,
            reocr_seconds=settings.track_reocr_seconds,
        )
        self._latest_annotated = None
        self._latest_jpeg = None
        self._frame_count = 0
        self._visible_tracks: list = []
        # plate_text -> last time it was seen, across all track ids. Only
        # ever touched by the OCR worker thread below, so no lock needed.
        self._recent_plates: dict[str, float] = {}

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
        frame = self.camera.read()
        if frame is None:
            return []

        self._frame_count += 1
        if self._frame_count % settings.process_every_n_frames == 0:
            self._detect_and_dispatch(frame)
        self._draw(frame)
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
                self._read(track_id, track, crop, box, frame)
            except Exception:
                logger.exception("OCR worker failed for track %s", track_id)
            finally:
                with self._in_flight_lock:
                    self._in_flight.discard(track_id)

    def _read(self, track_id, track, crop, box, frame):
        x1, y1, x2, y2 = box
        text, ocr_conf = self.reader.read(crop)
        track.last_ocr = time.time()

        if not text or ocr_conf < settings.min_log_confidence or not looks_like_thai_plate(text):
            return
        if track.logged and ocr_conf <= track.confidence:
            return

        now = time.time()
        # Only a track's *first* successful read is a "new sighting" --
        # later re-OCRs of the same track are refinements and should
        # always go through so the dashboard row keeps updating.
        is_new_sighting = not track.logged
        last_seen = self._recent_plates.get(text)
        is_recent_duplicate = (
            is_new_sighting and last_seen is not None and (now - last_seen) < _PLATE_COOLDOWN_SECONDS
        )
        self._recent_plates[text] = now
        self._prune_recent_plates()

        vehicle_boxes = self.vehicle_detector.detect(frame)
        vehicle_box = VehicleDetector.find_containing(vehicle_boxes, box)
        vehicle_crop = frame[vehicle_box[1]:vehicle_box[3], vehicle_box[0]:vehicle_box[2]] if vehicle_box else None

        track.plate_text = text
        track.confidence = ocr_conf
        track.vehicle_color = classify_vehicle_color(vehicle_crop)
        track.logged = True
        if not is_recent_duplicate:
            self._new_events.put(
                {
                    "id": track_id,
                    "plate_text": text,
                    "confidence": ocr_conf,
                    "vehicle_color": track.vehicle_color,
                    "bbox": [x1, y1, x2, y2],
                    "timestamp": now,
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
        stale = [text for text, seen in self._recent_plates.items() if seen < cutoff]
        for text in stale:
            del self._recent_plates[text]

    def _draw(self, frame):
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
