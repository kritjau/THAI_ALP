from __future__ import annotations

import os
import time

import cv2

from app.camera import CameraStream
from app.config import settings
from app.detector import PlateDetector
from app.ocr import PlateReader, looks_like_thai_plate
from app.tracker import PlateTracker

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
        self.reader = PlateReader()
        self.tracker = PlateTracker(
            ttl_seconds=settings.track_ttl_seconds,
            reocr_seconds=settings.track_reocr_seconds,
        )
        self._latest_annotated = None
        self._frame_count = 0
        self._visible_tracks: list = []
        # plate_text -> last time it was seen, across all track ids
        self._recent_plates: dict[str, float] = {}

    def step(self) -> list[dict]:
        frame = self.camera.read()
        if frame is None:
            return []

        self._frame_count += 1
        if self._frame_count % settings.process_every_n_frames == 0:
            events = self._process(frame)
        else:
            self._draw(frame)
            events = []
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

            if not text or ocr_conf < settings.min_log_confidence or not looks_like_thai_plate(text):
                continue
            if track.logged and ocr_conf <= track.confidence:
                continue

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

            track.plate_text = text
            track.confidence = ocr_conf
            track.logged = True
            if not is_recent_duplicate:
                events.append(
                    {
                        "id": track_id,
                        "plate_text": text,
                        "confidence": ocr_conf,
                        "bbox": [x1, y1, x2, y2],
                        "timestamp": now,
                    }
                )

        self._prune_recent_plates()
        self._visible_tracks = current_tracks
        self._draw(frame)
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

    def latest_jpeg(self) -> bytes | None:
        frame = self._latest_annotated if self._latest_annotated is not None else self.camera.read()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality])
        return buf.tobytes() if ok else None

    def stop(self):
        self.camera.stop()
