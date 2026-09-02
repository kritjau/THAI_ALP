from __future__ import annotations

import logging
import queue
import threading
import time

import cv2

from .camera import CameraStream
from .color import classify_color
from .config import settings
from .detector import PlateDetector
from .ocr import PlateReader, looks_like_thai_plate
from .plate_format import looks_like_valid_plate_number
from .plate_match import normalize_plate, vote_plate_text
from .tracker import PlateTracker
from .vehicle_detector import VehicleDetector

logger = logging.getLogger(__name__)

# Reads kept per track for majority voting (see plate_match.vote_plate_text)
# -- capped so a plate that lingers a long time doesn't let stale early
# reads outvote what the camera is seeing now.
MAX_READ_HISTORY = 5


def aggregate_rejection_stats(cameras: list["CameraWorker"]) -> dict:
    """Sums accepted/rejected plate-shape counts across every camera --
    shared by app/pipeline.py and app_live/pipeline.py so both dashboards
    show the same "how well is OCR doing" signal the same way."""
    accepted = sum(cam.accepted_count for cam in cameras)
    rejected = sum(cam.rejected_count for cam in cameras)
    return {"accepted": accepted, "rejected": rejected}


def build_camera_workers(configs: list[dict], on_new_read) -> list["CameraWorker"]:
    """Builds one CameraWorker per config, skipping (and logging) any whose
    source fails to open instead of letting one bad/offline camera take
    down every other camera too. Without this, a plain list comprehension
    over CameraWorker(...) lets one flaky/offline source (e.g. a dropped SSH
    tunnel) raise mid-construction and abort the whole pipeline -- observed
    in practice: an offline tunneled webcam killed the background pipeline
    thread outright, leaving *no* camera (including otherwise-healthy ones)
    and an empty /api/cameras indefinitely, since nothing restarts that
    thread on its own."""
    cameras = []
    for cfg in configs:
        try:
            cameras.append(CameraWorker(cfg["id"], cfg["name"], cfg["source"], on_new_read))
        except Exception:
            logger.exception(
                "Camera %s (%s) failed to start -- skipping it, other cameras continue",
                cfg["id"], cfg["name"],
            )
    return cameras


class CameraWorker:
    """One camera's full detect -> track -> OCR pipeline: its own camera read
    thread, plate detector (ByteTrack's tracker state is per-model-instance,
    so it can't be shared across cameras), tracker, OCR reader and vehicle
    detector. app/pipeline.py and app_live/pipeline.py each run one of these
    per configured camera (see Settings.camera_configs()) instead of
    duplicating this detect/track/OCR machinery per app -- the one thing
    that differs between them (what happens with a successful read: log to
    DB vs. check the gate whitelist) is left to the `on_new_read` callback.

    on_new_read(worker, track_id, track, box, crop, text, ocr_conf, color,
    vehicle_type, was_logged_before) is called from this camera's own OCR worker thread
    for every read that passes the confidence/plausibility filters.
    `was_logged_before` is `track.logged` from *before* this call updates
    it, i.e. whether this is the track's first successful read (a new
    sighting) or a refinement of one already reported.
    """

    def __init__(self, camera_id: str, name: str, source, on_new_read):
        self.camera_id = camera_id
        self.name = name
        self.camera = CameraStream(source).start()
        self.detector = PlateDetector()
        self.vehicle_detector = VehicleDetector()
        self.reader = PlateReader()
        self.tracker = PlateTracker(
            ttl_seconds=settings.track_ttl_seconds,
            reocr_seconds=settings.track_reocr_seconds,
        )
        self._on_new_read = on_new_read
        self._latest_jpeg = None
        self._frame_count = 0
        self._visible_tracks: list = []
        # Reads that passed the plate-shape check vs. ones rejected by it --
        # see _read() -- exposed via rejection_stats() so the dashboard can
        # show the real-world accept/reject split instead of it only being
        # visible by grepping the log.
        self.accepted_count = 0
        self.rejected_count = 0

        # See app/pipeline.py's original docstring for why this is a
        # decoupled background worker rather than inline OCR: it keeps a
        # slow CPU-bound read from ever blocking this camera's video draw.
        self._ocr_queue: queue.Queue = queue.Queue(maxsize=4)
        self._in_flight: set[int] = set()
        self._in_flight_lock = threading.Lock()
        threading.Thread(target=self._ocr_worker, daemon=True, name=f"ocr-{camera_id}").start()

    def step(self):
        """Grab one frame, run (fast, GPU) detection at most every Nth frame,
        and always draw immediately -- OCR for any plate that needs it is
        handed off to the background worker rather than run inline."""
        frame = self.camera.read()
        if frame is None:
            return
        self._frame_count += 1
        if self._frame_count % settings.process_every_n_frames == 0:
            self._detect_and_dispatch(frame)
        self._draw(frame)

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
                logger.exception("OCR worker failed for camera %s track %s", self.camera_id, track_id)
            finally:
                with self._in_flight_lock:
                    self._in_flight.discard(track_id)

    def _read(self, track_id, track, crop, box, frame):
        x1, y1, x2, y2 = box
        text, ocr_conf = self.reader.read(crop)
        track.last_ocr = time.time()

        # Discard noise (motion blur, false-positive detections without Thai
        # script) -- silently, these are expected and frequent enough that
        # logging every one would just be noise.
        if not text or ocr_conf < settings.min_log_confidence or not looks_like_thai_plate(text):
            return

        # Unlike the above, a read with Thai script and reasonable confidence
        # that still doesn't match a plausible plate shape is worth surfacing
        # -- it's evidence for whether OCR accuracy needs a bigger fix (e.g.
        # perspective correction on the crop before OCR) or whether this
        # shape check alone is already catching most of what would otherwise
        # have been a bad read.
        normalized = normalize_plate(text)
        if not looks_like_valid_plate_number(normalized):
            self.rejected_count += 1
            logger.info(
                "Camera %s track %s: rejected %r (normalized %r, conf %.2f) -- "
                "doesn't match a plausible Thai plate shape",
                self.camera_id, track_id, text, normalized, ocr_conf,
            )
            return
        self.accepted_count += 1

        track.read_history.append((text, ocr_conf))
        if len(track.read_history) > MAX_READ_HISTORY:
            track.read_history.pop(0)
        voted_text, voted_conf = vote_plate_text(track.read_history)

        if track.logged and voted_text == track.plate_text:
            return  # this read didn't change the voted-consensus text

        vehicle_boxes = self.vehicle_detector.detect(frame)
        vehicle_box = VehicleDetector.find_containing(vehicle_boxes, box)
        vehicle_crop = None
        plate_in_vehicle = None
        vehicle_type = ""
        if vehicle_box:
            (vx1, vy1, vx2, vy2), vehicle_type = vehicle_box
            vehicle_crop = frame[vy1:vy2, vx1:vx2]
            plate_in_vehicle = (x1 - vx1, y1 - vy1, x2 - vx1, y2 - vy1)
        color = classify_color(vehicle_crop, exclude_box=plate_in_vehicle)

        was_logged_before = track.logged
        track.plate_text = voted_text
        track.confidence = voted_conf
        track.color = color
        track.vehicle_type = vehicle_type
        track.logged = True

        self._on_new_read(
            self, track_id, track, (x1, y1, x2, y2), crop, voted_text, voted_conf, color,
            vehicle_type, was_logged_before,
        )

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
