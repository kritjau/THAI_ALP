from __future__ import annotations

import base64
import os
import queue
import time
from collections import defaultdict, deque

import cv2

from app import registered_plates_db
from app.camera_worker import CameraWorker, build_camera_workers, per_camera_rejection_stats
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
    and app/gate.py.

    A camera marked CAMERA_PAGE_N=admin is the "gate camera": it does gate
    access control *only* (whitelist check -> open gate -> log it) and is
    kept out of the monitoring stream entirely (no Detections rows, no
    vehicle-type stats -- those are about detection quality on the lot, not
    who's at the entrance). The monitoring cameras conversely stop
    triggering the gate once a gate camera exists. If no camera is marked
    admin, every camera both monitors and triggers the gate as before, so a
    one-camera setup still works.

    Each CameraWorker drives its own capture/detect/draw loop on its own
    thread (see CameraWorker._loop) rather than being stepped from here --
    step() below just drains whatever events those threads produced."""

    def __init__(self):
        self._new_events: queue.Queue = queue.Queue()
        # (camera_id, plate_text) -> last time it was seen. Only ever touched
        # by the OCR worker threads below (one per camera, but each key is
        # only ever written by its own camera's thread), so no lock needed.
        self._recent_plates: dict[tuple[str, str], float] = {}
        # Aggregate counts only (no plate text), so this doesn't conflict
        # with the no-storage design -- resets to zero on restart same as
        # everything else here, since nothing here is meant to persist.
        self._type_counts: dict[str, int] = defaultdict(int)
        # Same aggregate-only counts, bucketed by local calendar day, for the
        # Vehicle Types trend chart -- still no plate text, still gone on
        # restart. _prune_daily_counts() keeps this from growing unbounded
        # across a long uptime.
        self._type_counts_daily: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Recent gate opens (registered plate matched at the gate camera) --
        # in-memory, capped, gone on restart, same as everything else here.
        # Only whitelisted plates land here, and their owners opted in by
        # being on the list; passing traffic the gate camera sees but
        # doesn't open for is never recorded.
        self._gate_events: deque = deque(maxlen=50)
        self.cameras = build_camera_workers(settings.camera_configs(), self._on_new_read)
        self._cameras_by_id = {cam.camera_id: cam for cam in self.cameras}
        # Once any camera is the designated gate camera, it's the *only*
        # thing that opens the gate; without one, every camera does (a
        # single-camera setup).
        self._has_gate_camera = any(cam.page == "admin" for cam in self.cameras)

    def step(self) -> list[dict]:
        return self._drain_events()

    def _on_new_read(
        self, worker, track_id, track, box, crop, text, ocr_conf, color, vehicle_type, was_logged_before
    ):
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
        genuinely_new = is_new_sighting and not is_recent_duplicate

        if worker.page == "admin":
            # The gate camera: access control only. Nothing it sees reaches
            # the monitoring stream.
            self._handle_gate_read(worker, track, text, now, genuinely_new)
            return

        # A monitoring camera. It only opens the gate itself when there's no
        # dedicated gate camera to do it (see _has_gate_camera) -- otherwise
        # registration is not its concern and it shows no "GATE" badge.
        registered = False
        if not self._has_gate_camera:
            registered = registered_plates_db.is_registered_plate(text)
            if registered and not track.gate_opened:
                open_gate(text)
                track.gate_opened = True

        # Same "genuinely new, not a cooldown-deduped re-appearance"
        # condition the event emission below uses -- counting anything
        # looser would double-count a car re-OCR'd or briefly re-tracked.
        if genuinely_new and vehicle_type:
            self._type_counts[vehicle_type] += 1
            day = time.strftime("%Y-%m-%d", time.localtime(now))
            self._type_counts_daily[day][vehicle_type] += 1
            self._prune_daily_counts()

        if not is_recent_duplicate:
            self._new_events.put(
                {
                    "id": track_id,
                    "camera_id": worker.camera_id,
                    "camera_name": worker.name,
                    "plate_text": text,
                    "confidence": ocr_conf,
                    "color": color,
                    "vehicle_type": vehicle_type,
                    "bbox": list(box),
                    "timestamp": now,
                    "image": _as_data_uri(crop),
                    "registered": registered,
                }
            )

    def _handle_gate_read(self, worker, track, text, now, genuinely_new):
        if track.gate_opened:
            return  # already let this vehicle through
        match = registered_plates_db.lookup(text)
        if match is None:
            return  # not on the whitelist -- gate stays shut, nothing logged
        open_gate(text)
        track.gate_opened = True
        if genuinely_new:
            self._gate_events.appendleft(
                {
                    "plate_text": match["plate_text"],
                    "label": match["label"],
                    "camera_name": worker.name,
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
        stale = [key for key, seen in self._recent_plates.items() if seen < cutoff]
        for key in stale:
            del self._recent_plates[key]

    # A little more than the 7 days the trend chart actually shows, so a
    # viewer in a timezone slightly ahead of the server's never sees a day
    # drop off the chart's window a beat early.
    _DAILY_COUNTS_MAX_DAYS = 14

    def _prune_daily_counts(self):
        cutoff = time.strftime(
            "%Y-%m-%d", time.localtime(time.time() - self._DAILY_COUNTS_MAX_DAYS * 86400)
        )
        stale = [day for day in self._type_counts_daily if day < cutoff]
        for day in stale:
            del self._type_counts_daily[day]

    def camera_list(self) -> list[dict]:
        return [{"id": cam.camera_id, "name": cam.name, "page": cam.page} for cam in self.cameras]

    def type_counts(self) -> dict[str, int]:
        return dict(self._type_counts)

    def type_counts_daily(self, days: int = 7) -> dict[str, dict[str, int]]:
        """Day (YYYY-MM-DD, local time) -> vehicle_type -> count, for the
        last `days` calendar days including today -- powers the Vehicle
        Types trend chart."""
        cutoff = time.strftime("%Y-%m-%d", time.localtime(time.time() - (days - 1) * 86400))
        return {day: dict(counts) for day, counts in self._type_counts_daily.items() if day >= cutoff}

    def rejection_stats(self) -> list[dict]:
        return per_camera_rejection_stats(self.cameras)

    def gate_events(self) -> list[dict]:
        """Recent gate opens (registered plate matched at the gate camera),
        newest first -- powers the Gate Activity list on /admin."""
        return list(self._gate_events)

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
