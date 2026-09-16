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
from app.plate_match import plates_are_similar

# ByteTrack can still fragment a car into a new track id after an occlusion or
# a gap between processed frames, which would otherwise show up as a second
# "new" detection for the same physical car. This cooldown dedupes by the OCR
# text itself (per camera -- the same plate on two different cameras is two
# genuine sightings, not a fragmented track), independent of track id, so a
# re-appearing plate within this window is treated as a continuation, not a
# new sighting. Local to app_live (not a shared app/ setting) -- read straight
# from the environment so .env doesn't need an app/config.py field for it.
#
# Exact text match alone still misses the case where the fragmented track's
# own vote lands on a plate text that's almost, but not quite, what the
# first track already reported (a one-character OCR slip) -- see
# _find_similar_recent() below, which extends this same cooldown window to
# catch that with a fuzzy (edit-distance) match instead of an exact one.
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
        # (camera_id, plate_text) -> {"confidence", "last_seen", "event_id"}
        # for whatever was most recently shown for that exact text -- used
        # both for the exact-match cooldown and, via _find_similar_recent(),
        # to catch a fragmented track's near-identical misread of the same
        # plate (see plate_match.plates_are_similar()). "event_id" is
        # whatever "id" value was sent to the dashboard for it, so a later,
        # better-confidence near-duplicate can correct that same row
        # instead of adding a new one beside it. Only ever touched by the
        # OCR worker threads below (one per camera, but each key is only
        # ever written by its own camera's thread), so no lock needed.
        self._recent_plates: dict[tuple[str, str], dict] = {}
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
        # display_id is what gets sent to the dashboard as this event's "id"
        # -- normally the track's own id, but overridden below when this
        # read turns out to be a better version of something already shown
        # under a *different* (now stale/fragmented) track's id, so the
        # frontend corrects that existing row instead of adding a new one.
        display_id = track_id
        # Same physical car as something just shown on this camera --
        # whether by exact text match or a fuzzy one -- so it should never
        # count as a second new vehicle, regardless of whether the read
        # itself is worth displaying (see suppress_event below).
        is_repeat = False
        suppress_event = False

        if is_new_sighting:
            cooldown_key = (worker.camera_id, text)
            exact = self._recent_plates.get(cooldown_key)
            if exact is not None and now - exact["last_seen"] < _PLATE_COOLDOWN_SECONDS:
                is_repeat = True
                suppress_event = True
            else:
                similar_key, similar = self._find_similar_recent(worker.camera_id, text, now)
                if similar is not None:
                    is_repeat = True
                    if ocr_conf > similar["confidence"]:
                        # A better read of the same car a fragmented track
                        # already reported -- correct that row rather than
                        # add a near-duplicate-looking one beside it.
                        display_id = similar["event_id"]
                        del self._recent_plates[similar_key]
                    else:
                        suppress_event = True

            self._recent_plates[(worker.camera_id, text)] = {
                "confidence": ocr_conf, "last_seen": now, "event_id": display_id,
            }
            self._prune_recent_plates()

        genuinely_new = is_new_sighting and not is_repeat

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

        if not suppress_event:
            self._new_events.put(
                {
                    "id": display_id,
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

    def _find_similar_recent(self, camera_id: str, text: str, now: float):
        """The most confident recently-shown plate on this camera that reads
        as the same physical car as `text` (see plate_match.plates_are_similar)
        but isn't an exact string match -- i.e. a fragmented track's own
        near-identical misread of a plate another track already reported.
        Scoped to one camera and the same cooldown window an exact match
        already uses, so this never merges two genuinely different plates
        that simply happen to look alike -- that would need them to also
        appear on the same camera within seconds of each other."""
        best_key, best_entry = None, None
        for (cam_id, other_text), entry in self._recent_plates.items():
            if cam_id != camera_id or other_text == text:
                continue
            if now - entry["last_seen"] >= _PLATE_COOLDOWN_SECONDS:
                continue
            if not plates_are_similar(text, other_text):
                continue
            if best_entry is None or entry["confidence"] > best_entry["confidence"]:
                best_key, best_entry = (cam_id, other_text), entry
        return best_key, best_entry

    def _prune_recent_plates(self):
        cutoff = time.time() - _PLATE_COOLDOWN_SECONDS
        stale = [key for key, entry in self._recent_plates.items() if entry["last_seen"] < cutoff]
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
