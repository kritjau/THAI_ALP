from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Track:
    box: tuple
    plate_text: str = ""
    confidence: float = 0.0
    last_ocr: float = 0.0
    logged: bool = False
    db_id: int | None = None
    image_path: str | None = None
    last_seen: float = field(default_factory=time.time)


class PlateTracker:
    """Per-plate bookkeeping (best OCR read so far, its DB row, its saved crop)
    keyed by the persistent track ID ByteTrack (via the detector's built-in
    multi-object tracking) assigns to each plate across frames -- so a plate is
    only OCR'd/logged once while it's in view, not on every processed frame."""

    def __init__(self, ttl_seconds: float, reocr_seconds: float):
        self.ttl_seconds = ttl_seconds
        self.reocr_seconds = reocr_seconds
        self.tracks: dict[int, Track] = {}

    def get(self, track_id: int, box: tuple) -> Track:
        now = time.time()
        track = self.tracks.get(track_id)
        if track is None:
            track = Track(box=box, last_seen=now)
            self.tracks[track_id] = track
        else:
            track.box = box
            track.last_seen = now
        self._prune(now)
        return track

    def _prune(self, now: float):
        stale = [tid for tid, t in self.tracks.items() if now - t.last_seen > self.ttl_seconds]
        for tid in stale:
            del self.tracks[tid]

    def needs_ocr(self, track: Track) -> bool:
        return not track.logged or (time.time() - track.last_ocr) >= self.reocr_seconds
