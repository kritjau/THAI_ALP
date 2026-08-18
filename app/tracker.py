from __future__ import annotations

import time
from dataclasses import dataclass, field


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


@dataclass
class Track:
    box: tuple
    plate_text: str = ""
    confidence: float = 0.0
    last_ocr: float = 0.0
    logged: bool = False
    last_seen: float = field(default_factory=time.time)


class PlateTracker:
    """Matches detections to persisting plates by box overlap so a plate sitting
    in frame across many processed frames is OCR'd (and logged) once, not repeatedly."""

    def __init__(self, iou_threshold: float, ttl_seconds: float, reocr_seconds: float):
        self.iou_threshold = iou_threshold
        self.ttl_seconds = ttl_seconds
        self.reocr_seconds = reocr_seconds
        self.tracks: list[Track] = []

    def match(self, box) -> Track:
        now = time.time()
        self.tracks = [t for t in self.tracks if now - t.last_seen <= self.ttl_seconds]

        best, best_iou = None, 0.0
        for t in self.tracks:
            score = _iou(t.box, box)
            if score > best_iou:
                best, best_iou = t, score

        if best is not None and best_iou >= self.iou_threshold:
            best.box = box
            best.last_seen = now
            return best

        track = Track(box=box)
        self.tracks.append(track)
        return track

    def needs_ocr(self, track: Track) -> bool:
        return not track.logged or (time.time() - track.last_ocr) >= self.reocr_seconds
