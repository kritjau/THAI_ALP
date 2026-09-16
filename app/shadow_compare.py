from __future__ import annotations

"""Temporary A/B evaluation harness -- gathers evidence for the AGPL-vs-Apache
plate-detector decision (see shadow_detector/'s module docstring) by sending
a sample of frames the production detector already ran on to a second,
Apache-2.0-licensed candidate (shadow_detector/) and logging how their boxes
compare. Delete this module, shadow_detector/, and unset SHADOW_DETECTOR_URL
once that decision is made -- none of this is meant to be a permanent
second detector running alongside the real one.

Deliberately samples on a per-camera timer (sample_interval) rather than
every processed frame: at production's throughput, sending every frame to a
second detector would put real sustained load on whatever GPU it runs on,
and risk the same kind of unbounded backlog this project already hit once
(see ocr_service's OOM writeup) if that GPU or service ever falls behind.
A bounded queue (maxsize=4, dropping new samples instead of blocking when
full) and a short request timeout mean a slow/stuck shadow service only
ever loses samples -- it can never add latency to the real camera loop.

Logs structured comparison records only (box coordinates, confidence
scores, computed IoU) -- never the frame or crop itself, so this doesn't
accumulate images of real vehicles/plates/pedestrians on disk, consistent
with the rest of this project's no-image-persistence-by-default posture.
The log file is plain, append-only JSONL with no rotation -- it's sized for
a short (day-or-two) trial (see the module-level comment in
shadow_detector/main.py), not indefinite operation; delete it once the
trial's summarized.
"""

import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# A shadow detection only "counts" as matching a production box once they
# overlap this much -- loose enough that a slightly different box size for
# the same real plate still counts as a match, tight enough that two boxes
# on genuinely different objects (see the pedestrian false-positive found
# during the initial benchmark) don't get paired up by coincidence.
_MATCH_IOU_THRESHOLD = 0.3


def _iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


class ShadowComparator:
    def __init__(self, url: str, sample_interval: float, log_path: str, timeout: float):
        self._url = url.rstrip("/")
        self._sample_interval = sample_interval
        self._timeout = timeout
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_lock = threading.Lock()
        self._last_sample_at: dict[str, float] = {}
        self._queue: queue.Queue = queue.Queue(maxsize=4)
        threading.Thread(target=self._worker, daemon=True, name="shadow-compare").start()
        logger.info(
            "Shadow detector comparison enabled -- sampling every ~%.0fs per camera, logging to %s",
            sample_interval, self._log_path,
        )

    def maybe_submit(
        self, camera_id: str, camera_name: str, frame: np.ndarray, production_boxes: list[tuple]
    ) -> None:
        """Call after the production detector has already run on `frame` --
        throttled per-camera, so safe to call on every processed frame
        regardless of the configured interval."""
        now = time.time()
        if now - self._last_sample_at.get(camera_id, 0.0) < self._sample_interval:
            return
        self._last_sample_at[camera_id] = now
        try:
            # frame.copy(): the caller's frame is about to be reused/drawn
            # on by the video loop; this queue can outlive that.
            self._queue.put_nowait((camera_id, camera_name, frame.copy(), list(production_boxes), now))
        except queue.Full:
            pass  # a slow shadow service loses samples, never backs up the real pipeline

    def _worker(self):
        while True:
            camera_id, camera_name, frame, production_boxes, ts = self._queue.get()
            try:
                self._compare_one(camera_id, camera_name, frame, production_boxes, ts)
            except Exception:
                logger.exception("Shadow comparison failed")

    def _compare_one(self, camera_id, camera_name, frame, production_boxes, ts):
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return
        req = urllib.request.Request(
            f"{self._url}/detect", data=buf.tobytes(), method="POST",
            headers={"Content-Type": "application/octet-stream"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, ValueError):
            return  # shadow service down/slow -- skip this sample, not an error

        shadow_boxes = [(tuple(b["box"]), b["score"]) for b in data.get("boxes", [])]
        remaining_production = list(production_boxes)
        matched, shadow_only = [], []

        for sbox, sscore in shadow_boxes:
            best_iou, best_idx = 0.0, -1
            for i, pbox in enumerate(remaining_production):
                iou = _iou(sbox, pbox[:4])
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_iou >= _MATCH_IOU_THRESHOLD:
                pbox = remaining_production.pop(best_idx)
                matched.append({
                    "production_box": list(pbox[:4]),
                    "production_conf": round(float(pbox[4]), 3),
                    "shadow_box": list(sbox),
                    "shadow_conf": sscore,
                    "iou": round(best_iou, 3),
                })
            else:
                # No production box overlapped this one -- either a shadow
                # false positive, or a real plate production missed. This
                # log can't tell those apart automatically (see the
                # pedestrian false-positive found by eye during the initial
                # benchmark) -- that's exactly why disagreements need a
                # human look, not just this count.
                shadow_only.append({"box": list(sbox), "score": sscore})

        record = {
            "ts": ts,
            "camera_id": camera_id,
            "camera_name": camera_name,
            "matched": matched,
            # Boxes production found that no shadow box overlapped --
            # i.e. shadow missed something production caught.
            "production_only": [
                {"box": list(b[:4]), "conf": round(float(b[4]), 3)} for b in remaining_production
            ],
            "shadow_only": shadow_only,
        }
        with self._log_lock:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
