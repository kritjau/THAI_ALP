from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraStream:
    """Continuously reads frames from a webcam/RTSP source in a background thread
    so slow downstream processing never blocks capture and the video feed stays live."""

    def __init__(self, source):
        self.source = source
        self._cap = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> "CameraStream":
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {self.source!r}")
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                logger.warning("Failed to read frame from %r, retrying...", self.source)
                time.sleep(0.5)
                continue
            with self._lock:
                self._frame = frame

    def read(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
