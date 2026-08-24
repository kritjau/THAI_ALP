from __future__ import annotations

import logging
import os
import threading
import time

import cv2
import numpy as np

from .config import settings

logger = logging.getLogger(__name__)

# RTSP over UDP (FFmpeg's default) drops packets on real networks, which corrupts
# H.264/HEVC frames into visible block/tile artifacts; TCP avoids that. Only takes
# effect if the app hasn't already had this env var set to something else.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


def _call_with_timeout(fn, timeout_s: float):
    """Run a blocking call with a wall-clock timeout. Needed because a stalled
    network path (e.g. the TCP connect() itself never returning) can make
    cv2.VideoCapture hang indefinitely -- neither CAP_PROP_OPEN_TIMEOUT_MSEC/
    CAP_PROP_READ_TIMEOUT_MSEC nor FFmpeg's own `stimeout` option catch that,
    since it happens below both OpenCV's and FFmpeg's own timeout logic. On a
    timeout, the underlying call is abandoned running on a daemon thread (it
    will eventually return on its own and get garbage collected) so the caller
    can retry immediately instead of blocking for however long it takes."""
    result: dict = {}
    done = threading.Event()

    def run():
        try:
            result["value"] = fn()
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller below
            result["error"] = exc
        finally:
            done.set()

    threading.Thread(target=run, daemon=True).start()
    if not done.wait(timeout=timeout_s):
        return None, False
    if "error" in result:
        raise result["error"]
    return result.get("value"), True


class CameraStream:
    """Continuously reads frames from a webcam/RTSP source in a background thread
    so slow downstream processing never blocks capture and the video feed stays live.

    An RTSP source (flaky network, overloaded NVR) can stall indefinitely with no
    error -- bounded open/read timeouts turn that into a fast failure, and the
    read loop auto-reconnects (with a backoff retry) instead of hanging forever."""

    def __init__(self, source):
        self.source = source
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> "CameraStream":
        self._open()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _open(self) -> cv2.VideoCapture:
        def attempt():
            cap = cv2.VideoCapture()
            # Forcing FFmpeg only makes sense for URL/file sources; a local
            # webcam index needs the platform's native backend (e.g. V4L2).
            if isinstance(self.source, str):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.open(self.source, cv2.CAP_FFMPEG)
            else:
                cap.open(self.source)
            return cap

        cap, finished = _call_with_timeout(attempt, settings.camera_open_timeout_ms / 1000)
        if not finished or cap is None or not cap.isOpened():
            if cap:
                cap.release()
            raise RuntimeError(f"Could not open camera source: {self.source!r}")
        self._cap = cap
        return cap

    def _loop(self):
        while self._running:
            frame = None
            if self._cap is not None:
                try:
                    result, finished = _call_with_timeout(
                        self._cap.read, settings.camera_read_timeout_ms / 1000
                    )
                    # On a timeout `result` is None (not a (ok, frame) pair) --
                    # only unpack it once a read has actually finished.
                    if finished and result is not None:
                        ok, frame = result
                        frame = frame if ok else None
                except Exception:
                    # A read failing outright (not just timing out) shouldn't
                    # kill this thread either -- fall through to reconnect.
                    logger.exception("Camera read failed unexpectedly from %r", self.source)
            if frame is None:
                logger.warning("Lost camera stream from %r, reconnecting...", self.source)
                self._reconnect()
                continue
            with self._lock:
                self._frame = frame

    def _reconnect(self):
        if self._cap:
            self._cap.release()
            self._cap = None
        while self._running:
            try:
                self._open()
                logger.info("Reconnected to camera source: %r", self.source)
                return
            except RuntimeError:
                time.sleep(settings.camera_reconnect_delay_seconds)

    def read(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
