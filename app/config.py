from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass
class Settings:
    camera_source: str = os.environ.get("CAMERA_SOURCE", "0")
    device: str = os.environ.get("DEVICE", "cpu")  # detector (torch/ultralytics): "cpu" or e.g. "cuda:0"
    # paddlepaddle-gpu and torch's CUDA build install colliding versions of the
    # same `nvidia.*` shared libraries into one venv (confirmed: it breaks torch's
    # CUDA import outright) -- so PaddleOCR stays CPU-only by default even when
    # the detector runs on GPU. Only change this if paddlepaddle-gpu is installed
    # in a way that's actually verified compatible with the installed torch build.
    ocr_device: str = os.environ.get("OCR_DEVICE", "cpu")

    # An RTSP source that stalls (network drop, overloaded NVR) would otherwise
    # hang cv2.VideoCapture indefinitely; these bound how long it can block
    # before CameraStream treats it as failed and reconnects.
    camera_open_timeout_ms: int = _env_int("CAMERA_OPEN_TIMEOUT_MS", 8000)
    camera_read_timeout_ms: int = _env_int("CAMERA_READ_TIMEOUT_MS", 8000)
    camera_reconnect_delay_seconds: float = _env_float("CAMERA_RECONNECT_DELAY_SECONDS", 2.0)

    detector_model_size: str = os.environ.get("DETECTOR_MODEL_SIZE", "n")  # n, s, m, l, x
    detector_conf_threshold: float = _env_float("DETECTOR_CONF_THRESHOLD", 0.4)

    ocr_lang: str = os.environ.get("OCR_LANG", "th")

    process_every_n_frames: int = _env_int("PROCESS_EVERY_N_FRAMES", 3)
    track_ttl_seconds: float = _env_float("TRACK_TTL_SECONDS", 5.0)
    track_reocr_seconds: float = _env_float("TRACK_REOCR_SECONDS", 3.0)

    # Reads below this are noise (motion blur, false-positive detections) and are discarded.
    min_log_confidence: float = _env_float("MIN_LOG_CONFIDENCE", 0.5)
    # Individual OCR text segments (e.g. a garbled province line) below this are
    # dropped before the rest are joined, instead of dragging down the whole read.
    ocr_min_segment_confidence: float = _env_float("OCR_MIN_SEGMENT_CONFIDENCE", 0.5)
    # Plate crops are small at driving distance; upscale before OCR so small Thai
    # glyphs (esp. the province line) have enough pixels to be read reliably.
    ocr_upscale_height: int = _env_int("OCR_UPSCALE_HEIGHT", 200)

    models_dir: str = os.environ.get("MODELS_DIR", "models")
    captures_dir: str = os.environ.get("CAPTURES_DIR", "captures")
    db_path: str = os.environ.get("DB_PATH", "data/alpr.db")
    jpeg_quality: int = _env_int("JPEG_QUALITY", 80)
    # Detection/OCR always run on the full-resolution frame (accuracy matters
    # there); the *displayed* stream is downscaled to this width before JPEG
    # encoding -- a wide-FOV CCTV frame is much bigger than a dashboard needs,
    # and encoding/transferring/decoding it at full size is real, avoidable
    # cost that makes the video feel choppy independent of detection speed.
    stream_max_width: int = _env_int("STREAM_MAX_WIDTH", 1280)

    json_dir: str = os.environ.get("JSON_DIR", "json")
    json_export_interval_seconds: float = _env_float("JSON_EXPORT_INTERVAL_SECONDS", 20.0)

    # The ESP32 gate-relay board (firmware/gate_relay/) is reached over a USB
    # serial tether, not WiFi -- see app/gate.py. Left blank, a registered-
    # plate match still logs/shows on the dashboard, it just doesn't send a
    # signal anywhere.
    gate_serial_port: str = os.environ.get("GATE_SERIAL_PORT", "")
    gate_serial_baud: int = _env_int("GATE_SERIAL_BAUD", 115200)
    gate_serial_timeout_seconds: float = _env_float("GATE_SERIAL_TIMEOUT_SECONDS", 2.0)
    # A small, manually-curated whitelist -- deliberately its own database,
    # separate from DB_PATH (which logs every detected plate for app/'s
    # parking-lot use case). Only app_live's gate-access check reads this.
    registered_plates_db_path: str = os.environ.get(
        "REGISTERED_PLATES_DB_PATH", "data/registered_plates.db"
    )

    def camera_source_value(self):
        """Local webcams are given as an integer index; RTSP/HTTP/file sources stay strings."""
        return self._resolve_source(self.camera_source)

    @staticmethod
    def _resolve_source(value: str):
        try:
            return int(value)
        except ValueError:
            return value

    def camera_configs(self) -> list[dict]:
        """CAMERA_SOURCE is camera "1"; CAMERA_SOURCE_2, CAMERA_SOURCE_3, ...
        add more (with an optional CAMERA_NAME_N label each) for testing
        against multiple feeds side by side -- numbered .env keys rather than
        one JSON/CSV value so adding a camera is just one more plain line.
        Re-reads os.environ each call (like camera_source_value() effectively
        does) rather than being fixed at import time, so tests can monkeypatch
        environ without needing a fresh Settings instance."""
        configs = [
            {
                "id": "1",
                "name": os.environ.get("CAMERA_NAME_1", "Camera 1"),
                "source": self.camera_source_value(),
            }
        ]
        i = 2
        while True:
            source = os.environ.get(f"CAMERA_SOURCE_{i}")
            if not source:
                break
            configs.append(
                {
                    "id": str(i),
                    "name": os.environ.get(f"CAMERA_NAME_{i}", f"Camera {i}"),
                    "source": self._resolve_source(source),
                }
            )
            i += 1
        return configs


settings = Settings()
