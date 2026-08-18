from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass
class Settings:
    camera_source: str = os.environ.get("CAMERA_SOURCE", "0")
    device: str = os.environ.get("DEVICE", "cpu")  # "cpu" or e.g. "cuda:0"

    detector_model_size: str = os.environ.get("DETECTOR_MODEL_SIZE", "n")  # n, s, m, l, x
    detector_conf_threshold: float = _env_float("DETECTOR_CONF_THRESHOLD", 0.4)

    ocr_langs: list = field(
        default_factory=lambda: os.environ.get("OCR_LANGS", "th,en").split(",")
    )

    process_every_n_frames: int = _env_int("PROCESS_EVERY_N_FRAMES", 3)
    track_iou_threshold: float = _env_float("TRACK_IOU_THRESHOLD", 0.3)
    track_ttl_seconds: float = _env_float("TRACK_TTL_SECONDS", 5.0)
    track_reocr_seconds: float = _env_float("TRACK_REOCR_SECONDS", 3.0)

    models_dir: str = os.environ.get("MODELS_DIR", "models")
    captures_dir: str = os.environ.get("CAPTURES_DIR", "captures")
    db_path: str = os.environ.get("DB_PATH", "data/alpr.db")
    jpeg_quality: int = _env_int("JPEG_QUALITY", 80)

    def camera_source_value(self):
        """Local webcams are given as an integer index; RTSP/HTTP/file sources stay strings."""
        try:
            return int(self.camera_source)
        except ValueError:
            return self.camera_source


settings = Settings()
