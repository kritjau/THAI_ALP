from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from .config import settings

logger = logging.getLogger(__name__)

# Pretrained YOLOv11 license-plate localizer (country-agnostic: it finds the
# rectangular plate region, it does not need to know Thai script).
# https://huggingface.co/morsetechlab/yolov11-license-plate-detection
_REPO_ID = "morsetechlab/yolov11-license-plate-detection"
_WEIGHT_NAMES = {
    "n": "license-plate-finetune-v1n.pt",
    "s": "license-plate-finetune-v1s.pt",
    "m": "license-plate-finetune-v1m.pt",
    "l": "license-plate-finetune-v1l.pt",
    "x": "license-plate-finetune-v1x.pt",
}


class PlateDetector:
    def __init__(self, model_size: str | None = None, conf_threshold: float | None = None):
        size = model_size or settings.detector_model_size
        if size not in _WEIGHT_NAMES:
            raise ValueError(
                f"Unknown detector model size '{size}', expected one of {list(_WEIGHT_NAMES)}"
            )
        self.conf_threshold = (
            conf_threshold if conf_threshold is not None else settings.detector_conf_threshold
        )
        weights_path = self._ensure_weights(size)
        self.model = YOLO(weights_path)

    @staticmethod
    def _ensure_weights(size: str) -> str:
        Path(settings.models_dir).mkdir(parents=True, exist_ok=True)
        filename = _WEIGHT_NAMES[size]
        local_path = Path(settings.models_dir) / filename
        if local_path.exists():
            return str(local_path)
        logger.info("Downloading license plate detector weights (%s) from %s", filename, _REPO_ID)
        return hf_hub_download(repo_id=_REPO_ID, filename=filename, local_dir=settings.models_dir)

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int, float]]:
        results = self.model.predict(
            frame, conf=self.conf_threshold, device=settings.device, verbose=False
        )[0]
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
        return boxes
