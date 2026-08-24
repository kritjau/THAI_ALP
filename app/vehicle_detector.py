from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from .config import settings

# Stock COCO-pretrained detector (no fine-tuning) -- used only to find the
# extent of the whole vehicle body for color sampling, not to identify type
# or brand. COCO class ids: car=2, motorcycle=3, bus=5, truck=7.
_VEHICLE_CLASS_IDS = [2, 3, 5, 7]
_WEIGHTS_NAME = "yolo11n.pt"


class VehicleDetector:
    def __init__(self, conf_threshold: float = 0.35):
        self.conf_threshold = conf_threshold
        weights_path = Path(settings.models_dir) / _WEIGHTS_NAME
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(weights_path))  # ultralytics auto-downloads a recognized name if missing

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            classes=_VEHICLE_CLASS_IDS,
            device=settings.device,
            verbose=False,
        )[0]
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return boxes

    @staticmethod
    def find_containing(
        vehicle_boxes: list[tuple[int, int, int, int]], plate_box: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int] | None:
        """The smallest vehicle box whose area contains the plate's center --
        the vehicle this plate actually belongs to, not just the nearest or
        largest vehicle detected anywhere in frame."""
        px = (plate_box[0] + plate_box[2]) / 2
        py = (plate_box[1] + plate_box[3]) / 2
        best, best_area = None, float("inf")
        for (x1, y1, x2, y2) in vehicle_boxes:
            if x1 <= px <= x2 and y1 <= py <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area < best_area:
                    best, best_area = (x1, y1, x2, y2), area
        return best
