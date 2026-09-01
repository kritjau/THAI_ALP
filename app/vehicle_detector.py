from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

from .config import settings

# Stock COCO-pretrained detector (no fine-tuning) -- originally used only to
# find the extent of the whole vehicle body for color sampling; the class id
# COCO already gives us for free is now also kept to label the vehicle type
# (car/motorcycle/bus/truck), not just its box.
_VEHICLE_CLASS_LABELS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
_VEHICLE_CLASS_IDS = list(_VEHICLE_CLASS_LABELS)
_WEIGHTS_NAME = "yolo11n.pt"

VehicleBox = tuple[tuple[int, int, int, int], str]  # (box, vehicle_type)


class VehicleDetector:
    def __init__(self, conf_threshold: float = 0.35):
        self.conf_threshold = conf_threshold
        weights_path = Path(settings.models_dir) / _WEIGHTS_NAME
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = YOLO(str(weights_path))  # ultralytics auto-downloads a recognized name if missing

    def detect(self, frame: np.ndarray) -> list[VehicleBox]:
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            classes=_VEHICLE_CLASS_IDS,
            device=settings.device,
            verbose=False,
        )[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            vehicle_type = _VEHICLE_CLASS_LABELS.get(int(box.cls[0]), "vehicle")
            detections.append(((int(x1), int(y1), int(x2), int(y2)), vehicle_type))
        return detections

    @staticmethod
    def find_containing(
        vehicle_boxes: list[VehicleBox], plate_box: tuple[int, int, int, int]
    ) -> VehicleBox | None:
        """The smallest vehicle box whose area contains the plate's center --
        the vehicle this plate actually belongs to, not just the nearest or
        largest vehicle detected anywhere in frame."""
        px = (plate_box[0] + plate_box[2]) / 2
        py = (plate_box[1] + plate_box[3]) / 2
        best, best_area = None, float("inf")
        for box, vehicle_type in vehicle_boxes:
            x1, y1, x2, y2 = box
            if x1 <= px <= x2 and y1 <= py <= y2:
                area = (x2 - x1) * (y2 - y1)
                if area < best_area:
                    best, best_area = (box, vehicle_type), area
        return best
