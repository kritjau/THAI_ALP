from __future__ import annotations

import colorsys

import cv2
import numpy as np


def _dominant_color(crop: np.ndarray, k: int = 3) -> tuple[int, int, int] | None:
    """K-means over the crop's pixels; returns the largest cluster's centroid
    as RGB. A vehicle crop's body paint dominates spatially throughout the
    box, so majority pixel count is a reasonable stand-in for "the body
    color"."""
    pixels = crop.reshape(-1, 3).astype(np.float32)
    if len(pixels) < k:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _compactness, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    b, g, r = centers[np.argmax(counts)]  # crops are BGR (OpenCV)
    return (int(r), int(g), int(b))


def _classify(rgb: tuple[int, int, int]) -> str:
    """Classifies by hue/saturation/value rather than nearest-neighbor in raw
    RGB. RGB-distance confuses brightness with actual color -- a red car
    sampled a bit dark or desaturated (shadow, overcast light, JPEG/video
    compression -- all normal on outdoor CCTV footage) can land numerically
    closer to a muted "brown" reference point than a saturated "red" one,
    even though it's clearly red to the eye. Hue is stable under exactly that
    kind of brightness variation, so classifying on it directly avoids the
    confusion; brown gets its own rule since it's fundamentally a dark,
    desaturated orange/red rather than a distinct hue of its own.
    """
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    h_deg = h * 360

    if s < 0.15:  # achromatic -- distinguish only by brightness
        if v > 0.85:
            return "white"
        if v > 0.55:
            return "silver"
        if v > 0.25:
            return "gray"
        return "black"

    if v < 0.45 and (h_deg < 40 or h_deg > 345):
        return "brown"
    if h_deg < 15 or h_deg > 345:
        return "red"
    if h_deg < 45:
        return "orange"
    if h_deg < 70:
        return "yellow"
    if h_deg < 170:
        return "green"
    if h_deg < 260:
        return "blue"
    return "red"  # purple/magenta/pink region -- closest bucket available


def _body_region(vehicle_crop: np.ndarray) -> np.ndarray:
    """Trims the vehicle detector's box down toward the body panels: the
    bottom ~20% is usually wheels/road/shadow and the top ~8% is often
    background right above the roofline, both of which skew the dominant
    color away from the actual paint."""
    h, w = vehicle_crop.shape[:2]
    top, bottom = int(h * 0.08), int(h * 0.80)
    if bottom <= top:
        return vehicle_crop
    return vehicle_crop[top:bottom, :]


def classify_color(vehicle_crop: np.ndarray | None) -> str:
    if vehicle_crop is None or vehicle_crop.size == 0:
        return "unknown"
    rgb = _dominant_color(_body_region(vehicle_crop)) or _dominant_color(vehicle_crop)
    return _classify(rgb) if rgb else "unknown"
