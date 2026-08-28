from __future__ import annotations

import logging
import re

import cv2
import numpy as np
from paddleocr import PaddleOCR

from .config import settings
from .thai_provinces import closest_province

logger = logging.getLogger(__name__)

# Keep Thai script, Latin letters/digits (province names are occasionally
# transliterated) and basic separators; drop OCR noise from the plate border.
_ALLOWED_CHARS = re.compile(r"[^฀-๿A-Za-z0-9\- ]+")
_THAI_CHAR = re.compile(r"[ก-๙]")
_PLATE_SERIAL = re.compile(r"\d{3,4}")


def looks_like_thai_plate(text: str) -> bool:
    """Every real Thai plate carries Thai script (number line and/or province
    line), so a read with none is almost always a false-positive detection
    (signage, clothing text, etc.) rather than a plate."""
    return bool(_THAI_CHAR.search(text))


def _paddle_device(device: str) -> str:
    return device.replace("cuda", "gpu") if device.startswith("cuda") else device


def _group_into_lines(segments) -> list[list[tuple]]:
    """Group boxes (x1,y1,x2,y2) into text lines by y-center proximity (small
    per-box y jitter otherwise splits same-line segments apart), each line
    sorted left-to-right, lines ordered top-to-bottom."""
    def y_center(box):
        return (box[1] + box[3]) / 2

    def height(box):
        return box[3] - box[1]

    ordered = sorted(segments, key=lambda s: y_center(s[0]))
    lines: list[dict] = []
    for seg in ordered:
        yc = y_center(seg[0])
        if lines and abs(yc - lines[-1]["y"]) < height(seg[0]) * 0.6:
            lines[-1]["segs"].append(seg)
        else:
            lines.append({"y": yc, "segs": [seg]})

    return [sorted(line["segs"], key=lambda s: s[0][0]) for line in lines]


class PlateReader:
    def __init__(self, lang: str | None = None, device: str | None = None):
        self.reader = PaddleOCR(
            lang=lang or settings.ocr_lang,
            device=_paddle_device(device or settings.ocr_device),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # The default oneDNN CPU backend crashes (NotImplementedError) on
            # some detection-model ops; the plain CPU kernel is a safe fallback.
            enable_mkldnn=False,
        )

    def read(self, plate_crop: np.ndarray) -> tuple[str, float]:
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0

        crop = plate_crop
        if settings.ocr_deskew_enabled:
            crop = self._deskew(crop)
        result = self.reader.predict(self._upscale(crop))
        if not result:
            return "", 0.0
        page = result[0]
        texts, scores, boxes = page["rec_texts"], page["rec_scores"], page["rec_boxes"]
        if not texts:
            return "", 0.0

        # A car plate is the number line (top) then the province name below it;
        # a motorcycle plate adds a third line below *that* with the rest of the
        # serial number, often in a stylized font that OCR can misread entirely.
        # A plate frame/holder with printed dealer or district branding often
        # sits right below the real content too and gets picked up as another
        # "line". The number line is kept as read; a subsequent line is only
        # kept if it fuzzy-matches a real Thai province (corrected to the
        # canonical spelling, and only once) -- filtering out branding text a
        # plain line-count cutoff can't when it's packed tight against the real
        # province line -- or if its digit-only segments form a plausible 3-4
        # digit plate serial (a motorcycle's trailing number, possibly split
        # across boxes as e.g. "3" + "133"; unlike branding text, which always
        # has letters/punctuation in every case seen in practice). A line that
        # matches neither is skipped rather than stopping the scan outright --
        # the small province line is often too garbled to fuzzy-match even when
        # a clean serial line follows it -- but once both a province and a
        # serial have been found, anything further is assumed to be trailing
        # branding and stops the scan.
        lines = _group_into_lines(zip(boxes, texts, scores))

        kept_texts, kept_scores = [], []
        province_matched = False
        serial_found = False
        give_up = False
        for line_idx, line in enumerate(lines):
            valid = [
                (_ALLOWED_CHARS.sub("", text).strip(), float(score))
                for _box, text, score in line
                if score >= settings.ocr_min_segment_confidence
            ]
            valid = [(t, s) for t, s in valid if t]
            if not valid:
                continue
            avg_score = sum(s for _, s in valid) / len(valid)

            if line_idx == 0:
                kept_texts.append(" ".join(t for t, _ in valid))
                kept_scores.append(avg_score)
                continue
            if give_up:
                continue

            digits_only = "".join(t for t, _ in valid if t.isdigit())
            if not serial_found and _PLATE_SERIAL.fullmatch(digits_only):
                kept_texts.append(digits_only)
                kept_scores.append(avg_score)
                serial_found = True
                continue

            if not province_matched:
                province = closest_province(" ".join(t for t, _ in valid))
                if province:
                    kept_texts.append(province)
                    kept_scores.append(avg_score)
                    province_matched = True
                    continue

            if province_matched or serial_found:
                give_up = True

        if not kept_texts:
            return "", 0.0
        return " ".join(kept_texts), sum(kept_scores) / len(kept_scores)

    @staticmethod
    def _upscale(crop: np.ndarray) -> np.ndarray:
        target_h = settings.ocr_upscale_height
        h, w = crop.shape[:2]
        if h >= target_h or h == 0:
            return crop
        scale = target_h / h
        return cv2.resize(crop, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _deskew(crop: np.ndarray) -> np.ndarray:
        """Corrects in-plane rotation (a tilted camera, or a crookedly mounted
        plate) before OCR -- not full perspective/keystone distortion, which
        would need locating all 4 corners of the plate and a proper
        homography warp; a tight detector crop with no clean background makes
        that unreliable. Plain rotation covers the far more common real-world
        case on CCTV footage, at much lower risk of making a read worse.

        Estimates the tilt from the dominant near-horizontal line segments in
        the crop (the plate's own top/bottom border, or the text baseline)
        via a Hough transform, and falls back to the untouched crop if no
        reliable line is found rather than guessing at an angle."""
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return crop

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=max(20, w // 6),
            minLineLength=w * 0.4, maxLineGap=w * 0.1,
        )
        if lines is None:
            return crop

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            if dx == 0:
                continue
            angle = np.degrees(np.arctan2(dy, dx))
            # Only near-horizontal lines are plausibly the plate's own
            # edges/text baseline -- a near-vertical line is noise (a
            # character stroke, a mounting screw, a border seam).
            if abs(angle) <= 20:
                angles.append(angle)

        if not angles:
            return crop

        tilt = float(np.median(angles))
        if abs(tilt) < 1.0:
            return crop  # not worth a resample for a barely-there tilt

        center = (w / 2, h / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, tilt, 1.0)
        # Expand the output canvas so rotating doesn't clip the plate's
        # corners -- the crop is already tight, with little margin to spare.
        cos, sin = abs(rotation_matrix[0, 0]), abs(rotation_matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2
        return cv2.warpAffine(
            crop, rotation_matrix, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE
        )
