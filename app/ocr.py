from __future__ import annotations

import logging
import re

import easyocr
import numpy as np

from .config import settings

logger = logging.getLogger(__name__)

# Keep Thai script, Latin letters/digits (province names are occasionally
# transliterated) and basic separators; drop OCR noise from the plate border.
_ALLOWED_CHARS = re.compile(r"[^฀-๿A-Za-z0-9\- ]+")


class PlateReader:
    def __init__(self, langs: list[str] | None = None, gpu: bool | None = None):
        use_gpu = gpu if gpu is not None else settings.device != "cpu"
        self.reader = easyocr.Reader(langs or settings.ocr_langs, gpu=use_gpu)

    def read(self, plate_crop: np.ndarray) -> tuple[str, float]:
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0
        results = self.reader.readtext(plate_crop)
        if not results:
            return "", 0.0

        # Thai plates are two lines (plate number, then province name below it);
        # sort top-to-bottom then left-to-right so the joined text reads naturally.
        results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))

        texts, confs = [], []
        for _, text, conf in results:
            cleaned = _ALLOWED_CHARS.sub("", text).strip()
            if cleaned:
                texts.append(cleaned)
                confs.append(conf)

        if not texts:
            return "", 0.0
        return " ".join(texts), sum(confs) / len(confs)
