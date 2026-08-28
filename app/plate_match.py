from __future__ import annotations

import re

from .thai_provinces import THAI_PROVINCES

_WHITESPACE = re.compile(r"\s+")


def normalize_plate(plate_text: str) -> str:
    """Reduces an OCR'd read (and a stored registered plate) to a comparable
    form: strips a trailing province name -- present on most reads but not
    all, and irrelevant to which physical plate it is -- then collapses all
    whitespace, since PaddleOCR's line grouping can split the plate number
    itself across boxes (e.g. "5กย 9370" vs "5กย9370" for the same plate)."""
    text = plate_text.strip()
    for province in THAI_PROVINCES:
        suffix = " " + province
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return _WHITESPACE.sub("", text)
