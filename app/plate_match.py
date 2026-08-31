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


def vote_plate_text(history: list[tuple[str, float]]) -> tuple[str, float]:
    """Majority-votes a stable plate text across several independent reads of
    the same track. A single frame can misread a visually similar Thai
    consonant (e.g. ค/ต/ด) but the correct character is usually the
    plurality across the handful of reads a plate gets while it's in frame --
    this costs nothing extra since those reads already happen for re-OCR.

    Voting is per-character, confidence-weighted, and restricted to the
    largest group of reads sharing the same length (the modal length is
    almost always the correctly segmented text; a shorter/longer read is
    usually a partial miss, not a valid alternative to vote with)."""
    if len(history) == 1:
        return history[0]

    by_length: dict[int, list[tuple[str, float]]] = {}
    for text, conf in history:
        by_length.setdefault(len(text), []).append((text, conf))
    _, candidates = max(by_length.items(), key=lambda item: len(item[1]))

    voted_chars = []
    for i in range(len(candidates[0][0])):
        votes: dict[str, float] = {}
        for text, conf in candidates:
            votes[text[i]] = votes.get(text[i], 0.0) + conf
        voted_chars.append(max(votes.items(), key=lambda kv: kv[1])[0])
    voted_text = "".join(voted_chars)

    matching_confs = [conf for text, conf in candidates if text == voted_text]
    voted_conf = max(matching_confs) if matching_confs else max(conf for _, conf in candidates)
    return voted_text, voted_conf
