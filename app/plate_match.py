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
        best_single_conf: dict[str, float] = {}
        for text, conf in candidates:
            ch = text[i]
            votes[ch] = votes.get(ch, 0.0) + conf
            if conf > best_single_conf.get(ch, -1.0):
                best_single_conf[ch] = conf
        top_vote = max(votes.values())
        tied = [ch for ch, total in votes.items() if total == top_vote]
        if len(tied) == 1:
            voted_chars.append(tied[0])
        else:
            # An exact tie in summed confidence -- not rare, since reads of
            # the same clean plate often come back at identical confidence.
            # Falling through to dict order here would silently pick
            # whichever candidate happened to be read first, which carries
            # no accuracy signal; picking by whichever single read was most
            # confident does.
            voted_chars.append(max(tied, key=lambda ch: best_single_conf[ch]))
    voted_text = "".join(voted_chars)

    matching_confs = [conf for text, conf in candidates if text == voted_text]
    voted_conf = max(matching_confs) if matching_confs else max(conf for _, conf in candidates)
    return voted_text, voted_conf


def plates_are_similar(a: str, b: str, max_distance: int = 1) -> bool:
    """True if two normalized plate reads differ by at most `max_distance`
    single-character edits (insert/delete/substitute).

    vote_plate_text() already smooths out misreads *within* one track's own
    history, but ByteTrack can still fragment one physical car into a
    second track id after a brief occlusion or a gap between processed
    frames -- if that new track's own vote lands on a plate text that's
    almost, but not exactly, what the first track already reported, the
    exact-string cooldown in app_live/pipeline.py won't catch it, and the
    same car shows up as two near-identical rows in the Detections list.
    This is the looser check used to catch *that* case specifically.

    Deliberately conservative (max_distance=1 by default): plate serials
    are only a few characters, so two genuinely different real plates can
    be one edit apart (e.g. "1กก111" vs "1กก112") -- a larger distance
    would start merging distinct vehicles, not just a fragmented track's
    misread of the same one. Callers scope this further (same camera, a
    short time window) so it isn't relied on alone to tell two real plates
    apart -- see app_live/pipeline.py's _find_similar_recent()."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > max_distance:
        return False

    # Plain Levenshtein distance -- short strings (plate texts are a
    # handful of characters), so no need for anything smarter than the
    # textbook O(len(a) * len(b)) DP table.
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = curr
    return prev[-1] <= max_distance
