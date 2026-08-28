from __future__ import annotations

import re

# The common Thai private-vehicle/motorcycle plate shape: an optional
# leading digit (used once a province's classic 2-letter/4-digit
# combinations run out), 1-2 Thai consonants, then 1-4 digits. Applied to
# normalize_plate()'s output, which has already stripped the province and
# all whitespace, so no space needs to appear in the pattern itself.
#
# This is a structural plausibility check, not a real-plate registry the
# way THAI_PROVINCES is -- there's no fixed list of every valid plate
# number to validate/correct against, so this can only catch reads with an
# obviously wrong shape (wrong character counts, mixed scripts, all-digit),
# not confirm a read is the *correct* one. It also doesn't cover every
# plate category that exists (diplomatic, trailer, tractor, and other
# special-purpose plates follow different formats) -- only the common
# civilian car/motorcycle shape.
_PLATE_NUMBER_PATTERN = re.compile(r"^\d?[ก-ฮ]{1,2}\d{1,4}$")


def looks_like_valid_plate_number(normalized_text: str) -> bool:
    if not normalized_text:
        return False
    return bool(_PLATE_NUMBER_PATTERN.match(normalized_text))
