from __future__ import annotations

import logging
import urllib.request

from .config import settings

logger = logging.getLogger(__name__)

_warned_unconfigured = False


def open_gate(plate_text: str) -> bool:
    """Fires the gate-open signal for a recognized, registered plate.

    The actual trigger (an ESP-WROOM board wired to the existing remote,
    firmware unknown/lost) isn't discovered yet -- this sends a plain HTTP
    GET to GATE_TRIGGER_URL, the most common shape for a DIY ESP32 endpoint,
    as a placeholder. Once the board's real control protocol is confirmed
    (HTTP path/method, or something else like MQTT), replace the body of
    this function accordingly; nothing else in the app needs to change,
    since pipeline.py only ever calls this one function.
    """
    global _warned_unconfigured
    if not settings.gate_trigger_url:
        if not _warned_unconfigured:
            logger.warning(
                "Registered plate %r matched but GATE_TRIGGER_URL is not set -- "
                "gate signal not sent (set it in .env once the ESP32's control "
                "endpoint is known).",
                plate_text,
            )
            _warned_unconfigured = True
        return False

    try:
        with urllib.request.urlopen(settings.gate_trigger_url, timeout=settings.gate_trigger_timeout_seconds):
            pass
        logger.info("Gate signal sent for registered plate %r", plate_text)
        return True
    except Exception:
        logger.exception("Failed to send gate signal for registered plate %r", plate_text)
        return False
