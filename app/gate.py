from __future__ import annotations

import logging
import threading

import serial

from .config import settings

logger = logging.getLogger(__name__)

# The ESP32 board (see firmware/gate_relay/) is reached over a USB serial
# tether, not WiFi -- the server this runs on is on a university network
# with enterprise auth and likely client isolation, so a WiFi station
# connection to the board wouldn't have been reachable anyway. One
# long-lived connection is reused across calls; a lock guards it since the
# OCR worker thread is the only caller today, but a stale/broken connection
# after an unplug should still self-heal on the next call rather than wedge
# permanently.
_lock = threading.Lock()
_conn: serial.Serial | None = None
_warned_unconfigured = False


def _get_connection() -> serial.Serial | None:
    global _conn
    if _conn is not None and _conn.is_open:
        return _conn
    try:
        _conn = serial.Serial(
            settings.gate_serial_port,
            settings.gate_serial_baud,
            timeout=settings.gate_serial_timeout_seconds,
        )
        return _conn
    except Exception:
        logger.exception("Failed to open gate serial port %r", settings.gate_serial_port)
        _conn = None
        return None


def open_gate(plate_text: str) -> bool:
    """Sends the gate-open pulse command to the ESP32 board. See
    firmware/gate_relay/ for the board-side protocol."""
    global _warned_unconfigured
    if not settings.gate_serial_port:
        if not _warned_unconfigured:
            logger.warning(
                "Registered plate %r matched but GATE_SERIAL_PORT is not set -- "
                "gate signal not sent.",
                plate_text,
            )
            _warned_unconfigured = True
        return False

    with _lock:
        conn = _get_connection()
        if conn is None:
            return False
        try:
            conn.reset_input_buffer()
            conn.write(b"PULSE_OPEN\n")
            response = conn.readline().decode(errors="replace").strip()
        except Exception:
            logger.exception("Gate serial write/read failed for plate %r", plate_text)
            conn.close()
            global _conn
            _conn = None  # force a fresh connection next time
            return False

        if response.startswith("OK"):
            logger.info("Gate opened for registered plate %r (%s)", plate_text, response)
            return True
        logger.warning("Unexpected gate response %r for plate %r", response, plate_text)
        return False
