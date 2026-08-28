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
# long-lived connection is reused across calls; a lock guards it since it's
# shared between the OCR worker thread (open) and the timer thread below
# (the automatic close), and a stale/broken connection after an unplug
# should still self-heal on the next call rather than wedge permanently.
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


def _send_command(command: bytes, plate_text: str, action: str) -> bool:
    """Sends one command line to the board and checks for an `OK` reply. See
    firmware/gate_relay/ for the board-side protocol."""
    with _lock:
        conn = _get_connection()
        if conn is None:
            return False
        try:
            conn.reset_input_buffer()
            conn.write(command)
            response = conn.readline().decode(errors="replace").strip()
        except Exception:
            logger.exception("Gate serial write/read failed for plate %r (%s)", plate_text, action)
            conn.close()
            global _conn
            _conn = None  # force a fresh connection next time
            return False

    if response.startswith("OK"):
        logger.info("Gate %s for plate %r (%s)", action, plate_text, response)
        return True
    logger.warning("Unexpected gate response %r for plate %r (%s)", response, plate_text, action)
    return False


def open_gate(plate_text: str) -> bool:
    """Sends the gate-open pulse, then schedules the close pulse to follow
    automatically after GATE_CLOSE_DELAY_SECONDS -- callers don't need to
    call anything to close it back up."""
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

    opened = _send_command(b"PULSE_OPEN\n", plate_text, "opened")
    if opened:
        timer = threading.Timer(settings.gate_close_delay_seconds, _close_gate, args=(plate_text,))
        timer.daemon = True
        timer.start()
    return opened


def _close_gate(plate_text: str):
    _send_command(b"PULSE_CLOSE\n", plate_text, "closed")
