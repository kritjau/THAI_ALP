from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from .config import settings
from .plate_match import normalize_plate

# Deliberately its own SQLite file, separate from app/db.py's alpr.db (which
# stores every detected plate for the parking-lot use case). This is a small,
# manually-curated whitelist -- not detection history -- and belongs to the
# gate-access system (app_live), not the storage system (app), so it's kept
# in a database of its own rather than commingled with either app's data.


@contextmanager
def _connect():
    Path(settings.registered_plates_db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.registered_plates_db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS registered_plates (
                plate_text TEXT PRIMARY KEY,
                label TEXT,
                created_at REAL NOT NULL
            )
            """
        )


def add_registered_plate(plate_text: str, label: str | None = None) -> None:
    normalized = normalize_plate(plate_text)
    with _connect() as conn:
        conn.execute(
            "INSERT INTO registered_plates (plate_text, label, created_at) VALUES (?, ?, ?) "
            "ON CONFLICT(plate_text) DO UPDATE SET label = excluded.label",
            (normalized, label, time.time()),
        )


def remove_registered_plate(plate_text: str) -> None:
    with _connect() as conn:
        conn.execute(
            "DELETE FROM registered_plates WHERE plate_text = ?", (normalize_plate(plate_text),)
        )


def list_registered_plates() -> list[dict]:
    with _connect() as conn:
        cur = conn.execute(
            "SELECT plate_text, label, created_at FROM registered_plates ORDER BY created_at DESC"
        )
        rows = cur.fetchall()
    return [{"plate_text": r[0], "label": r[1], "created_at": r[2]} for r in rows]


def is_registered_plate(plate_text: str) -> bool:
    with _connect() as conn:
        cur = conn.execute(
            "SELECT 1 FROM registered_plates WHERE plate_text = ? LIMIT 1",
            (normalize_plate(plate_text),),
        )
        return cur.fetchone() is not None
