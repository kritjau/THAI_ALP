from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from .config import settings
from .plate_match import normalize_plate


@contextmanager
def _connect():
    conn = sqlite3.connect(settings.db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                plate_text TEXT NOT NULL,
                confidence REAL NOT NULL,
                bbox TEXT NOT NULL,
                image_path TEXT,
                color TEXT
            )
            """
        )
        # Migration for DBs created before the color column existed, and for
        # the earlier "vehicle_color" name (renamed here, keeping the data --
        # a prior version also added plate_color; that column may still exist
        # on older DBs as inert, unused data, not dropped, just not queried).
        existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(detections)")}
        if "color" not in existing_cols:
            if "vehicle_color" in existing_cols:
                conn.execute("ALTER TABLE detections RENAME COLUMN vehicle_color TO color")
            else:
                conn.execute("ALTER TABLE detections ADD COLUMN color TEXT")

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


def insert_detection(plate_text, confidence, bbox, image_path=None, timestamp=None, color=None) -> int:
    ts = timestamp if timestamp is not None else time.time()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO detections (timestamp, plate_text, confidence, bbox, image_path, color) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (ts, plate_text, confidence, ",".join(map(str, bbox)), image_path, color),
        )
        return cur.lastrowid


def update_detection(detection_id, plate_text, confidence, bbox, image_path=None, timestamp=None, color=None):
    ts = timestamp if timestamp is not None else time.time()
    with _connect() as conn:
        conn.execute(
            "UPDATE detections SET timestamp = ?, plate_text = ?, confidence = ?, "
            "bbox = ?, image_path = ?, color = ? WHERE id = ?",
            (ts, plate_text, confidence, ",".join(map(str, bbox)), image_path, color, detection_id),
        )


def recent_detections(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        cur = conn.execute(
            "SELECT id, timestamp, plate_text, confidence, bbox, image_path, color "
            "FROM detections ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "timestamp": r[1],
            "plate_text": r[2],
            "confidence": r[3],
            "bbox": r[4],
            "image_path": r[5],
            "color": r[6],
        }
        for r in rows
    ]
