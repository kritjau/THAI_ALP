from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from .config import settings


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
                image_path TEXT
            )
            """
        )


def insert_detection(plate_text, confidence, bbox, image_path=None, timestamp=None) -> int:
    ts = timestamp if timestamp is not None else time.time()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO detections (timestamp, plate_text, confidence, bbox, image_path) "
            "VALUES (?, ?, ?, ?, ?)",
            (ts, plate_text, confidence, ",".join(map(str, bbox)), image_path),
        )
        return cur.lastrowid


def update_detection(detection_id, plate_text, confidence, bbox, image_path=None, timestamp=None):
    ts = timestamp if timestamp is not None else time.time()
    with _connect() as conn:
        conn.execute(
            "UPDATE detections SET timestamp = ?, plate_text = ?, confidence = ?, "
            "bbox = ?, image_path = ? WHERE id = ?",
            (ts, plate_text, confidence, ",".join(map(str, bbox)), image_path, detection_id),
        )


def recent_detections(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        cur = conn.execute(
            "SELECT id, timestamp, plate_text, confidence, bbox, image_path "
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
        }
        for r in rows
    ]
