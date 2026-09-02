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
        # Migration for DBs created before multi-camera support -- existing
        # rows all predate it and get NULL, which the dashboard shows as
        # "Camera 1" (the only camera a single-camera setup ever had).
        if "camera_id" not in existing_cols:
            conn.execute("ALTER TABLE detections ADD COLUMN camera_id TEXT")
        if "camera_name" not in existing_cols:
            conn.execute("ALTER TABLE detections ADD COLUMN camera_name TEXT")
        # Migration for DBs created before vehicle-type classification --
        # existing rows simply have no type, shown by the dashboard as "?".
        if "vehicle_type" not in existing_cols:
            conn.execute("ALTER TABLE detections ADD COLUMN vehicle_type TEXT")


def insert_detection(
    plate_text, confidence, bbox, image_path=None, timestamp=None, color=None,
    vehicle_type=None, camera_id=None, camera_name=None,
) -> int:
    ts = timestamp if timestamp is not None else time.time()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO detections (timestamp, plate_text, confidence, bbox, image_path, color, "
            "vehicle_type, camera_id, camera_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, plate_text, confidence, ",".join(map(str, bbox)), image_path, color,
             vehicle_type, camera_id, camera_name),
        )
        return cur.lastrowid


def update_detection(
    detection_id, plate_text, confidence, bbox, image_path=None, timestamp=None, color=None,
    vehicle_type=None, camera_id=None, camera_name=None,
):
    ts = timestamp if timestamp is not None else time.time()
    with _connect() as conn:
        conn.execute(
            "UPDATE detections SET timestamp = ?, plate_text = ?, confidence = ?, "
            "bbox = ?, image_path = ?, color = ?, vehicle_type = ?, camera_id = ?, camera_name = ? WHERE id = ?",
            (ts, plate_text, confidence, ",".join(map(str, bbox)), image_path, color,
             vehicle_type, camera_id, camera_name, detection_id),
        )


def vehicle_type_counts() -> dict[str, int]:
    """Total detections per vehicle type across all history -- a DB query
    rather than an in-memory tally since app/ is the persistent app and this
    should survive a restart, and GROUP BY already gives an accurate count
    per unique tracked vehicle (one row per track, refined in place -- see
    update_detection) rather than double-counting re-OCR refinements."""
    with _connect() as conn:
        cur = conn.execute(
            "SELECT COALESCE(vehicle_type, 'unknown') AS vt, COUNT(*) "
            "FROM detections GROUP BY vt"
        )
        return {row[0]: row[1] for row in cur.fetchall()}


def recent_detections(limit: int = 50) -> list[dict]:
    with _connect() as conn:
        cur = conn.execute(
            "SELECT id, timestamp, plate_text, confidence, bbox, image_path, color, "
            "vehicle_type, camera_id, camera_name FROM detections ORDER BY id DESC LIMIT ?",
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
            "vehicle_type": r[7],
            "camera_id": r[8],
            # Rows logged before multi-camera support have no camera_name --
            # they all came from what is now "Camera 1", the only camera a
            # single-camera setup ever had.
            "camera_name": r[9] or "Camera 1",
        }
        for r in rows
    ]
