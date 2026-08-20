from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import settings


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


class JsonExporter:
    """Buffers plate reads deduplicated by plate text (keeping the best-confidence
    read seen) and, every `interval_seconds`, writes a timestamped snapshot of
    that interval plus updates one cumulative file merging every unique plate
    ever seen across intervals."""

    def __init__(self, interval_seconds: float | None = None, json_dir: str | None = None):
        self.interval_seconds = interval_seconds or settings.json_export_interval_seconds
        self.json_dir = Path(json_dir or settings.json_dir)
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.cumulative_path = self.json_dir / "plates_cumulative.json"
        self._buffer: dict[str, dict] = {}
        self._cumulative: dict[str, dict] = self._load_cumulative()
        self._last_flush = time.time()

    def record(self, plate_text: str, confidence: float, image_path: str | None, timestamp: float | None = None):
        ts = timestamp if timestamp is not None else time.time()
        existing = self._buffer.get(plate_text)
        if existing is None or confidence > existing["confidence"]:
            self._buffer[plate_text] = {
                "plate_text": plate_text,
                "confidence": confidence,
                "timestamp": ts,
                "image_path": image_path,
            }

    def maybe_flush(self) -> str | None:
        now = time.time()
        if now - self._last_flush < self.interval_seconds:
            return None
        self._last_flush = now
        return self._flush(now)

    def _flush(self, now: float) -> str:
        plates = list(self._buffer.values())
        interval_file = self.json_dir / f"interval_{datetime.fromtimestamp(now, tz=timezone.utc):%Y%m%d_%H%M%S}.json"
        interval_file.write_text(
            json.dumps(
                {
                    "interval_seconds": self.interval_seconds,
                    "generated_at": _iso(now),
                    "plate_count": len(plates),
                    "plates": plates,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        for plate_text, entry in self._buffer.items():
            existing = self._cumulative.get(plate_text)
            if existing is None:
                self._cumulative[plate_text] = {
                    **entry,
                    "first_seen": entry["timestamp"],
                    "last_seen": entry["timestamp"],
                    "times_seen": 1,
                }
            else:
                existing["last_seen"] = entry["timestamp"]
                existing["times_seen"] += 1
                if entry["confidence"] > existing["confidence"]:
                    existing["confidence"] = entry["confidence"]
                    existing["image_path"] = entry["image_path"]

        self.cumulative_path.write_text(
            json.dumps(
                {
                    "updated_at": _iso(now),
                    "unique_plate_count": len(self._cumulative),
                    "plates": list(self._cumulative.values()),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        self._buffer.clear()
        return str(interval_file)

    def _load_cumulative(self) -> dict[str, dict]:
        if not self.cumulative_path.exists():
            return {}
        try:
            data = json.loads(self.cumulative_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return {p["plate_text"]: p for p in data.get("plates", [])}
