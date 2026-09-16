from __future__ import annotations

"""Summarizes app/shadow_compare.py's JSONL log into a verdict on the
AGPL-vs-Apache detector decision. Temporary tool -- see this package's
module docstring in main.py.

Usage:
    python3 shadow_detector/summarize.py [path-to-log.jsonl]
Defaults to data/shadow_detector_log.jsonl (same default as
SHADOW_DETECTOR_LOG_PATH in app/config.py).

Prints per-camera counts, average confidence and IoU on matched boxes, and
disagreement rates -- but deliberately does NOT auto-declare a winner on
shadow_only/production_only counts alone: those numbers can't tell a real
missed plate apart from a false positive (see the pedestrian false-positive
found by eye during the initial benchmark). Use --show-disagreements to
print the raw records for a human to actually look at.
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", nargs="?", default="data/shadow_detector_log.jsonl")
    parser.add_argument(
        "--show-disagreements", type=int, default=0,
        help="print this many production_only/shadow_only records for manual review",
    )
    args = parser.parse_args()

    try:
        lines = open(args.log_path, encoding="utf-8").readlines()
    except FileNotFoundError:
        print(f"No log at {args.log_path} yet -- has SHADOW_DETECTOR_URL been set and app_live restarted?")
        sys.exit(1)

    per_camera = defaultdict(lambda: {
        "samples": 0, "matched": 0, "shadow_only": 0, "production_only": 0,
        "conf_deltas": [], "ious": [],
    })
    disagreements = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        cam = per_camera[rec["camera_name"]]
        cam["samples"] += 1
        for m in rec["matched"]:
            cam["matched"] += 1
            cam["ious"].append(m["iou"])
            cam["conf_deltas"].append(m["shadow_conf"] - m["production_conf"])
        cam["shadow_only"] += len(rec["shadow_only"])
        cam["production_only"] += len(rec["production_only"])
        if rec["shadow_only"] or rec["production_only"]:
            disagreements.append(rec)

    print(f"{len(lines)} total samples across {len(per_camera)} camera(s)\n")
    for name, s in per_camera.items():
        print(f"== {name} ==")
        print(f"  samples: {s['samples']}, matched boxes: {s['matched']}")
        print(f"  shadow_only (possible false positive OR a plate production missed): {s['shadow_only']}")
        print(f"  production_only (a plate shadow missed): {s['production_only']}")
        if s["ious"]:
            print(f"  avg IoU on matches: {statistics.mean(s['ious']):.3f}")
        if s["conf_deltas"]:
            avg_delta = statistics.mean(s["conf_deltas"])
            sign = "higher" if avg_delta > 0 else "lower"
            print(f"  avg confidence delta (shadow - production): {avg_delta:+.3f} (shadow runs {sign})")
        print()

    if args.show_disagreements:
        print(f"--- {min(args.show_disagreements, len(disagreements))} disagreement record(s) ---")
        for rec in disagreements[: args.show_disagreements]:
            print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
