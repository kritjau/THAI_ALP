# Thai ALPR

Live automatic license plate recognition for Thai vehicles: a webcam/RTSP feed
is detected with a YOLOv11 plate localizer and read with PaddleOCR (Thai),
then shown as an annotated MJPEG stream with a live detection log in the
browser.

## How it works

1. `app/camera.py` pulls frames from the configured source in a background thread.
   RTSP opens/reads are wrapped with a wall-clock timeout (`CAMERA_OPEN_TIMEOUT_MS`/
   `CAMERA_READ_TIMEOUT_MS`) and auto-reconnect, since a stalled network path or
   overloaded NVR can otherwise hang `cv2.VideoCapture` indefinitely with no error.
2. `app/detector.py` runs a pretrained YOLOv11 license-plate detector
   ([morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection),
   auto-downloaded to `models/` on first run) to find plate bounding boxes, and
   tracks them across frames with ByteTrack (bundled with ultralytics) so each
   plate gets a stable ID -- a proper motion-aware multi-object tracker instead
   of naive box-overlap matching, at negligible extra cost (detection still
   runs once per frame either way; tracking only adds cheap Kalman-filter +
   assignment math on top).
3. `app/tracker.py` keeps per-plate bookkeeping (best OCR read so far, its DB
   row, its saved crop) keyed by that track ID, so a plate sitting in frame is
   only OCR'd (and logged) once, not on every processed frame.
4. `app/ocr.py` crops each new plate and reads it with PaddleOCR (`th`),
   discarding individual text segments below `OCR_MIN_SEGMENT_CONFIDENCE`
   instead of letting them drag down the whole read. The plate-number line is
   kept as read; anything below it is only kept if `app/thai_provinces.py`
   fuzzy-matches it to one of Thailand's 77 real provinces (correcting it to
   the canonical spelling) -- this filters out non-province text a plate frame
   or holder often has printed on it (dealer/district branding).
5. `app/db.py` logs each new read (timestamp, text, confidence, crop) to SQLite
   at `data/alpr.db`, refining the row in place if a better read of the same
   plate comes in while it's still in frame.
6. `app/json_export.py` buffers reads deduplicated by plate text and, every
   `JSON_EXPORT_INTERVAL_SECONDS` (default 20s), writes a timestamped snapshot
   file to `json/interval_<timestamp>.json` plus updates `json/plates_cumulative.json`,
   which merges every unique plate ever seen across all intervals.
7. `app/main.py` (FastAPI) serves the annotated stream at `/video_feed` and
   pushes new detections over a WebSocket to the dashboard at `/`.

## Multiple cameras

Steps 1-4 above (camera read thread, detector+tracker, OCR, vehicle/color
classification) are one `app/camera_worker.py:CameraWorker` per camera --
`app/pipeline.py` and `app_live/pipeline.py` each run one `CameraWorker` per
entry in `Settings.camera_configs()` (`CAMERA_SOURCE` plus however many
`CAMERA_SOURCE_2`, `CAMERA_SOURCE_3`, ... are set) rather than assuming a
single camera. The dashboard renders one video panel per camera, and
Detections/Captures carry a Camera column/tag so a plate can be tied back to
which feed saw it. Each camera gets its own detector/tracker/OCR model
instances (ByteTrack's tracker state, in particular, can't be shared across
streams), so this is real added CPU/GPU/RAM cost per camera, not just a
display change.

## Setup

PaddleOCR's backend (`paddlepaddle`) isn't on PyPI and has no wheel for
Python 3.14+ as of writing — this project targets **Python 3.11/3.12**.

```bash
python3.11 -m venv .venv
source .venv/bin/activate

# paddlepaddle first, from its own package index (CPU build shown; see
# https://www.paddlepaddle.org.cn/ for a GPU/CUDA build if you have one)
pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

pip install -r requirements.txt
cp .env.example .env   # edit CAMERA_SOURCE etc.
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000. First run downloads the detector weights
(~5-115MB depending on `DETECTOR_MODEL_SIZE`) and PaddleOCR's Thai detection +
recognition weights; both are cached locally afterwards.

## `app_live/` -- no-storage variant (PDPA)

`app/` logs every read to SQLite, saves a crop image, and exports JSON --
useful for a parking-lot use case, but each of those is a retained personal
data record. `app_live/` is a separate, standalone version of the same
detect -> track -> OCR pipeline (reuses `app/camera_worker.py` and the
dashboard's static assets unchanged, but its own `pipeline.py`/`main.py`)
that never writes any of that to disk: no
database, no saved crop, no JSON export. Recognized text only exists in
memory for as long as a plate's track is alive (plus a capped 50-item
in-RAM buffer for the dashboard's initial load), and is gone on restart.

```bash
uvicorn app_live.main:app --reload --port 8002
```

Open http://localhost:8002.

Uses the same `.env` as `app/` (camera, detector, OCR settings) since none of
that is personal data on its own -- only the storage step is removed. If you
run this alongside `app/` against the same RTSP source, note some
cameras/NVRs cap concurrent client connections per channel.

## Configuration (`.env`)

- `CAMERA_SOURCE` — `0`/`1` for a local webcam, `rtsp://user:pass@host/stream`
  for an IP camera, or a path to a video file. `CAMERA_SOURCE_2`,
  `CAMERA_SOURCE_3`, ... (with optional `CAMERA_NAME_2`, etc. labels) run
  more cameras side by side in the same dashboard -- see Multiple cameras
  above.
- `DEVICE` — plate detector (torch/ultralytics): `cpu` or `cuda:0` if you have a GPU.
- `OCR_DEVICE` — PaddleOCR device, kept `cpu` by default even when `DEVICE` uses
  a GPU; see Known limitations below for why.
- `DETECTOR_MODEL_SIZE` — `n` (fastest, default) through `x` (most accurate).
- `PROCESS_EVERY_N_FRAMES` — raise this on slower/CPU-only machines to keep
  the video smooth at the cost of slower plate recognition.
- `CAMERA_OPEN_TIMEOUT_MS` / `CAMERA_READ_TIMEOUT_MS` / `CAMERA_RECONNECT_DELAY_SECONDS`
  — how an RTSP source recovers from a stalled/dropped connection.

See `.env.example` for the full list.

## Known limitations

- OpenCV can't render Thai glyphs, so the video overlay only shows a box and
  confidence percentage; the recognized Thai text is shown in the dashboard's
  detection log instead.
- The plate-number line is checked against the common civilian plate shape
  (`app/plate_format.py`: an optional leading digit, 1-2 Thai consonants,
  1-4 digits) and rejected if it doesn't match -- but unlike the province
  line, there's no fixed registry of every real plate number to *correct*
  a read against, only reject an implausible one. Special-purpose plates
  (diplomatic, trailer, tractor, etc.) follow different shapes and aren't
  covered by this check.
- The plate detector is a general (non-Thai-specific) model — it localizes
  any rectangular plate; accuracy depends on camera angle, distance and
  lighting like any single-model ALPR setup.
- ByteTrack is motion-aware but not infallible -- a plate can still get a new
  ID (logged as a second entry) after a long enough occlusion or gap between
  processed frames (raising `PROCESS_EVERY_N_FRAMES`'s frequency, i.e. lowering
  the value, gives the tracker more frames to follow the motion with).
- If the network path to an RTSP camera is itself unreliable (not just the
  app), reconnects will keep the feed alive but reads still won't be reliably
  fast — `PROCESS_EVERY_N_FRAMES` and a lower-resolution substream (if the
  camera/NVR offers one) help more than any app-side setting can.
- PaddleOCR's CPU backend crashes on this project's test machine with its
  default oneDNN (mkldnn) kernel (`NotImplementedError` in the ONEDNN executor);
  `app/ocr.py` disables it (`enable_mkldnn=False`) as a stable workaround.
- `paddlepaddle-gpu` and a CUDA build of `torch` can't safely coexist in the
  same venv: installing `paddlepaddle-gpu` overwrote files in torch's `nvidia.*`
  package namespace (they share the same on-disk paths regardless of the `-cuXX`
  suffix in the package name), breaking torch's CUDA import outright until both
  were reinstalled clean. The detector (`DEVICE`) can use GPU; PaddleOCR
  (`OCR_DEVICE`) stays CPU-only unless you've verified a specific
  paddlepaddle-gpu/torch version pair actually coexists.
