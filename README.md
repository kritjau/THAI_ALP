# Thai ALPR

Live automatic license plate recognition for Thai vehicles: a webcam/RTSP feed
is detected with a YOLOv11 plate localizer and read with EasyOCR (Thai +
English), then shown as an annotated MJPEG stream with a live detection log
in the browser.

## How it works

1. `app/camera.py` pulls frames from the configured source in a background thread.
2. `app/detector.py` runs a pretrained YOLOv11 license-plate detector
   ([morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection),
   auto-downloaded to `models/` on first run) to find plate bounding boxes.
3. `app/tracker.py` matches boxes across frames by IoU so a plate sitting in
   frame is only OCR'd (and logged) once, not on every processed frame.
4. `app/ocr.py` crops each new plate and reads it with EasyOCR (`th`+`en`).
5. `app/db.py` logs each new read (timestamp, text, confidence, crop) to SQLite.
6. `app/main.py` (FastAPI) serves the annotated stream at `/video_feed` and
   pushes new detections over a WebSocket to the dashboard at `/`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # edit CAMERA_SOURCE etc.
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000. First run downloads the detector weights
(~5-115MB depending on `DETECTOR_MODEL_SIZE`) and EasyOCR's Thai+English
recognition weights (~100MB); both are cached locally afterwards.

## Configuration (`.env`)

- `CAMERA_SOURCE` — `0`/`1` for a local webcam, `rtsp://user:pass@host/stream`
  for an IP camera, or a path to a video file.
- `DEVICE` — `cpu` or `cuda:0` if you have a matching CUDA build of torch.
- `DETECTOR_MODEL_SIZE` — `n` (fastest, default) through `x` (most accurate).
- `PROCESS_EVERY_N_FRAMES` — raise this on slower/CPU-only machines to keep
  the video smooth at the cost of slower plate recognition.

See `.env.example` for the full list.

## Known limitations

- OpenCV can't render Thai glyphs, so the video overlay only shows a box and
  confidence percentage; the recognized Thai text is shown in the dashboard's
  detection log instead.
- OCR output is the raw text EasyOCR reads (province name + plate number as
  two lines); there's no Thai plate-grammar validation/correction layer.
- The plate detector is a general (non-Thai-specific) model — it localizes
  any rectangular plate; accuracy depends on camera angle, distance and
  lighting like any single-model ALPR setup.
