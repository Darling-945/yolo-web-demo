# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Flask-based web application for YOLO object detection (v5/v8/v11). Supports PyTorch (.pt), ONNX (.onnx), and TensorRT (.engine) model formats. Provides a web UI and RESTful API for image/video detection, batch processing, and real-time camera detection.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app (development mode, default)
python run.py

# Run with options
python run.py --debug          # Force debug mode
python run.py --production     # Production mode (requires SECRET_KEY in .env)
python run.py --port 8080      # Custom port
python run.py --config         # Show current config
python run.py --manage show    # Show config and exit

# Run tests
python utils/test_app.py

# Convert models (PyTorch -> ONNX)
python convert_models.py
```

## Architecture

### Entry Point & Config
- **`run.py`** — Entry point. Defines `Config`/`DevelopmentConfig`/`ProductionConfig` classes, manages `.env` file creation, parses CLI args, and starts Flask. Config values are loaded from environment variables with hardcoded defaults in `DEFAULT_CONFIG`.

### Application Layer
- **`app.py`** — Flask app factory. Routes (web pages + API), async task processing with `threading`, batch upload/inference endpoints, rate limiting via `Flask-Limiter`. Uses a custom `NumpyJSONProvider` for JSON serialization of numpy types.

- **`model_inference.py`** — Core inference engine. Loads YOLO models via Ultralytics, caches them in a module-level `MODEL_CACHE` dict (thread-safe with `Lock`). Handles image and video inference with FP16 on GPU. Contains `yolo_inference()` (main entry) and `get_available_models()`.

### Utilities
- **`utils/`** — Re-exported via `utils/__init__.py`:
  - `utils.py` — File upload validation, secure filename generation, file cleanup scheduling, security logging, inference parameter processing
  - `path_utils.py` — Static path normalization and URL path helpers
  - `detect.py` — Detection-related utilities
  - `test_app.py` — Integration tests (import verification, API endpoint tests)

### Frontend
- **`templates/`** — Jinja2 templates. `base.html` is the layout; pages: `index.html` (main detection), `camera.html` (live webcam), `inference.html` (results), `batch_inference.html`, `about.html`
- **`static/js/`** — `camera.js` (webcam detection), `multi-file-upload.js` (batch uploads)
- **`static/css/main.css`** — Single stylesheet
- All UI must follow DESIGN.md. Do not invent new colors/typography outside the spec.

### Models
- **`models/`** — Custom model files directory. Predefined models (yolo11n.pt, etc.) are auto-downloaded by Ultralytics on first use.

## Key Patterns

- **Model caching**: Models loaded once and stored in `MODEL_CACHE` with thread-safe `Lock`. Don't re-load models per request.
- **Async tasks**: Long-running operations (video/batch processing) run in background threads with progress tracked via task IDs. Clients poll `GET /api/task/<task_id>` for status.
- **Config flow**: `run.py` → `.env` file → `Config` class → `app.py` reads config. The `.env` file is auto-created on first run if missing.
- **Windows compatibility**: Flask reloader is disabled on Windows (`use_reloader = config.DEBUG and sys.platform != 'win32'`) to avoid threading issues. Console encoding is forced to UTF-8.
- **Video processing timeout**: 300 seconds (`VIDEO_PROCESSING_TIMEOUT_SECONDS` in `app.py`).

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/detect` | POST | Image/video detection |
| `/api/models` | GET | List available models |
| `/api/batch_upload` | POST | Multi-file upload |
| `/api/batch_inference` | POST | Batch processing |
| `/api/task/<task_id>` | GET | Check async task progress |
| `/api/camera_detect` | POST | Real-time camera frame detection |

## Configuration

All config is via `.env` file (auto-created). Key variables: `HOST`, `PORT`, `DEFAULT_MODEL`, `DEFAULT_CONFIDENCE`, `DEFAULT_IOU`, `MAX_CONTENT_LENGTH` (500MB default), `RATELIMIT_DEFAULT`, `RATELIMIT_API`.
