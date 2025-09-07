# app/utils.py
import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# Resolve project root as the folder ABOVE /app
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent

MODEL_DIR = ROOT / "model"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Default model path (we'll check existence later)
DEFAULT_MODEL_PATH = MODEL_DIR / "best.pt"

# Globals
_model = None
_conf = 0.35  # default confidence


def set_conf_threshold(v: float):
    """Set global confidence threshold used by run_detection()."""
    global _conf
    _conf = float(v)


def load_model(model_path: Path | str | None = None):
    """Load YOLO model once and cache it globally."""
    global _model
    if _model is not None:
        return _model

    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at '{path}'. "
            f"Place your trained weight at '{DEFAULT_MODEL_PATH}', "
            "or pass a valid path to load_model()."
        )
    _model = YOLO(str(path))
    return _model


def _imwrite_unicode(path: str | Path, img):
    """Robust imwrite for any path."""
    path = str(path)
    ext = os.path.splitext(path)[1].lower() or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError("Failed to encode image for saving.")
    with open(path, "wb") as f:
        buf.tofile(f)


def run_detection(image_path: str | Path, save_path: str | Path | None = None) -> str:
    """
    Run YOLOv11 on an image and save annotated result.
    Returns the output image path (string).
    """
    model = load_model()
    results = model.predict(source=str(image_path), save=False, conf=_conf, verbose=False)
    annotated = results[0].plot()  # BGR ndarray

    if save_path is None:
        save_path = OUTPUT_DIR / ("pred_" + Path(image_path).name)
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    _imwrite_unicode(save_path, annotated)
    return str(save_path)


def pil_resize(image_path: str | Path, max_size=(640, 480)) -> Image.Image:
    """Load and resize for GUI display."""
    img = Image.open(image_path).convert("RGB")
    img.thumbnail(max_size)
    return img


def ensure_dirs():
    (ROOT / "input").mkdir(exist_ok=True)
    (ROOT / "output").mkdir(exist_ok=True)
