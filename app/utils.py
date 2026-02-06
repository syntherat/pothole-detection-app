# app/utils.py
import os
import cv2
import numpy as np
import logging
import time
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pothole_detection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    logger.info(f"Confidence threshold set to {_conf:.2f}")


def load_model(model_path: Path | str | None = None):
    """Load YOLO model once and cache it globally."""
    global _model
    if _model is not None:
        logger.debug("Using cached model")
        return _model

    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists():
        error_msg = (
            f"Model not found at '{path}'. "
            f"Place your trained weight at '{DEFAULT_MODEL_PATH}', "
            "or pass a valid path to load_model()."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        logger.info(f"Loading model from {path}")
        _model = YOLO(str(path))
        logger.info(f"Model loaded successfully")
        return _model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def _imwrite_unicode(path: str | Path, img):
    """Robust imwrite for any path."""
    path = str(path)
    ext = os.path.splitext(path)[1].lower() or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError("Failed to encode image for saving.")
    with open(path, "wb") as f:
        buf.tofile(f)


def apply_roi_mask(img, roi: dict | None):
    """Apply a ROI mask (supports both rectangular and polygon formats)."""
    if not roi:
        return img

    h, w = img.shape[:2]
    
    # Check if it's the new polygon format
    if "vertices" in roi:
        # Polygon/trapezoid format
        vertices = roi["vertices"]
        pts = np.array([
            [int(x * w), int(y * h)] for x, y in vertices
        ], dtype=np.int32)
        
        # Create mask and apply
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        
        # Apply mask to image
        masked = cv2.bitwise_and(img, img, mask=mask)
        return masked
    else:
        # Legacy rectangular format
        left = int(max(0.0, min(1.0, float(roi.get("left", 0.0)))) * w)
        right = int(max(0.0, min(1.0, float(roi.get("right", 1.0)))) * w)
        top = int(max(0.0, min(1.0, float(roi.get("top", 0.0)))) * h)
        bottom = int(max(0.0, min(1.0, float(roi.get("bottom", 1.0)))) * h)

        if right <= left or bottom <= top:
            logger.warning("Invalid ROI bounds; skipping ROI mask")
            return img

        masked = img.copy()
        if top > 0:
            masked[:top, :] = 0
        if bottom < h:
            masked[bottom:, :] = 0
        if left > 0:
            masked[:, :left] = 0
        if right < w:
            masked[:, right:] = 0
        return masked


def run_detection(
    image_path: str | Path,
    save_path: str | Path | None = None,
    roi: dict | None = None,
) -> tuple[str, dict]:
    """
    Run YOLOv11 on an image and save annotated result.
    Returns tuple: (output_image_path, detection_stats_dict)
    
    Detection stats include:
        - num_potholes: Number of potholes detected
        - confidences: List of confidence scores for each detection
        - avg_confidence: Average confidence of all detections
        - inference_time: Time taken for inference in milliseconds
    """
    model = load_model()
    
    # Validate image path
    image_path = Path(image_path)
    if not image_path.exists():
        error_msg = f"Image file not found: {image_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Check image can be read
    test_img = cv2.imread(str(image_path))
    if test_img is None:
        error_msg = f"Failed to read image: {image_path}. Ensure it's a valid image format."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Running detection on: {image_path}")
    start_time = time.time()
    
    try:
        if roi:
            detection_source = apply_roi_mask(test_img, roi)
        else:
            detection_source = str(image_path)

        results = model.predict(source=detection_source, save=False, conf=_conf, verbose=False)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        result = results[0]
        annotated = result.plot()  # BGR ndarray
        
        # Extract detection statistics
        num_potholes = len(result.boxes) if result.boxes is not None else 0
        confidences = result.boxes.conf.cpu().numpy().tolist() if result.boxes is not None and result.boxes.conf is not None else []
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        stats = {
            "num_potholes": num_potholes,
            "confidences": confidences,
            "avg_confidence": float(avg_confidence),
            "inference_time": inference_time
        }
        
        logger.info(f"Detection complete: {num_potholes} potholes found (avg confidence: {avg_confidence:.2f})")
        
        if save_path is None:
            save_path = OUTPUT_DIR / ("pred_" + image_path.name)
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

        _imwrite_unicode(save_path, annotated)
        logger.info(f"Annotated image saved to: {save_path}")
        
        return str(save_path), stats
    
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise


def pil_resize(image_path: str | Path, max_size=(640, 480)) -> Image.Image:
    """Load and resize for GUI display. Validates image before loading."""
    try:
        img = Image.open(image_path)
        # Validate image is actually an image
        img.verify()
        # Re-open since verify() closes it
        img = Image.open(image_path).convert("RGB")
        img.thumbnail(max_size)
        return img
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        raise


def ensure_dirs():
    """Ensure required directories exist."""
    (ROOT / "input").mkdir(exist_ok=True)
    (ROOT / "output").mkdir(exist_ok=True)
    logger.debug(f"Ensured required directories exist at {ROOT}")
