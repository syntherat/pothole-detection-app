# app/enhanced_utils.py
import os
import cv2
import numpy as np
import logging
import time
import csv
import json
from pathlib import Path
from PIL import Image, ImageEnhance
from ultralytics import YOLO
from datetime import datetime

logger = logging.getLogger(__name__)

# Keep existing imports
from utils import (
    APP_DIR, ROOT, MODEL_DIR, OUTPUT_DIR, DEFAULT_MODEL_PATH,
    _model, _conf, set_conf_threshold, load_model, _imwrite_unicode,
    pil_resize, ensure_dirs, apply_roi_mask
)

# Batch processing
def batch_process_images(image_folder, output_folder=None, save_csv=True, save_json=True, roi=None):
    """
    Process all images in a folder
    
    Args:
        image_folder: Path to folder containing images
        output_folder: Path to save results (default: OUTPUT_DIR / batch_<timestamp>)
        save_csv: Save results to CSV
        save_json: Save results to JSON
    
    Returns:
        dict: Summary statistics
    """
    image_folder = Path(image_folder)
    if output_folder is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_folder = OUTPUT_DIR / f"batch_{timestamp}"
    else:
        output_folder = Path(output_folder)
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    images = []
    for ext in image_extensions:
        images.extend(list(image_folder.glob(f"*{ext}")))
        images.extend(list(image_folder.glob(f"*{ext.upper()}")))
    
    if not images:
        logger.warning(f"No images found in {image_folder}")
        return None
    
    logger.info(f"Processing {len(images)} images from {image_folder}")
    
    # Load model
    model = load_model()
    
    results_data = []
    total_potholes = 0
    total_time = 0
    
    for img_path in images:
        try:
            start_time = time.time()
            
            # Run detection
            if roi:
                img = cv2.imread(str(img_path))
                if img is None:
                    raise ValueError(f"Could not read image: {img_path}")
                img = apply_roi_mask(img, roi)
                results = model.predict(source=img, save=False, conf=_conf, verbose=False)
            else:
                results = model.predict(source=str(img_path), save=False, conf=_conf, verbose=False)
            result = results[0]
            inference_time = (time.time() - start_time) * 1000
            
            # Extract data
            num_potholes = len(result.boxes) if result.boxes is not None else 0
            confidences = result.boxes.conf.cpu().numpy().tolist() if result.boxes is not None and result.boxes.conf is not None else []
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Save annotated image
            annotated = result.plot()
            out_path = output_folder / f"det_{img_path.name}"
            _imwrite_unicode(out_path, annotated)
            
            # Store data
            results_data.append({
                'filename': img_path.name,
                'num_potholes': num_potholes,
                'avg_confidence': avg_confidence,
                'confidences': confidences,
                'inference_time_ms': inference_time,
                'output_path': str(out_path)
            })
            
            total_potholes += num_potholes
            total_time += inference_time
            
        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {e}")
            results_data.append({
                'filename': img_path.name,
                'error': str(e)
            })
    
    # Save results
    if save_csv:
        csv_path = output_folder / "results.csv"
        with open(csv_path, 'w', newline='') as f:
            if results_data:
                writer = csv.DictWriter(f, fieldnames=['filename', 'num_potholes', 'avg_confidence', 'inference_time_ms', 'output_path'])
                writer.writeheader()
                for row in results_data:
                    if 'error' not in row:
                        writer.writerow({
                            'filename': row['filename'],
                            'num_potholes': row['num_potholes'],
                            'avg_confidence': f"{row['avg_confidence']:.4f}",
                            'inference_time_ms': f"{row['inference_time_ms']:.2f}",
                            'output_path': row['output_path']
                        })
        logger.info(f"CSV results saved to {csv_path}")
    
    if save_json:
        json_path = output_folder / "results.json"
        summary = {
            'processed_images': len(results_data),
            'total_potholes': total_potholes,
            'avg_inference_time_ms': total_time / len(images) if images else 0,
            'results': results_data
        }
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"JSON results saved to {json_path}")
    
    logger.info(f"Batch processing complete: {len(images)} images, {total_potholes} potholes detected")
    
    return {
        'total_images': len(images),
        'total_potholes': total_potholes,
        'avg_inference_time': total_time / len(images) if images else 0,
        'output_folder': str(output_folder)
    }

# Image preprocessing
def preprocess_image(image_path, enhance=True, denoise=True, resize_max=None):
    """
    Preprocess image for better detection
    
    Args:
        image_path: Path to image
        enhance: Apply brightness/contrast enhancement
        denoise: Apply denoising
        resize_max: Maximum dimension for resizing (None to skip)
    
    Returns:
        np.ndarray: Preprocessed image
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Resize if too large
    if resize_max:
        h, w = img.shape[:2]
        if max(h, w) > resize_max:
            scale = resize_max / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Denoise
    if denoise:
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # Enhance
    if enhance:
        # Convert to PIL for enhancement
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Auto-enhance brightness and contrast
        enhancer = ImageEnhance.Brightness(img_pil)
        img_pil = enhancer.enhance(1.1)
        
        enhancer = ImageEnhance.Contrast(img_pil)
        img_pil = enhancer.enhance(1.2)
        
        # Convert back to OpenCV
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    return img

# Performance monitoring
class PerformanceMonitor:
    """Monitor detection performance metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.detections = []
        self.inference_times = []
        self.start_time = time.time()
    
    def add_detection(self, num_potholes, inference_time, confidence):
        self.detections.append({
            'num_potholes': num_potholes,
            'inference_time': inference_time,
            'confidence': confidence,
            'timestamp': time.time()
        })
        self.inference_times.append(inference_time)
    
    def get_stats(self):
        if not self.detections:
            return None
        
        return {
            'total_detections': len(self.detections),
            'total_potholes': sum(d['num_potholes'] for d in self.detections),
            'avg_inference_time': np.mean(self.inference_times),
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'fps': len(self.detections) / (time.time() - self.start_time),
            'avg_confidence': np.mean([d['confidence'] for d in self.detections if d['confidence'] > 0])
        }

# Export functions
def export_detections_csv(detections, output_path):
    """Export detection results to CSV"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Image', 'Potholes', 'Avg_Confidence', 'Inference_Time_ms'])
        for d in detections:
            writer.writerow([
                d.get('timestamp', ''),
                d.get('image', ''),
                d.get('num_potholes', 0),
                f"{d.get('avg_confidence', 0):.4f}",
                f"{d.get('inference_time', 0):.2f}"
            ])
    logger.info(f"Exported {len(detections)} detections to {output_path}")

def export_detections_json(detections, output_path):
    """Export detection results to JSON"""
    with open(output_path, 'w') as f:
        json.dump({
            'export_time': datetime.now().isoformat(),
            'total_detections': len(detections),
            'detections': detections
        }, f, indent=2)
    logger.info(f"Exported {len(detections)} detections to {output_path}")
