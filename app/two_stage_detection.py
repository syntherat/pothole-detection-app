# app/two_stage_detection.py
"""
Two-stage detection: Road segmentation → Pothole detection
Reduces false positives by only detecting potholes on road surfaces.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)

class TwoStageDetector:
    """Combines road segmentation with pothole detection."""
    
    def __init__(self, pothole_model_path, road_model_path=None):
        """
        Initialize two-stage detector.
        
        Args:
            pothole_model_path: Path to trained pothole detection model
            road_model_path: Path to road segmentation model (optional)
        """
        self.pothole_model = YOLO(pothole_model_path)
        
        self.road_model = None
        self.use_road_seg = False
        
        if road_model_path and Path(road_model_path).exists():
            try:
                self.road_model = YOLO(road_model_path)
                self.use_road_seg = True
                logger.info(f"Two-stage detection enabled with road segmentation: {road_model_path}")
            except Exception as e:
                logger.warning(f"Failed to load road segmentation model: {e}")
                logger.info("Falling back to single-stage pothole detection")
        else:
            logger.info("Road segmentation model not found, using single-stage detection")
    
    def get_road_mask(self, frame, lowres_width: int | None = None):
        """
        Extract road mask from frame using segmentation model.
        
        Args:
            frame: Input image (BGR)
            lowres_width: Optional width to downscale for faster segmentation
            
        Returns:
            Binary mask (uint8) where 255=road, 0=non-road
        """
        if not self.use_road_seg:
            # Return full frame mask if no segmentation
            return np.ones(frame.shape[:2], dtype=np.uint8) * 255
        
        try:
            h, w = frame.shape[:2]
            seg_frame = frame
            seg_h, seg_w = h, w

            if lowres_width and w > lowres_width:
                scale = lowres_width / float(w)
                seg_w = int(w * scale)
                seg_h = int(h * scale)
                seg_frame = cv2.resize(frame, (seg_w, seg_h), interpolation=cv2.INTER_AREA)

            # Run segmentation
            results = self.road_model(seg_frame, verbose=False)[0]
            
            road_mask = np.zeros((seg_h, seg_w), dtype=np.uint8)
            
            if results.masks is not None and len(results.masks) > 0:
                # Strategy: Combine all detected segments
                # For generic models, we assume road is the largest continuous region
                # in the lower 2/3 of the frame
                
                for i in range(len(results.masks)):
                    mask = results.masks.data[i].cpu().numpy()
                    mask_resized = cv2.resize(mask, (seg_w, seg_h))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                    
                    # Check if mask is in lower portion (likely road)
                    lower_portion = mask_binary[seg_h//3:, :]
                    if lower_portion.sum() > mask_binary.sum() * 0.3:
                        road_mask = cv2.bitwise_or(road_mask, mask_binary)
                
                # Post-process: keep only the largest connected component
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(road_mask, connectivity=8)
                if num_labels > 1:
                    # Find largest component (excluding background=0)
                    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                    road_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
                
                # Morphological operations to clean up mask
                kernel = np.ones((5, 5), np.uint8)
                road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
                road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
            
            # If no mask generated, use lower 60% of frame (fallback)
            if road_mask.max() == 0:
                logger.warning("No road mask generated, using lower 60% of frame as fallback")
                road_mask[int(seg_h * 0.4):, :] = 255

            if seg_h != h or seg_w != w:
                road_mask = cv2.resize(road_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
            return road_mask
            
        except Exception as e:
            logger.error(f"Error in road segmentation: {e}")
            # Fallback: return full frame
            return np.ones(frame.shape[:2], dtype=np.uint8) * 255
    
    def detect_potholes(self, frame, conf=0.35, return_mask=False, road_mask=None, lowres_width: int | None = None):
        """
        Detect potholes using two-stage approach.
        
        Args:
            frame: Input image (BGR)
            conf: Confidence threshold for pothole detection
            return_mask: If True, also return the road mask used
            road_mask: Optional precomputed road mask
            lowres_width: Optional width to downscale for faster segmentation
            
        Returns:
            results: YOLO detection results
            road_mask (optional): Binary road mask if return_mask=True
        """
        # Stage 1: Get road mask
        if road_mask is None:
            road_mask = self.get_road_mask(frame, lowres_width=lowres_width)
        
        # Stage 2: Detect potholes on full frame (not masked)
        # This preserves image quality and context for better detection
        results = self.pothole_model(frame, conf=conf, verbose=False)[0]
        
        if return_mask:
            return results, road_mask
        return results
    
    def visualize(self, frame, results, road_mask=None, show_mask=True):
        """
        Create visualization with detections and optional road mask overlay.
        
        Args:
            frame: Original frame
            results: YOLO detection results
            road_mask: Road mask (optional)
            show_mask: Whether to show road mask overlay
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw detections with bounding boxes and confidence scores
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                # Handle both tensor and numpy arrays
                xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
                conf_val = box.conf[0].cpu().numpy() if hasattr(box.conf[0], 'cpu') else box.conf[0]
                
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                conf = float(conf_val)
                
                # Filter by road mask if available (only show if center is on road)
                if self.use_road_seg and road_mask is not None:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if not (0 <= cy < road_mask.shape[0] and 0 <= cx < road_mask.shape[1] and road_mask[cy, cx] > 127):
                        continue  # Skip this box if not in road area
                
                # Draw bounding box (green for pothole)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label with confidence score
                label = f"Pothole {conf:.2f}"
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                
                # Draw label background
                cv2.rectangle(
                    annotated,
                    (x1, y1 - label_size[1] - baseline),
                    (x1 + label_size[0], y1),
                    (0, 255, 0),
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated, label,
                    (x1, y1 - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2
                )
        
        # Overlay road mask if available
        if show_mask and road_mask is not None and self.use_road_seg:
            # Create green semi-transparent overlay for road area
            overlay = annotated.copy()
            overlay[road_mask > 0] = cv2.addWeighted(
                overlay[road_mask > 0], 0.7,
                np.full_like(overlay[road_mask > 0], (0, 255, 0)), 0.3,
                0
            )
            annotated = overlay
            
            # Draw road boundary
            contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(annotated, contours, -1, (0, 255, 0), 2)
        
        # Add label
        status = "Two-Stage" if self.use_road_seg else "Single-Stage"
        cv2.putText(annotated, f"{status} Detection", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated


def create_two_stage_detector(pothole_model_path="model/best.pt", 
                              road_model_path="model/road_seg.pt"):
    """
    Factory function to create a two-stage detector.
    
    Args:
        pothole_model_path: Path to pothole detection model
        road_model_path: Path to road segmentation model
        
    Returns:
        TwoStageDetector instance
    """
    return TwoStageDetector(pothole_model_path, road_model_path)
