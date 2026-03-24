#!/usr/bin/env python
"""
Test the pothole detection logic without GUI
"""
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

root_dir = Path(__file__).resolve().parent
MODEL_DIR = root_dir / "model"
OUTPUT_DIR = root_dir / "output"

print("[INFO] Testing pothole detection with road masking...\n")

# Load models
print("[1] Loading models...")
pothole_model_path = MODEL_DIR / "best.pt"
road_model_path = MODEL_DIR / "road_seg.pt"

if not pothole_model_path.exists():
    print(f"[ERROR] Pothole model not found at {pothole_model_path}")
    sys.exit(1)

pothole_model = YOLO(str(pothole_model_path))
print(f"    ✓ Pothole model loaded: {pothole_model_path.name}")

road_model = None
if road_model_path.exists():
    road_model = YOLO(str(road_model_path))
    print(f"    ✓ Road segmentation model loaded: {road_model_path.name}")
else:
    print(f"    ⚠ Road model not found at {road_model_path}")

# Find a test image
print("\n[2] Looking for test image...")
test_images = list((root_dir / "input").glob("*.jpg")) + \
              list((root_dir / "input").glob("*.png")) + \
              list((root_dir / "data/visible_road_seg_public_full/images/val").glob("*.jpg"))

if not test_images:
    print("[ERROR] No test images found")
    sys.exit(1)

test_image_path = test_images[0]
print(f"    ✓ Found test image: {test_image_path}")

# Load image
frame = cv2.imread(str(test_image_path))
if frame is None:
    print(f"[ERROR] Could not load image: {test_image_path}")
    sys.exit(1)

h, w = frame.shape[:2]
print(f"    Image size: {w}x{h}")

# Get road mask
print("\n[3] Extracting road segmentation mask...")
road_mask = None
if road_model:
    results = road_model.predict(frame, conf=0.5, verbose=False)
    if results and results[0].masks:
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for mask in results[0].masks.data:
            mask_resized = cv2.resize(
                mask.cpu().numpy().astype(np.uint8),
                (w, h),
                interpolation=cv2.INTER_NEAREST
            )
            combined_mask = np.maximum(combined_mask, mask_resized * 255)
        road_mask = combined_mask
        road_pixels = np.count_nonzero(road_mask)
        print(f"   ✓ Road mask created: {road_pixels} pixels on road ({100*road_pixels/(h*w):.1f}%)")

# Run pothole detection
print("\n[4] Running pothole detection...")
results = pothole_model.predict(frame, conf=0.35, verbose=False)

if results and results[0].boxes:
    total_detections = len(results[0].boxes)
    road_detections = 0
    
    for box, class_id in zip(results[0].boxes.xyxy, results[0].boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Check if inside road
        if road_mask is not None and road_mask[center_y, center_x] > 0:
            road_detections += 1
    
    print(f"    ✓ Total detections: {total_detections}")
    if road_mask is not None:
        print(f"    ✓ Detections on road: {road_detections} ({100*road_detections/total_detections:.1f}%)")
else:
    print("    No detections found")

# Save example result
print("\n[5] Creating visual result...")
display = frame.copy()

if results and results[0].boxes and road_mask is not None:
    for box, class_id in zip(results[0].boxes.xyxy, results[0].boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Only draw if on road
        if road_mask[center_y, center_x] > 0:
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Add road overlay
if road_mask is not None:
    mask_colored = cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR)
    mask_colored[:, :, 0] = 0
    mask_colored[:, :, 2] = 0
    display = cv2.addWeighted(display, 0.8, mask_colored, 0.2, 0)

output_path = OUTPUT_DIR / "test_result.jpg"
cv2.imwrite(str(output_path), display)
print(f"    ✓ Result saved: {output_path}")

print("\n[SUCCESS] Test completed successfully!")
print(f"\nTo use the GUI app, run:")
print(f"  cd {root_dir}")
print(f"  {sys.executable} run_app.py")
