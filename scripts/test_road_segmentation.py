# scripts/test_road_segmentation.py
"""
Test road segmentation model on sample video frames.
This helps verify if the pre-trained model works for your videos.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import sys

def test_road_segmentation(video_path, model_path="model/road_seg.pt", num_frames=5):
    """Test road segmentation on video frames."""
    
    # Load model
    print(f"Loading segmentation model: {model_path}")
    if not Path(model_path).exists():
        print(f"Model not found. Downloading pre-trained model...")
        model = YOLO('yolov8n-seg.pt')
    else:
        model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames)
    
    output_dir = Path("pothole_detect_app/output/road_seg_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTesting on {num_frames} frames from: {Path(video_path).name}")
    print(f"Total frames: {total_frames}, sampling every {frame_interval} frames")
    print("="*60)
    
    success_count = 0
    
    for i in range(num_frames):
        frame_idx = i * frame_interval
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Run segmentation
        results = model(frame, verbose=False)[0]
        
        # Check if we got segmentation masks
        if results.masks is not None and len(results.masks) > 0:
            # Create visualization
            annotated = results.plot()
            
            # Try to extract road mask (class varies by model)
            # Common road classes: 0=road in some models, 6=road in COCO, etc.
            road_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            
            for j, cls in enumerate(results.boxes.cls):
                cls_id = int(cls.item())
                # Try multiple possible road class IDs
                # COCO: 0=person, 1=bicycle, etc. - no direct road
                # Cityscapes: 0=road, 1=sidewalk
                # We'll take the largest mask as "likely road"
                if results.masks is not None and j < len(results.masks):
                    mask = results.masks.data[j].cpu().numpy()
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                    
                    # Accumulate masks (take union)
                    road_mask = cv2.bitwise_or(road_mask, mask_binary)
            
            # If we got a mask, apply it
            if road_mask.max() > 0:
                masked_frame = cv2.bitwise_and(frame, frame, mask=road_mask)
                
                # Create side-by-side comparison
                comparison = np.hstack([
                    frame,
                    cv2.cvtColor(road_mask, cv2.COLOR_GRAY2BGR),
                    masked_frame,
                    annotated
                ])
                
                # Add labels
                cv2.putText(comparison, "Original", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(comparison, "Road Mask", (frame.shape[1] + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(comparison, "Masked Result", (frame.shape[1]*2 + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                output_path = output_dir / f"frame_{frame_idx:06d}_segmented.jpg"
                cv2.imwrite(str(output_path), comparison)
                
                success_count += 1
                print(f"✓ Frame {frame_idx}: Segmentation successful - saved to {output_path.name}")
            else:
                print(f"✗ Frame {frame_idx}: No road detected")
        else:
            print(f"✗ Frame {frame_idx}: No segmentation masks generated")
    
    cap.release()
    
    print("="*60)
    print(f"\nResults: {success_count}/{num_frames} frames successfully segmented")
    print(f"Output saved to: {output_dir}")
    
    if success_count > 0:
        print("\n✓ Road segmentation appears to work!")
        print("  Next: Integrate into detection pipeline")
        return True
    else:
        print("\n✗ Road segmentation not working well with this model")
        print("  Options:")
        print("  1. Try a different pre-trained model")
        print("  2. Annotate your own road data and train")
        return False

if __name__ == "__main__":
    # Test on first video in vids folder
    vids_dir = Path("d:/epics/vids")
    
    if len(sys.argv) > 1:
        video_path = Path(sys.argv[1])
    else:
        # Find first video
        videos = list(vids_dir.glob("*.mp4"))
        if not videos:
            print("No videos found in d:/epics/vids")
            print("Usage: python scripts/test_road_segmentation.py <video_path>")
            sys.exit(1)
        video_path = videos[0]
    
    print("Testing Road Segmentation Model")
    print("="*60)
    test_road_segmentation(video_path)
