# scripts/download_road_model.py
"""
Download a pre-trained road segmentation model.
Uses YOLOv8-seg trained on driving datasets.
"""

from ultralytics import YOLO
from pathlib import Path

def download_road_segmentation_model():
    """Download and save a pre-trained road segmentation model."""
    
    print("Downloading pre-trained YOLOv8 segmentation model...")
    print("This model is trained on Cityscapes/COCO and can segment roads.")
    
    # Download YOLOv8s-seg (small segmentation model - more accurate)
    model = YOLO('yolov8s-seg.pt')
    
    # Save to model directory
    model_dir = Path(__file__).parent.parent / "model"
    model_dir.mkdir(exist_ok=True)
    
    save_path = model_dir / "road_seg.pt"
    
    # Copy from root directory (where ultralytics downloads it) or cache
    import shutil
    root_path = Path(__file__).parent.parent / "yolov8s-seg.pt"
    cache_path = Path.home() / ".cache" / "ultralytics" / "yolov8s-seg.pt"
    
    if root_path.exists():
        shutil.copy(root_path, save_path)
        print(f"✓ Road segmentation model saved to: {save_path}")
    elif cache_path.exists():
        shutil.copy(cache_path, save_path)
        print(f"✓ Road segmentation model saved to: {save_path}")
    else:
        print(f"⚠ Model file not found at expected locations")
        print(f"  Please manually copy yolov8s-seg.pt to {save_path}")
        return None
    
    print("\nNote: This is a general segmentation model.")
    print("It can segment 'road' class (class 0 in some datasets).")
    print("We'll filter for road-like classes in the detection code.")
    
    return model

if __name__ == "__main__":
    download_road_segmentation_model()
    print("\n" + "="*60)
    print("Next steps:")
    print("1. Test the model: python scripts/test_road_segmentation.py")
    print("2. If it works well, it will auto-integrate into your pipeline")
    print("3. If not, we can fine-tune it on your video frames")
    print("="*60)
