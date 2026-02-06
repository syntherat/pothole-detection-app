"""
Merge Original Dataset (665 VOC XML) + New Dataset (1243 YOLO)
Converts VOC XML to YOLO format and combines both datasets
Total: 1908 images for training
"""
import os
import shutil
import random
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml

# Paths
ROOT = Path(__file__).resolve().parent.parent
RAW_IMAGES = ROOT / "data" / "raw" / "images"
RAW_ANNOTATIONS = ROOT / "data" / "raw" / "annotations"
EXISTING_YOLO = ROOT / "data" / "dataset_v2"
OUTPUT_DIR = ROOT / "data" / "dataset_v3"

def voc_to_yolo(xml_file, image_width, image_height):
    """
    Convert VOC XML annotation to YOLO format
    Returns list of [class_id, x_center, y_center, width, height] (normalized 0-1)
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        annotations = []
        for obj in root.findall('object'):
            try:
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)
                
                # Convert to YOLO format (normalized)
                x_center = ((xmin + xmax) / 2) / image_width
                y_center = ((ymin + ymax) / 2) / image_height
                width = (xmax - xmin) / image_width
                height = (ymax - ymin) / image_height
                
                # Clamp to valid range
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                # Class 0 = pothole
                annotations.append([0, x_center, y_center, width, height])
            except Exception as e:
                print(f"  Error parsing object in {xml_file}: {e}")
                continue
        
        return annotations
    except Exception as e:
        print(f"  Error reading XML {xml_file}: {e}")
        return []

def get_image_dimensions(image_path):
    """Get image width and height"""
    try:
        from PIL import Image
        img = Image.open(image_path)
        return img.width, img.height
    except:
        return 640, 480  # Default fallback

def convert_voc_to_yolo():
    """Convert all VOC XML annotations to YOLO format"""
    print("\n" + "="*70)
    print("CONVERTING VOC XML TO YOLO FORMAT")
    print("="*70)
    
    converted = 0
    failed = 0
    
    for xml_file in sorted(RAW_ANNOTATIONS.glob("*.xml")):
        image_name = xml_file.stem + ".png"
        image_path = RAW_IMAGES / image_name
        
        if not image_path.exists():
            print(f"⚠️  Image not found: {image_name}")
            failed += 1
            continue
        
        # Get image dimensions
        width, height = get_image_dimensions(image_path)
        
        # Convert annotations
        annotations = voc_to_yolo(xml_file, width, height)
        
        if not annotations:
            print(f"⚠️  No annotations found in {xml_file.name}")
            failed += 1
            continue
        
        # Save as YOLO txt
        txt_file = RAW_ANNOTATIONS / (xml_file.stem + ".txt")
        with open(txt_file, 'w') as f:
            for ann in annotations:
                f.write(f"{int(ann[0])} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")
        
        converted += 1
        if converted % 100 == 0:
            print(f"  Converted {converted} images...")
    
    print(f"\n✅ Conversion complete: {converted} converted, {failed} failed")
    return converted

def merge_datasets():
    """Merge original dataset (665) + new dataset (1243) = 1908 total"""
    print("\n" + "="*70)
    print("MERGING DATASETS")
    print("="*70)
    
    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "train" / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "val" / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "val" / "labels").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "test" / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "test" / "labels").mkdir(parents=True, exist_ok=True)
    
    # Collect all image-label pairs
    all_pairs = []
    
    # From original dataset (665 images in raw/)
    print("\nScanning original dataset (VOC XML)...")
    for img_file in sorted(RAW_IMAGES.glob("*.png")):
        txt_file = RAW_ANNOTATIONS / (img_file.stem + ".txt")
        if txt_file.exists():
            all_pairs.append((img_file, txt_file))
    
    original_count = len(all_pairs)
    print(f"  Found {original_count} original images with annotations")
    
    # From new dataset (1243 images in dataset_v2)
    print("Scanning new dataset (YOLO format)...")
    for split in ["train", "val", "test"]:
        split_path = EXISTING_YOLO / split
        if split_path.exists():
            for img_file in sorted((split_path / "images").glob("*")):
                txt_file = split_path / "labels" / (img_file.stem + ".txt")
                if txt_file.exists():
                    all_pairs.append((img_file, txt_file))
    
    new_count = len(all_pairs) - original_count
    print(f"  Found {new_count} new dataset images")
    print(f"\n📊 Total pairs: {len(all_pairs)} ({original_count} + {new_count})")
    
    # Shuffle and split: 70% train, 15% val, 15% test
    random.seed(42)
    random.shuffle(all_pairs)
    
    train_count = int(len(all_pairs) * 0.70)
    val_count = int(len(all_pairs) * 0.15)
    
    train_pairs = all_pairs[:train_count]
    val_pairs = all_pairs[train_count:train_count + val_count]
    test_pairs = all_pairs[train_count + val_count:]
    
    # Copy files to new structure
    print("\nCopying files...")
    
    def copy_split(pairs, split_name):
        for img_file, txt_file in pairs:
            # Convert PNG to JPG if needed for consistency
            img_dest = OUTPUT_DIR / split_name / "images" / f"{img_file.stem}.jpg"
            txt_dest = OUTPUT_DIR / split_name / "labels" / f"{img_file.stem}.txt"
            
            # Copy annotation
            shutil.copy2(txt_file, txt_dest)
            
            # Copy image (convert PNG to JPG if needed)
            if img_file.suffix.lower() == ".png":
                from PIL import Image
                img = Image.open(img_file)
                if img.mode == 'RGBA':
                    # Convert RGBA to RGB
                    rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                    rgb_img.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                    rgb_img.save(img_dest, 'JPEG', quality=95)
                else:
                    img.convert('RGB').save(img_dest, 'JPEG', quality=95)
            else:
                shutil.copy2(img_file, img_dest)
        
        print(f"  {split_name}: {len(pairs)} images")
    
    copy_split(train_pairs, "train")
    copy_split(val_pairs, "val")
    copy_split(test_pairs, "test")
    
    print(f"\n✅ Dataset merge complete!")
    print(f"  Train: {len(train_pairs)} images ({len(train_pairs)/len(all_pairs)*100:.1f}%)")
    print(f"  Val:   {len(val_pairs)} images ({len(val_pairs)/len(all_pairs)*100:.1f}%)")
    print(f"  Test:  {len(test_pairs)} images ({len(test_pairs)/len(all_pairs)*100:.1f}%)")
    
    # Create data.yaml
    data_yaml = {
        'path': str(OUTPUT_DIR),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': 1,
        'names': {0: 'pothole'}
    }
    
    with open(OUTPUT_DIR / "data.yaml", 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\n📁 Dataset ready at: {OUTPUT_DIR}")
    return OUTPUT_DIR

if __name__ == "__main__":
    print("\n" + "="*70)
    print("POTHOLE DATASET MERGER")
    print("="*70)
    print(f"Original dataset: {RAW_IMAGES}")
    print(f"New dataset: {EXISTING_YOLO}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Step 1: Convert VOC XML to YOLO
    converted = convert_voc_to_yolo()
    
    if converted > 0:
        # Step 2: Merge datasets
        merge_datasets()
        
        print("\n" + "="*70)
        print("✅ MERGE COMPLETE! Ready to train with dataset_v3")
        print("="*70)
        print("\nNext steps:")
        print("1. Train: python scripts/train_model.py --model medium --hyperparams baseline")
        print("2. Evaluate: python scripts/evaluate_model.py")
    else:
        print("\n❌ No images converted. Check original dataset.")
