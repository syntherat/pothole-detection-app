"""
Dataset Organization Script
Organizes new pothole dataset with train/val/test split (70/15/15)
"""
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

random.seed(42)

# Paths
SOURCE_DIR = Path(r"C:\Users\palso\OneDrive\Desktop\VITB\epics\Pothole Dataset")
TARGET_DIR = Path(__file__).resolve().parent.parent / "data" / "dataset_v2"

# Create directory structure
SPLITS = {
    'train': 0.70,
    'val': 0.15,
    'test': 0.15
}

def create_directory_structure():
    """Create YOLO format directory structure"""
    for split in SPLITS.keys():
        (TARGET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (TARGET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    print(f"Created directory structure at {TARGET_DIR}")

def get_image_label_pairs(source_dir):
    """Find all image-label pairs"""
    pairs = []
    for img_file in source_dir.glob("*.jpg"):
        label_file = img_file.with_suffix('.txt')
        if label_file.exists():
            # Validate label file has content
            if label_file.stat().st_size > 0:
                pairs.append((img_file, label_file))
    return pairs

def split_dataset(pairs):
    """Split dataset into train/val/test"""
    random.shuffle(pairs)
    total = len(pairs)
    
    train_end = int(total * SPLITS['train'])
    val_end = train_end + int(total * SPLITS['val'])
    
    splits = {
        'train': pairs[:train_end],
        'val': pairs[train_end:val_end],
        'test': pairs[val_end:]
    }
    
    return splits

def copy_files(splits):
    """Copy files to respective directories"""
    for split_name, pairs in splits.items():
        print(f"\nCopying {split_name} set ({len(pairs)} images)...")
        img_dir = TARGET_DIR / split_name / "images"
        lbl_dir = TARGET_DIR / split_name / "labels"
        
        for img_path, lbl_path in tqdm(pairs):
            # Copy image
            shutil.copy2(img_path, img_dir / img_path.name)
            # Copy label
            shutil.copy2(lbl_path, lbl_dir / lbl_path.name)

def create_yaml_config():
    """Create data.yaml for YOLO training"""
    yaml_content = f"""# Pothole Detection Dataset v2
# {sum(SPLITS.values()) * 100:.0f}% of data used

path: {str(TARGET_DIR.resolve()).replace(chr(92), '/')}
train: train/images
val: val/images
test: test/images

nc: 1
names: ['pothole']

# Dataset statistics will be logged during training
"""
    
    yaml_path = TARGET_DIR / "data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nCreated data.yaml at {yaml_path}")
    return yaml_path

def main():
    print("=" * 60)
    print("POTHOLE DATASET ORGANIZATION")
    print("=" * 60)
    
    # Check source directory
    if not SOURCE_DIR.exists():
        print(f"ERROR: Source directory not found: {SOURCE_DIR}")
        return
    
    print(f"\nSource: {SOURCE_DIR}")
    print(f"Target: {TARGET_DIR}")
    
    # Get all valid pairs
    print("\nScanning for image-label pairs...")
    pairs = get_image_label_pairs(SOURCE_DIR)
    print(f"Found {len(pairs)} valid image-label pairs")
    
    if len(pairs) == 0:
        print("ERROR: No valid pairs found!")
        return
    
    # Create directories
    create_directory_structure()
    
    # Split dataset
    print("\nSplitting dataset...")
    splits = split_dataset(pairs)
    
    for split_name, split_pairs in splits.items():
        percentage = (len(split_pairs) / len(pairs)) * 100
        print(f"  {split_name}: {len(split_pairs)} images ({percentage:.1f}%)")
    
    # Copy files
    copy_files(splits)
    
    # Create YAML config
    yaml_path = create_yaml_config()
    
    print("\n" + "=" * 60)
    print("DATASET ORGANIZATION COMPLETE!")
    print("=" * 60)
    print(f"\nDataset ready at: {TARGET_DIR}")
    print(f"Config file: {yaml_path}")
    print(f"\nNext step: Run training script")
    print("  python scripts/train_model.py")

if __name__ == "__main__":
    main()
