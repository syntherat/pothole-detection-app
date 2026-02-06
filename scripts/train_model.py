"""
Advanced Training Script for Pothole Detection
Includes multiple model sizes, hyperparameter tuning, and detailed logging
"""
import os
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml
from datetime import datetime

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "dataset_v3"
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "training_results"

# Model configurations
MODELS = {
    'nano': 'yolo11n.pt',      # Fastest, less accurate
    'small': 'yolo11s.pt',     # Balanced
    'medium': 'yolo11m.pt',    # More accurate, slower
}

# Hyperparameter presets
HYPERPARAMS = {
    'baseline': {
        'epochs': 100,
        'batch': 16,
        'imgsz': 640,
        'lr0': 0.01,
        'lrf': 0.01,
        'patience': 20,
        'augment': True,
    },
    'aggressive': {
        'epochs': 150,
        'batch': 16,
        'imgsz': 640,
        'lr0': 0.02,
        'lrf': 0.001,
        'patience': 30,
        'augment': True,
        'mosaic': 1.0,
        'mixup': 0.1,
    },
    'conservative': {
        'epochs': 80,
        'batch': 16,
        'imgsz': 640,
        'lr0': 0.005,
        'lrf': 0.01,
        'patience': 15,
        'augment': True,
    }
}

def train_model(model_size='small', hyperparams='baseline', resume=False, device=None):
    """
    Train YOLO model with specified configuration
    
    Args:
        model_size: 'nano', 'small', or 'medium'
        hyperparams: 'baseline', 'aggressive', or 'conservative'
        resume: Resume from last checkpoint
        device: CUDA device (None for auto-detect GPU/CPU, 'cpu' for CPU only)
    """
    
    # Auto-detect GPU if available
    if device is None:
        device = 0 if torch.cuda.is_available() else 'cpu'
        
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"pothole_{model_size}_{hyperparams}_{timestamp}"
    
    print("=" * 70)
    print(f"TRAINING: {exp_name}")
    print("=" * 70)
    
    # Check data config
    data_yaml = DATA_DIR / "data.yaml"
    if not data_yaml.exists():
        print(f"ERROR: Data config not found at {data_yaml}")
        print("Run: python scripts/organize_dataset.py first")
        return None
    
    # Load model
    model_path = MODELS.get(model_size)
    if not model_path:
        print(f"ERROR: Invalid model size '{model_size}'")
        return None
    
    print(f"\nModel: {model_path}")
    print(f"Hyperparameters: {hyperparams}")
    print(f"Device: {device if device else 'auto-detect'}")
    
    model = YOLO(model_path)
    
    # Get hyperparameters
    hp = HYPERPARAMS.get(hyperparams, HYPERPARAMS['baseline'])
    
    # Training arguments
    train_args = {
        'data': str(data_yaml),
        'epochs': hp['epochs'],
        'batch': hp['batch'],
        'imgsz': hp['imgsz'],
        'patience': hp['patience'],
        'save': True,
        'save_period': 10,  # Save every 10 epochs
        'device': device,
        'workers': 8,
        'project': str(RESULTS_DIR),
        'name': exp_name,
        'exist_ok': False,
        'pretrained': True,
        'optimizer': 'AdamW',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'val': True,
        'plots': True,
        'cache': False,  # Set to True if you have enough RAM
        
        # Learning rate
        'lr0': hp['lr0'],
        'lrf': hp['lrf'],
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Augmentation
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': hp.get('mosaic', 1.0),
        'mixup': hp.get('mixup', 0.0),
        'copy_paste': 0.0,
        
        # Loss weights
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # Resume
        'resume': resume,
    }
    
    print("\n" + "-" * 70)
    print("TRAINING CONFIGURATION:")
    print("-" * 70)
    for key, value in train_args.items():
        if key not in ['data', 'project', 'name']:
            print(f"  {key:20s}: {value}")
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING...")
    print("=" * 70 + "\n")
    
    # Train
    try:
        results = model.train(**train_args)
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        
        # Save best model to main model directory
        best_model_src = RESULTS_DIR / exp_name / "weights" / "best.pt"
        if best_model_src.exists():
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            best_model_dest = MODEL_DIR / f"best_{model_size}_{timestamp}.pt"
            import shutil
            shutil.copy2(best_model_src, best_model_dest)
            
            # Also copy as default best.pt
            shutil.copy2(best_model_src, MODEL_DIR / "best.pt")
            
            print(f"\nBest model saved to:")
            print(f"  {best_model_dest}")
            print(f"  {MODEL_DIR / 'best.pt'} (default)")
        
        print(f"\nTraining results: {RESULTS_DIR / exp_name}")
        return exp_name
        
    except Exception as e:
        print(f"\nERROR during training: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='Train YOLO pothole detection model')
    parser.add_argument(
        '--model', 
        type=str, 
        default='small',
        choices=['nano', 'small', 'medium'],
        help='Model size (default: small)'
    )
    parser.add_argument(
        '--hyperparams',
        type=str,
        default='baseline',
        choices=['baseline', 'aggressive', 'conservative'],
        help='Hyperparameter preset (default: baseline)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to train on (default: auto GPU if available, use "cpu" for CPU only)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Train all model sizes with baseline hyperparameters'
    )
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("No GPU detected - training will use CPU (slower)")
    
    print()
    
    if args.all:
        print("Training all model sizes...")
        for model_size in ['nano', 'small', 'medium']:
            print(f"\n{'='*70}")
            print(f"Training {model_size} model...")
            print(f"{'='*70}\n")
            train_model(
                model_size=model_size,
                hyperparams='baseline',
                resume=False,
                device=args.device
            )
    else:
        train_model(
            model_size=args.model,
            hyperparams=args.hyperparams,
            resume=args.resume,
            device=args.device
        )
    
    print("\n" + "=" * 70)
    print("ALL TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nNext step: Evaluate model performance")
    print("  python scripts/evaluate_model.py")

if __name__ == "__main__":
    main()
