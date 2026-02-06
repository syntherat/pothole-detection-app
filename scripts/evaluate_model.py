"""
Model Evaluation Script
Comprehensive evaluation with mAP, precision, recall, confusion matrix
"""
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

# Project paths
ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data" / "dataset_v2"
EVAL_DIR = ROOT / "evaluation_results"

def evaluate_model(model_path=None, data_yaml=None, conf_threshold=0.25):
    """
    Evaluate YOLO model performance
    
    Args:
        model_path: Path to model weights (.pt file)
        data_yaml: Path to data configuration
        conf_threshold: Confidence threshold for predictions
    """
    
    # Setup
    if model_path is None:
        model_path = MODEL_DIR / "best.pt"
    else:
        model_path = Path(model_path)
    
    if data_yaml is None:
        data_yaml = DATA_DIR / "data.yaml"
    else:
        data_yaml = Path(data_yaml)
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return
    
    if not data_yaml.exists():
        print(f"ERROR: Data config not found at {data_yaml}")
        return
    
    # Create evaluation directory
    eval_name = f"eval_{model_path.stem}"
    eval_path = EVAL_DIR / eval_name
    eval_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"MODEL EVALUATION: {model_path.name}")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Data: {data_yaml}")
    print(f"Confidence: {conf_threshold}")
    print(f"Output: {eval_path}")
    print()
    
    # Load model
    print("Loading model...")
    model = YOLO(str(model_path))
    
    # Validate on test set
    print("\n" + "-" * 70)
    print("RUNNING VALIDATION ON TEST SET")
    print("-" * 70)
    
    results = model.val(
        data=str(data_yaml),
        split='test',
        conf=conf_threshold,
        iou=0.6,
        max_det=300,
        plots=True,
        save_json=True,
        save_hybrid=False,
        project=str(eval_path.parent),
        name=eval_path.name,
    )
    
    # Extract metrics
    metrics = {
        'model': str(model_path),
        'confidence_threshold': conf_threshold,
        'metrics': {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
            'f1_score': 2 * (float(results.box.mp) * float(results.box.mr)) / (float(results.box.mp) + float(results.box.mr) + 1e-6)
        }
    }
    
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"mAP@0.5:     {metrics['metrics']['mAP50']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['metrics']['mAP50-95']:.4f}")
    print(f"Precision:    {metrics['metrics']['precision']:.4f}")
    print(f"Recall:       {metrics['metrics']['recall']:.4f}")
    print(f"F1-Score:     {metrics['metrics']['f1_score']:.4f}")
    
    # Save metrics to JSON
    metrics_file = eval_path / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to: {metrics_file}")
    print(f"Plots saved to: {eval_path}")
    
    # Performance interpretation
    print("\n" + "-" * 70)
    print("PERFORMANCE INTERPRETATION")
    print("-" * 70)
    
    map50 = metrics['metrics']['mAP50']
    if map50 >= 0.90:
        rating = "Excellent"
        comment = "Model performs very well, ready for deployment"
    elif map50 >= 0.80:
        rating = "Good"
        comment = "Model performs well, minor improvements possible"
    elif map50 >= 0.70:
        rating = "Fair"
        comment = "Model is functional but needs improvement"
    elif map50 >= 0.60:
        rating = "Poor"
        comment = "Model needs significant improvement"
    else:
        rating = "Very Poor"
        comment = "Model requires retraining with better data/hyperparameters"
    
    print(f"Overall Rating: {rating}")
    print(f"Comment: {comment}")
    
    # Recommendations
    print("\n" + "-" * 70)
    print("RECOMMENDATIONS")
    print("-" * 70)
    
    precision = metrics['metrics']['precision']
    recall = metrics['metrics']['recall']
    
    if precision < 0.80:
        print("- Low precision: Model has many false positives")
        print("  Solution: Increase confidence threshold or collect more negative samples")
    
    if recall < 0.80:
        print("- Low recall: Model misses many potholes")
        print("  Solution: More training data, better augmentation, or longer training")
    
    if map50 < 0.80:
        print("- Overall performance needs improvement")
        print("  Solution: Try larger model (medium), more epochs, or hyperparameter tuning")
    
    if map50 >= 0.85 and precision >= 0.85 and recall >= 0.85:
        print("- Model performance is strong across all metrics")
        print("  Ready for real-world deployment")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    
    return metrics

def compare_models(model_paths):
    """Compare multiple trained models"""
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    results = []
    for model_path in model_paths:
        if not Path(model_path).exists():
            print(f"Skipping {model_path} - not found")
            continue
        
        print(f"\nEvaluating: {model_path}")
        metrics = evaluate_model(model_path)
        if metrics:
            results.append({
                'name': Path(model_path).stem,
                **metrics['metrics']
            })
    
    if not results:
        print("No valid models to compare")
        return
    
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Model':<20} {'mAP@0.5':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<20} {r['mAP50']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1_score']:>10.4f}")
    
    # Find best model
    best = max(results, key=lambda x: x['mAP50'])
    print(f"\nBest model: {best['name']} (mAP@0.5: {best['mAP50']:.4f})")

def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLO pothole detection model')
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model weights (default: model/best.pt)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to data.yaml (default: data/dataset_v2/data.yaml)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--compare',
        nargs='+',
        help='Compare multiple models (provide paths)'
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(args.compare)
    else:
        evaluate_model(
            model_path=args.model,
            data_yaml=args.data,
            conf_threshold=args.conf
        )

if __name__ == "__main__":
    main()
