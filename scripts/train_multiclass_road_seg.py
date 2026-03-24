"""
Train a multi-class segmentation model for visible-road masking.

Expected dataset:
    data/visible_road_seg_public/data.yaml

Classes:
  0 visible_road
  1 vehicle
  2 pedestrian
  3 shadow
"""

from __future__ import annotations

import argparse
import ast
import os
import shutil
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
RESULTS_DIR = ROOT / "training_results"
DEFAULT_DATASET = ROOT / "data" / "visible_road_seg_public" / "data.yaml"

MODEL_PRESETS = {
    "nano": "yolo11n-seg.pt",
    "small": "yolo11s-seg.pt",
    "medium": "yolo11m-seg.pt",
}


def read_names_from_data_yaml(data_yaml: Path) -> list[str]:
    """Read class names from Ultralytics data.yaml without extra dependencies."""
    text = data_yaml.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            payload = stripped.split("names:", 1)[1].strip()
            if not payload:
                return []
            try:
                parsed = ast.literal_eval(payload)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                return []
    return []


def default_workers() -> int:
    if os.name == "nt":
        return 0
    return 8


def train(
    model_size: str,
    data_yaml: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str | None,
    workers: int,
) -> Path:
    if model_size not in MODEL_PRESETS:
        raise ValueError(f"Invalid model size: {model_size}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    if device is None:
        device = "0" if torch.cuda.is_available() else "cpu"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"road_seg_multiclass_{model_size}_{ts}"

    model = YOLO(MODEL_PRESETS[model_size])
    
    model.train(
        data=str(data_yaml),
        task="segment",
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(RESULTS_DIR),
        name=run_name,
        optimizer="AdamW",
        lr0=0.003,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        patience=30,
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.3,
        translate=0.1,
        scale=0.35,
        fliplr=0.5,
        mosaic=0.2,
        mixup=0.0,
        copy_paste=0.0,
        workers=workers,
        deterministic=True,
        seed=42,
        verbose=True,
        val=True,
        plots=True,
        cls=3.0,  # High weight on class loss to prioritize road vs non-road distinction
    )

    best = RESULTS_DIR / run_name / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"Training finished but best.pt not found at {best}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = MODEL_DIR / f"road_seg_multiclass_{model_size}_{ts}.pt"
    shutil.copy2(best, archive_path)
    shutil.copy2(best, MODEL_DIR / "road_seg.pt")

    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-class visible-road segmentation model")
    parser.add_argument("--model", default="small", choices=["nano", "small", "medium"], help="Segmentation backbone")
    parser.add_argument("--data", default=str(DEFAULT_DATASET), help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=704, help="Image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default=None, help="Device like '0' or 'cpu'")
    parser.add_argument("--workers", type=int, default=default_workers(), help="Dataloader worker processes")
    args = parser.parse_args()

    data_yaml = Path(args.data).resolve()
    out = train(
        model_size=args.model,
        data_yaml=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
    )

    names = read_names_from_data_yaml(data_yaml)
    print("Classes:")
    if names:
        for idx, name in enumerate(names):
            print(f"  {idx}: {name}")
    else:
        print("  Could not parse names from data.yaml")
    print("\nNOTE: Using cls=3.0 to prioritize road vs non-road accuracy (all 7 classes kept)")
    print(f"\nTraining complete")
    print(f"Archived model: {out}")
    print(f"Active model: {MODEL_DIR / 'road_seg.pt'}")


if __name__ == "__main__":
    main()
