# scripts/predict_videos.py

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from two_stage_detection import create_two_stage_detector

ROAD_MASK_STRIDE = 10
ROAD_MASK_WIDTH = 800


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_vids = root.parent / "vids"
    default_model = root / "model" / "best.pt"
    default_road_model = root / "model" / "road_seg.pt"
    default_output = root / "output" / "videos"

    parser = argparse.ArgumentParser(description="Run YOLO pothole detection on videos with road segmentation.")
    parser.add_argument("--vids-dir", type=Path, default=default_vids, help="Folder with .mp4 videos")
    parser.add_argument("--model", type=Path, default=default_model, help="Path to pothole model weights (.pt)")
    parser.add_argument("--road-model", type=Path, default=default_road_model, help="Path to road segmentation model (.pt)")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="Output folder for annotated videos")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--vid-stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream results to reduce RAM usage",
    )
    parser.add_argument(
        "--use-road-seg",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use road segmentation to filter false positives",
    )
    parser.add_argument(
        "--show-road-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show road mask overlay in output videos",
    )
    return parser.parse_args()

def _process_video_two_stage(detector, vid: Path, out_dir: Path, args) -> tuple[Path, dict]:
    """Process video using two-stage detection (road segmentation + pothole detection)"""
    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {vid}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / vid.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    frame_idx = 0
    total_potholes = 0
    use_road_seg = args.use_road_seg and detector.use_road_seg
    last_road_mask = None
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        if args.vid_stride > 1 and (frame_idx - 1) % args.vid_stride != 0:
            writer.write(frame)
            continue

        # Two-stage detection
        if use_road_seg:
            if frame_idx == 1 or frame_idx % ROAD_MASK_STRIDE == 0 or last_road_mask is None:
                last_road_mask = detector.get_road_mask(frame, lowres_width=ROAD_MASK_WIDTH)
            results = detector.detect_potholes(
                frame,
                conf=args.conf,
                return_mask=False,
                road_mask=last_road_mask,
                lowres_width=ROAD_MASK_WIDTH
            )
            annotated = detector.visualize(frame, results, last_road_mask, show_mask=args.show_road_mask)
        else:
            results = detector.pothole_model(frame, conf=args.conf, verbose=False)[0]
            annotated = results.plot()
        
        total_potholes += len(results.boxes) if results.boxes else 0
        writer.write(annotated)

    cap.release()
    writer.release()
    
    stats = {
        'frames': frame_idx,
        'potholes': total_potholes
    }
    return out_path, stats


def main() -> int:
    args = _parse_args()

    if not args.vids_dir.exists():
        print(f"Videos folder not found: {args.vids_dir}")
        return 1

    videos = sorted(args.vids_dir.glob("*.mp4"))
    if not videos:
        print(f"No .mp4 videos found in: {args.vids_dir}")
        return 1

    if not args.model.exists():
        print(f"Pothole model not found: {args.model}")
        return 1

    # Initialize two-stage detector
    print("Initializing two-stage detector...")
    detector = create_two_stage_detector(
        str(args.model),
        str(args.road_model) if args.road_model.exists() else None
    )
    
    if detector.use_road_seg:
        print("✓ Road segmentation enabled")
    else:
        print("⚠ Road segmentation model not found, using single-stage detection")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "_summary.csv"

    print(f"\nProcessing {len(videos)} videos...")
    print(f"Output directory: {args.output_dir}")
    print("="*60)

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "status", "frames", "potholes", "elapsed_sec"])

        for idx, vid in enumerate(videos, 1):
            print(f"\n[{idx}/{len(videos)}] Processing: {vid.name}")
            out_dir = args.output_dir / vid.stem
            t0 = time.time()
            status = "ok"
            frames = 0
            potholes = 0
            
            try:
                out_path, stats = _process_video_two_stage(detector, vid, out_dir, args)
                frames = stats['frames']
                potholes = stats['potholes']
                print(f"  ✓ Complete: {potholes} potholes in {frames} frames")
            except Exception as exc:
                status = f"error: {exc}"
                print(f"  ✗ Error: {exc}")
                
            elapsed = round(time.time() - t0, 2)
            writer.writerow([vid.name, status, frames, potholes, elapsed])
            f.flush()

    print("\n" + "="*60)
    print(f"✓ Done. Summary saved to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
