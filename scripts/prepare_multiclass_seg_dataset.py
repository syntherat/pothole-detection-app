"""
Prepare a YOLO segmentation dataset for visible-road masking from CARLA semantic labels.

Input expected:
  <input>/rgb/*.png|jpg
  <input>/semantic/*.png  (single-channel CARLA semantic class ids)

Output:
  data/visible_road_seg/
    images/{train,val,test}
    labels/{train,val,test}
    data.yaml

Classes:
  0 visible_road
  1 vehicle
  2 pedestrian
  3 shadow
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# CARLA semantic tags (raw ids from semantic camera)
CARLA_ROAD = 7
CARLA_PEDESTRIAN = 4
CARLA_VEHICLE = 10

CLASS_NAMES = ["visible_road", "vehicle", "pedestrian", "shadow"]


def find_contours(mask: np.ndarray, min_area: int) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kept = []
    for c in contours:
        if cv2.contourArea(c) >= min_area and len(c) >= 3:
            kept.append(c)
    return kept


def contour_to_yolo_polygon(contour: np.ndarray, w: int, h: int) -> str:
    pts = contour.reshape(-1, 2)
    norm = []
    for x, y in pts:
        norm.append(f"{x / float(w):.6f}")
        norm.append(f"{y / float(h):.6f}")
    return " ".join(norm)


def estimate_shadow_mask(rgb_bgr: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    # Dark + low saturation regions on road are treated as shadow candidates.
    dark = v < 65
    low_sat = s < 80
    road = road_mask > 0
    shadow = (dark & low_sat & road).astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, kernel)
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, kernel)
    return shadow


def build_label_lines(rgb_bgr: np.ndarray, sem: np.ndarray, min_area: int) -> list[str]:
    h, w = sem.shape[:2]

    road_mask = (sem == CARLA_ROAD).astype(np.uint8) * 255
    vehicle_mask = (sem == CARLA_VEHICLE).astype(np.uint8) * 255
    pedestrian_mask = (sem == CARLA_PEDESTRIAN).astype(np.uint8) * 255
    shadow_mask = estimate_shadow_mask(rgb_bgr, road_mask)

    visible_road = cv2.bitwise_and(road_mask, cv2.bitwise_not(vehicle_mask))
    visible_road = cv2.bitwise_and(visible_road, cv2.bitwise_not(pedestrian_mask))
    visible_road = cv2.bitwise_and(visible_road, cv2.bitwise_not(shadow_mask))

    class_masks = [visible_road, vehicle_mask, pedestrian_mask, shadow_mask]

    lines: list[str] = []
    for cls_id, mask in enumerate(class_masks):
        contours = find_contours(mask, min_area=min_area)
        for contour in contours:
            poly = contour_to_yolo_polygon(contour, w, h)
            if poly:
                lines.append(f"{cls_id} {poly}")

    return lines


def split_items(items: list[Path], train_ratio: float, val_ratio: float) -> tuple[list[Path], list[Path], list[Path]]:
    random.shuffle(items)
    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]
    return train_items, val_items, test_items


def write_yaml(path: Path, root: Path) -> None:
    content = (
        f"path: {root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names: {CLASS_NAMES}\n"
    )
    path.write_text(content, encoding="utf-8")


def process_split(items: list[Path], split: str, rgb_dir: Path, sem_dir: Path, out_root: Path, min_area: int) -> int:
    img_out = out_root / "images" / split
    lbl_out = out_root / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    written = 0
    for rgb_path in items:
        sem_path = sem_dir / f"{rgb_path.stem}.png"
        if not sem_path.exists():
            continue

        rgb = cv2.imread(str(rgb_path))
        sem = cv2.imread(str(sem_path), cv2.IMREAD_GRAYSCALE)
        if rgb is None or sem is None:
            continue
        if rgb.shape[:2] != sem.shape[:2]:
            continue

        lines = build_label_lines(rgb, sem, min_area=min_area)
        if not lines:
            continue

        dst_img = img_out / rgb_path.name
        dst_lbl = lbl_out / f"{rgb_path.stem}.txt"

        shutil.copy2(rgb_path, dst_img)
        dst_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare multi-class road segmentation dataset")
    parser.add_argument("--input", default="data/raw/carla_multiclass", help="Input root with rgb/ and semantic/")
    parser.add_argument("--output", default="data/visible_road_seg", help="Output YOLO-seg dataset root")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-area", type=int, default=80, help="Minimum contour area in pixels")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    input_root = (root / args.input).resolve()
    output_root = (root / args.output).resolve()

    rgb_dir = input_root / "rgb"
    sem_dir = input_root / "semantic"

    if not rgb_dir.exists() or not sem_dir.exists():
        raise FileNotFoundError(f"Expected {rgb_dir} and {sem_dir}")

    rgb_items = sorted([p for p in rgb_dir.glob("*.*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not rgb_items:
        raise RuntimeError("No RGB images found in input/rgb")

    random.seed(args.seed)

    if output_root.exists():
        shutil.rmtree(output_root)

    (output_root / "images").mkdir(parents=True, exist_ok=True)
    (output_root / "labels").mkdir(parents=True, exist_ok=True)

    train_items, val_items, test_items = split_items(rgb_items, args.train_ratio, args.val_ratio)

    n_train = process_split(train_items, "train", rgb_dir, sem_dir, output_root, args.min_area)
    n_val = process_split(val_items, "val", rgb_dir, sem_dir, output_root, args.min_area)
    n_test = process_split(test_items, "test", rgb_dir, sem_dir, output_root, args.min_area)

    write_yaml(output_root / "data.yaml", output_root)

    print("Dataset prepared")
    print(f"Output: {output_root}")
    print(f"train={n_train}, val={n_val}, test={n_test}")
    print(f"classes={CLASS_NAMES}")


if __name__ == "__main__":
    main()
