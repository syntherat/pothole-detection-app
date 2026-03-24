"""
Build a unified YOLO-seg dataset for visible-road masking from public datasets.

Sources supported:
- Cityscapes (gtFine labelIds + leftImg8bit)
- ACDC (gt_labelIds + rgb_anon)
- IDD Segmentation (gtFine polygons + leftImg8bit)
- Mapillary Vistas v1.2 and v2.0 (indexed label PNG + config)

Class presets:
- core: visible_road, vehicle, pedestrian, shadow
- extended: core + vegetation, roadside_object, road_obstacle
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

PRESETS = {
    "core": ["visible_road", "vehicle", "pedestrian", "shadow"],
    "extended": [
        "visible_road",
        "vehicle",
        "pedestrian",
        "shadow",
        "vegetation",
        "roadside_object",
        "road_obstacle",
    ],
}

# Cityscapes-style ids used by Cityscapes and ACDC.
CITYSCAPES_ROAD_IDS = {7}
CITYSCAPES_VEHICLE_IDS = {26, 27, 28, 29, 30, 31, 32, 33}
CITYSCAPES_PEDESTRIAN_IDS = {24, 25}
CITYSCAPES_VEGETATION_IDS = {21, 22}
CITYSCAPES_ROADSIDE_OBJECT_IDS = {17, 19, 20}
CITYSCAPES_ROAD_OBSTACLE_IDS = {4, 5}

# IDD polygon label keywords (lowercased substring matching).
IDD_ROAD_KEYWORDS = {"road", "drivable", "fallback", "lane"}
IDD_VEHICLE_KEYWORDS = {
    "car", "truck", "bus", "vehicle", "van", "autorickshaw", "auto",
    "motorcycle", "motorbike", "bicycle", "bike", "trailer", "tractor", "train",
}
IDD_PEDESTRIAN_KEYWORDS = {"person", "pedestrian", "rider"}
IDD_VEGETATION_KEYWORDS = {"vegetation", "tree", "bush", "grass", "terrain"}
IDD_ROADSIDE_OBJECT_KEYWORDS = {
    "pole", "street light", "traffic light", "traffic sign", "guard rail", "barrier",
    "billboard", "fence", "wall", "signboard",
}
IDD_ROAD_OBSTACLE_KEYWORDS = {
    "obstacle", "debris", "garbage", "trash", "bin", "cone", "animal", "construction",
}

# Mapillary class-name keyword matching.
MAP_ROAD_KEYWORDS = {"construction--flat--road", "construction--flat--service-lane"}
MAP_VEHICLE_PREFIXES = {"object--vehicle--", "object--vehicle-group--"}
MAP_PEDESTRIAN_PREFIXES = {"human--person", "human--rider--"}
MAP_VEGETATION_KEYWORDS = {"nature--vegetation", "nature--terrain"}
MAP_ROADSIDE_OBJECT_KEYWORDS = {
    "traffic-light", "traffic-sign", "support--pole", "street-light", "sign-frame",
    "bench", "barrier", "fence", "guard-rail",
}
MAP_ROAD_OBSTACLE_KEYWORDS = {
    "traffic-cone", "trash", "debris", "object--temporary", "construction", "animal",
}


@dataclass
class Sample:
    dataset: str
    split: str  # train | val
    image_path: Path
    ann_path: Path
    ann_type: str  # city_mask | acdc_mask | idd_poly | mapillary_mask


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def mask_to_lines(mask_by_class: list[np.ndarray], min_area: int) -> list[str]:
    lines: list[str] = []
    for cls_id, mask in enumerate(mask_by_class):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < min_area or len(c) < 3:
                continue
            pts = c.reshape(-1, 2)
            h, w = mask.shape[:2]
            coords = []
            for x, y in pts:
                coords.append(f"{float(x) / float(w):.6f}")
                coords.append(f"{float(y) / float(h):.6f}")
            lines.append(f"{cls_id} {' '.join(coords)}")
    return lines


def estimate_shadow_mask(img_bgr: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    dark = v < 65
    low_sat = s < 85
    road = road_mask > 0

    shadow = (dark & low_sat & road).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, kernel)
    shadow = cv2.morphologyEx(shadow, cv2.MORPH_CLOSE, kernel)
    return shadow


def _class_index(class_names: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(class_names)}


def _zeros(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _subtract_visible(
    road: np.ndarray,
    veh: np.ndarray,
    ped: np.ndarray,
    shadow: np.ndarray,
    vegetation: np.ndarray,
    roadside_object: np.ndarray,
    road_obstacle: np.ndarray,
    include_extra: bool,
) -> np.ndarray:
    visible = cv2.bitwise_and(road, cv2.bitwise_not(veh))
    visible = cv2.bitwise_and(visible, cv2.bitwise_not(ped))
    visible = cv2.bitwise_and(visible, cv2.bitwise_not(shadow))
    if include_extra:
        visible = cv2.bitwise_and(visible, cv2.bitwise_not(vegetation))
        visible = cv2.bitwise_and(visible, cv2.bitwise_not(roadside_object))
        visible = cv2.bitwise_and(visible, cv2.bitwise_not(road_obstacle))
    return visible


def build_masks_from_label_ids(label_img: np.ndarray, img_bgr: np.ndarray, class_names: list[str]) -> list[np.ndarray]:
    h, w = label_img.shape[:2]
    idx = _class_index(class_names)
    include_extra = "vegetation" in idx

    road = np.isin(label_img, list(CITYSCAPES_ROAD_IDS)).astype(np.uint8) * 255
    veh = np.isin(label_img, list(CITYSCAPES_VEHICLE_IDS)).astype(np.uint8) * 255
    ped = np.isin(label_img, list(CITYSCAPES_PEDESTRIAN_IDS)).astype(np.uint8) * 255
    shadow = estimate_shadow_mask(img_bgr, road)

    vegetation = np.isin(label_img, list(CITYSCAPES_VEGETATION_IDS)).astype(np.uint8) * 255 if include_extra else _zeros(h, w)
    roadside_object = np.isin(label_img, list(CITYSCAPES_ROADSIDE_OBJECT_IDS)).astype(np.uint8) * 255 if include_extra else _zeros(h, w)
    road_obstacle = np.isin(label_img, list(CITYSCAPES_ROAD_OBSTACLE_IDS)).astype(np.uint8) * 255 if include_extra else _zeros(h, w)

    visible = _subtract_visible(road, veh, ped, shadow, vegetation, roadside_object, road_obstacle, include_extra)

    out = [_zeros(h, w) for _ in class_names]
    out[idx["visible_road"]] = visible
    out[idx["vehicle"]] = veh
    out[idx["pedestrian"]] = ped
    out[idx["shadow"]] = shadow

    if include_extra:
        out[idx["vegetation"]] = vegetation
        out[idx["roadside_object"]] = roadside_object
        out[idx["road_obstacle"]] = road_obstacle

    return out


def build_masks_from_idd_polygons(
    json_path: Path,
    img_shape: tuple[int, int, int],
    class_names: list[str],
) -> list[np.ndarray]:
    h, w = img_shape[:2]
    idx = _class_index(class_names)
    include_extra = "vegetation" in idx

    road = _zeros(h, w)
    veh = _zeros(h, w)
    ped = _zeros(h, w)
    vegetation = _zeros(h, w)
    roadside_object = _zeros(h, w)
    road_obstacle = _zeros(h, w)

    data = json.loads(json_path.read_text(encoding="utf-8"))
    objects = data.get("objects", [])

    for obj in objects:
        if obj.get("deleted", 0) == 1:
            continue

        label = str(obj.get("label", "")).lower()
        polygon = obj.get("polygon", [])
        if len(polygon) < 3:
            continue

        pts = np.array([[int(round(p[0])), int(round(p[1]))] for p in polygon], dtype=np.int32)
        if pts.shape[0] < 3:
            continue

        target = None
        if any(k in label for k in IDD_ROAD_KEYWORDS):
            target = road
        elif any(k in label for k in IDD_VEHICLE_KEYWORDS):
            target = veh
        elif any(k in label for k in IDD_PEDESTRIAN_KEYWORDS):
            target = ped
        elif include_extra and any(k in label for k in IDD_VEGETATION_KEYWORDS):
            target = vegetation
        elif include_extra and any(k in label for k in IDD_ROADSIDE_OBJECT_KEYWORDS):
            target = roadside_object
        elif include_extra and any(k in label for k in IDD_ROAD_OBSTACLE_KEYWORDS):
            target = road_obstacle

        if target is not None:
            cv2.fillPoly(target, [pts], 255)

    shadow = _zeros(h, w)
    out = [_zeros(h, w) for _ in class_names]
    out[idx["visible_road"]] = road
    out[idx["vehicle"]] = veh
    out[idx["pedestrian"]] = ped
    out[idx["shadow"]] = shadow

    if include_extra:
        out[idx["vegetation"]] = vegetation
        out[idx["roadside_object"]] = roadside_object
        out[idx["road_obstacle"]] = road_obstacle

    return out


def mapillary_ids_from_config(config_path: Path, class_names: list[str]) -> dict[str, set[int]]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    labels = cfg.get("labels", [])

    out = {
        "road": set(),
        "vehicle": set(),
        "pedestrian": set(),
        "vegetation": set(),
        "roadside_object": set(),
        "road_obstacle": set(),
    }

    include_extra = "vegetation" in class_names

    for cls_id, item in enumerate(labels):
        name = str(item.get("name", "")).lower()

        if name in MAP_ROAD_KEYWORDS:
            out["road"].add(cls_id)

        if any(name.startswith(p) for p in MAP_VEHICLE_PREFIXES):
            out["vehicle"].add(cls_id)

        if name == "human--person" or name.startswith("human--rider--"):
            out["pedestrian"].add(cls_id)

        if include_extra:
            if any(k in name for k in MAP_VEGETATION_KEYWORDS):
                out["vegetation"].add(cls_id)
            if any(k in name for k in MAP_ROADSIDE_OBJECT_KEYWORDS):
                out["roadside_object"].add(cls_id)
            if any(k in name for k in MAP_ROAD_OBSTACLE_KEYWORDS):
                out["road_obstacle"].add(cls_id)

    return out


def build_masks_from_mapillary_label(
    label_img: np.ndarray,
    img_bgr: np.ndarray,
    class_names: list[str],
    map_ids: dict[str, set[int]],
) -> list[np.ndarray]:
    h, w = label_img.shape[:2]
    idx = _class_index(class_names)
    include_extra = "vegetation" in idx

    road = np.isin(label_img, list(map_ids["road"])).astype(np.uint8) * 255
    veh = np.isin(label_img, list(map_ids["vehicle"])).astype(np.uint8) * 255
    ped = np.isin(label_img, list(map_ids["pedestrian"])).astype(np.uint8) * 255
    shadow = estimate_shadow_mask(img_bgr, road)

    vegetation = np.isin(label_img, list(map_ids["vegetation"])).astype(np.uint8) * 255 if include_extra else _zeros(h, w)
    roadside_object = np.isin(label_img, list(map_ids["roadside_object"])).astype(np.uint8) * 255 if include_extra else _zeros(h, w)
    road_obstacle = np.isin(label_img, list(map_ids["road_obstacle"])).astype(np.uint8) * 255 if include_extra else _zeros(h, w)

    visible = _subtract_visible(road, veh, ped, shadow, vegetation, roadside_object, road_obstacle, include_extra)

    out = [_zeros(h, w) for _ in class_names]
    out[idx["visible_road"]] = visible
    out[idx["vehicle"]] = veh
    out[idx["pedestrian"]] = ped
    out[idx["shadow"]] = shadow

    if include_extra:
        out[idx["vegetation"]] = vegetation
        out[idx["roadside_object"]] = roadside_object
        out[idx["road_obstacle"]] = road_obstacle

    return out


def add_cityscapes_samples(root: Path) -> list[Sample]:
    out: list[Sample] = []
    gt_root = root / "gtFine_trainvaltest" / "gtFine"
    img_root = root / "leftImg8bit_trainvaltest" / "leftImg8bit"

    for split in ("train", "val"):
        for ann in gt_root.glob(f"{split}/**/*_gtFine_labelIds.png"):
            rel = ann.relative_to(gt_root / split)
            stem = ann.name.replace("_gtFine_labelIds.png", "")
            img = img_root / split / rel.parent / f"{stem}_leftImg8bit.png"
            if img.exists():
                out.append(Sample("cityscapes", split, img, ann, "city_mask"))
    return out


def add_acdc_samples(root: Path) -> list[Sample]:
    out: list[Sample] = []
    gt_root = root / "gt_trainval" / "gt"
    img_root = root / "rgb_anon_trainvaltest" / "rgb_anon"

    for split in ("train", "val"):
        for ann in gt_root.glob(f"*/{split}/**/*_gt_labelIds.png"):
            rel = ann.relative_to(gt_root)
            cond = rel.parts[0]
            seq = rel.parts[2]
            stem = ann.name.replace("_gt_labelIds.png", "")
            img = img_root / cond / split / seq / f"{stem}_rgb_anon.png"
            if img.exists():
                out.append(Sample("acdc", split, img, ann, "acdc_mask"))
    return out


def add_idd_samples(root: Path) -> list[Sample]:
    out: list[Sample] = []

    for base in [root / "idd-segmentation" / "IDD_Segmentation", root / "idd-20k-II" / "idd20kII"]:
        gt_root = base / "gtFine"
        img_root = base / "leftImg8bit"
        if not gt_root.exists() or not img_root.exists():
            continue

        for split in ("train", "val"):
            for ann in gt_root.glob(f"{split}/**/*_gtFine_polygons.json"):
                rel = ann.relative_to(gt_root / split)
                stem = ann.name.replace("_gtFine_polygons.json", "")
                img = img_root / split / rel.parent / f"{stem}_leftImg8bit.png"
                if img.exists():
                    out.append(Sample("idd", split, img, ann, "idd_poly"))

    return out


def add_mapillary_samples(root: Path, use_v2: bool) -> tuple[list[Sample], Path]:
    out: list[Sample] = []

    # Prefer v2 package when available, fallback to v1 package.
    v2_base = root / "An-ZjB1Zm61yAZG0ozTymz8I8NqI4x0MrYrh26dq7kPgfu8vf9ImrdaOAVOFYbJ2pNAgUnVGBmbue9lTgdBOb5BbKXIpFs0fpYWqACbrQDChAA2fdX0zS9PcHu7fY8c-FOvyBVxPNYNFQuM"
    v1_base = root / "An_o5cmHOsS1VbLdaKx_zfMdi0No5LUpL2htRxMwCjY_bophtOkM0-6yTKB2T2sa0yo1oP086sqiaCjmNEw5d_pofWyaE9LysYJagH8yXw_GZPzK2wfiQ9u4uAKrVcEIrkJiVuTn7JBumrA"

    if use_v2 and v2_base.exists():
        config = v2_base / "config_v2.0.json"
        for split_src, split_dst in (("training", "train"), ("validation", "val")):
            img_dir = v2_base / split_src / "images"
            lbl_dir = v2_base / split_src / "v2.0" / "labels"
            for img in img_dir.glob("*.jpg"):
                ann = lbl_dir / f"{img.stem}.png"
                if ann.exists():
                    out.append(Sample("mapillary", split_dst, img, ann, "mapillary_mask"))
        return out, config

    if v1_base.exists():
        config = v1_base / "config.json"
        for split_src, split_dst in (("training", "train"), ("validation", "val")):
            img_dir = v1_base / split_src / "images"
            lbl_dir = v1_base / split_src / "labels"
            for img in img_dir.glob("*.jpg"):
                ann = lbl_dir / f"{img.stem}.png"
                if ann.exists():
                    out.append(Sample("mapillary", split_dst, img, ann, "mapillary_mask"))
        return out, config

    return out, Path()


def write_data_yaml(out_root: Path, class_names: list[str]) -> None:
    yaml_text = (
        f"path: {out_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        f"names: {class_names}\n"
    )
    (out_root / "data.yaml").write_text(yaml_text, encoding="utf-8")


def limited(samples: list[Sample], max_train: int, max_val: int, seed: int) -> list[Sample]:
    by_split = {"train": [], "val": []}
    for s in samples:
        by_split[s.split].append(s)

    rnd = random.Random(seed)
    out: list[Sample] = []

    for split, limit in (("train", max_train), ("val", max_val)):
        arr = by_split[split]
        rnd.shuffle(arr)
        if limit > 0:
            arr = arr[:limit]
        out.extend(arr)

    return out


def process_sample(
    sample: Sample,
    out_root: Path,
    idx: int,
    class_names: list[str],
    map_ids: dict[str, set[int]],
    min_area: int,
) -> bool:
    img = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    if img is None:
        return False

    if sample.ann_type in {"city_mask", "acdc_mask"}:
        lbl = cv2.imread(str(sample.ann_path), cv2.IMREAD_GRAYSCALE)
        if lbl is None or lbl.shape[:2] != img.shape[:2]:
            return False
        masks = build_masks_from_label_ids(lbl, img, class_names)

    elif sample.ann_type == "idd_poly":
        masks = build_masks_from_idd_polygons(sample.ann_path, img.shape, class_names)

        idx_map = _class_index(class_names)
        shadow = estimate_shadow_mask(img, masks[idx_map["visible_road"]])
        masks[idx_map["shadow"]] = shadow

        include_extra = "vegetation" in idx_map
        vegetation = masks[idx_map["vegetation"]] if include_extra else _zeros(*shadow.shape)
        roadside_obj = masks[idx_map["roadside_object"]] if include_extra else _zeros(*shadow.shape)
        road_obstacle = masks[idx_map["road_obstacle"]] if include_extra else _zeros(*shadow.shape)

        masks[idx_map["visible_road"]] = _subtract_visible(
            masks[idx_map["visible_road"]],
            masks[idx_map["vehicle"]],
            masks[idx_map["pedestrian"]],
            shadow,
            vegetation,
            roadside_obj,
            road_obstacle,
            include_extra,
        )

    elif sample.ann_type == "mapillary_mask":
        lbl = cv2.imread(str(sample.ann_path), cv2.IMREAD_GRAYSCALE)
        if lbl is None or lbl.shape[:2] != img.shape[:2]:
            return False
        masks = build_masks_from_mapillary_label(lbl, img, class_names, map_ids)
    else:
        return False

    lines = mask_to_lines(masks, min_area=min_area)
    if not lines:
        return False

    split = sample.split
    img_out = out_root / "images" / split
    lbl_out = out_root / "labels" / split
    ensure_dir(img_out)
    ensure_dir(lbl_out)

    stem = f"{sample.dataset}_{idx:08d}"
    out_img = img_out / f"{stem}.jpg"
    out_lbl = lbl_out / f"{stem}.txt"

    cv2.imwrite(str(out_img), img)
    out_lbl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True




def _worker(args: tuple) -> bool:
    """Top-level wrapper so ProcessPoolExecutor can pickle the function on Windows."""
    sample, out_root, idx, class_names, map_ids, min_area = args
    return process_sample(sample, out_root, idx, class_names, map_ids, min_area)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare unified visible-road YOLO-seg dataset")
    parser.add_argument("--data-root", required=True, help="Root path containing downloaded datasets")
    parser.add_argument("--output", default="data/visible_road_seg_public", help="Output dataset directory")
    parser.add_argument("--class-preset", choices=["core", "extended"], default="extended")
    parser.add_argument("--use-mapillary-v2", action="store_true", help="Use Mapillary v2.0 package when available")
    parser.add_argument("--max-per-source-train", type=int, default=12000, help="Limit train samples per source (0=all)")
    parser.add_argument("--max-per-source-val", type=int, default=1500, help="Limit val samples per source (0=all)")
    parser.add_argument("--min-area", type=int, default=80, help="Minimum polygon contour area in pixels")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1),
                        help="Parallel worker processes (default: CPU count - 1)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing output directory before building")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    data_root = Path(args.data_root).resolve()
    out_root = (root / args.output).resolve()
    class_names = PRESETS[args.class_preset]

    if out_root.exists():
        if args.overwrite:
            shutil.rmtree(out_root)
        else:
            raise RuntimeError(
                f"Output directory already exists: {out_root}. "
                "Use --overwrite to remove it first or choose a different --output."
            )

    ensure_dir(out_root / "images" / "train")
    ensure_dir(out_root / "images" / "val")
    ensure_dir(out_root / "images" / "test")
    ensure_dir(out_root / "labels" / "train")
    ensure_dir(out_root / "labels" / "val")
    ensure_dir(out_root / "labels" / "test")

    city = add_cityscapes_samples(data_root)
    acdc = add_acdc_samples(data_root)
    idd = add_idd_samples(data_root)
    mapillary, map_config = add_mapillary_samples(data_root, use_v2=args.use_mapillary_v2)

    if not mapillary:
        raise RuntimeError("Mapillary dataset not found under data-root")
    if not map_config.exists():
        raise RuntimeError(f"Mapillary config not found: {map_config}")

    map_ids = mapillary_ids_from_config(map_config, class_names)

    sources = {
        "cityscapes": city,
        "acdc": acdc,
        "idd": idd,
        "mapillary": mapillary,
    }

    # Collect and pre-index all samples across sources.
    all_indexed: list[tuple[int, Sample]] = []
    for src_name, items in sources.items():
        chosen = limited(items, args.max_per_source_train, args.max_per_source_val, args.seed)
        print(f"[{src_name}] discovered={len(items)} selected={len(chosen)}", flush=True)
        for s in chosen:
            all_indexed.append((len(all_indexed), s))

    total = len(all_indexed)
    print(f"\nTotal samples to process: {total}  workers={args.workers}", flush=True)

    worker_args = [
        (sample, out_root, idx, class_names, map_ids, args.min_area)
        for idx, sample in all_indexed
    ]

    written = 0
    skipped = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_worker, a): a[2] for a in worker_args}
        for done_count, fut in enumerate(as_completed(futures), 1):
            if fut.result():
                written += 1
            else:
                skipped += 1
            if done_count % 200 == 0 or done_count == total:
                elapsed = time.time() - t0
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (total - done_count) / rate if rate > 0 else 0
                print(
                    f"  {done_count}/{total}  written={written} skipped={skipped} "
                    f"rate={rate:.1f}/s  eta={eta/60:.1f}min",
                    flush=True,
                )

    elapsed_total = time.time() - t0
    write_data_yaml(out_root, class_names)

    n_train = len(list((out_root / "images" / "train").glob("*.jpg")))
    n_val = len(list((out_root / "images" / "val").glob("*.jpg")))

    print(f"\nDone in {elapsed_total/60:.1f} min")
    print(f"Output: {out_root}")
    print(f"train={n_train}, val={n_val}, total={n_train + n_val}")
    print(f"classes={class_names}")


if __name__ == "__main__":
    main()
