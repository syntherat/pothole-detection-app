import os, random, shutil
from glob import glob
from lxml import etree
from tqdm import tqdm

random.seed(42)

# paths
ROOT = os.path.dirname(__file__)
RAW_IMG_DIR = os.path.join(ROOT, "raw", "images")
RAW_XML_DIR = os.path.join(ROOT, "raw", "annotations")
YOLO_IMG_TRAIN = os.path.join(ROOT, "yolo", "images", "train")
YOLO_IMG_VAL   = os.path.join(ROOT, "yolo", "images", "val")
YOLO_LBL_TRAIN = os.path.join(ROOT, "yolo", "labels", "train")
YOLO_LBL_VAL   = os.path.join(ROOT, "yolo", "labels", "val")

for d in [YOLO_IMG_TRAIN, YOLO_IMG_VAL, YOLO_LBL_TRAIN, YOLO_LBL_VAL]:
    os.makedirs(d, exist_ok=True)

# helper: convert one xml to YOLO lines
def voc_xml_to_yolo_lines(xml_path, class_map={"pothole":0}):
    tree = etree.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    w = float(size.findtext("width"))
    h = float(size.findtext("height"))

    lines = []
    for obj in root.findall("object"):
        cls = obj.findtext("name").strip()
        if cls not in class_map:
            # skip unknown classes (dataset is single-class but this keeps it safe)
            continue
        cid = class_map[cls]
        bnd = obj.find("bndbox")
        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))

        # convert VOC -> YOLO (normalized)
        x_center = ((xmin + xmax) / 2.0) / w
        y_center = ((ymin + ymax) / 2.0) / h
        box_w    = (xmax - xmin) / w
        box_h    = (ymax - ymin) / h

        # clamp to [0,1] just in case
        def clamp(z): return max(0.0, min(1.0, z))
        x_center, y_center, box_w, box_h = map(clamp, (x_center, y_center, box_w, box_h))

        lines.append(f"{cid} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
    return lines

# collect base names present in both folders
xmls = sorted(glob(os.path.join(RAW_XML_DIR, "*.xml")))
pairs = []
for xp in xmls:
    base = os.path.splitext(os.path.basename(xp))[0]
    # try common image extensions
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        ip = os.path.join(RAW_IMG_DIR, base + ext)
        if os.path.exists(ip):
            pairs.append((ip, xp))
            break

print(f"Found {len(pairs)} image-xml pairs.")

# split 80/20
random.shuffle(pairs)
split_idx = int(0.8 * len(pairs))
train_pairs = pairs[:split_idx]
val_pairs   = pairs[split_idx:]

def write_split(pairs, img_out_dir, lbl_out_dir):
    for img_path, xml_path in tqdm(pairs):
        lines = voc_xml_to_yolo_lines(xml_path)
        # skip images with no boxes to avoid training warnings
        if not lines:
            continue

        # copy image
        dst_img = os.path.join(img_out_dir, os.path.basename(img_path))
        shutil.copy2(img_path, dst_img)

        # write label
        lbl_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        dst_lbl = os.path.join(lbl_out_dir, lbl_name)
        with open(dst_lbl, "w") as f:
            f.write("\n".join(lines))

write_split(train_pairs, YOLO_IMG_TRAIN, YOLO_LBL_TRAIN)
write_split(val_pairs,   YOLO_IMG_VAL,   YOLO_LBL_VAL)

print("Done. YOLO images/labels written to ./yolo/ .")