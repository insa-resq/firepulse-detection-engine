import os
from pathlib import Path
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
import random
import yaml

# ============================================================
#                      CONFIGURATION
# ============================================================
INPUT_DIR = "../ts-satfire"
MASK_DIRS = ["../ts-satfire"]          # where to look for masks
OUTPUT_DIR = "../dataset-yolo"

TILE_SIZE = 1024
TARGET_SIZE = (1024, 1024)
RESAMPLE_METHOD = Image.BILINEAR

FIRE_CLASS_ID = 0
THRESHOLD_MASK = 0               # mask > this → fire pixel
MIN_BBOX_AREA = 20

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Create output dirs
IMAGES_OUT = Path(OUTPUT_DIR) / "images"
LABELS_OUT = Path(OUTPUT_DIR) / "labels"
SPLIT_DIR = Path(OUTPUT_DIR) / "splits"
for d in [IMAGES_OUT, LABELS_OUT, SPLIT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
#                   UTILITY FUNCTIONS
# ============================================================

def normalize_array(arr):
    """Normalize band to uint8 safely."""
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx - mn == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = (arr - mn) / (mx - mn) * 255
    return np.clip(scaled, 0, 255).astype(np.uint8)

def read_band_in_chunks(src, band_index):
    """Read one band using small memory tiles."""
    w, h = src.width, src.height
    result = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, TILE_SIZE):
        for x in range(0, w, TILE_SIZE):
            win = Window(x, y, min(TILE_SIZE, w-x), min(TILE_SIZE, h-y))
            arr = src.read(band_index, window=win)
            result[y:y+win.height, x:x+win.width] = normalize_array(arr)
    return result

def find_mask(image_path: Path):
    """
    Find a mask file *that is not the same TIFF file* and which contains
    obvious fire/mask keywords.
    """
    keywords = ["mask", "fire", "burn", "active", "af", "ba"]

    for folder in [image_path.parent] + [Path(p) for p in MASK_DIRS]:
        for f in folder.glob("*.tif"):
            if f == image_path:
                continue  # do NOT treat image as mask
            name = f.name.lower()
            if any(k in name for k in keywords):
                return f
    return None

def mask_to_bboxes(mask):
    """Convert binary mask → list of (xmin, ymin, xmax, ymax)."""
    binary = (mask > THRESHOLD_MASK).astype(np.uint8)
    if binary.max() == 0:
        return []

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w*h >= MIN_BBOX_AREA:
            out.append((x, y, x+w, y+h))
    return out

def bbox_to_yolo(b, W, H):
    x1, y1, x2, y2 = b
    bw, bh = x2-x1, y2-y1
    cx, cy = x1 + bw/2, y1 + bh/2
    return cx/W, cy/H, bw/W, bh/H

# ============================================================
#                     MAIN PROCESSING LOOP
# ============================================================

tiff_files = list(Path(INPUT_DIR).rglob("*.tif"))
print("Found", len(tiff_files), "TIFF files.")

records = []  # (image_path, has_fire)

for tif in tqdm(tiff_files, desc="Processing"):
    try:
        # -----------------------
        # Read image as RGB
        # -----------------------
        with rasterio.open(tif) as src:
            bands = [read_band_in_chunks(src, i) for i in range(1, min(3, src.count)+1)]
            while len(bands) < 3:
                bands.append(bands[-1])

            rgb = np.stack(bands[:3], axis=2)
            pil = Image.fromarray(rgb)

            # Resize
            if TARGET_SIZE:
                pil = pil.resize(TARGET_SIZE, RESAMPLE_METHOD)
            W, H = pil.size

        # Save image
        out_img = IMAGES_OUT / f"{tif.stem}.jpg"
        pil.save(out_img, "JPEG", quality=90)

        # Prepare empty label file
        out_lbl = LABELS_OUT / f"{tif.stem}.txt"
        open(out_lbl, "w").close()

        # -----------------------
        # Load mask if exists
        # -----------------------
        mask_path = find_mask(tif)
        if mask_path is None:
            records.append((str(out_img), False))
            continue

        with rasterio.open(mask_path) as msrc:
            mask_raw = msrc.read(1)

        # Resize mask to match image
        mask_resized = cv2.resize((mask_raw > 0).astype(np.uint8), (W, H), cv2.INTER_NEAREST)

        # -----------------------
        # Extract bounding boxes
        # -----------------------
        bboxes = mask_to_bboxes(mask_resized)

        if len(bboxes) == 0:
            records.append((str(out_img), False))
            continue

        # -----------------------
        # Write YOLO labels
        # -----------------------
        with open(out_lbl, "w") as f:
            for bbox in bboxes:
                xc, yc, w, h = bbox_to_yolo(bbox, W, H)
                f.write(f"{FIRE_CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        records.append((str(out_img), True))

    except Exception as e:
        print("ERROR:", tif, e)
        continue

# ============================================================
#                   TRAIN / VAL / TEST SPLITS
# ============================================================

# Shuffle consistently
random.shuffle(records)

paths = [r[0] for r in records]
N = len(paths)
n_train = int(N * TRAIN_RATIO)
n_val = int(N * VAL_RATIO)
n_test = N - n_train - n_val

train_list = paths[:n_train]
val_list = paths[n_train:n_train+n_val]
test_list = paths[n_train+n_val:]

def write_list(lst, path):
    with open(path, "w") as f:
        for p in lst:
            f.write(p + "\n")

write_list(train_list, SPLIT_DIR / "train.txt")
write_list(val_list, SPLIT_DIR / "val.txt")
write_list(test_list, SPLIT_DIR / "test.txt")

# ============================================================
#                        DATA.YAML
# ============================================================

data_yaml = {
    "path": str(Path(OUTPUT_DIR).absolute()),
    "train": str((SPLIT_DIR/"train.txt").absolute()),
    "val": str((SPLIT_DIR/"val.txt").absolute()),
    "test": str((SPLIT_DIR/"test.txt").absolute()),
    "nc": 1,
    "names": {0: "fire"},
}

with open(Path(OUTPUT_DIR) / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f)

print("\nDONE!")
print("Images:", IMAGES_OUT)
print("Labels:", LABELS_OUT)
print("Splits:", SPLIT_DIR)
