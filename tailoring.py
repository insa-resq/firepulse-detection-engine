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

# --------- CONFIG -----------
INPUT_DIR = r"../ts-satfire"
MASK_DIRS = [r"../ts-satfire"]
OUTPUT_DIR = r"cropped-data-newer-yolo"

IMAGES_OUT = Path(OUTPUT_DIR) / "images"
LABELS_OUT = Path(OUTPUT_DIR) / "labels"
COORD_OUT = Path(OUTPUT_DIR) / "coordinates"
SPLIT_DIR = Path(OUTPUT_DIR) / "splits"

for p in [IMAGES_OUT, LABELS_OUT, COORD_OUT, SPLIT_DIR]:
    os.makedirs(p, exist_ok=True)

TILE_SIZE = 1024
TARGET_SIZE = (1024, 1024)
RESAMPLE_METHOD = Image.BILINEAR
FIRE_CLASS_ID = 0
THRESHOLD_MASK = 1
RANDOM_SEED = 42

TRAIN_RATIO = 0.8
TEST_RATIO = 0.1
VAL_RATIO = 0.1   # images sans feu vont aussi ici !

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ------------------------------------------
# NORMALISATION
# ------------------------------------------
def normalize_array(arr):
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx - mn == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = ((arr - mn) / (mx - mn) * 255)
    return np.clip(scaled, 0, 255).astype(np.uint8)

def read_band_in_chunks(src, band_index):
    width, height = src.width, src.height
    result = np.zeros((height, width), dtype=np.uint8)
    for y in range(0, height, TILE_SIZE):
        for x in range(0, width, TILE_SIZE):
            win = Window(x, y, min(TILE_SIZE, width - x), min(TILE_SIZE, height - y))
            arr = src.read(band_index, window=win)
            result[y:y+win.height, x:x+win.width] = normalize_array(arr)
    return result

# ------------------------------------------
# FIND MASK
# ------------------------------------------
def find_mask_for_image(image_path: Path):
    """Heuristics: try same-stem masks, or files containing keywords."""
    stem = image_path.stem
    candidates = []

    # check same directory and provided mask dirs
    search_dirs = [image_path.parent] + [Path(d) for d in MASK_DIRS if Path(d).exists()]
    keywords = [stem, 'fire', 'active_fire', 'AF', 'mask', 'burn', 'burned', 'BA']

    for d in search_dirs:
        for f in d.glob("*.tif"):
            fname = f.name.lower()
            # exact stem match or contains any keyword
            if stem.lower() in fname:
                candidates.append(f)
            else:
                for kw in keywords:
                    if kw in fname:
                        candidates.append(f)
                        break

    # choose best candidate (prefer exact stem match)
    exact = [c for c in candidates if c.stem.lower() == stem.lower() or stem.lower() in c.stem.lower()]
    if exact:
        return exact[0]
    if candidates:
        return candidates[0]
    return None

# ------------------------------------------
# MASK → BBOXES
# ------------------------------------------
def mask_to_bboxes(mask, min_area=20):
    binary = (mask > THRESHOLD_MASK).astype(np.uint8)

    if binary.max() == 0:
        return []

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            bboxes.append((x, y, x+w, y+h))
    return bboxes

def bbox_to_yolo(bbox, W, H):
    x_min, y_min, x_max, y_max = bbox
    bw = x_max - x_min
    bh = y_max - y_min
    xc = x_min + bw/2
    yc = y_min + bh/2
    return xc/W, yc/H, bw/W, bh/H

# ------------------------------------------
# GEO COORD OF CENTER
# ------------------------------------------
def get_center_latlon(src):
    transform: Affine = src.transform
    cx = src.width / 2
    cy = src.height / 2
    lon, lat = transform * (cx, cy)
    return lat, lon

# ------------------------------------------
# PROCESS ALL TIFF FILES
# ------------------------------------------
tiff_files = list(Path(INPUT_DIR).rglob("*.tif"))
print(f"Found {len(tiff_files)} TIFF files")

train_items = []
test_items = []
val_items = []

for tif in tqdm(tiff_files, desc="Processing TIFFs"):

    try:
        with rasterio.open(tif) as src:
            # --- read RGB bands ---
            channels = [read_band_in_chunks(src, i) for i in range(1, min(3, src.count)+1)]
            while len(channels) < 3:
                channels.append(channels[-1])

            rgb = np.stack(channels[:3], axis=2)
            pil = Image.fromarray(rgb)

            # Resize
            if TARGET_SIZE:
                pil = pil.resize(TARGET_SIZE, Image.BILINEAR)
            out_w, out_h = pil.size

            # Save image
            out_img = IMAGES_OUT / (tif.stem + ".jpg")
            pil.save(out_img, "JPEG", quality=85)

            # Save geographical center
            lat, lon = get_center_latlon(src)
            with open(COORD_OUT / f"{tif.stem}.txt", "w") as f:
                f.write(f"{lat} {lon}\n")

        # --- Find mask ---
        mask_path = find_mask_for_image(tif)
        out_lbl = LABELS_OUT / (tif.stem + ".txt")

        if mask_path is None:
            # No fire → empty label + put in VAL
            open(out_lbl, "w").close()
            val_items.append(str(out_img))
            continue

        # --- Read mask ---
        with rasterio.open(mask_path) as msrc:
            mask = msrc.read(1)

        # Resize mask to match output image size
        mask_resized = cv2.resize(
            (mask > 0).astype(np.uint8),
            (out_w, out_h),
            interpolation=cv2.INTER_NEAREST
        )

        # Compute bboxes
        bboxes = mask_to_bboxes(mask_resized)

        if len(bboxes) == 0:
            open(out_lbl, "w").close()
            val_items.append(str(out_img))
            continue

        # Write YOLO label
        with open(out_lbl, "w") as f:
            for bbox in bboxes:
                xc, yc, w, h = bbox_to_yolo(bbox, out_w, out_h)
                f.write(f"{FIRE_CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        # fire images are put into train/test/val by ratios
        r = random.random()
        if r < TRAIN_RATIO:
            train_items.append(str(out_img))
        elif r < TRAIN_RATIO + TEST_RATIO:
            test_items.append(str(out_img))
        else:
            val_items.append(str(out_img))

    except Exception as e:
        print("ERROR:", tif, e)
        continue

# ------------------------------------------
# SAVE SPLITS
# ------------------------------------------
def wlist(lst, path):
    with open(path, "w") as f:
        for x in lst:
            f.write(x + "\n")

wlist(train_items, SPLIT_DIR / "train.txt")
wlist(test_items, SPLIT_DIR / "test.txt")
wlist(val_items, SPLIT_DIR / "val.txt")

# ------------------------------------------
# YOLO YAML
# ------------------------------------------
data_yaml = {
    "path": str(Path(OUTPUT_DIR).absolute()),
    "train": str((SPLIT_DIR / "train.txt").absolute()),
    "val": str((SPLIT_DIR / "val.txt").absolute()),
    "test": str((SPLIT_DIR / "test.txt").absolute()),
    "nc": 1,
    "names": {0: "fire"},
}

with open(Path(OUTPUT_DIR) / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f)

print("\nDone!")
print("Images →", IMAGES_OUT)
print("Labels →", LABELS_OUT)
print("Coordinates →", COORD_OUT)
print("Splits →", SPLIT_DIR)
