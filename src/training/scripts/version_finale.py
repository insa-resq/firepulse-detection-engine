import random
import shutil
from pathlib import Path
import numpy as np
import rasterio
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_DIR = Path("../ts-satfire")
OUTPUT_DIR = Path("../dataset-final-test")

IMAGE_SIZE = (1024, 1024)
FIRE_CLASS_ID = 0
MIN_BBOX_AREA = 50

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

THRESHOLD_FACTOR = 4.0  # Seuil adaptatif pour fire_band (non AF)

DEBUG_VIS = True
DEBUG_DIR = OUTPUT_DIR / "debug"
if DEBUG_VIS:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# OUTPUT FOLDERS
# ============================================================

IMAGES_DIR = OUTPUT_DIR / "images"
LABELS_DIR = OUTPUT_DIR / "labels"

for d in [
    IMAGES_DIR / "train", IMAGES_DIR / "val", IMAGES_DIR / "test",
    LABELS_DIR / "train", LABELS_DIR / "val", LABELS_DIR / "test",
]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# UTILITAIRES
# ============================================================

def normalize(band):
    if band.size == 0 or np.all(np.isnan(band)):
        return np.zeros((band.shape[0], band.shape[1]), dtype=np.uint8)
    mn, mx = np.nanmin(band), np.nanmax(band)
    if mx - mn == 0:
        return np.zeros_like(band, dtype=np.uint8)
    return ((band - mn) / (mx - mn) * 255).astype(np.uint8)

def extract_bboxes(mask):
    contours, _ = cv2.findContours((mask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= MIN_BBOX_AREA:
            boxes.append((x, y, x + w, y + h))
    return boxes

def to_yolo(box, width, height):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return cx / width, cy / height, w / width, h / height

def save_debug_combined(date_key, imgs_dict, masks_dict, bboxes_dict, out_dir, af_masks_dict):
    """Plot combiné pour VIIRS DAY / VIIRS NIGHT avec AF mask si présent."""
    keys = ["viirs_day", "viirs_night"]
    n = sum([k in imgs_dict for k in keys])
    if n == 0:
        return

    fig, axes = plt.subplots(2, n, figsize=(5*n, 10))
    if n == 1:
        axes = np.array(axes).reshape(2, 1)

    idx = 0
    for key in keys:
        if key not in imgs_dict:
            continue
        img = np.array(imgs_dict[key]).copy()
        # Draw bounding boxes
        for (x1, y1, x2, y2) in bboxes_dict.get(key, []):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        axes[0, idx].imshow(img)
        axes[0, idx].set_title(key.replace("_", " ").title())
        axes[0, idx].axis('off')

        # Determine mask to show
        mask_to_show = masks_dict.get(key)
        if key == "viirs_day" and key in af_masks_dict and af_masks_dict[key] is not None:
            mask_to_show = af_masks_dict[key]

        if mask_to_show is not None:
            axes[1, idx].imshow(mask_to_show, cmap='hot', alpha=0.6)
            axes[1, idx].set_title("AF Mask" if key=="viirs_day" and key in af_masks_dict and af_masks_dict[key] is not None else "Fire Mask")
        axes[1, idx].axis('off')

        idx += 1

    plt.tight_layout()
    out_path = out_dir / f"{date_key}_debug_combined.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

# ============================================================
# TRAITEMENT DES TIFFS
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

images_with_labels = []
tif_files = list(INPUT_DIR.rglob("*.tif"))
print(f"Found {len(tif_files)} TIFF files")

# Organiser les fichiers par date
files_by_date = {}
for tif in tif_files:
    date_key = tif.stem.split("_")[0]
    files_by_date.setdefault(date_key, []).append(tif)

for date_key, tifs in tqdm(files_by_date.items(), desc="Processing TS-SatFire"):

    imgs_dict = {}
    masks_dict = {}
    bboxes_dict = {}
    af_masks_dict = {}

    for tif in tifs:
        fname = tif.name.lower()
        if any(x in fname for x in ["esri_lulc"]):
            continue

        with rasterio.open(tif) as src:
            fire_band_af = None
            water_mask = None
            fire_band = None

            if "viirs_day" in fname:
                if src.count >= 7:
                    fire_band_af = src.read(7).astype(np.float32)
                    if np.all(np.isnan(fire_band_af)) or fire_band_af.size == 0:
                        fire_band = src.read(6).astype(np.float32)
                    else:
                        fire_band = fire_band_af.copy()
                else:
                    if src.count >= 4:
                        i_nir = src.read(2).astype(np.float32)
                        i_mir = src.read(4).astype(np.float32)
                        fire_band = i_mir - i_nir
                    else:
                        continue
                rgb = np.stack([normalize(src.read(i)) for i in [1,2,3]], axis=2)
                water_mask = np.zeros_like(fire_band, dtype=bool)
                key = "viirs_day"

            elif "viirs_night" in fname:
                fire_band = src.read(1).astype(np.float32)
                band_norm = normalize(fire_band)
                rgb = np.stack([band_norm]*3, axis=2)
                water_mask = np.zeros_like(fire_band, dtype=bool)
                key = "viirs_night"

            else:
                continue  # Ignorer firepred

        if fire_band is None or np.all(np.isnan(fire_band)) or fire_band.size == 0:
            continue

        pil_img = Image.fromarray(rgb).resize(IMAGE_SIZE, Image.BILINEAR)
        W, H = pil_img.size

        # Fire mask
        if fire_band_af is not None and not np.all(np.isnan(fire_band_af)) and fire_band_af.size > 0:
            mask_fire = (fire_band_af > 0).astype(np.uint8)
            af_masks_dict[key] = mask_fire.copy()
        else:
            median_val = np.nanmedian(fire_band)
            std_val = np.nanstd(fire_band)
            threshold = median_val + THRESHOLD_FACTOR * std_val
            mask_fire = (fire_band > threshold).astype(np.uint8)
            af_masks_dict[key] = None

        if water_mask is not None:
            mask_fire[water_mask] = 0

        mask_fire_resized = cv2.resize(mask_fire, (W, H), interpolation=cv2.INTER_NEAREST)
        bboxes = extract_bboxes(mask_fire_resized)
        if not bboxes:
            continue

        imgs_dict[key] = pil_img
        masks_dict[key] = mask_fire_resized
        bboxes_dict[key] = bboxes

        # Save images/labels
        out_img = IMAGES_DIR / f"{tif.stem}.jpg"
        out_lbl = LABELS_DIR / f"{tif.stem}.txt"
        pil_img.save(out_img, "JPEG", quality=90)
        with open(out_lbl, "w") as f:
            for b in bboxes:
                xc, yc, bw, bh = to_yolo(b, W, H)
                f.write(f"{FIRE_CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
        images_with_labels.append(str(out_img))

    if DEBUG_VIS:
        save_debug_combined(date_key, imgs_dict, masks_dict, bboxes_dict, DEBUG_DIR, af_masks_dict)

# ============================================================
# SPLIT TRAIN / VAL / TEST PAR DATE
# ============================================================

# Regrouper les images par date
images_by_date = {}
for img_path in images_with_labels:
    date_key = Path(img_path).stem.split("_")[0]
    images_by_date.setdefault(date_key, []).append(img_path)

# Mélanger les dates
dates = list(images_by_date.keys())
random.shuffle(dates)
N_dates = len(dates)
n_train = int(N_dates * TRAIN_RATIO)
n_val = int(N_dates * VAL_RATIO)

train_dates = dates[:n_train]
val_dates = dates[n_train:n_train+n_val]
test_dates = dates[n_train+n_val:]

splits = {"train": [], "val": [], "test": []}
for date in train_dates:
    splits["train"].extend(images_by_date[date])
for date in val_dates:
    splits["val"].extend(images_by_date[date])
for date in test_dates:
    splits["test"].extend(images_by_date[date])

# Déplacer les fichiers vers les dossiers correspondants
for split, files in splits.items():
    for src_img_path in files:
        name = Path(src_img_path).name
        dst_img = IMAGES_DIR / split / name
        dst_lbl = LABELS_DIR / split / f"{Path(name).stem}.txt"
        src_lbl = LABELS_DIR / f"{Path(name).stem}.txt"
        if not Path(src_img_path).exists() or not Path(src_lbl).exists():
            continue
        shutil.move(src_img_path, dst_img)
        shutil.move(src_lbl, dst_lbl)

print(f"\n✅ SPLIT PAR DATE EFFECTUÉ")
print(f"Train images: {len(splits['train'])}, Val images: {len(splits['val'])}, Test images: {len(splits['test'])}")

# ============================================================
# DATA.YAML
# ============================================================

data_yaml = {
    "path": str(OUTPUT_DIR.resolve()),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": 1,
    "names": ["fire"],
}

with open(OUTPUT_DIR / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f)

print("\n✅ DONE")
