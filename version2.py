
import os
from pathlib import Path
import rasterio
from rasterio.windows import Window
from PIL import Image
import numpy as np
from tqdm import tqdm
import cv2
import random
import yaml

# --------- CONFIG -----------
INPUT_DIR = r"ts-satfire"
MASK_DIRS = [
    r"ts-satfire",  # same folder as images
]
OUTPUT_DIR = r"cropped-data-newer-yolo"
IMAGES_OUT = Path(OUTPUT_DIR) / "images"
LABELS_OUT = Path(OUTPUT_DIR) / "labels"
SPLIT_DIR = Path(OUTPUT_DIR) / "splits"   # will contain train.txt val.txt test.txt (lists)
os.makedirs(IMAGES_OUT, exist_ok=True)
os.makedirs(LABELS_OUT, exist_ok=True)
os.makedirs(SPLIT_DIR, exist_ok=True)

TILE_SIZE = 1024           # chunking tile size (keeps memory low)
TARGET_SIZE = (1024, 1024) # size to save images for YOLO; set to None to keep native size
RESAMPLE_METHOD = Image.BILINEAR  # resizing method for RGB images
FIRE_CLASS_ID = 0          # YOLO class id for 'Fire'
THRESHOLD_MASK = 1         # threshold > value is considered fire
RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
# ----------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def normalize_array(arr):
    """Normalize to 8-bit safely (per-chunk)."""
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx - mn == 0 or np.isnan(mn) or np.isnan(mx):
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = ((arr - mn) / (mx - mn) * 255.0)
    scaled = np.clip(scaled, 0, 255)
    return scaled.astype(np.uint8)

def read_band_in_chunks(src, band_index):
    width, height = src.width, src.height
    result = np.zeros((height, width), dtype=np.uint8)
    for y in range(0, height, TILE_SIZE):
        for x in range(0, width, TILE_SIZE):
            w = min(TILE_SIZE, width - x)
            h = min(TILE_SIZE, height - y)
            window = Window(x, y, w, h)
            arr = src.read(band_index, window=window)
            arr8 = normalize_array(arr)
            result[y:y+h, x:x+w] = arr8
    return result

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

def mask_to_bboxes(mask_array, min_area=10):
    """
    Convert binary mask to list of bounding boxes (x_min, y_min, x_max, y_max) in pixel coords.
    Uses OpenCV contours.
    """
    # ensure uint8
    m = (mask_array > THRESHOLD_MASK).astype(np.uint8) * 255
    if m.max() == 0:
        return []
    # find contours
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w*h >= min_area:
            bboxes.append((x, y, x + w, y + h))
    return bboxes

def bbox_to_yolo(bbox, img_w, img_h):
    """
    Convert bbox (x_min,y_min,x_max,y_max) to YOLO normalized format:
    class x_center_norm y_center_norm width_norm height_norm
    """
    x_min, y_min, x_max, y_max = bbox
    bw = x_max - x_min
    bh = y_max - y_min
    x_center = x_min + bw / 2.0
    y_center = y_min + bh / 2.0
    # normalize
    return (x_center / img_w, y_center / img_h, bw / img_w, bh / img_h)

# collect tiffs
tiff_files = list(Path(INPUT_DIR).rglob("*.tif"))
print(f"Found {len(tiff_files)} TIFF files to process.")

records = []  # keep track of produced image paths for splitting

for tif in tqdm(tiff_files, desc="Converting TIFFs and generating labels"):
    try:
        with rasterio.open(tif) as src:
            band_count = min(3, src.count)

            # read bands one-by-one to avoid OOM
            channels = []
            for b in range(1, band_count + 1):
                arr8 = read_band_in_chunks(src, b)
                channels.append(arr8)

            # build RGB
            if len(channels) == 1:
                rgb = np.stack([channels[0]]*3, axis=2)
            else:
                while len(channels) < 3:
                    channels.append(channels[-1])
                rgb = np.stack(channels[:3], axis=2)

            # Convert to PIL image (uint8)
            pil = Image.fromarray(rgb)

            # optionally resize to TARGET_SIZE (keeps aspect ratio by default, but we force exact size)
            if TARGET_SIZE is not None:
                pil = pil.resize(TARGET_SIZE, resample=RESAMPLE_METHOD)
                out_w, out_h = TARGET_SIZE
            else:
                out_w, out_h = pil.size

            # save image
            out_image_name = tif.stem + ".jpg"
            out_label_name = tif.stem + ".txt"
            out_image_path = IMAGES_OUT / out_image_name
            out_label_path = LABELS_OUT / out_label_name

            # save as JPG
            pil.save(out_image_path, format="JPEG", quality=85)

            # ensure a .txt exists for every image (empty by default)
            open(out_label_path, "w").close()

            # locate mask
            mask_path = find_mask_for_image(tif)
            if mask_path is None:
                # no mask found → keep empty label file (NoFire)
                records.append((str(out_image_path), False))
                continue

            # open mask and read first band, resample to image size if needed
            with rasterio.open(mask_path) as msrc:
                mask_band = msrc.read(1)  # read first band
                # If sizes differ, resample mask to image pixel dims
                mask_uint8 = (mask_band > 0).astype(np.uint8) * 255
                if (msrc.width, msrc.height) != (src.width, src.height) or TARGET_SIZE is not None:
                    # resize to out_w/out_h using nearest to preserve binary classes
                    mask_resized = cv2.resize(mask_uint8, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                else:
                    mask_resized = mask_uint8

            # compute bboxes on mask_resized
            bboxes = mask_to_bboxes(mask_resized, min_area=10)

            if len(bboxes) == 0:
                # no fire pixels → keep empty label
                records.append((str(out_image_path), False))
            else:
                # write YOLO .txt (overwrite the empty file)
                with open(out_label_path, "w") as f:
                    for bbox in bboxes:
                        xcn, ycn, wn, hn = bbox_to_yolo(bbox, out_w, out_h)
                        # clamp to [0,1]
                        xcn = float(np.clip(xcn, 0.0, 1.0))
                        ycn = float(np.clip(ycn, 0.0, 1.0))
                        wn = float(np.clip(wn, 0.0, 1.0))
                        hn = float(np.clip(hn, 0.0, 1.0))
                        f.write(f"{FIRE_CLASS_ID} {xcn:.6f} {ycn:.6f} {wn:.6f} {hn:.6f}\n")
                records.append((str(out_image_path), True))

    except Exception as e:
        print(f"ERROR processing {tif}: {e}")
        # continue to next file

# --- Create train/val/test splits (file lists with absolute or relative paths)
image_paths = [r[0] for r in records]
random.shuffle(image_paths)

n = len(image_paths)
n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)
n_test = n - n_train - n_val

train_list = image_paths[:n_train]
val_list = image_paths[n_train:n_train+n_val]
test_list = image_paths[n_train+n_val:]

def write_list(lst, path):
    with open(path, "w") as f:
        for p in lst:
            f.write(f"{p}\n")

write_list(train_list, SPLIT_DIR / "train.txt")
write_list(val_list, SPLIT_DIR / "val.txt")
write_list(test_list, SPLIT_DIR / "test.txt")  # created and filled, but NOT referenced in data.yaml

# --- create data.yaml for YOLOv11 training (only train & val)
data_yaml = {
    'path': str(Path(OUTPUT_DIR).absolute()),  # base dataset path
    'train': str((Path(SPLIT_DIR) / "train.txt").absolute()),
    'val': str((Path(SPLIT_DIR) / "val.txt").absolute()),
    'nc': 1,
    'names': {0: 'fire'}
}

with open(Path(OUTPUT_DIR) / "data.yaml", "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print("Done. Images saved to:", IMAGES_OUT)
print("Labels saved to:", LABELS_OUT)
print("Splits saved to:", SPLIT_DIR)
print("YOLO data file (train/val only):", Path(OUTPUT_DIR) / "data.yaml")
