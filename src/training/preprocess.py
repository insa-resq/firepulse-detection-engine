import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import rasterio
from tqdm import tqdm

from src.common.config import settings


SPLIT_RATIOS = { "train": 0.7, "val": 0.2, "test": 0.1 }
BAND_MAP = { "I1": 0, "I2": 1, "I4": 3, "LABEL": 6 }
SEED = 42


def create_raw_images(dataset_dir: Path, raw_images_dir: Path) -> None:
    raw_images_dir.mkdir(parents=True, exist_ok=True)

    day_files = list(dataset_dir.rglob("*_Day.tif"))
    total_size = sum(
        file.stat().st_size
        for file in day_files
    )
    print(f"Found {len(day_files)} GeoTIFF files of total size {total_size / 1e9:.2f}GB at {str(dataset_dir.absolute())}.")

    unique_files = set([
        file.name
        for file in day_files
    ])
    print(f"{len(unique_files)} files are unique.")

    files_occurrences = {
        name: [
            file
            for file in day_files
            if file.name == name
        ]
        for name in unique_files
    }
    files_paths = [
        sorted(occurrences, key=lambda o: str(o.absolute()).lstrip(str(dataset_dir.absolute())))[0]
        for name, occurrences in files_occurrences.items()
    ]
    for path in tqdm(files_paths, desc=f"Copying {len(files_paths)} files to {str(raw_images_dir.absolute())}"):
        shutil.copy(path, raw_images_dir)

    print("Raw images created successfully.")


def normalize_band(band):
    """Normalize the 16-bit / float band to 0-255 uint8, handling NaNs."""
    # 1. FIX: Replace NaNs with 0 before any calculation
    band = np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)

    lower = np.percentile(band, 1)
    upper = np.percentile(band, 99)

    # Avoid division by zero if flat image
    if upper == lower:
        return np.zeros_like(band, dtype=np.uint8)

    # Clip to remove outliers
    band = np.clip(band, lower, upper)

    # 2. Normalize and Safe Cast
    normalized = ((band - lower) / (upper - lower) * 255.0)
    return normalized.astype(np.uint8)


def create_yolo_label(mask, img_width, img_height):
    """Convert binary mask to YOLO bounding boxes."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yolo_lines = []

    for cnt in contours:
        if cv2.contourArea(cnt) < 2: continue
        x, y, width, height = cv2.boundingRect(cnt)

        # Normalize (0-1)
        x_center = (x + width / 2) / img_width
        y_center = (y + height / 2) / img_height
        norm_w = width / img_width
        norm_h = height / img_height

        # Ensure values stay within [0, 1] (rare edge case with contours at the border)
        x_center = min(max(x_center, 0), 1)
        y_center = min(max(y_center, 0), 1)
        norm_w = min(norm_w, 1)
        norm_h = min(norm_h, 1)

        yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

    return yolo_lines

if __name__ == "__main__":
    input_dir = Path(settings.RAW_IMAGES_DIR)
    output_dir = Path(settings.PROCESSED_IMAGES_DIR)

    parser = argparse.ArgumentParser()
    parser.add_argument("--create-raw", action="store_true", help="Filter the initial dataset to create the raw dataset")
    parser.add_argument("--dataset-dir", type=str, help="Path to the original dataset")

    args = parser.parse_args()

    if args.create_raw:
        if args.dataset_dir is None:
            print("Please specify the path to the original dataset with '--dataset-dir'.")
            exit(1)

        create_raw_images(dataset_dir=Path(args.dataset_dir), raw_images_dir=input_dir)

    raw_images = list(input_dir.rglob("*.tif"))

    if len(raw_images) == 0:
        print("No GeoTIFFs found in the raw dataset. Please run the script with '--create-raw' to create it.")
        exit(1)

    random.seed(SEED)
    random.shuffle(raw_images)

    total_files = len(raw_images)

    train_count = int(total_files * SPLIT_RATIOS["train"])
    val_count = int(total_files * SPLIT_RATIOS["val"])

    print(f"Train size: {train_count} | Val size: {val_count} | Test size: {total_files - train_count - val_count}")

    for split_name in SPLIT_RATIOS.keys():
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    errors = []
    index = 0

    for tif_path in tqdm(raw_images, desc=f"Processing {len(raw_images)} files into {str(output_dir.absolute())}"):
        try:
            if index < train_count:
                split = "train"
            elif index < train_count + val_count:
                split = "val"
            else:
                split = "test"

            with rasterio.open(tif_path) as src:
                w, h = src.width, src.height

                # Read Bands & Handle NaNs
                r = src.read(BAND_MAP["I4"] + 1)
                g = src.read(BAND_MAP["I2"] + 1)
                b = src.read(BAND_MAP["I1"] + 1)

                raw_mask = src.read(BAND_MAP["LABEL"] + 1)
                clean_mask = np.nan_to_num(raw_mask, nan=0.0)
                label_mask = clean_mask.astype(np.uint8)

                # Create Composite
                img_merged = cv2.merge([
                    normalize_band(b),
                    normalize_band(g),
                    normalize_band(r)
                ])

                # Generate Labels
                yolo_labels = create_yolo_label(label_mask, w, h)

                # --- Save to Split Directory ---
                out_name = tif_path.stem

                # Save Image
                img_out_path = output_dir / "images" / split / f"{out_name}.jpg"
                cv2.imwrite(str(img_out_path), img_merged)

                # Save Label
                lbl_out_path = output_dir / "labels" / split / f"{out_name}.txt"
                with open(lbl_out_path, "w") as f:
                    if yolo_labels:
                        f.write("\n".join(yolo_labels))
        except Exception as e:
            errors.append(f"Error processing {tif_path.name}: {e}")
        index += 1

    print("Dataset processed successfully.")

    for error in errors:
        print(error)
