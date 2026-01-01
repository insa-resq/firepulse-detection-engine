import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.common.config import settings
from src.common.utils import geotiff_to_jpg, get_geotiff_metadata


SPLIT_RATIOS = { "train": 0.7, "val": 0.2, "test": 0.1 }
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


def create_yolo_label(mask, img_width, img_height):
    """Convert binary mask to YOLO Segmentation Polygons."""

    # Retrieves external contours only
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yolo_lines = []

    for cnt in contours:
        # Filter noise (less than 2 pixels area)
        if cv2.contourArea(cnt) < 2:
            continue

        # Simplify contour slightly to reduce file size (optional but recommended)
        # 0.005 is the epsilon factor - higher means more simplification
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # If simplification reduced it to a line or point, skip
        if len(approx) < 3:
            continue

        # Flatten the array: [[x1, y1], [x2, y2]] -> [x1, y1, x2, y2]
        flattened = approx.flatten().astype(float)

        # Normalize coordinates
        # Even indices (0, 2, 4...) are X, Odd indices (1, 3, 5...) are Y
        flattened[0::2] /= img_width
        flattened[1::2] /= img_height

        # Clamp values to [0, 1] to strictly avoid errors
        flattened = np.clip(flattened, 0, 1)

        # Format string
        poly_str = " ".join([f"{coord:.6f}" for coord in flattened])
        yolo_lines.append(f"0 {poly_str}")

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
        split_images_dir = output_dir / "images" / split_name
        split_labels_dir = output_dir / "labels" / split_name

        split_images_dir.mkdir(parents=True, exist_ok=True)
        split_labels_dir.mkdir(parents=True, exist_ok=True)

        for existing_file in split_images_dir.rglob("*"):
            existing_file.unlink()
        for existing_file in split_labels_dir.rglob("*"):
            existing_file.unlink()

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

            out_name = tif_path.stem

            # Create and save the image
            jpg_image = geotiff_to_jpg(image_path=tif_path)
            cv2.imwrite(
                str(output_dir / "images" / split / f"{out_name}.jpg"),
                jpg_image
            )

            metadata = get_geotiff_metadata(image_path=tif_path)

            # Generate Labels
            raw_mask = metadata["bands"]["label"]
            clean_mask = np.nan_to_num(raw_mask, nan=0.0)
            label_mask = clean_mask.astype(np.uint8)
            yolo_labels = create_yolo_label(label_mask, metadata["width"], metadata["height"])

            # Save Label
            label_out_path = output_dir / "labels" / split / f"{out_name}.txt"
            with open(label_out_path, "w") as f:
                if yolo_labels:
                    f.write("\n".join(yolo_labels))

        except Exception as e:
            errors.append(f"Error processing {tif_path.name}: {e}")

        index += 1

    print("Dataset processed successfully.")

    for error in errors:
        print(error)
