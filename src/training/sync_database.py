import asyncio
from pathlib import Path
from typing import List

from tqdm import tqdm

from src.common.config import settings
from src.common.models import ImageSplit, ImageMetadata, ImageCreationDto
from src.common.remote_client import remote_client
from src.common.utils import get_geotiff_metadata
from src.detection.inference import detector


def get_split_from_path(path: Path) -> ImageSplit:
    parent_folder = path.parent.name
    match parent_folder:
        case "train": return ImageSplit.TRAIN
        case "val": return ImageSplit.VALIDATION
        case "test": return ImageSplit.TEST
    return ImageSplit.TRAIN


def get_images_to_sync() -> List[ImageCreationDto]:
    base_path = Path(settings.PROCESSED_IMAGES_DIR) / "images"
    images_to_sync = []

    print(f"Scanning {base_path} for images...")

    files_paths = list(base_path.rglob("*.jpg"))

    print(f"Found {len(files_paths)} images to process.")

    if len(files_paths) == 0:
        return images_to_sync

    tif_files_paths = [
        Path(settings.RAW_IMAGES_DIR) / f"{file_path.stem}.tif"
        for file_path in files_paths
    ]

    for tif_file_path in tqdm(tif_files_paths, desc="Processing images for sync"):
        detection_result = detector.detect(images_paths=[tif_file_path.resolve()])[0]

        file_metadata = get_geotiff_metadata(image_path=tif_file_path)

        image = ImageCreationDto(
            url=f"{settings.REMOTE_IMAGES_SERVE_BASE_URL}/raw/{tif_file_path.stem}",
            split=get_split_from_path(tif_file_path),
            containsFire=detection_result["has_fire"],
            metadata=ImageMetadata(
                localPath=str(tif_file_path),
                width=file_metadata["width"],
                height=file_metadata["height"],
                latitude=file_metadata["latitude"],
                longitude=file_metadata["longitude"],
                polygons=detection_result["polygons"]
            )
        )

        images_to_sync.append(image)

    return images_to_sync


async def sync_images() -> None:
    images_to_sync = get_images_to_sync()

    if len(images_to_sync) > 0:
        chunk_size = 100

        chunks_range = list(range(0, len(images_to_sync), chunk_size))

        print("Clearing existing remote images...")

        existing_images = await remote_client.get_images()

        errors = []
        for i in tqdm(chunks_range, desc="Clearing existing remote images in batches"):
            chunk = existing_images[i:i+chunk_size]
            try:
                await remote_client.delete_images(imageIds=[image["id"] for image in chunk])
            except Exception as e:
                errors.append(f"  - Error clearing batch n°{i // chunk_size + 1}: {e}")

        if len(errors) > 0:
            print("Errors encountered while clearing existing remote images:")
            for error in errors:
                print(error)
            exit(1)

        print("Existing remote images cleared.")

        print(f"Registering {len(images_to_sync)} remote images...")

        errors = []
        for i in tqdm(chunks_range, desc="Syncing images in batches"):
            chunk = images_to_sync[i:i+chunk_size]
            try:
                await remote_client.register_images(chunk)
            except Exception as e:
                errors.append(f"  - Error syncing batch n°{i // chunk_size + 1}: {e}")

        if len(errors) > 0:
            print("Errors encountered while syncing images:")
            for error in errors:
                print(error)
            exit(1)

        print("Images registered successfully.")

if __name__ == "__main__":
    asyncio.run(sync_images())
