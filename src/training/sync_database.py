import asyncio
from pathlib import Path
from typing import List

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
    base_path = Path(settings.PROCESSED_IMAGES_DIR)
    images_to_sync = []

    print(f"Scanning {base_path} for images...")

    absolute_files_paths = [p.resolve() for p in base_path.rglob("*.tif")]

    detection_results = detector.detect(absolute_files_paths)

    for file_path, detection_result in zip(absolute_files_paths, detection_results):
        file_metadata = get_geotiff_metadata(file_path)

        image = ImageCreationDto(
            url=f"{settings.IMAGES_BASE_URL}/raw/{file_path.name}",
            split=get_split_from_path(file_path),
            containsFire=detection_result["has_fire"],
            metadata=ImageMetadata(
                localPath=str(file_path),
                width=file_metadata["width"],
                height=file_metadata["height"],
                latitude=file_metadata["latitude"],
                longitude=file_metadata["longitude"],
                boundingBoxes=detection_result["bounding_boxes"]
            )
        )

        images_to_sync.append(image)

    return images_to_sync


async def sync_images() -> None:
    images_to_sync = get_images_to_sync()

    if images_to_sync:
        print(f"Registering {len(images_to_sync)} images...")

        chunk_size = 100

        for i in range(0, len(images_to_sync), chunk_size):
            chunk = images_to_sync[i:i+chunk_size]
            try:
                await remote_client.register_images(chunk)
                print(f"Batch n°{i // chunk_size + 1} synced.")
            except Exception as e:
                print(f"Error syncing batch n°{i // chunk_size + 1}: {e}")


if __name__ == "__main__":
    asyncio.run(sync_images())
