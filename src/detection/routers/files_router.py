import logging
from enum import StrEnum
from io import BytesIO
from pathlib import Path

import cv2
from PIL import Image
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.common.config import settings
from src.common.utils import geotiff_to_jpg

logger = logging.getLogger(__name__)

class FileGroup(StrEnum):
    LIVE = "live"
    RAW = "raw"


def _find_tif_file_path(file_group: FileGroup, filename: str) -> Path:
    """
    Helper function to find the corresponding GeoTiff file for a given filename.
    """
    directory = ""

    match file_group:
        case FileGroup.RAW:
            directory = settings.RAW_IMAGES_DIR
        case FileGroup.LIVE:
            directory = settings.LIVE_IMAGES_DIR
        case _:
            raise ValueError(f"Invalid file group: {file_group}")

    tif_file_path = Path(directory) / f"{filename}.tif"

    if tif_file_path.exists():
        return tif_file_path

    raise FileNotFoundError(f"File {filename} not found in any of the expected directories.")


router = APIRouter()

@router.get("/{file_group}/{filename}")
async def get_file(file_group: FileGroup, filename: str) -> StreamingResponse:
    """
    Retrieve an image file by filename.
    """
    try:
        tif_file_path = _find_tif_file_path(file_group=file_group, filename=Path(filename).name)
        jpg_array = geotiff_to_jpg(tif_file_path)
        image = Image.fromarray(cv2.cvtColor(jpg_array, cv2.COLOR_BGR2RGB))
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        return StreamingResponse(
            content=buffer,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"}
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid file group. Expected '{FileGroup.LIVE.value}' or '{FileGroup.RAW.value}'.")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="File not found.")
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail="Failed to retrieve file.")
