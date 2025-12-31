import logging
import os
import shutil
import uuid
from typing import cast, TypedDict, Optional, Any

from fastapi import APIRouter, UploadFile, File, HTTPException

from src.common.config import settings
from src.common.models import ImageSplit, ImageMetadata, ImageCreationDto, FireAlertCreationDto
from src.common.remote_client import remote_client
from src.common.utils import get_geotiff_metadata
from src.detection.inference import detector

RunDetectionResponse = TypedDict(
    "RunDetectionResponse",
    {
        "filename": str,
        "image_id": str,
        "fire_detected": bool,
        "alert_id": Optional[str]
    }
)

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/run-detection")
async def detect_fire(file: UploadFile = File(...)) -> RunDetectionResponse:
    """
    Upload GeoTIFF -> Detect -> Extract Coords -> Alert
    """
    if file.content_type != "image/tiff":
        raise HTTPException(status_code=400, detail="Invalid file type. Only GeoTIFF files are supported.")

    filename = f"{uuid.uuid4()}.tif"

    logger.info(f"Received file {filename} ({file.size} bytes)")

    file_path = os.path.join(settings.LIVE_IMAGES_DIR, filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, cast(Any, buffer))
        logger.info(f"File {filename} saved to {file_path}")

    file_metadata = get_geotiff_metadata(file_path)

    logger.info(f"File metadata: {file_metadata}")

    absolute_file_path = os.path.abspath(file_path)

    detection_result = detector.detect([absolute_file_path])[0]

    logger.info(f"Detection result: {detection_result}")

    try:
        image = ImageCreationDto(
            url=f"{settings.IMAGES_BASE_URL}/live/{filename}",
            split=ImageSplit.NONE,
            containsFire=detection_result["has_fire"],
            metadata=ImageMetadata(
                localPath=str(file_path),
                width=file_metadata["width"] ,
                height=file_metadata["height"],
                latitude=file_metadata["latitude"],
                longitude=file_metadata["longitude"],
                boundingBoxes=detection_result["bounding_boxes"]
            )
        )

        registered_images = await remote_client.register_images([image])
        remote_image = registered_images[0]

        logger.info(f"Remote image {filename} registered successfully (ID: {remote_image["id"]})")

        if not detection_result["has_fire"]:
            logger.info(f"No fire detected on image {filename}")
            return {
                "filename": filename,
                "image_id": remote_image["id"],
                "fire_detected": False,
                "alert_id": None
            }

        logger.info(f"Fire detected on image {filename}. Creating alert.")

        alert_payload = FireAlertCreationDto(
            description=f"Fire detected via API run.",
            confidence=detection_result["confidence"],
            severity=detection_result["severity"],
            latitude=remote_image["metadata"]["latitude"],
            longitude=remote_image["metadata"]["longitude"],
            imageId=remote_image["id"]
        )

        alert_response = await remote_client.create_fire_alert(alert_payload)

        logger.info(f"Alert sent for image with ID {remote_image["id"]}")

        return {
            "filename": filename,
            "image_id": remote_image["id"],
            "fire_detected": True,
            "alert_id": alert_response["id"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
