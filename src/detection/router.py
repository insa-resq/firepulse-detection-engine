import os
import shutil
import uuid
from typing import cast, Dict, TypedDict, Optional, Any

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
        "fire_detected": bool,
        "alert_id": Optional[str]
    }
)

router = APIRouter()

@router.get("/")
def root() -> Dict[str, str]:
    return {"message": "Welcome to the Firepulse Detection Engine API"}

@router.get("/health")
def health() -> Dict[str, str]:
    return {"status": "UP"}

@router.post("/run-detection")
async def detect_fire(file: UploadFile = File(...)) -> RunDetectionResponse:
    """
    Upload GeoTIFF -> Detect -> Extract Coords -> Alert
    """
    filename = f"{uuid.uuid4()}.tif"

    file_path = os.path.join(settings.LIVE_IMAGES_PATH, filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, cast(Any, buffer))

    file_metadata = get_geotiff_metadata(file_path)

    detection_result = detector.detect(file_path)

    if not detection_result["has_fire"]:
        return {
            "filename": filename,
            "fire_detected": False,
            "alert_id": None
        }

    try:
        image = ImageCreationDto(
            url=f"{settings.IMAGES_BASE_URL}/live/{filename}",
            width=file_metadata["width"],
            height=file_metadata["height"],
            split=ImageSplit.TEST,
            metadata=ImageMetadata(
                local_path=str(file_path),
                contains_fire=True,
                latitude=file_metadata["latitude"],
                longitude=file_metadata["longitude"]
            )
        )

        registered_images = await remote_client.register_images([image])
        remote_image = registered_images[0]

        alert_payload = FireAlertCreationDto(
            description=f"Fire detected via API run.",
            confidence=detection_result["confidence"],
            severity=detection_result["severity"],
            latitude=remote_image["metadata"]["latitude"],
            longitude=remote_image["metadata"]["longitude"],
            imageId=remote_image["id"]
        )

        alert_response = await remote_client.create_fire_alert(alert_payload)

        return {
            "filename": filename,
            "fire_detected": True,
            "alert_id": alert_response["id"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
