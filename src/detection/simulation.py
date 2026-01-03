import logging
import os
import random

from src.common.models import FireAlertCreationDto, ImageDto, ImagesFilters, ImageSplit
from src.common.remote_client import remote_client
from src.detection.inference import detector

logger = logging.getLogger(__name__)

async def simulation_job() -> None:
    """
    Periodically fetches a random TEST image and triggers an alert.
    """
    logger.info("Starting simulation cycle...")
    try:
        candidate_images = await remote_client.get_images(
            filters=ImagesFilters(splits=[ImageSplit.TEST], containsFire=True)
        )

        if not candidate_images:
            logger.warning("No test images available for simulation.")
            return

        selected_image: ImageDto = random.choice(candidate_images)

        local_path = selected_image["metadata"]["localPath"]
        absolute_file_path = os.path.abspath(local_path)

        detection_result = detector.detect([absolute_file_path])[0]

        if not detection_result["has_fire"]:
            logger.warning(
                f"No fire detected on image with ID {selected_image["id"]}."
                + " This is not expected as the image is marked as containing fire."
            )
            return

        alert_payload = FireAlertCreationDto(
            description=f"SIMULATION: Fire detected on image {selected_image["id"]}",
            confidence=detection_result["confidence"],
            severity=detection_result["severity"],
            latitude=selected_image["metadata"]["latitude"],
            longitude=selected_image["metadata"]["longitude"],
            imageId=selected_image["id"]
        )

        await remote_client.create_fire_alert(alert_payload)

        logger.info(f"Simulation alert sent for image with ID {selected_image["id"]}")

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
