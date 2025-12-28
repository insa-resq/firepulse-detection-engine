import asyncio
import logging
import random

from src.common.models import FireAlertCreationDto
from src.common.remote_client import remote_client
from src.detection.inference import detector

logger = logging.getLogger(__name__)

async def run_simulation_cycle() -> None:
    """
    Periodically fetches a random TEST image and triggers an alert.
    """
    logger.info("Starting simulation cycle...")
    try:
        images = await remote_client.get_test_images()

        if not images:
            logger.warning("No test images available for simulation.")
            return

        candidate_images = list(filter(lambda image: image["metadata"]["contains_fire"], images))

        if not candidate_images:
            logger.warning("No candidate images found for simulation.")
            return

        selected_image = random.choice(candidate_images)

        detection_result = detector.detect(selected_image["local_path"])

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


def simulation_job() -> None:
    """
    Synchronous wrapper for APScheduler
    """
    asyncio.run(run_simulation_cycle())
