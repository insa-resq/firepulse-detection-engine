import os
import logging
from typing import List, TypedDict, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from src.common.config import settings
from src.common.models import FireSeverity
from src.common.utils import geotiff_to_jpg

Polygon = List[float]
DetectionResultItem = TypedDict(
    "DetectionResultItem",
    {
        "has_fire": bool,
        "confidence": Optional[float],
        "severity": Optional[FireSeverity],
        "polygons": List[Polygon]
    }
)
DetectionResult = List[DetectionResultItem]

logger = logging.getLogger(__name__)

class FireDetector:
    def __init__(self) -> None:
        self.model = YOLO(settings.BEST_MODEL_WEIGHTS_PATH)

    def detect(self, images_paths: List[str]) -> DetectionResult:
        """
        Runs YOLO inference. Returns aggregated result for the image.
        """
        for image_path in images_paths:
            if not os.path.isabs(image_path):
                raise Exception(f"Image path {image_path} is not absolute!")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image {image_path} does not exist!")

        logger.info("Converting GeoTIFF images to JPEG format for inference...")

        jpg_images = [
            geotiff_to_jpg(image_path)
            for image_path in images_paths
        ]

        logger.info(f"Running inference on {len(images_paths)} image(s)...")

        results = self.model.predict(
            source=jpg_images,
            task="segment",
            conf=settings.CONFIDENCE_THRESHOLD,
            save=False,
            verbose=False
        )

        logger.info("Inference completed.")

        all_results = []

        for result in results:
            # Negative result
            if result.masks is None:
                all_results.append({
                    "has_fire": False,
                    "confidence": None,
                    "severity": None,
                    "polygons": []
                })
                continue

            max_confidence = 0.0
            worst_severity = FireSeverity.LOW
            all_polygons = []

            # Image dimensions for relative area calculation
            img_height, img_width = result.orig_shape
            img_area = img_height * img_width

            for i, confidence_tensor in enumerate(result.boxes.conf):
                confidence = float(confidence_tensor)
                max_confidence = max(max_confidence, confidence)

                # Get normalized polygon (List of [x, y])
                # result.masks.xyn is a list of arrays
                poly_norm = result.masks.xyn[i]

                # Convert to a flat list for the API [x1, y1, x2, y2...]
                flat_poly = poly_norm.flatten().tolist()
                all_polygons.append(flat_poly)

                # Calculate Exact Fire Area
                # We can use the Polygon area formula (Shoelace formula)
                # or approximate using the mask object if needed.
                # Ultralytics masks have a .data attribute (bitmap), but xyn is faster.

                # Simple approach: Denormalize and calculate area using OpenCV
                # (poly_norm * [w, h])
                denorm_poly = (poly_norm * np.array([img_width, img_height])).astype(np.float32)

                # cv2.contourArea requires float32/int
                fire_area = cv2.contourArea(denorm_poly)

                # Recalculate severity with the exact area
                current_severity = self._calculate_severity(confidence, fire_area, img_area)
                if current_severity > worst_severity:
                    worst_severity = current_severity

            all_results.append({
                "has_fire": True,
                "confidence": max_confidence,
                "severity": worst_severity,
                "polygons": all_polygons
            })

        return all_results

    @staticmethod
    def _calculate_severity(confidence: float, fire_area: float, img_area: float) -> FireSeverity:
        """
        Determines severity based on:
        1. Area ratio: How much of the image is burning?
        2. Confidence: How sure is the model?
        """
        area_ratio = fire_area / img_area
        
        if area_ratio > 0.0005 and confidence > 0.5:
            return FireSeverity.CRITICAL

        if area_ratio > 0.0002:
            if confidence > 0.4:
                return FireSeverity.HIGH
            return FireSeverity.MEDIUM

        if confidence > 0.4:
            return FireSeverity.MEDIUM

        return FireSeverity.LOW


detector = FireDetector()
