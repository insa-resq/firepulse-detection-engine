import os
import logging
from typing import List, TypedDict, Optional

from ultralytics import YOLO

from src.common.config import settings
from src.common.models import FireSeverity

BoundingBox = List[float]
DetectionResultItem = TypedDict(
    "DetectionResultItem",
    {
        "has_fire": bool,
        "confidence": Optional[float],
        "severity": Optional[FireSeverity],
        "bounding_boxes": List[BoundingBox]
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

        logger.info(f"Running inference on {len(images_paths)} image(s)...")

        results = self.model.predict(
            source=images_paths,
            conf=settings.CONFIDENCE_THRESHOLD,
            save=False,
            verbose=False
        )

        logger.info("Inference completed.")

        all_results = []

        for result in results:
            # Negative result
            if len(result.boxes) == 0:
                all_results.append({
                    "has_fire": False,
                    "confidence": None,
                    "severity": None,
                    "bounding_boxes": []
                })
                continue

            max_confidence = 0.0
            worst_severity = FireSeverity.LOW
            all_boxes = []

            # Image dimensions for relative area calculation
            # result.orig_shape is (height, width)
            img_height, img_width = result.orig_shape
            img_area = img_height * img_width

            for box in result.boxes:
                # 1. Extract confidence
                confidence = float(box.conf[0])
                if confidence > max_confidence:
                    max_confidence = confidence

                # 2. Extract bounding box formatted as (x_center, y_center, width, height) to easily calculate the area
                x, y, w, h = box.xywh[0].tolist()
                all_boxes.append([x, y, w, h])

                # 3. Calculate the severity for this specific fire
                box_area = w * h
                severity = self._calculate_severity(confidence, box_area, img_area)

                if severity > worst_severity:
                    worst_severity = severity

            all_results.append({
                "has_fire": True,
                "confidence": max_confidence,
                "severity": worst_severity,
                "bounding_boxes": all_boxes
            })

        return all_results

    @staticmethod
    def _calculate_severity(confidence: float, box_area: float, img_area: float) -> FireSeverity:
        """
        Determines severity based on:
        1. Area ratio: How much of the image is burning?
        2. Confidence: How sure is the model?
        """
        area_ratio = box_area / img_area

        if area_ratio > 0.10 and confidence > 0.7:
            return FireSeverity.CRITICAL

        if area_ratio > 0.02:
            if confidence > 0.6:
                return FireSeverity.HIGH
            return FireSeverity.MEDIUM

        if confidence > 0.8:
            return FireSeverity.MEDIUM

        return FireSeverity.LOW


detector = FireDetector()
