from typing import List, TypedDict, Optional
from ultralytics import YOLO

from src.common.config import settings
from src.common.models import FireSeverity

BoundingBox = List[float]
DetectionResult = TypedDict(
    "DetectionResult",
    {
        "has_fire": bool,
        "confidence": Optional[float],
        "severity": Optional[FireSeverity],
        "bounding_boxes": List[BoundingBox]
    }
)

class FireDetector:
    def __init__(self) -> None:
        self.model = YOLO(settings.BEST_MODEL_WEIGHTS_PATH)

    def detect(self, image_path: str) -> DetectionResult:
        """
        Runs YOLO inference. Returns aggregated result for the image.
        """
        results = self.model.predict(
            source=image_path,
            conf=settings.CONFIDENCE_THRESHOLD,
            save=False,
            verbose=False
        )

        result = results[0] # Single-image inference so the list is always of length 1

        # Negative result
        if len(result.boxes) == 0:
            return {
                "has_fire": False,
                "confidence": None,
                "severity": None,
                "bounding_boxes": []
            }

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

        return {
            "has_fire": True,
            "confidence": max_confidence,
            "severity": worst_severity,
            "bounding_boxes": all_boxes
        }

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
