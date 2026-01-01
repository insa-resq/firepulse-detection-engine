import functools
from datetime import datetime
from enum import StrEnum
from typing import List, Optional, Dict

from pydantic import BaseModel, Field

class ImageSplit(StrEnum):
    TRAIN = "TRAIN"
    TEST = "TEST"
    VALIDATION = "VALIDATION"
    NONE = "NONE"

@functools.total_ordering
class FireSeverity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    def __lt__(self, other):
        if isinstance(other, FireSeverity):
            hierarchy = {
                self.LOW: 0,
                self.MEDIUM: 1,
                self.HIGH: 2,
                self.CRITICAL: 3
            }
            return hierarchy[self] < hierarchy[other]
        return NotImplemented

class FireStatus(StrEnum):
    NEW = "NEW"
    IN_PROGRESS = "IN_PROGRESS"
    RESOLVED = "RESOLVED"
    DISMISSED = "DISMISSED"

class ImageMetadata(BaseModel):
    localPath: str
    width: int
    height: int
    latitude: float
    longitude: float
    polygons: List[List[float]]

class ImageCreationDto(BaseModel):
    url: str
    split: ImageSplit
    containsFire: bool
    metadata: ImageMetadata

    def as_dict(self) -> Dict[str, str]:
        return self.model_dump(
            exclude_unset=True,
            exclude_none=True,
        )

class ImageDto(BaseModel):
    id: str
    createdAt: datetime
    updatedAt: datetime
    url: str
    split: ImageSplit
    containsFire: bool
    metadata: ImageMetadata

class FireAlertCreationDto(BaseModel):
    description: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    severity: FireSeverity
    imageId: str = Field(..., min_length=1)

    def as_dict(self) -> Dict[str, str]:
        return self.model_dump(
            exclude_unset=True,
            exclude_none=True,
        )

class FireAlertDto(BaseModel):
    id: int
    createdAt: datetime
    updatedAt: datetime
    description: str
    confidence: float
    latitude: float
    longitude: float
    severity: FireSeverity
    status: FireStatus
    imageId: str

class ImagesFilters(BaseModel):
    splits: Optional[List[ImageSplit]] = None
    containsFire: Optional[bool] = None

    def as_dict(self) -> Dict[str, str]:
        return self.model_dump(
            exclude_unset=True,
            exclude_none=True,
        )
