import functools
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

class ImageSplit(StrEnum):
    TRAIN = "TRAIN"
    TEST = "TEST"
    VALIDATION = "VALIDATION"

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
    local_path: str
    contains_fire: bool
    latitude: float
    longitude: float

class ImageCreationDto(BaseModel):
    url: str
    width: int
    height: int
    split: ImageSplit
    metadata: ImageMetadata

class ImageDto(BaseModel):
    id: str
    createdAt: datetime
    updatedAt: datetime
    url: str
    width: int
    height: int
    imageSplit: ImageSplit
    metadata: ImageMetadata

class FireAlertCreationDto(BaseModel):
    description: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    severity: FireSeverity
    imageId: str = Field(..., min_length=1)

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
