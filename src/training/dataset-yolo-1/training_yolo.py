import os
from ultralytics import YOLO

from src.common.config import settings

model = YOLO(settings.BASE_MODEL)

results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640
)

os.remove(settings.BASE_MODEL)
