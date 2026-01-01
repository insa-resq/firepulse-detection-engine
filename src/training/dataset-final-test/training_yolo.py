import os
from ultralytics import YOLO

from src.common.config import settings

model = YOLO(settings.BASE_MODEL)

results = model.train(
    data="src/training/dataset-final-test/data.yaml",
    epochs=30,
    imgsz=640,
    device=0,
    batch=8,
    patience=10,
    save=True,
    project="src/training/dataset-final-test/runs/detect",
    name="train"
)

os.remove(settings.BASE_MODEL)
