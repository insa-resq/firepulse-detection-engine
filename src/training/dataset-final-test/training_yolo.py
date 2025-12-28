import os
from ultralytics import YOLO

from src.common.config import settings

model = YOLO(settings.BASE_MODEL_WEIGHTS_PATH)

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

model_name = os.path.basename(settings.BASE_MODEL_WEIGHTS_PATH)
os.remove(model_name)
