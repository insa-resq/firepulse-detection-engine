import os
from ultralytics import YOLO

from src.common.config import settings

model = YOLO(settings.BASE_MODEL_WEIGHTS_PATH)

results = model.train(
    data="data.yaml",
    epochs=100,
    imgsz=640
)

model_name = os.path.basename(settings.BASE_MODEL_WEIGHTS_PATH)
os.remove(model_name)
