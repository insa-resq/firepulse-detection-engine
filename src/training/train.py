import os
from typing import Dict, Any

from ultralytics import YOLO

from src.common.config import settings

def get_new_version_name() -> str:
    previous_versions = [
        folder_name
        for folder_name in os.listdir(settings.MODELS_DIR)
        if folder_name != "base"
    ]

    for version in previous_versions:
        if not "best.pt" in os.listdir(os.path.join(settings.MODELS_DIR, version, "weights")):
            raise Exception(f"Incomplete version found: {version}. No 'best.pt' file found in it. Please delete it and try again.")

    version_numbers = [
        int(folder_name.lstrip("v"))
        for folder_name in previous_versions
    ]

    if len(version_numbers) == 0:
        return "v1"
    else:
        return f"v{max(version_numbers) + 1}"


def train_model(version: str) -> Dict[str, Any]:
    model = YOLO(settings.BASE_MODEL_WEIGHTS_PATH)
    return model.train(
        data=settings.DATA_YAML_PATH,
        epochs=30,
        imgsz=640,
        device=0,
        batch=8,
        patience=10,
        save=True,
        project=settings.MODELS_DIR,
        name=version
    )


def cleanup() -> None:
    base_model_name = os.path.basename(settings.BASE_MODEL_WEIGHTS_PATH)
    os.remove(base_model_name)

if __name__ == "__main__":
    version_name = get_new_version_name()
    print(f"Starting training ({version_name})...")
    train_model(version_name)
    print("Training finished")
    cleanup()
    print("Cleanup finished")
