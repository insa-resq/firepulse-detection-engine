from argparse import ArgumentParser
import glob
import os
from typing import Dict, Any, List

from codecarbon import track_emissions
from ultralytics import YOLO

from src.common.config import settings

def get_new_version_name() -> str:
    previous_versions = [
        folder_name
        for folder_name in os.listdir(settings.MODELS_DIR)
    ]

    for version in previous_versions:
        if not "best.pt" in os.listdir(os.path.join(settings.MODELS_DIR, version, "weights")):
            raise Exception(f"Incomplete version found: {version}. No 'best.pt' file found in it. Please delete it and try again.")

    version_numbers = [
        int(folder_name.lstrip("v").split("_")[0])
        for folder_name in previous_versions
    ]

    if len(version_numbers) == 0:
        return "v1"
    else:
        return f"v{max(version_numbers) + 1}"


@track_emissions()
def train_model(version: str) -> Dict[str, Any]:
    model = YOLO(settings.BASE_MODEL)
    return model.train(
        data=settings.DATA_YAML_PATH,
        task="segment",
        epochs=100,
        imgsz=608, # First multiple of 32 greater than 595
        device=0,
        batch=8,
        patience=5,
        save=True,
        project=settings.MODELS_DIR,
        name=version
    )

@track_emissions()
def compare_training_fractions(
    version: str,
    fractions: List[float]
) -> Dict[float, Dict[str, Any]]:
    results = {}

    for fraction in fractions:
        print(f"\n{'='*50}")
        print(f"Training with {fraction*100}% of data")
        print(f"{'='*50}\n")
        
        version = f"{version}_{int(fraction*100)}percent"
        
        model = YOLO(settings.BASE_MODEL)
        training_results = model.train(
            data=settings.DATA_YAML_PATH,
            task="segment",
            epochs=100,
            imgsz=608, # First multiple of 32 greater than 595
            device=0,
            batch=8,
            patience=5,
            save=True,
            project=settings.MODELS_DIR,
            name=version,
            fraction=fraction
        )
        
        results[fraction] = {
            'version': version,
            'metrics': training_results,
            'fraction': fraction
        }
    
    return results

def cleanup() -> None:
    for weights_file in glob.glob("yolo*.pt"):
        os.remove(weights_file)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--version", type=str, required=False, default=get_new_version_name(), help="Base version name for training with different data fractions")
    parser.add_argument("--fractions", type=float, nargs='+', required=False, help="List of data fractions to use for training")
    args = parser.parse_args()
    print(f"Starting training ({args.version})...")
    
    if args.fractions is None or len(args.fractions) == 0:
        train_model(args.version)
    else:
        results = compare_training_fractions(args.version, args.fractions)
    print("Training finished")
    cleanup()
    print("Cleanup finished")
