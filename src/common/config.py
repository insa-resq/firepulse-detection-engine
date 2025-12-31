import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv(dotenv_path=".env")

class Settings(BaseSettings):
    APP_NAME: str = "Firepulse Detection Engine API"
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    REMOTE_API_BASE_URL: str = os.getenv("REMOTE_API_BASE_URL")
    REMOTE_API_EMAIL: str = os.getenv("REMOTE_API_EMAIL")
    REMOTE_API_PASSWORD: str = os.getenv("REMOTE_API_PASSWORD")

    MODELS_DIR: str = "model"
    BASE_MODEL_WEIGHTS_PATH: str = f"{MODELS_DIR}/base/yolo11n.pt"
    BEST_MODEL_VERSION: str = os.getenv("BEST_MODEL_VERSION")
    BEST_MODEL_WEIGHTS_PATH: str = f"{MODELS_DIR}/{BEST_MODEL_VERSION}/weights/best.pt"

    RAW_IMAGES_DIR: str = "data/raw"
    LIVE_IMAGES_DIR: str = "data/live"
    PROCESSED_IMAGES_DIR: str = "data/processed"

    DATA_YAML_PATH: str = "src/training/data.yaml"

    IMAGES_SERVE_HOST: str = os.getenv("IMAGES_SERVE_HOST")
    IMAGES_SERVE_BASE_PATH: str = "/static"
    IMAGES_SERVE_BASE_URL: str = f"{IMAGES_SERVE_HOST}{IMAGES_SERVE_BASE_PATH}"

    CONFIDENCE_THRESHOLD: float = 0.5

    SIMULATION_INTERVAL_SECONDS: int = 60


settings = Settings()
