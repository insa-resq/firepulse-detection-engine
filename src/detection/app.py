from contextlib import asynccontextmanager
from typing import AsyncGenerator

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.common.config import settings
from src.detection.router import router
from src.detection.simulation import simulation_job

scheduler = BackgroundScheduler()

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    scheduler.add_job(
        simulation_job,
        "interval",
        seconds=settings.SIMULATION_INTERVAL_SECONDS
    )
    scheduler.start()
    yield
    # Shutdown
    scheduler.shutdown()

app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

app.mount(
    f"/{settings.IMAGES_SERVE_BASE_PATH}/raw",
    StaticFiles(directory=settings.RAW_IMAGES_PATH),
    name="static_raw"
)

app.mount(
    f"/{settings.IMAGES_SERVE_BASE_PATH}/live",
    StaticFiles(directory=settings.LIVE_IMAGES_PATH),
    name="static_live"
)

app.include_router(router, prefix="/api")
