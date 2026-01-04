import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Literal, TypedDict

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI

from src.common.config import settings
from src.common.remote_client import remote_client
from src.detection.routers import detection_router, files_router
from src.detection.simulation import simulation_job


RootResponse = TypedDict("RootResponse", {"message": str})
HealthCheckResponse = TypedDict("HealthCheckResponse", {"status": Literal["UP"]})

logging.basicConfig(
    format="%(asctime)s [%(name)-30s] %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)

scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    scheduler.add_job(
        func=simulation_job,
        name="Simulated Alert Generation",
        trigger="interval",
        seconds=settings.SIMULATION_INTERVAL_SECONDS
    )
    scheduler.start()
    yield
    scheduler.shutdown()
    await remote_client.close()

app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

@app.get("/")
def root() -> RootResponse:
    return {"message": "Welcome to the Firepulse Detection Engine API"}

@app.get("/health")
def health() -> HealthCheckResponse:
    return {"status": "UP"}

app.include_router(detection_router.router)

app.include_router(files_router.router, prefix=settings.IMAGES_SERVE_BASE_PATH)
