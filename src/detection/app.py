from contextlib import asynccontextmanager
from typing import AsyncGenerator, Literal, TypedDict

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI

from src.common.config import settings
from src.detection.routers import detection_router, files_router
from src.detection.simulation import simulation_job

scheduler = BackgroundScheduler()

@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    scheduler.add_job(
        func=simulation_job,
        trigger="interval",
        seconds=settings.SIMULATION_INTERVAL_SECONDS
    )
    scheduler.start()
    yield
    # Shutdown
    scheduler.shutdown()

app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

RootResponse = TypedDict("RootResponse", {"message": str})
@app.get("/")
def root() -> RootResponse:
    return {"message": "Welcome to the Firepulse Detection Engine API"}

HealthCheckResponse = TypedDict("HealthCheckResponse", {"status": Literal["UP"]})
@app.get("/health")
def health() -> HealthCheckResponse:
    return {"status": "UP"}

app.include_router(detection_router.router)

app.include_router(files_router.router, prefix=settings.IMAGES_SERVE_BASE_PATH)
