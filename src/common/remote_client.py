import logging
import time
from typing import Dict, List, Optional

import httpx

from src.common.config import settings
from src.common.models import ImageCreationDto, ImageDto, FireAlertCreationDto, FireAlertDto, ImagesFilters

HttpHeaders = Dict[str, str]

logger = logging.getLogger(__name__)

class RemoteClient:
    _CACHE_TTL_SECONDS = 300 # 5 minutes

    def __init__(self) -> None:
        self.base_url = settings.REMOTE_API_BASE_URL
        self.auth_token: Optional[str] = None
        self._images_cache: Dict[str, (int, List[ImageDto])] = {}

    async def _get_headers(self, with_auth: bool = True) -> HttpHeaders:
        if self.auth_token is None:
            if with_auth:
                await self.login()
            else:
                return {
                    "Content-Type": "application/json"
                }

        return {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json"
        }

    async def login(self) -> None:
        """
        Authenticates the client with the remote API.
        """
        logger.info("Logging in to the remote API...")
        headers = await self._get_headers(with_auth=False)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/accounts-service/auth/login",
                json={
                    "email": settings.REMOTE_API_EMAIL,
                    "password": settings.REMOTE_API_PASSWORD
                },
                headers=headers
            )
            response.raise_for_status()
            self.auth_token = response.json()["token"]
            logger.info("Logged in successfully!")

    async def register_images(self, payload: List[ImageCreationDto]) -> List[ImageDto]:
        """
        Registers a batch of images.
        """
        logger.info(f"Registering {len(payload)} images...")
        headers = await self._get_headers()
        async with httpx.AsyncClient() as client:
            payload = {"images": payload}
            response = await client.post(
                f"{self.base_url}/detection-service/images",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return response.json()

    async def create_fire_alert(self, payload: FireAlertCreationDto) -> FireAlertDto:
        """
        Registers a new fire alert.
        """
        logger.info(f"Creating fire alert for image with ID {payload.image_id}...")
        headers = await self._get_headers()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/detection-service/fire-alerts",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return response.json()

    async def get_images(self, filters: Optional[ImagesFilters] = None) -> List[ImageDto]:
        """
        Fetches images for the periodic simulation.
        """
        cache_key = str(filters.as_dict()) if filters else "None"

        if cache_key in self._images_cache:
            timestamp, data = self._images_cache[cache_key]
            if time.time() - timestamp < self._CACHE_TTL_SECONDS:
                logger.info("Returning cached test images")
                return data

        logger.info("Fetching test images...")
        headers = await self._get_headers()
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/detection-service/images",
                params=filters.as_dict() if filters is not None else None,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            self._images_cache[cache_key] = (time.time(), data)
            return data


remote_client = RemoteClient()
