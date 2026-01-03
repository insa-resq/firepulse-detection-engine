import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Tuple

import httpx

from src.common.config import settings
from src.common.models import ImageCreationDto, ImageDto, FireAlertCreationDto, FireAlertDto, ImagesFilters

HttpHeaders = Dict[str, str]

logger = logging.getLogger(__name__)

class RemoteClient:
    _CACHE_TTL_SECONDS = 5 * 60 # 5 minutes
    _CACHE_MAX_SIZE = 5
    _AUTH_REFRESH_INTERVAL_SECONDS = 5 * 60 * 60 # 5 hours

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(base_url=settings.REMOTE_API_BASE_URL)
        self._auth_token: Optional[str] = None
        self._last_auth_refresh_time: float = 0.0
        self._auth_refresh_lock = asyncio.Lock()
        self._images_cache: Dict[str, Tuple[float, List[ImageDto]]] = {}
        self._images_cache_lock = asyncio.Lock()

    async def close(self) -> None:
        """
        Closes the underlying HTTP client.
        """
        if not self._client.is_closed:
            await self._client.aclose()

    async def _get_headers(self, with_auth: bool = True) -> HttpHeaders:
        if not with_auth:
            return {
                "Content-Type": "application/json",
            }

        async with self._auth_refresh_lock:
            current_time = time.time()
            if (self._auth_token is None) or (current_time - self._last_auth_refresh_time > self._AUTH_REFRESH_INTERVAL_SECONDS):
                self._auth_token = await self.login()
                self._last_auth_refresh_time = current_time

            return {
                "Authorization": f"Bearer {self._auth_token}",
                "Content-Type": "application/json"
            }

    async def login(self) -> str:
        """
        Authenticates the client with the remote API.
        """
        logger.info("Logging in to the remote API...")
        headers = await self._get_headers(with_auth=False)
        response = await self._client.post(
            url="accounts-service/auth/login",
            json={
                "email": settings.REMOTE_API_EMAIL,
                "password": settings.REMOTE_API_PASSWORD
            },
            headers=headers
        )
        response.raise_for_status()
        logger.info("Logged in successfully!")
        token: Optional[str] = response.json().get("token")
        if token is None:
            raise ValueError("No token returned from the remote API.")
        return token

    async def register_images(self, payload: List[ImageCreationDto]) -> List[ImageDto]:
        """
        Registers a batch of images.
        """
        if len(payload) == 0:
            return []
        headers = await self._get_headers()
        response = await self._client.post(
            url="detection-service/images",
            json={"images": [image.as_dict() for image in payload]},
            headers=headers
        )
        response.raise_for_status()
        return response.json()

    async def create_fire_alert(self, payload: FireAlertCreationDto) -> FireAlertDto:
        """
        Registers a new fire alert.
        """
        logger.info(f"Creating fire alert for image with ID {payload.imageId}...")
        headers = await self._get_headers()
        response = await self._client.post(
            url="detection-service/fire-alerts",
            json=payload.as_dict(),
            headers=headers
        )
        response.raise_for_status()
        return response.json()

    async def get_images(self, filters: Optional[ImagesFilters] = None) -> List[ImageDto]:
        """
        Fetches images for the periodic simulation.
        """
        cache_key = json.dumps(filters.as_dict(), sort_keys=True) if filters else "None"
        current_time = time.time()

        async with self._images_cache_lock:
            if cache_key in self._images_cache:
                timestamp, data = self._images_cache[cache_key]
                if current_time - timestamp < self._CACHE_TTL_SECONDS:
                    logger.info("Returning cached test images")
                    return data

            logger.info("Fetching test images...")
            headers = await self._get_headers()
            response = await self._client.get(
                url="detection-service/images",
                params=filters.as_dict() if filters is not None else None,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()

            if len(self._images_cache) >= self._CACHE_MAX_SIZE:
                oldest_key = min(self._images_cache, key=lambda k: self._images_cache[k][0])
                del self._images_cache[oldest_key]

            self._images_cache[cache_key] = (current_time, data)

            return data

    async def delete_images(self, image_ids: List[str]) -> None:
        """
        Deletes images from the remote API.
        """
        if len(image_ids) == 0:
            return
        headers = await self._get_headers()
        response = await self._client.delete(
            url="detection-service/images",
            params={"imageIds": image_ids},
            headers=headers
        )
        response.raise_for_status()
        return


remote_client = RemoteClient()
