import logging
from typing import Dict, List, Optional

import httpx

from src.common.config import settings
from src.common.models import ImageSplit, ImageCreationDto, ImageDto, FireAlertCreationDto, FireAlertDto

logger = logging.getLogger(__name__)

HttpHeaders = Dict[str, str]

class RemoteClient:
    def __init__(self) -> None:
        self.base_url = settings.REMOTE_API_BASE_URL
        self.auth_token: Optional[str] = None

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

    async def register_images(self, payload: List[ImageCreationDto]) -> List[ImageDto]:
        """
        Registers a batch of images.
        """
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
        headers = await self._get_headers()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/detection-service/fire-alerts",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return response.json()

    async def get_test_images(self) -> List[ImageDto]:
        """
        Fetches images for the periodic simulation.
        """
        headers = await self._get_headers()
        async with httpx.AsyncClient() as client:
            params = {
                "filters": {
                    "splits": [ImageSplit.TEST.value]
                }
            }
            response = await client.get(
                f"{self.base_url}/detection-service/images",
                params=params,
                headers=headers
            )
            response.raise_for_status()
            return response.json()


remote_client = RemoteClient()
