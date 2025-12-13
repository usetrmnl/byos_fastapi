import asyncio
import logging
import os
from io import BytesIO
from typing import Tuple

import httpx
from PIL import Image, ImageOps

from .base import PhotographicPlugin, PluginOutput

logger = logging.getLogger(__name__)


class BingWallpaperPlugin(PhotographicPlugin):
    """Download the latest Bing wallpaper and adapt it for TRMNL displays."""

    BASENAME = "bing_wallpaper"
    OUTPUT_SUBDIR = "bing"
    REGISTRY_ORDER = 20
    REFRESH_INTERVAL = 3600
    DISPLAY_NAME = "Bing Wallpaper"

    API_URL = "https://www.bing.com/HPImageArchive.aspx"
    BASE_URL = "https://www.bing.com"

    def __init__(self, market: str = "en-US"):
        super().__init__()
        self.market = market

    async def run(self, **kwargs):
        """Fetch, process, and persist the latest Bing wallpaper image."""
        market = kwargs.get("market", self.market)
        target_size: Tuple[int, int] = kwargs.get("target_size", (800, 480))
        output_dir = kwargs.get("output_dir", "web")
        os.makedirs(output_dir, exist_ok=True)

        try:
            async with httpx.AsyncClient(timeout=20) as client:
                metadata = await self._fetch_metadata(client, market)
                if not metadata:
                    logger.warning("No Bing metadata returned")
                    return None

                image_url = metadata.get("url") or metadata.get("urlbase")
                if not image_url:
                    logger.warning("Bing metadata missing URL")
                    return None

                full_url = image_url
                if not full_url.startswith("http"):
                    full_url = f"{self.BASE_URL}{image_url}"

                logger.info("Downloading Bing wallpaper from %s", full_url)
                image_bytes = await self._download_image(client, full_url)
            processed = await asyncio.to_thread(self._prepare_image, image_bytes, target_size)

            output = await asyncio.to_thread(
                self.save_assets,
                processed,
                output_dir,
                'bing_wallpaper',
                dither_mode='floyd-steinberg'
            )
            logger.info(
                "Saved Bing wallpaper assets to %s and %s",
                output.monochrome_path,
                output.grayscale_path
            )
            return output
        except Exception as exc:
            logger.error("Failed to fetch Bing wallpaper: %s", exc)
            raise

    def get_content_ttl(self) -> int:
        return 43200  # 12 hours

    async def _fetch_metadata(self, client: httpx.AsyncClient, market: str) -> dict:
        params = {
            "format": "js",
            "idx": 0,
            "n": 1,
            "mkt": market
        }
        response = await client.get(self.API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        images = data.get("images", [])
        return images[0] if images else {}

    async def _download_image(self, client: httpx.AsyncClient, url: str) -> bytes:
        response = await client.get(url)
        response.raise_for_status()
        return response.content

    def _prepare_image(self, data: bytes, target_size: Tuple[int, int]) -> Image.Image:
        with Image.open(BytesIO(data)) as img:
            rgb_image = img.convert("RGB")
            fitted = ImageOps.fit(
                rgb_image,
                target_size,
                method=Image.Resampling.LANCZOS,
                bleed=0.0,
                centering=(0.5, 0.5)
            )
            return fitted.convert("L")
