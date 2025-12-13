import asyncio
import logging
import os
from io import BytesIO
from typing import Optional

import httpx
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .base import PluginBase, PluginOutput

logger = logging.getLogger(__name__)


class XKCDPlugin(PluginBase):
    """Fetch the current XKCD comic and render a summary panel."""

    BASENAME = "xkcd"
    OUTPUT_SUBDIR = "xkcd"
    REGISTRY_ORDER = 30
    REFRESH_INTERVAL = 3600
    DISPLAY_NAME = "XKCD"

    INFO_URL = "https://xkcd.com/info.0.json"

    async def run(self, **kwargs) -> Optional[PluginOutput]:
        """Download the comic of the day, render metadata, and store it as BMP."""
        target_size = kwargs.get("target_size", (800, 480))
        output_dir = kwargs.get("output_dir", "web")
        os.makedirs(output_dir, exist_ok=True)

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                metadata = await self._fetch_metadata(client)
                if not metadata:
                    logger.warning("No XKCD metadata available")
                    return None
                image_bytes = await self._download_image(client, metadata.get("img"))
            comic_image = await asyncio.to_thread(self._load_image, image_bytes)
            rendered = await asyncio.to_thread(self._render_panel, comic_image, metadata, target_size)
            output = await asyncio.to_thread(self.save_assets, rendered, output_dir, 'xkcd')
            logger.info(
                "Saved XKCD assets to %s and %s",
                output.monochrome_path,
                output.grayscale_path
            )
            return output
        except Exception as exc:
            logger.error("Failed to fetch XKCD comic: %s", exc)
            raise

    def get_content_ttl(self) -> int:
        return 43200  # 12 hours; comics update daily

    async def _fetch_metadata(self, client: httpx.AsyncClient) -> dict:
        response = await client.get(self.INFO_URL)
        response.raise_for_status()
        return response.json()

    async def _download_image(self, client: httpx.AsyncClient, url: Optional[str]) -> bytes:
        if not url:
            raise ValueError("XKCD metadata missing image URL")
        response = await client.get(url)
        response.raise_for_status()
        return response.content

    def _load_image(self, data: bytes) -> Image.Image:
        return Image.open(BytesIO(data)).convert("RGB")

    def _render_panel(self, comic: Image.Image, metadata: dict, target_size) -> Image.Image:
        base_canvas = Image.new("L", target_size, color=255)
        base_draw = ImageDraw.Draw(base_canvas)

        title = metadata.get("safe_title") or metadata.get("title", "XKCD")
        alt_text = metadata.get("alt", "")
        transcript = metadata.get("transcript", "")

        title_font = self.load_font(32)
        body_font = self.load_font(18)

        max_comic_height = 280
        max_comic_width = target_size[0] - 60
        fitted = ImageOps.contain(comic, (max_comic_width, max_comic_height))
        fitted = fitted.convert("L")

        top_margin = 30
        comic_x = max(0, (target_size[0] - fitted.width) // 2)
        base_canvas.paste(fitted, (comic_x, top_margin))
        draw = ImageDraw.Draw(base_canvas)

        text_y = top_margin + fitted.height + 20
        draw.text((30, text_y), title, fill=0, font=title_font)
        text_y += title_font.size + 10

        wrapped_alt = self._wrap_text(alt_text or transcript, body_font, target_size[0] - 60)
        for line in wrapped_alt:
            draw.text((30, text_y), line, fill=0, font=body_font)
            text_y += body_font.size + 4

        return base_canvas

    def _wrap_text(self, text: str, font: ImageFont.ImageFont, max_width: int):
        words = text.split()
        lines = []
        current = []

        for word in words:
            test_line = " ".join(current + [word]).strip()
            width = font.getlength(test_line)
            if width <= max_width:
                current.append(word)
            else:
                if current:
                    lines.append(" ".join(current))
                current = [word]
        if current:
            lines.append(" ".join(current))
        return lines