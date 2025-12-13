import asyncio
import logging
import os
import re
from html import unescape
from html.parser import HTMLParser
from typing import List, Optional, Tuple

import feedparser
import httpx
from PIL import Image, ImageDraw, ImageFont

from .base import PluginBase, PluginOutput

logger = logging.getLogger(__name__)


class HNPlugin(PluginBase):
    """Render an e-ink friendly snapshot of Hacker News front page headlines."""

    BASENAME = "hackernews"
    OUTPUT_SUBDIR = "news"
    REGISTRY_ORDER = 40
    REFRESH_INTERVAL = 600
    DISPLAY_NAME = "Hacker News"

    FEED_URL = "https://hnrss.org/frontpage"

    def get_content_ttl(self) -> int:
        return 1800  # 30 minutes

    async def run(self, **kwargs) -> Optional[PluginOutput]:
        target_size = kwargs.get("target_size", (800, 480))
        max_items = kwargs.get("max_items", 10)
        output_dir = kwargs.get("output_dir", "web")
        os.makedirs(output_dir, exist_ok=True)

        try:
            entries = await self._fetch_entries(max_items)
            if not entries:
                logger.warning("HN feed returned no entries")
                return None

            rendered = await asyncio.to_thread(self._render_entries, entries, target_size)
            output = await asyncio.to_thread(self.save_assets, rendered, output_dir, 'hackernews')
            logger.info(
                "Saved Hacker News assets to %s and %s",
                output.monochrome_path,
                output.grayscale_path
            )
            return output
        except Exception as exc:
            logger.error("Failed to render HN feed: %s", exc)
            raise

    async def _fetch_entries(self, max_items: int) -> List[dict]:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(self.FEED_URL)
            response.raise_for_status()
            payload = response.content
        feed = feedparser.parse(payload)
        entries = []
        for entry in feed.entries[:max_items]:
            points, comments = self._parse_metadata(entry)
            entries.append({
                'title': entry.get('title', 'Untitled'),
                'points': points,
                'comments': comments
            })
        return entries

    def _parse_metadata(self, entry) -> Tuple[int, int]:
        points = 0
        comments = 0
        summary = entry.get('summary', '')
        text = self._strip_html(summary)
        points_match = re.search(r'Points:\s*(\d+)', text)
        comments_match = re.search(r'#\s*Comments:\s*(\d+)', text)
        if points_match:
            points = int(points_match.group(1))
        if comments_match:
            comments = int(comments_match.group(1))
        return points, comments

    def _strip_html(self, html: str) -> str:
        parser = _HTMLStripper()
        parser.feed(html or '')
        return unescape(parser.get_text())

    def _render_entries(self, entries: List[dict], target_size) -> Image.Image:
        canvas = Image.new('L', target_size, color=255)
        draw = ImageDraw.Draw(canvas)

        title_font = self.load_font(28)
        item_font = self.load_font(20)
        meta_font = self.load_font(16)

        draw.text((30, 20), "Hacker News", fill=0, font=title_font)
        draw.line([(30, 60), (target_size[0] - 30, 60)], fill=0, width=1)

        y = 80
        line_height = item_font.size + 10

        for idx, entry in enumerate(entries, start=1):
            title = f"{entry['title']}"
            wrapped = self._wrap_text(title, item_font, target_size[0] - 60)
            for line in wrapped:
                draw.text((40, y), line, fill=0, font=item_font)
                y += line_height

            meta = f"{entry['points']} points | {entry['comments']} comments"
            draw.text((40, y), meta, fill=80, font=meta_font)
            y += meta_font.size + 12

            if y > target_size[1] - line_height:
                break

        return canvas

    def _wrap_text(self, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
        words = text.split()
        lines: List[str] = []
        current: List[str] = []

        for word in words:
            test_line = " ".join(current + [word]).strip()
            if font.getlength(test_line) <= max_width:
                current.append(word)
            else:
                if current:
                    lines.append(" ".join(current))
                current = [word]
        if current:
            lines.append(" ".join(current))
        return lines


class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: List[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def get_text(self) -> str:
        return "\n".join(self.parts)