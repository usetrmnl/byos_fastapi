import logging
from typing import List, Sequence, Tuple

import httpx

from .base import ChartPlugin

logger = logging.getLogger(__name__)

# Node-RED API endpoints for local web stats
PAGEVIEWS_URL = "http://192.168.1.100:1880/api/site/pageviews"
VISITORS_URL = "http://192.168.1.100:1880/api/site/visitors"

# Node-RED API endpoint for total power consumption
TOTAL_POWER_URL = "http://192.168.1.100:1880/api/power/all"


async def _fetch_series(url: str) -> List[Tuple[str, int]]:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(url)
        response.raise_for_status()
        payload = response.json()
    entries = payload.get('data', [])
    parsed: List[Tuple[str, int]] = []
    for entry in entries:
        name = str(entry.get('name', '')).zfill(2)
        try:
            value = int(entry.get('value', 0))
        except (TypeError, ValueError):
            value = 0
        parsed.append((name, value))
    return parsed


class PageviewsPlugin(ChartPlugin):
    """Render hourly pageviews chart."""

    DISPLAY_NAME = "Web Stats - Pageviews"
    SERIES_LABEL = "Pageviews"
    BASENAME = "webstats_pageviews"
    OUTPUT_SUBDIR = "webstats"
    REGISTRY_ORDER = 60
    REFRESH_INTERVAL = 1800
    CHART_STYLE = "area"
    CAPTION_TEXT = "Estimated pageviews - last 24h"

    async def _fetch_series(self) -> Sequence[Tuple[str, int]]:
        return await _fetch_series(PAGEVIEWS_URL)


class VisitorsPlugin(ChartPlugin):
    """Render hourly visitors chart."""

    DISPLAY_NAME = "Web Stats - Visitors"
    SERIES_LABEL = "Visitors"
    BASENAME = "webstats_visitors"
    OUTPUT_SUBDIR = "webstats"
    REGISTRY_ORDER = 70
    REFRESH_INTERVAL = 1800
    CHART_STYLE = "area"
    CAPTION_TEXT = "Estimated unique visitors - last 24h"

    async def _fetch_series(self) -> Sequence[Tuple[str, int]]:
        return await _fetch_series(VISITORS_URL)


class TotalPowerPlugin(ChartPlugin):
    """Render the total power chart from the local power feed."""

    DISPLAY_NAME = "Power - Total"
    SERIES_LABEL = "Total Power (W)"
    BASENAME = "power_total"
    OUTPUT_SUBDIR = "power"
    REGISTRY_ORDER = 80
    REFRESH_INTERVAL = 1800
    CHART_STYLE = "area"
    CAPTION_TEXT = "All smart outlets combined - last 24h"

    async def _fetch_series(self) -> Sequence[Tuple[str, int]]:
        return await _fetch_series(TOTAL_POWER_URL)
