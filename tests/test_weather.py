from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest

from trmnl_server.plugins.base import PluginOutput
from trmnl_server.plugins.weather import WeatherPlugin


def _sample_weather_payload() -> Dict[str, Any]:
    # Minimal structure consumed by WeatherPlugin
    times = [f"2025-12-10T{str(h).zfill(2)}:00" for h in range(24)]
    temps = [10 + (h % 5) for h in range(24)]
    precip = [0.1 * (h % 3) for h in range(24)]
    return {
        "current_weather": {
            "temperature": 21.5,
            "windspeed": 12.3
        },
        "hourly": {
            "time": times,
            "temperature_2m": temps,
            "precipitation": precip
        }
    }


@pytest.mark.asyncio
async def test_weather_plugin_generates_assets(tmp_path: Path):
    payload = _sample_weather_payload()
    with patch.object(
        WeatherPlugin,
        '_fetch_weather_data',
        new=AsyncMock(return_value=payload)
    ):
        plugin = WeatherPlugin()
        output = await plugin.run(output_dir=str(tmp_path))

    assert isinstance(output, PluginOutput)
    mono = Path(output.monochrome_path)
    gray = Path(output.grayscale_path)
    assert mono.is_file() and mono.stat().st_size > 0
    assert gray.is_file() and gray.stat().st_size > 0
