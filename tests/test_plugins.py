from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from trmnl_server.plugins.base import PluginOutput
from trmnl_server.plugins.bing import BingWallpaperPlugin
from trmnl_server.plugins.hn import HNPlugin
from trmnl_server.plugins.random_image import RandomImagePlugin
try:
    from trmnl_server.plugins.charts import PageviewsPlugin, VisitorsPlugin  # type: ignore[attr-defined]
except ModuleNotFoundError:
    PageviewsPlugin = None
    VisitorsPlugin = None
from trmnl_server.plugins.xkcd import XKCDPlugin
from trmnl_server.plugins.weather import WeatherPlugin
from trmnl_server.services import plugins as plugin_service
from trmnl_server import config

pytestmark = pytest.mark.asyncio

def _make_image_bytes(size: Tuple[int, int] = (32, 32), color: Tuple[int, int, int] = (255, 0, 0)) -> bytes:
    img = Image.new('RGB', size, color=color)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


def _make_image(size: Tuple[int, int] = (32, 32), color: Tuple[int, int, int] = (0, 255, 0)) -> Image.Image:
    return Image.open(BytesIO(_make_image_bytes(size=size, color=color))).convert('RGB')


def _plugin_available(plugin_name: str) -> bool:
    available = {name.lower() for name in plugin_service.list_available_plugins()}
    return plugin_name.lower() in available


def _assert_output_assets(output: PluginOutput) -> None:
    mono = Path(output.monochrome_path)
    gray = Path(output.grayscale_path)
    assert mono.is_file() and mono.stat().st_size > 0
    assert gray.is_file() and gray.stat().st_size > 0


async def test_bing_plugin_generates_assets(tmp_path: Path) -> None:
    if not _plugin_available(BingWallpaperPlugin.__name__):
        pytest.skip('BingWallpaperPlugin not available')
    plugin_cls = plugin_service.get_plugin_schedule(BingWallpaperPlugin.__name__).plugin_cls
    plugin = plugin_cls()
    fake_metadata = {"url": "http://example.com/test_wallpaper.png"}

    with patch.object(plugin, '_fetch_metadata', new=AsyncMock(return_value=fake_metadata)), \
         patch.object(plugin, '_download_image', new=AsyncMock(return_value=_make_image_bytes())):
        output = await plugin.run(output_dir=str(tmp_path))

    assert isinstance(output, PluginOutput)
    _assert_output_assets(output)


async def test_xkcd_plugin_generates_assets(tmp_path: Path) -> None:
    if not _plugin_available(XKCDPlugin.__name__):
        pytest.skip('XKCDPlugin not available')
    plugin_cls = plugin_service.get_plugin_schedule(XKCDPlugin.__name__).plugin_cls
    plugin = plugin_cls()
    fake_metadata = {
        "img": "http://example.com/comic.png",
        "safe_title": "Test XKCD",
        "alt": "Alt text",
        "transcript": "Transcript text"
    }

    with patch.object(plugin, '_fetch_metadata', new=AsyncMock(return_value=fake_metadata)), \
         patch.object(plugin, '_download_image', new=AsyncMock(return_value=_make_image_bytes())), \
         patch.object(plugin, '_load_image', return_value=_make_image()):
        output = await plugin.run(output_dir=str(tmp_path))

    assert isinstance(output, PluginOutput)
    _assert_output_assets(output)


async def test_hn_plugin_generates_assets(tmp_path: Path) -> None:
    if not _plugin_available(HNPlugin.__name__):
        pytest.skip('HNPlugin not available')
    plugin_cls = plugin_service.get_plugin_schedule(HNPlugin.__name__).plugin_cls
    plugin = plugin_cls()
    fake_entries: List[Dict[str, Any]] = [
        {"title": "Test headline", "points": 123, "comments": 45},
        {"title": "Another headline", "points": 456, "comments": 78}
    ]

    with patch.object(plugin, '_fetch_entries', new=AsyncMock(return_value=fake_entries)):
        output = await plugin.run(output_dir=str(tmp_path))

    assert isinstance(output, PluginOutput)
    _assert_output_assets(output)


def _fake_stats_data() -> List[Tuple[str, int]]:
    return [("00", 0), ("01", 10), ("02", 20), ("03", 5)]


async def test_pageviews_plugin_generates_assets(tmp_path: Path) -> None:
    if PageviewsPlugin is None or not _plugin_available('PageviewsPlugin'):
        pytest.skip('PageviewsPlugin not available')
    plugin_cls = plugin_service.get_plugin_schedule('PageviewsPlugin').plugin_cls
    plugin = plugin_cls()
    fake_data = _fake_stats_data()

    with patch.object(plugin, '_fetch_series', new=AsyncMock(return_value=fake_data)):
        output = await plugin.run(output_dir=str(tmp_path))

    assert isinstance(output, PluginOutput)
    _assert_output_assets(output)


async def test_visitors_plugin_generates_assets(tmp_path: Path) -> None:
    if VisitorsPlugin is None or not _plugin_available('VisitorsPlugin'):
        pytest.skip('VisitorsPlugin not available')
    plugin_cls = plugin_service.get_plugin_schedule('VisitorsPlugin').plugin_cls
    plugin = plugin_cls()
    fake_data = _fake_stats_data()

    with patch.object(plugin, '_fetch_series', new=AsyncMock(return_value=fake_data)):
        output = await plugin.run(output_dir=str(tmp_path))

    assert isinstance(output, PluginOutput)
    _assert_output_assets(output)


async def test_random_image_plugin_generates_assets(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    img_path = image_root / "sample.png"
    img = Image.new('RGB', (64, 64), color=(0, 0, 255))
    img.save(img_path)

    if not _plugin_available(RandomImagePlugin.__name__):
        pytest.skip('RandomImagePlugin not available')
    plugin_cls = plugin_service.get_plugin_schedule(RandomImagePlugin.__name__).plugin_cls
    plugin = plugin_cls()
    output = await plugin.run(output_dir=str(tmp_path), image_root=str(image_root))

    assert isinstance(output, PluginOutput)
    _assert_output_assets(output)


async def test_photo_grading_toggle_changes_prepare_image() -> None:
    if not _plugin_available(BingWallpaperPlugin.__name__):
        pytest.skip('BingWallpaperPlugin not available')
    plugin_cls = plugin_service.get_plugin_schedule(BingWallpaperPlugin.__name__).plugin_cls
    plugin = plugin_cls()

    gradient = Image.linear_gradient('L').resize((64, 64))
    rgb = gradient.convert('RGB')

    original = config.PHOTO_GRADING_ENABLED
    try:
        config.PHOTO_GRADING_ENABLED = False
        disabled = plugin.prepare_image(rgb)
        assert disabled.mode == 'L'
        assert disabled.tobytes() == rgb.convert('L').tobytes()

        config.PHOTO_GRADING_ENABLED = True
        enabled = plugin.prepare_image(rgb)
        assert enabled.mode == 'L'
        assert enabled.tobytes() != rgb.convert('L').tobytes()
    finally:
        config.PHOTO_GRADING_ENABLED = original


async def test_plugin_runner_passes_extra_kwargs(monkeypatch, tmp_path: Path) -> None:
    bmp = tmp_path / 'weather.bmp'
    png = tmp_path / 'weather.png'
    bmp.write_bytes(b'0')
    png.write_bytes(b'0')

    async def fake_run(self, **kwargs):  # type: ignore[override]
        assert kwargs['output_dir'] == str(tmp_path)
        assert kwargs['extra'] == 'value'
        return PluginOutput(monochrome_path=str(bmp), grayscale_path=str(png))

    if not _plugin_available(WeatherPlugin.__name__):
        pytest.skip('WeatherPlugin not available')

    monkeypatch.setattr(WeatherPlugin, 'run', fake_run)

    result = await plugin_service.run_single_plugin_by_name(
        'WeatherPlugin',
        output_dir=str(tmp_path),
        plugin_kwargs={'extra': 'value'}
    )

    assert result.monochrome_path == str(bmp)
    assert result.grayscale_path == str(png)


async def test_plugin_runner_errors_on_unknown_plugin(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        await plugin_service.run_single_plugin_by_name('NotAPlugin', output_dir=str(tmp_path))


async def test_calibration_plugins_can_be_disabled(monkeypatch) -> None:
    original = config.CALIBRATION_PLUGIN_ENABLED
    try:
        monkeypatch.setattr(config, 'CALIBRATION_PLUGIN_ENABLED', False)
        names = plugin_service.list_available_plugins()
        assert all('calibration' not in name.lower() for name in names)
        with pytest.raises(ValueError):
            plugin_service.get_plugin_schedule('CalibrationPlugin')
    finally:
        monkeypatch.setattr(config, 'CALIBRATION_PLUGIN_ENABLED', original)
