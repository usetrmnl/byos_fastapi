"""Plugin refresh scheduling, orchestration, and CLI helpers."""

from __future__ import annotations

import inspect
from asyncio import CancelledError, Task, create_task, gather, sleep
from dataclasses import dataclass
from importlib import import_module
from os.path import exists
from pkgutil import walk_packages
from time import time
from typing import Any, Dict, List, Optional, Tuple, Type

from .. import config, plugins as plugins_pkg
from ..utils import asset_path, get_generated_assets_root
from ..plugins.base import PluginBase, PluginOutput

from . import state

logger = config.logger

PLUGIN_REFRESH_RETRY = 300
plugin_tasks: Dict[str, Task] = {}
_ttl_cache: Dict[str, int] = {}


@dataclass(frozen=True)
class PluginSchedule:
    """Typed plugin configuration entry used by the scheduler and CLI."""

    plugin_cls: Type[PluginBase]
    basename: str
    set_primary: bool = False
    refresh_interval: Optional[int] = None
    output_subdir: Optional[str] = None

    @property
    def name(self) -> str:
        return self.plugin_cls.__name__

    def fallback_assets(self) -> Tuple[str, str]:
        return _asset_paths(self.basename)

    def resolved_refresh_interval(self) -> int:
        if self.refresh_interval:
            return self.refresh_interval
        return _default_plugin_ttl(self.plugin_cls)

    def resolved_output_directory(self) -> str:
        root = get_generated_assets_root()
        target = root / self.output_subdir if self.output_subdir else root
        target.mkdir(parents=True, exist_ok=True)
        return target.as_posix()

    @classmethod
    def from_plugin(cls, plugin_cls: Type[PluginBase]) -> PluginSchedule:
        basename = getattr(plugin_cls, 'BASENAME', plugin_cls.__name__.lower())
        output_subdir = getattr(plugin_cls, 'OUTPUT_SUBDIR', None) or basename
        return cls(
            plugin_cls=plugin_cls,
            basename=basename,
            set_primary=bool(getattr(plugin_cls, 'SET_PRIMARY', False)),
            refresh_interval=getattr(plugin_cls, 'REFRESH_INTERVAL', None),
            output_subdir=output_subdir
        )


def _discover_plugin_classes() -> List[Type[PluginBase]]:
    classes: List[Type[PluginBase]] = []
    prefix = f"{plugins_pkg.__name__}."
    for _, module_name, _ in walk_packages(plugins_pkg.__path__, prefix):
        if module_name.endswith('.base'):
            continue
        if (not config.CALIBRATION_PLUGIN_ENABLED) and module_name.endswith('.calibration'):
            continue
        try:
            module = import_module(module_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to import plugin module %s: %s", module_name, exc)
            continue
        for attribute in vars(module).values():
            if not inspect.isclass(attribute):
                continue
            if not issubclass(attribute, PluginBase):
                continue
            if attribute is PluginBase:
                continue
            if attribute.__module__ != module.__name__:
                continue
            if inspect.isabstract(attribute):
                continue
            if not getattr(attribute, 'AUTO_REGISTER', True):
                continue
            classes.append(attribute)
    classes.sort(key=lambda cls: (getattr(cls, 'REGISTRY_ORDER', 100), cls.__name__))
    return classes


def _is_calibration_plugin_class(plugin_cls: Type[PluginBase]) -> bool:
    return plugin_cls.__module__.endswith('.calibration') or getattr(plugin_cls, 'BASENAME', '').startswith('calibration')


def _build_plugin_registry() -> List[PluginSchedule]:
    return [PluginSchedule.from_plugin(plugin_cls) for plugin_cls in _discover_plugin_classes()]


PLUGIN_REGISTRY: List[PluginSchedule] = _build_plugin_registry()


def _asset_paths(basename: str) -> Tuple[str, str]:
    bmp_path = asset_path(f'{basename}.bmp')
    png_path = asset_path(f'{basename}.png')
    return (bmp_path.as_posix(), png_path.as_posix())


def _default_plugin_ttl(plugin_cls: Type[PluginBase]) -> int:
    cached = _ttl_cache.get(plugin_cls.__name__)
    if cached:
        return cached
    try:
        instance = plugin_cls()
        ttl_default = max(60, int(instance.get_content_ttl()))
    except Exception:  # noqa: BLE001
        ttl_default = 900
    _ttl_cache[plugin_cls.__name__] = ttl_default
    return ttl_default


def plugin_schedules() -> List[PluginSchedule]:
    """Return a shallow copy of the plugin registry for iteration."""
    schedules = list(PLUGIN_REGISTRY)
    if not config.CALIBRATION_PLUGIN_ENABLED:
        schedules = [schedule for schedule in schedules if not _is_calibration_plugin_class(schedule.plugin_cls)]
    return schedules


def list_available_plugins() -> List[str]:
    return [schedule.name for schedule in plugin_schedules()]


def get_plugin_schedule(plugin_name: str) -> PluginSchedule:
    normalized = plugin_name.lower()
    for schedule in plugin_schedules():
        if schedule.name.lower() == normalized:
            return schedule
    raise ValueError(f"Unknown plugin '{plugin_name}'. Available: {', '.join(list_available_plugins())}")


async def run_single_plugin_by_name(
    plugin_name: str,
    *,
    output_dir: Optional[str] = None,
    plugin_kwargs: Optional[Dict[str, Any]] = None
) -> PluginOutput:
    schedule = get_plugin_schedule(plugin_name)
    plugin_instance = schedule.plugin_cls()
    kwargs: Dict[str, Any] = dict(plugin_kwargs or {})
    kwargs.setdefault('output_dir', output_dir or schedule.resolved_output_directory())
    result = await plugin_instance.run(**kwargs)
    if not result or not _assets_exist(result):
        raise RuntimeError(f"Plugin {plugin_name} produced no output")
    return result


def _assets_exist(assets: Optional[PluginOutput]) -> bool:
    if not assets:
        return False
    return exists(assets.monochrome_path) and exists(assets.grayscale_path)


def _seconds_until_plugin_refresh(
    plugin_name: str,
    default_interval: int = 900,
    min_interval: int = 60
) -> int:
    with state.STATE_LOCK:
        cache_entry = state.global_state.get('plugins', {}).get(plugin_name, {})
        expires_at = cache_entry.get('expires_at')
    if expires_at:
        delay = max(0, expires_at - time())
        return max(min_interval, int(delay))
    return default_interval


async def process_plugin_output(schedule: PluginSchedule) -> None:
    plugin_cls = schedule.plugin_cls
    plugin_name = plugin_cls.__name__
    fallback_assets = schedule.fallback_assets()
    with state.STATE_LOCK:
        cache_entry = state.global_state['plugins'].setdefault(plugin_name, {})
        cached_assets: Optional[PluginOutput] = cache_entry.get('assets')
        expires_at = cache_entry.get('expires_at', 0)
        display_name: Optional[str] = cache_entry.get('display_name') or getattr(plugin_cls, 'DISPLAY_NAME', None)
    now = time()

    assets_valid = _assets_exist(cached_assets)
    needs_refresh = (not assets_valid) or (expires_at <= now)
    assets = cached_assets if assets_valid else None

    if needs_refresh:
        plugin_instance = plugin_cls()
        output_dir = schedule.resolved_output_directory()
        try:
            refreshed_assets = await plugin_instance.run(output_dir=output_dir)
            if _assets_exist(refreshed_assets):
                ttl_seconds = max(1, plugin_instance.get_content_ttl())
                try:
                    display_name = plugin_instance.get_display_name()
                except Exception:  # noqa: BLE001
                    display_name = getattr(plugin_cls, 'DISPLAY_NAME', None) or plugin_name
                with state.STATE_LOCK:
                    cache_entry = state.global_state['plugins'].setdefault(plugin_name, cache_entry)
                    cache_entry['expires_at'] = time() + ttl_seconds
                    cache_entry['assets'] = refreshed_assets
                    cache_entry['display_name'] = display_name
                assets = refreshed_assets
                logger.info("%s refreshed; next refresh in %.0f seconds", plugin_name, ttl_seconds)
            else:
                logger.warning("%s produced no output; attempting fallback", plugin_name)
                with state.STATE_LOCK:
                    cache_entry = state.global_state['plugins'].setdefault(plugin_name, cache_entry)
                    cache_entry['expires_at'] = time() + PLUGIN_REFRESH_RETRY
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to run %s: %s", plugin_name, exc)
            with state.STATE_LOCK:
                cache_entry = state.global_state['plugins'].setdefault(plugin_name, cache_entry)
                cache_entry['expires_at'] = time() + PLUGIN_REFRESH_RETRY

    if (not assets) and fallback_assets:
        bmp_path, png_path = fallback_assets
        if exists(bmp_path) and exists(png_path):
            assets = PluginOutput(monochrome_path=bmp_path, grayscale_path=png_path)
            with state.STATE_LOCK:
                cache_entry = state.global_state['plugins'].setdefault(plugin_name, cache_entry)
                cache_entry['assets'] = assets

    if not assets:
        logger.warning("No valid asset available for %s", plugin_name)
        return

    if schedule.set_primary:
        state.set_primary_rotation_assets(plugin_name, assets, display_name)
    else:
        state.append_rotation_assets(plugin_name, assets, display_name)


async def refresh_plugin_assets() -> None:
    """Run all plugin refresh operations concurrently off the event loop."""
    schedules = plugin_schedules()
    tasks = [process_plugin_output(schedule) for schedule in schedules]
    if not tasks:
        return
    results = await gather(*tasks, return_exceptions=True)
    for schedule, result in zip(schedules, results):
        if isinstance(result, Exception):
            logger.error("Plugin refresh task failed for %s: %s", schedule.name, result)


async def _plugin_refresh_worker(
    schedule: PluginSchedule
) -> None:
    plugin_name = schedule.name
    logger.info("Starting refresh worker for %s", plugin_name)
    try:
        while True:
            try:
                await process_plugin_output(schedule)
            except Exception as exc:  # noqa: BLE001
                logger.error("Plugin refresh task failed for %s: %s", plugin_name, exc)
            delay = _seconds_until_plugin_refresh(plugin_name, schedule.resolved_refresh_interval(), 60)
            try:
                await sleep(delay)
            except CancelledError:
                logger.info("Plugin refresh worker cancelled for %s", plugin_name)
                raise
    except CancelledError:
        pass


async def start_plugin_refreshers() -> None:
    """Launch background tasks to refresh each plugin independently."""
    if plugin_tasks:
        return
    for schedule in plugin_schedules():
        plugin_name = schedule.name
        task = create_task(_plugin_refresh_worker(schedule))
        plugin_tasks[plugin_name] = task


async def stop_plugin_refreshers() -> None:
    """Cancel all plugin refresh tasks and wait for them to stop."""
    if not plugin_tasks:
        return
    for task in plugin_tasks.values():
        task.cancel()
    await gather(*plugin_tasks.values(), return_exceptions=True)
    plugin_tasks.clear()