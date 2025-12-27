"""Shared device, rotation, and state management utilities."""

from __future__ import annotations

from hashlib import sha1
from os.path import abspath
from time import time
from io import BytesIO
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

from fastapi import Request

from .. import config, models, utils
from ..plugins.base import PluginOutput

logger = config.logger

DEFAULT_DEVICE_ID = 'default'
STATE_LOCK = RLock()
start_time = time()

_IMAGE_TOKEN_TTL_SECONDS = 600.0

_server_base_url = ''


class RotationUnavailableError(RuntimeError):
    """Raised when no rotation entries are available for a device."""
    pass


def set_server_base_url(base_url: str) -> None:
    """Persist the server base URL so freshly created device states have a default."""
    global _server_base_url
    _server_base_url = base_url.rstrip('/') if base_url else ''


def get_server_base_url() -> str:
    return _server_base_url or f"{config.SERVER_SCHEME}://{utils.get_ip_address()}:{config.SERVER_PORT}"


def update_device_preview(
    device_id: str,
    device_state: Optional[Dict[str, Any]],
    entry_index: Optional[int]
) -> Tuple[Optional[str], Optional[str]]:
    if entry_index is None or entry_index < 0:
        return None, None
    target_state = device_state or get_device_state(device_id)
    with STATE_LOCK:
        previous_index = target_state.get('current_preview_entry_index')
        target_state['current_preview_entry_index'] = entry_index
        existing_token = target_state.get('current_preview_token')
        if previous_index == entry_index and existing_token:
            token = str(existing_token)
        else:
            sequence = int(target_state.get('token_sequence', 0))
            token = sha1(f"{device_id}-{entry_index}-{sequence}".encode('utf-8')).hexdigest()[:16]
            target_state['current_preview_token'] = token
        target_state['current_preview_url'] = f"/preview/{quote(device_id)}?token={token}"
        return target_state['current_preview_url'], token


def _client_metrics_store() -> Dict[str, Dict[str, Any]]:
    return global_state.setdefault('client_metrics', {})


def get_client_metrics(device_id: str) -> Dict[str, Any]:
    with STATE_LOCK:
        store = _client_metrics_store()
        metrics = store.get(device_id)
        if metrics is None:
            metrics = {
                'refresh_rate': config.REFRESH_TIME,
                'battery_voltage': config.BATTERY_MAX_VOLTAGE,
                'rssi': -100,
                'last_contact': 0.0
            }
            store[device_id] = metrics
        return metrics


def update_client_metrics(
    device_id: str,
    *,
    refresh_rate: Optional[int] = None,
    battery_voltage: Optional[float] = None,
    rssi: Optional[int] = None
) -> None:
    with STATE_LOCK:
        metrics = get_client_metrics(device_id)
        if refresh_rate is not None:
            metrics['refresh_rate'] = refresh_rate
        if battery_voltage is not None:
            metrics['battery_voltage'] = battery_voltage
        if rssi is not None:
            metrics['rssi'] = rssi
        metrics['last_contact'] = time()


def get_all_client_metrics() -> Dict[str, Dict[str, Any]]:
    with STATE_LOCK:
        return {device_id: dict(metrics) for device_id, metrics in _client_metrics_store().items()}


def _device_profile_cache() -> Dict[str, Dict[str, Any]]:
    return global_state.setdefault('device_profiles', {})


def ensure_device_profile(device_id: str) -> Dict[str, Any]:
    with STATE_LOCK:
        cache = _device_profile_cache()
        profile = cache.get(device_id)
    if profile is None:
        profile = models.ensure_device_profile(device_id)
        with STATE_LOCK:
            cache = _device_profile_cache()
            cache[device_id] = profile
    return profile


def refresh_device_profile(device_id: str) -> Dict[str, Any]:
    profile = models.ensure_device_profile(device_id)
    with STATE_LOCK:
        cache = _device_profile_cache()
        cache[device_id] = profile
    return profile


def update_device_profile(
    device_id: str,
    *,
    friendly_name: Optional[str] = None,
    refresh_interval: Optional[int] = None,
    time_zone: Optional[str] = None
) -> Dict[str, Any]:
    profile = models.update_device_profile(
        device_id,
        friendly_name=friendly_name,
        refresh_interval=refresh_interval,
        time_zone=time_zone
    )
    with STATE_LOCK:
        cache = _device_profile_cache()
        cache[device_id] = profile
    return profile


def get_refresh_interval(device_id: str) -> int:
    profile = ensure_device_profile(device_id)
    refresh_interval = profile.get('refresh_interval')
    if isinstance(refresh_interval, int) and refresh_interval > 0:
        return refresh_interval
    metrics = get_client_metrics(device_id)
    return int(metrics.get('refresh_rate', config.REFRESH_TIME))


def _build_device_state() -> Dict[str, Any]:
    return {
        'bmp_send_switch': True,
        'current_preview_url': None,
        'current_preview_token': None,
        'current_preview_entry_index': None,
        'playlist_ids': [],
        'playlist_indexes': [],
        'playlist_media': [],
        'current_playlist_index': -1,
        'current_entry_index': -1,
        'current_entry_media': 'auto',
        'request_count': 0,
        'last_entry_hash': None,
        'last_entry_plugin': None,
        'pending_entry_index': None,
        'pending_entry_media': None,
        'supports_grayscale': False,
        'grayscale_send_switch': True,
        'token_sequence': 0
    }


global_state: Dict[str, Any] = {
    'rotation_master': {
        'bmp_entries': [],
        'png_entries': [],
        'hashes': [],
        'meta': [],
        'selected_ids': [],
        'version': 0,
        'has_persistent_playlist': False
    },
    'devices': {},
    'device_playlists': {},
    'device_profiles': {},
    'client_metrics': {},
    'server': {
        'uptime': 0,
        'cpu_load': 0,
        'current_time': 0
    },
    'client': {
        'battery_voltage': 0,
        'battery_voltage_max': 0
    },
    'plugins': {}
}


def _image_token_store() -> Dict[str, Dict[str, Any]]:
    return global_state.setdefault('image_tokens', {})


def register_image_token(device_id: str, token: str) -> None:
    normalized_device = (device_id or '').strip() or DEFAULT_DEVICE_ID
    if normalized_device == DEFAULT_DEVICE_ID:
        return
    token_value = (token or '').strip()
    if not token_value:
        return
    now = time()
    cutoff = now - _IMAGE_TOKEN_TTL_SECONDS
    with STATE_LOCK:
        store = _image_token_store()
        store[token_value] = {'device_id': normalized_device, 'ts': now}
        for key, payload in list(store.items()):
            if not isinstance(payload, dict):
                store.pop(key, None)
                continue
            ts = payload.get('ts')
            if not isinstance(ts, (int, float)) or ts < cutoff:
                store.pop(key, None)


def resolve_device_id_from_token(token: Optional[str]) -> Optional[str]:
    token_value = (token or '').strip()
    if not token_value:
        return None
    now = time()
    cutoff = now - _IMAGE_TOKEN_TTL_SECONDS
    with STATE_LOCK:
        store = _image_token_store()
        payload = store.get(token_value)
        if not isinstance(payload, dict):
            return None
        ts = payload.get('ts')
        if not isinstance(ts, (int, float)) or ts < cutoff:
            store.pop(token_value, None)
            return None
        device_id = payload.get('device_id')
        if isinstance(device_id, str) and device_id.strip() and device_id.strip() != DEFAULT_DEVICE_ID:
            return device_id.strip()
    return None


def _read_image_bytes(image_path: str) -> bytes:
    with open(image_path, 'rb') as file:
        return file.read()


def _digest_bytes(data: bytes) -> str:
    return sha1(data).hexdigest()


def _rotation_entry_id(plugin_name: str, bmp_path: str, png_path: str) -> str:
    plugin = (plugin_name or '').strip()
    bmp_url = utils.path_to_web_url(bmp_path)
    png_url = utils.path_to_web_url(png_path)
    return f"{plugin}:{bmp_url}|{png_url}"


def _collision_safe_rotation_id(preferred: str, disallowed: set[str]) -> str:
    if preferred not in disallowed:
        return preferred
    for idx in range(1, 100):
        candidate = f"{preferred}#{idx}"
        if candidate not in disallowed:
            return candidate
    return f"{preferred}#{len(disallowed) + 1}"


def rotation_master() -> Dict[str, Any]:
    """Return the shared rotation master structure."""
    return global_state['rotation_master']


def _selected_matches_all(master: Dict[str, Any]) -> bool:
    selected_ids = master.get('selected_ids') or []
    meta = master.get('meta') or []
    ids = [entry.get('id') for entry in meta if entry.get('id')]
    if len(selected_ids) != len(ids):
        return False
    return all(lhs == rhs for lhs, rhs in zip(selected_ids, ids))


def _prune_missing_selected_ids(master: Dict[str, Any]) -> bool:
    selected_ids = master.get('selected_ids') or []
    if not selected_ids:
        return False
    meta = master.get('meta') or []
    valid_ids = {entry.get('id') for entry in meta if entry.get('id')}
    filtered_ids: List[str] = []
    for entry in selected_ids:
        base_id, _ = _parse_playlist_entry_id(entry)
        if base_id in valid_ids:
            filtered_ids.append(entry)
    if len(filtered_ids) == len(selected_ids):
        return False
    master['selected_ids'] = filtered_ids
    return True


def persist_default_playlist(selected_ids: List[str]) -> None:
    try:
        models.save_rotation_playlist(selected_ids)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to persist default playlist: %s", exc)


def persist_device_playlist(device_id: str, selected_ids: List[str]) -> None:
    try:
        models.save_rotation_playlist(selected_ids, device_id=device_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to persist playlist for %s: %s", device_id, exc)


def cache_device_playlist(device_id: str, selected_ids: Optional[List[str]]) -> None:
    with STATE_LOCK:
        playlists = global_state.setdefault('device_playlists', {})
        if selected_ids is None:
            playlists.pop(device_id, None)
        else:
            playlists[device_id] = selected_ids


def _named_playlists_store() -> Dict[str, List[str]]:
    return global_state.setdefault('named_playlists', {})


def cache_named_playlist(name: str, selected_ids: Optional[List[str]]) -> None:
    normalized = (name or '').strip()
    if not normalized or normalized == DEFAULT_DEVICE_ID:
        return
    with STATE_LOCK:
        store = _named_playlists_store()
        if selected_ids is None:
            store.pop(normalized, None)
        else:
            store[normalized] = list(selected_ids)


def _device_playlist_binding_store() -> Dict[str, str]:
    return global_state.setdefault('device_playlist_bindings', {})


def cache_device_playlist_binding(device_id: str, playlist_name: Optional[str]) -> None:
    normalized_device = (device_id or '').strip() or DEFAULT_DEVICE_ID
    normalized_name = (playlist_name or '').strip() or DEFAULT_DEVICE_ID
    if normalized_device == DEFAULT_DEVICE_ID:
        return
    with STATE_LOCK:
        store = _device_playlist_binding_store()
        if normalized_name == DEFAULT_DEVICE_ID:
            store.pop(normalized_device, None)
        else:
            store[normalized_device] = normalized_name


def get_device_playlist_binding_name(device_id: str) -> Optional[str]:
    normalized_device = (device_id or '').strip() or DEFAULT_DEVICE_ID
    if normalized_device == DEFAULT_DEVICE_ID:
        return None
    with STATE_LOCK:
        store = _device_playlist_binding_store()
        cached = store.get(normalized_device)
        if cached:
            return cached
    bound = models.get_device_playlist_binding(normalized_device)
    cache_device_playlist_binding(normalized_device, bound)
    if bound and bound.strip() and bound.strip() != DEFAULT_DEVICE_ID:
        return bound.strip()
    return None


def get_named_playlist_selection(name: str) -> Optional[List[str]]:
    normalized = (name or '').strip()
    if not normalized or normalized == DEFAULT_DEVICE_ID:
        return None
    with STATE_LOCK:
        store = _named_playlists_store()
        cached = store.get(normalized)
        if cached is not None:
            return list(cached)
    selected = models.get_rotation_playlist(device_id=None, name=normalized)
    cache_named_playlist(normalized, selected)
    if selected is None:
        return None
    return list(selected)


def _ensure_device_playlist_cached(device_id: str) -> Optional[List[str]]:
    with STATE_LOCK:
        playlists = global_state.setdefault('device_playlists', {})
        if device_id in playlists:
            return playlists[device_id]
    selected_ids = models.get_rotation_playlist(device_id=device_id)
    cache_device_playlist(device_id, selected_ids)
    return selected_ids


def initialize_rotation_playlists_from_storage() -> None:
    default_playlist = models.get_rotation_playlist()
    if default_playlist is not None:
        with STATE_LOCK:
            master = rotation_master()
            master['selected_ids'] = default_playlist
            master['has_persistent_playlist'] = True
    for device_id, selected_ids in models.list_device_playlists():
        cache_device_playlist(device_id, selected_ids)

    for name, selected_ids in models.list_named_rotation_playlists():
        cache_named_playlist(name, selected_ids)

    for device_id, playlist_name in models.list_device_playlist_bindings():
        cache_device_playlist_binding(device_id, playlist_name)


def _replace_selected_hash(master: Dict[str, Any], previous_hash: Optional[str], new_hash: str) -> None:
    if not previous_hash:
        return
    selected_ids = master.get('selected_ids')
    if not selected_ids:
        return
    master['selected_ids'] = [new_hash if entry == previous_hash else entry for entry in selected_ids]


def _selected_ids_for_device(device_id: Optional[str]) -> List[str]:
    if device_id:
        bound_name = get_device_playlist_binding_name(device_id)
        if bound_name:
            bound_playlist = get_named_playlist_selection(bound_name)
            if bound_playlist:
                return list(bound_playlist)
        cached = _ensure_device_playlist_cached(device_id)
        if cached is not None:
            return list(cached)
    master = rotation_master()
    selected_ids = master.get('selected_ids') or []
    if selected_ids:
        return list(selected_ids)
    meta = master.get('meta') or []
    return [entry.get('id') for entry in meta if entry.get('id')]


def get_playlist_selection(device_id: Optional[str] = None) -> List[str]:
    return _selected_ids_for_device(device_id)


def known_device_ids(include_default: bool = True) -> List[str]:
    with STATE_LOCK:
        ids = set(global_state.get('devices', {}).keys())
        ids.update((global_state.get('device_playlists') or {}).keys())
        ids.update((global_state.get('device_playlist_bindings') or {}).keys())
        ids.update((global_state.get('client_metrics') or {}).keys())
        ids.update((global_state.get('device_profiles') or {}).keys())
    if include_default:
        ids.add(DEFAULT_DEVICE_ID)
    return sorted(ids)


def list_playlist_targets(include_default: bool = True) -> List[Dict[str, Any]]:
    assignments: List[Dict[str, Any]] = []
    if include_default:
        default_playlist = _selected_ids_for_device(None)
        assignments.append({
            'device_id': DEFAULT_DEVICE_ID,
            'friendly_name': 'Default playlist',
            'playlist': list(default_playlist),
            'count': len(default_playlist)
        })
    with STATE_LOCK:
        overrides = list((global_state.get('device_playlists') or {}).items())
    for device_id, playlist_ids in sorted(overrides, key=lambda item: item[0]):
        current_playlist = list(playlist_ids or [])
        profile = ensure_device_profile(device_id)
        assignments.append({
            'device_id': device_id,
            'friendly_name': profile.get('friendly_name') or device_id,
            'playlist': current_playlist,
            'count': len(current_playlist)
        })
    return assignments


def set_named_playlist(name: str, playlist_ids: List[str]) -> None:
    normalized = (name or '').strip()
    if not normalized or normalized == DEFAULT_DEVICE_ID:
        raise ValueError('invalid playlist name')
    if not playlist_ids:
        raise ValueError('playlist must contain at least one active entry')
    validate_playlist_ids(playlist_ids)
    models.save_rotation_playlist(list(playlist_ids), device_id=None, name=normalized)
    cache_named_playlist(normalized, list(playlist_ids))


def delete_named_playlist(name: str) -> None:
    normalized = (name or '').strip()
    if not normalized or normalized == DEFAULT_DEVICE_ID:
        raise ValueError('default playlist cannot be deleted')

    with STATE_LOCK:
        bindings = dict(_device_playlist_binding_store())
    for device_id, playlist_name in bindings.items():
        if playlist_name == normalized:
            models.delete_device_playlist_binding(device_id)
            cache_device_playlist_binding(device_id, None)
            with STATE_LOCK:
                device_state = get_device_state(device_id)
                device_state['request_count'] = 0
                device_state['current_playlist_index'] = -1
                device_state['current_entry_index'] = -1
                device_state['last_entry_hash'] = None
                device_state['last_entry_plugin'] = None
            persist_device_state(device_id, 0, -1, [], None, None)

    models.delete_named_rotation_playlist(normalized)
    cache_named_playlist(normalized, None)


def set_device_playlist_binding(device_id: str, playlist_name: Optional[str]) -> None:
    normalized_device = (device_id or '').strip() or DEFAULT_DEVICE_ID
    if normalized_device == DEFAULT_DEVICE_ID:
        raise ValueError('invalid device id')

    normalized_name = (playlist_name or '').strip() or DEFAULT_DEVICE_ID
    # Binding is the primary assignment mechanism; clear any legacy per-device overrides
    # so the binding is reflected immediately in rotation selection.
    cache_device_playlist(normalized_device, None)
    try:
        models.delete_rotation_playlist(device_id=normalized_device)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to delete legacy rotation playlist for %s: %s", normalized_device, exc)

    if normalized_name != DEFAULT_DEVICE_ID:
        selection = get_named_playlist_selection(normalized_name)
        if not selection:
            raise ValueError('unknown playlist')
        models.set_device_playlist_binding(normalized_device, normalized_name)
        cache_device_playlist_binding(normalized_device, normalized_name)
    else:
        models.delete_device_playlist_binding(normalized_device)
        cache_device_playlist_binding(normalized_device, None)

    with STATE_LOCK:
        device_state = get_device_state(normalized_device)
        device_state['request_count'] = 0
        device_state['current_playlist_index'] = -1
        device_state['current_entry_index'] = -1
        device_state['pending_entry_index'] = None
        device_state['last_entry_hash'] = None
        device_state['last_entry_plugin'] = None
    persist_device_state(normalized_device, 0, -1, [], None, None)


def _playlist_index_map(master: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    if master is None:
        master = rotation_master()
    meta = master.get('meta') or []
    return {entry.get('id'): idx for idx, entry in enumerate(meta) if entry.get('id')}


def _parse_playlist_entry_id(entry_id: str) -> Tuple[str, Optional[str]]:
    """Parse playlist entry IDs with optional media directives.

    Supported suffixes: '@bmp'/'@mono' forces 1-bit BMP delivery; '@png'/'@gray'
    forces grayscale PNG delivery; '@auto' uses the default device behavior.

    Returns (base_id, media) where media is 'bmp', 'png', or None.
    """
    raw = (entry_id or '').strip()
    if not raw or '@' not in raw:
        return raw, None

    base, suffix = raw.rsplit('@', 1)
    suffix_norm = suffix.strip().lower()
    if suffix_norm in ('bmp', 'mono', '1bit', '1-bit'):
        return base, 'bmp'
    if suffix_norm in ('png', 'gray', 'grayscale'):
        return base, 'png'
    if suffix_norm in ('auto',):
        return base, None
    return raw, None


def _resolved_playlist_for_device(device_id: Optional[str]) -> Tuple[List[str], List[int]]:
    master = rotation_master()
    playlist_ids = _selected_ids_for_device(device_id)
    if not playlist_ids:
        playlist_ids = [entry.get('id') for entry in master.get('meta') or [] if entry.get('id')]
    mapping = _playlist_index_map(master)
    resolved_ids: List[str] = []
    resolved_indexes: List[int] = []
    resolved_media: List[str] = []
    for entry_id in playlist_ids:
        base_id, media = _parse_playlist_entry_id(entry_id)
        index = mapping.get(base_id)
        if index is None:
            continue
        resolved_ids.append(entry_id)
        resolved_indexes.append(index)
        resolved_media.append(media or 'auto')
    return resolved_ids, resolved_indexes, resolved_media


def _update_playlist_snapshot(device_id: str, device_state: Dict[str, Any]) -> Tuple[List[str], List[int]]:
    resolved_ids, resolved_indexes, resolved_media = _resolved_playlist_for_device(device_id)
    device_state['playlist_ids'] = list(resolved_ids)
    device_state['playlist_indexes'] = list(resolved_indexes)
    device_state['playlist_media'] = list(resolved_media)
    if resolved_indexes:
        max_index = len(resolved_indexes) - 1
        if device_state.get('current_playlist_index', -1) > max_index:
            device_state['current_playlist_index'] = -1
    else:
        device_state['current_playlist_index'] = -1
        device_state['current_entry_index'] = -1
        device_state['current_entry_media'] = 'auto'
    return device_state['playlist_ids'], device_state['playlist_indexes']


def _select_next_playlist_entry(device_id: str, device_state: Dict[str, Any]) -> Tuple[int, str]:
    with STATE_LOCK:
        _, playlist_indexes = _update_playlist_snapshot(device_id, device_state)
        if not playlist_indexes:
            raise RotationUnavailableError(f'No rotation entries are available for {device_id}')
        request_count = int(device_state.get('request_count') or 0)
        position = request_count % len(playlist_indexes)
        entry_index = playlist_indexes[position]
        playlist_media = device_state.get('playlist_media') or []
        media = str(playlist_media[position]) if position < len(playlist_media) else 'auto'
        if media not in ('bmp', 'png'):
            media = 'auto'
        master = rotation_master()
        meta_list = master.get('meta') or []
        if entry_index < 0 or entry_index >= len(meta_list):
            raise RotationUnavailableError(f'Playlist entry {entry_index} is invalid for {device_id}')
        entry = meta_list[entry_index]
        entry_id = entry.get('id')
        plugin_id = entry.get('plugin')
        device_state['request_count'] = request_count + 1
        device_state['current_playlist_index'] = position
        device_state['current_entry_index'] = entry_index
        device_state['current_entry_media'] = media
        device_state['last_entry_hash'] = entry_id
        device_state['last_entry_plugin'] = plugin_id
        playlist_snapshot = list(playlist_indexes)
        request_snapshot = device_state['request_count']
    persist_device_state(
        device_id,
        request_snapshot,
        position,
        playlist_snapshot,
        entry_id,
        plugin_id
    )
    return entry_index, media


def validate_playlist_ids(playlist_ids: List[str]) -> None:
    mapping = _playlist_index_map()
    unknown: List[str] = []
    for pid in playlist_ids:
        base_id, _ = _parse_playlist_entry_id(pid)
        if base_id not in mapping:
            unknown.append(pid)
    if unknown:
        raise ValueError(f'unknown playlist ids: {unknown}')


def set_default_playlist(playlist_ids: List[str]) -> None:
    master = rotation_master()
    mapping = _playlist_index_map(master)
    unknown: List[str] = []
    for pid in playlist_ids:
        base_id, _ = _parse_playlist_entry_id(pid)
        if base_id not in mapping:
            unknown.append(pid)
    if unknown:
        raise ValueError(f'unknown playlist ids: {unknown}')
    with STATE_LOCK:
        master = rotation_master()
        master['selected_ids'] = playlist_ids
        master['has_persistent_playlist'] = False  # Reset flag when user explicitly sets playlist
        master['version'] += 1
        selection_snapshot = list(master['selected_ids'])
    persist_default_playlist(selection_snapshot)


def set_device_playlist(device_id: str, playlist_ids: List[str]) -> None:
    master = rotation_master()
    mapping = _playlist_index_map(master)
    unknown: List[str] = []
    resolved_indexes: List[int] = []
    for pid in playlist_ids:
        base_id, _ = _parse_playlist_entry_id(pid)
        idx = mapping.get(base_id)
        if idx is None:
            unknown.append(pid)
        else:
            resolved_indexes.append(idx)
    if unknown:
        raise ValueError(f'unknown playlist ids: {unknown}')

    playlist_indexes = resolved_indexes
    with STATE_LOCK:
        device_state = get_device_state(device_id)
        device_state['playlist_ids'] = list(playlist_ids)
        device_state['playlist_indexes'] = list(playlist_indexes)
        device_state['playlist_media'] = []
        device_state['request_count'] = 0
        device_state['current_playlist_index'] = -1
        device_state['current_entry_index'] = -1
        device_state['current_entry_media'] = 'auto'
        device_state['last_entry_hash'] = None
        device_state['last_entry_plugin'] = None
        device_state['pending_entry_media'] = None
        cache_device_playlist(device_id, list(playlist_ids))
    persist_device_playlist(device_id, list(playlist_ids))
    persist_device_state(device_id, 0, -1, playlist_indexes, None, None)


def clear_device_playlist(device_id: str) -> None:
    with STATE_LOCK:
        playlists = global_state.setdefault('device_playlists', {})
        playlists.pop(device_id, None)
        devices = global_state.setdefault('devices', {})
        device_state = devices.setdefault(device_id, _build_device_state())
        device_state['playlist_ids'] = []
        device_state['playlist_indexes'] = []
        device_state['request_count'] = 0
        device_state['current_playlist_index'] = -1
        device_state['current_entry_index'] = -1
        device_state['last_entry_hash'] = None
        device_state['last_entry_plugin'] = None

    models.delete_rotation_playlist(device_id=device_id)
    persist_device_state(device_id, 0, -1, [], None, None)


def persist_device_state(
    device_id: str,
    request_count: int,
    playlist_position: int,
    playlist_indexes: List[int],
    current_entry_id: Optional[str],
    current_plugin_id: Optional[str]
) -> None:
    try:
        snapshot = [str(idx) for idx in playlist_indexes]
        models.save_device_state(
            device_id=device_id,
            rotation_version=request_count,
            rotation_index=playlist_position,
            rotation_hash_order=snapshot,
            last_entry_hash=current_entry_id,
            current_plugin_id=current_plugin_id
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to persist device state for %s: %s", device_id, exc)


def schedule_next_rotation_entry(device_id: str, device_state: Dict[str, Any]) -> int:
    entry_idx, media = _select_next_playlist_entry(device_id, device_state)
    with STATE_LOCK:
        device_state['pending_entry_index'] = entry_idx
        device_state['pending_entry_media'] = media
    return entry_idx


def _pop_pending_entry_index(device_state: Dict[str, Any]) -> Optional[int]:
    with STATE_LOCK:
        pending = device_state.get('pending_entry_index')
        device_state['pending_entry_index'] = None
        device_state['pending_entry_media'] = None
        return pending


def _resolve_next_entry(device_id: str, device_state: Dict[str, Any]) -> int:
    pending = _pop_pending_entry_index(device_state)
    if pending is not None:
        return pending
    entry_idx, _ = _select_next_playlist_entry(device_id, device_state)
    return entry_idx


def _rotation_frame_bytes(entry_idx: int, media: str) -> bytes:
    key = 'bmp_entries' if media == 'bmp' else 'png_entries'
    with STATE_LOCK:
        master = rotation_master()
        entries = master.get(key) or []
        if entry_idx < 0 or entry_idx >= len(entries):
            raise RotationUnavailableError(f'Rotation index {entry_idx} is out of range for {media.upper()} frames')
        return entries[entry_idx]


def get_rotation_bmp_bytes(entry_idx: int) -> bytes:
    return _rotation_frame_bytes(entry_idx, 'bmp')


def get_rotation_png_bytes(entry_idx: int) -> bytes:
    return _rotation_frame_bytes(entry_idx, 'png')


def current_frame_entry_index(device_state: Dict[str, Any]) -> Optional[int]:
    with STATE_LOCK:
        pending = device_state.get('pending_entry_index')
        if isinstance(pending, int) and pending >= 0:
            return pending
        current = device_state.get('current_entry_index')
        if isinstance(current, int) and current >= 0:
            return current
    return None


def preview_frame_entry_index(device_state: Dict[str, Any]) -> Optional[int]:
    with STATE_LOCK:
        preview = device_state.get('current_preview_entry_index')
        if isinstance(preview, int) and preview >= 0:
            return preview
    return None


def get_next_bmp_frame(device_id: str, device_state: Dict[str, Any]) -> BytesIO:
    entry_idx = _resolve_next_entry(device_id, device_state)
    bmp_bytes = get_rotation_bmp_bytes(entry_idx)
    return BytesIO(bmp_bytes)


def get_next_png_frame(device_id: str, device_state: Dict[str, Any]) -> BytesIO:
    entry_idx = _resolve_next_entry(device_id, device_state)
    png_bytes = get_rotation_png_bytes(entry_idx)
    return BytesIO(png_bytes)


def _build_rotation_meta(
    plugin_name: str,
    bmp_path: str,
    png_path: str,
    entry_id: str,
    content_hash: str,
    display_name: Optional[str] = None
) -> Dict[str, Any]:
    label = display_name or plugin_name
    return {
        'id': entry_id,
        'hash': content_hash,
        'plugin': plugin_name,
        'label': label,
        'bmp_path': abspath(bmp_path),
        'png_path': abspath(png_path),
        'url_bmp': utils.path_to_web_url(bmp_path),
        'url_png': utils.path_to_web_url(png_path)
    }


def set_primary_rotation_assets(
    plugin_name: str,
    assets: PluginOutput,
    display_name: Optional[str] = None
) -> None:
    bmp_bytes = _read_image_bytes(assets.monochrome_path)
    png_bytes = _read_image_bytes(assets.grayscale_path)
    content_hash = _digest_bytes(b"|".join((bmp_bytes, png_bytes)))
    preferred_id = _rotation_entry_id(plugin_name, assets.monochrome_path, assets.grayscale_path)

    selection_snapshot: Optional[List[str]] = None
    with STATE_LOCK:
        master = rotation_master()
        if _prune_missing_selected_ids(master):
            selection_snapshot = list(master.get('selected_ids') or [])
        bmp_entries = master.setdefault('bmp_entries', [])
        png_entries = master.setdefault('png_entries', [])
        hashes = master.setdefault('hashes', [])
        meta_list = master.setdefault('meta', [])
        had_entries = bool(bmp_entries)
        auto_fill_enabled = _selected_matches_all(master)

        previous_id = None
        if had_entries and meta_list:
            previous_id = meta_list[0].get('id')
        disallowed_ids = {entry.get('id') for entry in meta_list if entry.get('id')}
        if previous_id:
            disallowed_ids.discard(previous_id)
        entry_id = previous_id or preferred_id
        entry_id = _collision_safe_rotation_id(entry_id, disallowed_ids)
        meta_entry = _build_rotation_meta(
            plugin_name,
            assets.monochrome_path,
            assets.grayscale_path,
            entry_id,
            content_hash,
            display_name
        )

        if bmp_entries:
            bmp_entries[0] = bmp_bytes
            png_entries[0] = png_bytes
            hashes[0] = content_hash
            if meta_list:
                meta_list[0] = meta_entry
            else:
                meta_list.append(meta_entry)
        else:
            bmp_entries.append(bmp_bytes)
            png_entries.append(png_bytes)
            hashes.append(content_hash)
            meta_list.append(meta_entry)

        if not had_entries:
            master['version'] += 1

        # Only auto-fill if there's no existing playlist AND no persistent playlist was loaded
        # This prevents overwriting user-defined playlists during plugin refresh
        if not master.get('selected_ids') and not master.get('has_persistent_playlist'):
            master['selected_ids'] = [entry.get('id') for entry in meta_list if entry.get('id')]
            selection_snapshot = list(master['selected_ids'])

    if selection_snapshot is not None:
        persist_default_playlist(selection_snapshot)


def append_rotation_assets(
    plugin_name: str,
    assets: PluginOutput,
    display_name: Optional[str] = None
) -> None:
    bmp_bytes = _read_image_bytes(assets.monochrome_path)
    png_bytes = _read_image_bytes(assets.grayscale_path)
    content_hash = _digest_bytes(b"|".join((bmp_bytes, png_bytes)))
    preferred_id = _rotation_entry_id(plugin_name, assets.monochrome_path, assets.grayscale_path)
    selection_snapshot: Optional[List[str]] = None
    replaced_entry = False
    skip_append = False
    with STATE_LOCK:
        master = rotation_master()
        if _prune_missing_selected_ids(master):
            selection_snapshot = list(master.get('selected_ids') or [])
        hashes = master.setdefault('hashes', [])
        meta_list = master.setdefault('meta', [])
        bmp_entries = master.setdefault('bmp_entries', [])
        png_entries = master.setdefault('png_entries', [])
        auto_fill_enabled = _selected_matches_all(master)

        replace_index = None
        for idx, meta_entry in enumerate(meta_list):
            if meta_entry.get('plugin') == plugin_name:
                replace_index = idx
                break

        if replace_index is not None:
            previous_id = meta_list[replace_index].get('id') if replace_index < len(meta_list) else None
            disallowed_ids = {entry.get('id') for entry in meta_list if entry.get('id')}
            if previous_id:
                disallowed_ids.discard(previous_id)
            entry_id = previous_id or preferred_id
            entry_id = _collision_safe_rotation_id(entry_id, disallowed_ids)
            bmp_entries[replace_index] = bmp_bytes
            png_entries[replace_index] = png_bytes
            hashes[replace_index] = content_hash
            meta_list[replace_index] = _build_rotation_meta(
                plugin_name,
                assets.monochrome_path,
                assets.grayscale_path,
                entry_id,
                content_hash,
                display_name
            )
            selection_snapshot = list(master.get('selected_ids') or [])
            replaced_entry = True

        if replace_index is None:
            disallowed_ids = {entry.get('id') for entry in meta_list if entry.get('id')}
            entry_id = _collision_safe_rotation_id(preferred_id, disallowed_ids)

            bmp_entries.append(bmp_bytes)
            png_entries.append(png_bytes)
            hashes.append(content_hash)
            meta_list.append(
                _build_rotation_meta(
                    plugin_name,
                    assets.monochrome_path,
                    assets.grayscale_path,
                    entry_id,
                    content_hash,
                    display_name
                )
            )
            master['version'] += 1
            # Only auto-fill if there's no existing playlist AND no persistent playlist was loaded
            # This prevents overwriting user-defined playlists during plugin refresh
            if not master.get('selected_ids') and not master.get('has_persistent_playlist'):
                master['selected_ids'] = [entry.get('id') for entry in meta_list if entry.get('id')]
                selection_snapshot = list(master['selected_ids'])

    if selection_snapshot is not None:
        persist_default_playlist(selection_snapshot)

    if skip_append or replaced_entry:
        return


def build_rotation_snapshot() -> Dict[str, Any]:
    persist_default: Optional[List[str]] = None
    persist_devices: Dict[str, List[str]] = {}
    persist_named: Dict[str, List[str]] = {}

    with STATE_LOCK:
        master = rotation_master()
        if _prune_missing_selected_ids(master):
            persist_default = list(master.get('selected_ids') or [])

        meta = list(master.get('meta') or [])
        selected_ids = list(master.get('selected_ids') or [])
        version = master.get('version', 0)

        device_playlists = {
            device_id: list(playlist or [])
            for device_id, playlist in (global_state.get('device_playlists') or {}).items()
            if playlist is not None
        }
        named_playlists = {
            name: list(value or [])
            for name, value in (global_state.get('named_playlists') or {}).items()
            if value is not None
        }
        bindings = {
            device_id: playlist_name
            for device_id, playlist_name in (global_state.get('device_playlist_bindings') or {}).items()
            if playlist_name and playlist_name != DEFAULT_DEVICE_ID
        }
    entries: List[Dict[str, Any]] = []
    for entry in meta:
        entries.append({
            'id': entry.get('id'),
            'label': entry.get('label'),
            'plugin': entry.get('plugin'),
            'url_png': entry.get('url_png'),
            'url_bmp': entry.get('url_bmp')
        })

    valid_ids = {e.get('id') for e in entries if e.get('id')}
    entry_fallback = [e.get('id') for e in entries if e.get('id')]

    # Hygiene: prune IDs that no longer exist in the current rotation entries.
    original_default = list(selected_ids)
    selected_ids = [entry_id for entry_id in selected_ids if _parse_playlist_entry_id(entry_id)[0] in valid_ids]
    if selected_ids != original_default:
        persist_default = list(selected_ids)
        with STATE_LOCK:
            master = rotation_master()
            master['selected_ids'] = list(selected_ids)

    for device_id, playlist in list(device_playlists.items()):
        original = list(playlist or [])
        filtered = [entry_id for entry_id in original if _parse_playlist_entry_id(entry_id)[0] in valid_ids]
        if filtered != original:
            device_playlists[device_id] = filtered
            persist_devices[device_id] = filtered
            with STATE_LOCK:
                global_state.setdefault('device_playlists', {})[device_id] = list(filtered)

    for name, playlist in list(named_playlists.items()):
        original = list(playlist or [])
        filtered = [entry_id for entry_id in original if _parse_playlist_entry_id(entry_id)[0] in valid_ids]
        if filtered != original:
            named_playlists[name] = filtered
            persist_named[name] = filtered
            with STATE_LOCK:
                global_state.setdefault('named_playlists', {})[name] = list(filtered)

    default_playlist = selected_ids or entry_fallback
    payload = {
        'version': version,
        'playlists': {
            'default': default_playlist
        },
        'entries': entries
    }
    if device_playlists:
        payload['playlists']['devices'] = device_playlists
    if named_playlists:
        payload['playlists']['named'] = named_playlists
    if bindings:
        payload['playlists']['bindings'] = bindings

    # Persist pruned playlists back to SQLite (outside the state lock).
    if persist_default is not None:
        persist_default_playlist(list(persist_default))
    for device_id, playlist in persist_devices.items():
        persist_device_playlist(device_id, list(playlist))
    for name, playlist in persist_named.items():
        try:
            models.save_rotation_playlist(list(playlist), device_id=None, name=name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to persist named playlist '%s': %s", name, exc)

    return payload


def get_rotation_entry(entry_hash: Optional[str]) -> Optional[Dict[str, Any]]:
    if not entry_hash:
        return None
    master = rotation_master()
    for entry in master.get('meta') or []:
        if entry.get('hash') == entry_hash or entry.get('id') == entry_hash:
            return entry
    return None


def _decode_persisted_playlist_indexes(device_id: str, stored_indexes: List[Any]) -> List[int]:
    parsed: List[int] = []
    if not stored_indexes:
        return parsed
    mapping: Optional[Dict[str, int]] = None
    for value in stored_indexes:
        if isinstance(value, int):
            parsed.append(value)
            continue
        try:
            parsed.append(int(value))
            continue
        except (TypeError, ValueError):
            entry_id = str(value).strip()
            if not entry_id:
                continue
            if mapping is None:
                mapping = _playlist_index_map()
            resolved_index = mapping.get(entry_id)
            if resolved_index is None:
                logger.info("Dropping legacy rotation entry '%s' for %s", entry_id, device_id)
                continue
            parsed.append(resolved_index)
    return parsed


def get_device_state(device_id: str) -> Dict[str, Any]:
    created = False
    with STATE_LOCK:
        devices = global_state.setdefault('devices', {})
        if device_id not in devices:
            devices[device_id] = _build_device_state()
            created = True
        state = devices[device_id]
    ensure_device_profile(device_id)
    get_client_metrics(device_id)
    if created:
        persisted = models.get_device_state(device_id)
        if persisted:
            with STATE_LOCK:
                state['request_count'] = int(persisted.get('rotation_version') or 0)
                state['current_playlist_index'] = int(persisted.get('rotation_index', -1))
                stored_indexes = persisted.get('rotation_hash_order') or []
                parsed_indexes = _decode_persisted_playlist_indexes(device_id, stored_indexes)
                state['playlist_indexes'] = parsed_indexes
                state['last_entry_hash'] = persisted.get('last_entry_hash')
                state['last_entry_plugin'] = persisted.get('current_plugin_id')
                if 0 <= state['current_playlist_index'] < len(parsed_indexes):
                    state['current_entry_index'] = parsed_indexes[state['current_playlist_index']]
                else:
                    state['current_playlist_index'] = -1
                    state['current_entry_index'] = -1
    if 'grayscale_send_switch' not in state:
        state['grayscale_send_switch'] = True
    return state


def _extract_device_id(request: Optional[Request]) -> str:
    if not request:
        return DEFAULT_DEVICE_ID
    for header in ('x-device-id', 'device-id', 'id'):
        value = request.headers.get(header)
        if value:
            return value.strip()
    query_params = request.query_params
    for key in ('device', 'device_id', 'id', 'deviceId'):
        candidate = query_params.get(key)
        if candidate:
            return candidate.strip()
    return DEFAULT_DEVICE_ID


def request_base_url(request: Optional[Request]) -> str:
    if not request:
        return get_server_base_url()
    base = str(request.base_url)
    if base:
        return base[:-1] if base.endswith('/') else base
    host = request.headers.get('host')
    scheme = request.url.scheme if request.url.scheme else config.SERVER_SCHEME
    if host:
        return f"{scheme}://{host}"
    return get_server_base_url()


def get_device_state_from_request(request: Request) -> Tuple[str, Dict[str, Any]]:
    device_id = _extract_device_id(request)
    return device_id, get_device_state(device_id)
