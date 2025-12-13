"""API routes for TRMNL local server."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from hashlib import sha256
from time import time
import psutil
from fastapi import APIRouter, Body, Header, Query, Request
from fastapi.responses import JSONResponse, Response

from .. import config, models, utils
from ..services import plugins, state

router = APIRouter()
logger = config.logger


def _serialize_device_payload(device_id: str) -> Dict[str, Any]:
    profile = state.ensure_device_profile(device_id)
    metrics = state.get_client_metrics(device_id)
    device_state = state.get_device_state(device_id)
    playlist = state.get_playlist_selection(device_id)
    binding_name = state.get_device_playlist_binding_name(device_id) or state.DEFAULT_DEVICE_ID
    current_entry_hash = device_state.get('last_entry_hash')
    preview_url = device_state.get('current_preview_url') or f"/preview/{quote(device_id)}"
    preview_token = device_state.get('current_preview_token')
    return {
        'device_id': device_id,
        'friendly_name': profile.get('friendly_name') or device_id,
        'refresh_interval': state.get_refresh_interval(device_id),
        'playlist_name': binding_name,
        'playlist': playlist,
        'metrics': {
            'refresh_rate': metrics.get('refresh_rate'),
            'battery_voltage': metrics.get('battery_voltage'),
            'battery_state': utils.get_battery_state(metrics.get('battery_voltage')),
            'rssi': metrics.get('rssi'),
            'last_contact': utils.to_iso_timestamp(metrics.get('last_contact'))
        },
        'profile': {
            'refresh_interval': profile.get('refresh_interval'),
            'time_zone': profile.get('time_zone'),
            'last_seen': utils.to_iso_datetime(profile.get('last_seen')) if profile.get('last_seen') else None
        },
        'state': {
            'supports_grayscale': device_state.get('supports_grayscale'),
            'current_entry_hash': current_entry_hash,
            'current_plugin_id': device_state.get('last_entry_plugin'),
            'current_preview_url': preview_url,
            'current_preview_token': preview_token
        }
    }


@router.get('/api/display')
@router.get('/api/display/')
async def display(
    request: Request,
    refresh_rate: Optional[str] = Header(None, alias='Refresh-Rate'),
    battery_voltage: Optional[str] = Header(None, alias='Battery-Voltage'),
    rssi: Optional[str] = Header(None, alias='RSSI')
) -> JSONResponse:
    """Main firmware endpoint returning the next image URL and metadata."""
    device_id, device_state = state.get_device_state_from_request(request)
    state.ensure_device_profile(device_id)
    models.add_log_entry(
        'Request received at /api/display',
        f'Headers: {request.headers},URL: {request.url}, device: {device_id}'
    )
    logger.info('[API] /api/display - URL: %s device: %s', request.url, device_id)

    fw_version = request.headers.get('fw-version')
    grayscale_ready = utils.firmware_supports_grayscale(fw_version)
    with state.STATE_LOCK:
        device_state['supports_grayscale'] = grayscale_ready

    update_kwargs = {
        'refresh_rate': int(refresh_rate) if refresh_rate is not None else None,
        'battery_voltage': float(battery_voltage) if battery_voltage is not None else None,
        'rssi': int(rssi) if rssi is not None else None
    }
    state.update_client_metrics(device_id, **update_kwargs)
    models.touch_device_last_seen(device_id)
    if battery_voltage is not None and rssi is not None:
        models.add_battery_status(float(battery_voltage), int(rssi))

    base_url = state.request_base_url(request)

    grayscale_path: Optional[str] = None
    sequence = 0
    with state.STATE_LOCK:
        send_switch = device_state['bmp_send_switch']
        adapted_url = base_url + ('/image/screen.bmp' if send_switch else '/image/screen1.bmp')
        device_state['bmp_send_switch'] = not send_switch
        if grayscale_ready:
            grayscale_switch = device_state.get('grayscale_send_switch', True)
            grayscale_path = '/image/grayscale.png' if grayscale_switch else '/image/grayscale1.png'
            device_state['grayscale_send_switch'] = not grayscale_switch
        sequence = device_state.get('token_sequence', 0) + 1
        device_state['token_sequence'] = sequence

    entry_idx: Optional[int] = None
    try:
        entry_idx = state.schedule_next_rotation_entry(device_id, device_state)
    except state.RotationUnavailableError as exc:
        logger.warning('[API] rotation unavailable for %s: %s; attempting plugin refresh', device_id, exc)
        await plugins.refresh_plugin_assets()
        try:
            entry_idx = state.schedule_next_rotation_entry(device_id, device_state)
        except state.RotationUnavailableError as exc2:
            logger.error('[API] rotation still unavailable for %s after refresh: %s', device_id, exc2)
            entry_idx = None

    if entry_idx is None:
        base_url = state.request_base_url(request)
        sequence = 0
        with state.STATE_LOCK:
            send_switch = device_state['bmp_send_switch']
            device_state['bmp_send_switch'] = not send_switch
            if grayscale_ready:
                grayscale_switch = device_state.get('grayscale_send_switch', True)
                device_state['grayscale_send_switch'] = not grayscale_switch
            sequence = device_state.get('token_sequence', 0) + 1
            device_state['token_sequence'] = sequence

        request_version = device_state.get('request_count', 0)
        salt = f"{device_id}-{sequence}-{request_version}-dummy"
        token = sha256(salt.encode('utf-8')).hexdigest()[:16]
        state.register_image_token(device_id, token)

        separator = '&' if '?' in base_url else '?'
        image_url = f"{base_url}/image/dummy.bmp{separator}token={token}"
        response = {
            'status': 0,
            'image_url': image_url,
            'filename': token,
            'update_firmware': False,
            'firmware_url': base_url + '/fw/update',
            'refresh_rate': state.get_refresh_interval(device_id),
            'reset_firmware': False,
            'special_function': '',
            'action': ''
        }
        models.add_log_entry('send json /api/display', f'response: {response}')
        return JSONResponse(response)

    state.update_device_preview(device_id, device_state, entry_idx)

    entry_hash = device_state.get('last_entry_hash') or str(entry_idx)
    request_version = device_state.get('request_count', 0)
    salt = f"{device_id}-{sequence}-{request_version}-{entry_hash}"
    token = sha256(salt.encode('utf-8')).hexdigest()[:16]

    state.register_image_token(device_id, token)

    separator = '&' if '?' in adapted_url else '?'
    image_url = f"{adapted_url}{separator}token={token}"
    pending_media = None
    with state.STATE_LOCK:
        pending_media = device_state.get('pending_entry_media')

    if grayscale_ready and grayscale_path and pending_media != 'bmp':
        image_url = f"{base_url}{grayscale_path}?token={token}"

    response = {
        'status': 0,
        'image_url': image_url,
        'filename': token,
        'update_firmware': False,
        'firmware_url': base_url + '/fw/update',
        'refresh_rate': state.get_refresh_interval(device_id),
        'reset_firmware': False,
        'special_function': '',
        'action': ''
    }

    models.add_log_entry('send json /api/display', f'response: {response}')
    return JSONResponse(response)


@router.get('/api/setup')
@router.get('/api/setup/')
def api_setup(request: Request) -> JSONResponse:
    """Provide setup metadata expected by TRMNL clients."""
    base_url = state.request_base_url(request)
    payload = {
        'status': 200,
        'api_key': config.SETUP_API_KEY,
        'friendly_id': config.SETUP_FRIENDLY_ID,
        'image_url': base_url + '/image/screen.bmp',
        'message': config.SETUP_MESSAGE
    }
    models.add_log_entry('send json /api/setup', f'response: {payload}')
    return JSONResponse(payload)


@router.get('/settings')
def get_settings() -> JSONResponse:
    """Retrieve the current settings for the terminal server."""
    return JSONResponse({
        'config_image_path': config.IMAGE_PATH,
        'config_refresh_time': config.REFRESH_TIME
    })


@router.post('/settings/refreshtime')
def update_refresh_time(data: Dict[str, Any] = Body(...)) -> JSONResponse:
    """Update the refresh time in the configuration."""
    new_refresh_time = data.get('refresh_rate')
    if new_refresh_time is None:
        return JSONResponse({'status': 'error', 'message': 'Invalid refresh rate'}, status_code=400)

    config.update_config('refresh_time', new_refresh_time)
    models.save_config_entry('refresh_time', str(config.REFRESH_TIME))
    return JSONResponse({'status': 'success', 'new_refresh_time': config.REFRESH_TIME}, status_code=200)


@router.post('/settings/imagepath')
def update_image_path(data: Dict[str, Any] = Body(...)) -> JSONResponse:
    """Update the fallback image path in the configuration."""
    new_image_path = data.get('bmp_path')
    if new_image_path is None:
        return JSONResponse({'status': 'error', 'message': 'Invalid new_image_path'}, status_code=400)

    config.update_config('image_path', new_image_path)
    models.save_config_entry('image_path', config.IMAGE_PATH)
    return JSONResponse({'status': 'success', 'new_image_path': config.IMAGE_PATH}, status_code=200)


@router.post('/api/log')
@router.post('/api/log/')
async def api_log(request: Request) -> JSONResponse:
    """Capture and persist client log payloads while echoing them to stdout."""
    raw_body = await request.body()
    body_text = raw_body.decode('utf-8', errors='replace') if raw_body else '<empty>'

    logger.info('[API] /api/log - payload: %s', body_text)
    print(body_text)
    models.add_log_entry('Request received at /api/log', body_text)

    parsed_content: Dict[str, Any] = {}
    try:
        parsed_content = await request.json()
    except ValueError:
        pass

    logs_array: List[Any] = []
    if isinstance(parsed_content, dict):
        log_block = parsed_content.get('log')
        if isinstance(log_block, dict):
            logs_array = log_block.get('logs_array') or []

    for log_entry in logs_array:
        models.add_log_entry('Client Log', str(log_entry))
        print(str(log_entry))

    return JSONResponse({'status': 'logged'}, status_code=200)


@router.get('/rotation')
def get_rotation_playlist() -> JSONResponse:
    """Expose the current rotation entries and default playlist selection."""
    snapshot = state.build_rotation_snapshot()
    return JSONResponse(snapshot)


@router.post('/rotation')
def update_rotation_playlist(data: Dict[str, Any] = Body(...)) -> JSONResponse:
    """Update the default or per-device rotation playlist using entry IDs."""
    playlist_ids = data.get('playlist')
    device_id = data.get('device_id')

    if not isinstance(playlist_ids, list) or not all(isinstance(pid, str) for pid in playlist_ids):
        return JSONResponse({'status': 'error', 'message': 'playlist must be a list of IDs'}, status_code=400)

    try:
        if device_id:
            state.set_device_playlist(device_id, playlist_ids)
        else:
            state.set_default_playlist(playlist_ids)
    except ValueError as exc:
        return JSONResponse({'status': 'error', 'message': str(exc)}, status_code=400)

    snapshot = state.build_rotation_snapshot()
    return JSONResponse(snapshot)


@router.post('/playlists')
def upsert_named_playlist(data: Dict[str, Any] = Body(...)) -> JSONResponse:
    """Create or update a named rotation playlist entity."""
    name = data.get('name')
    playlist_ids = data.get('playlist')

    if not isinstance(name, str) or not name.strip() or name.strip() == state.DEFAULT_DEVICE_ID:
        return JSONResponse({'status': 'error', 'message': 'name must be a non-default string'}, status_code=400)
    if not isinstance(playlist_ids, list) or not all(isinstance(pid, str) for pid in playlist_ids):
        return JSONResponse({'status': 'error', 'message': 'playlist must be a list of IDs'}, status_code=400)

    try:
        state.set_named_playlist(name.strip(), playlist_ids)
    except ValueError as exc:
        return JSONResponse({'status': 'error', 'message': str(exc)}, status_code=400)

    snapshot = state.build_rotation_snapshot()
    return JSONResponse(snapshot)


@router.delete('/playlists/{name}')
def delete_named_playlist(name: str) -> JSONResponse:
    """Delete a named playlist and unbind any devices using it."""
    if not name or name.strip() == state.DEFAULT_DEVICE_ID:
        return JSONResponse({'status': 'error', 'message': 'default playlist cannot be deleted'}, status_code=400)
    try:
        state.delete_named_playlist(name.strip())
    except ValueError as exc:
        return JSONResponse({'status': 'error', 'message': str(exc)}, status_code=400)
    snapshot = state.build_rotation_snapshot()
    return JSONResponse(snapshot)


@router.delete('/rotation/{device_id}')
def delete_rotation_playlist(device_id: str) -> JSONResponse:
    """Remove a per-device playlist override and fall back to the default selection."""
    normalized = (device_id or '').strip() or state.DEFAULT_DEVICE_ID
    if normalized == state.DEFAULT_DEVICE_ID:
        return JSONResponse({'status': 'error', 'message': 'default playlist cannot be deleted'}, status_code=400)

    state.clear_device_playlist(normalized)
    snapshot = state.build_rotation_snapshot()
    return JSONResponse(snapshot)


@router.get('/devices')
def list_devices(include_default: bool = Query(True, alias='include_default')) -> JSONResponse:
    """List known devices along with their profiles, metrics, and playlists."""
    device_ids = state.known_device_ids(include_default=include_default)
    devices = [_serialize_device_payload(device_id) for device_id in device_ids]
    return JSONResponse({'devices': devices})


@router.get('/devices/{device_id}')
def get_device(device_id: str) -> JSONResponse:
    """Return metadata and metrics for a specific device."""
    normalized_id = device_id.strip() or state.DEFAULT_DEVICE_ID
    payload = _serialize_device_payload(normalized_id)
    return JSONResponse(payload)


@router.patch('/devices/{device_id}')
def update_device(
    device_id: str,
    data: Dict[str, Any] = Body(...)
) -> JSONResponse:
    """Update device profile fields and optionally override its playlist."""
    normalized_id = device_id.strip() or state.DEFAULT_DEVICE_ID
    friendly_name = data.get('friendly_name')
    refresh_interval = data.get('refresh_interval')
    time_zone = data.get('time_zone')
    playlist_ids = data.get('playlist')
    has_playlist_name = 'playlist_name' in data
    playlist_name = data.get('playlist_name')

    refresh_override: Optional[int] = None
    if refresh_interval is not None:
        if not isinstance(refresh_interval, int) or refresh_interval <= 0:
            return JSONResponse({'status': 'error', 'message': 'refresh_interval must be a positive integer'}, status_code=400)
        refresh_override = refresh_interval

    if playlist_ids is not None:
        if not isinstance(playlist_ids, list) or not all(isinstance(pid, str) for pid in playlist_ids):
            return JSONResponse({'status': 'error', 'message': 'playlist must be a list of IDs'}, status_code=400)

    state.update_device_profile(
        normalized_id,
        friendly_name=friendly_name,
        refresh_interval=refresh_override,
        time_zone=time_zone
    )

    if playlist_ids is not None:
        try:
            state.set_device_playlist(normalized_id, playlist_ids)
        except ValueError as exc:
            return JSONResponse({'status': 'error', 'message': str(exc)}, status_code=400)

    if has_playlist_name:
        if playlist_name is not None and not isinstance(playlist_name, str):
            return JSONResponse({'status': 'error', 'message': 'playlist_name must be a string or null'}, status_code=400)
        try:
            state.set_device_playlist_binding(normalized_id, playlist_name)
        except ValueError as exc:
            return JSONResponse({'status': 'error', 'message': str(exc)}, status_code=400)

    payload = _serialize_device_payload(normalized_id)
    return JSONResponse(payload)


@router.get('/server/log')
def log_view(
    request: Request,
    limit: int = Query(30, ge=1, le=200),
    after: Optional[int] = Query(None),
    response_format: str = Query('text', alias='format')
) -> Response:
    """Return recent logs with optional cursor-based pagination."""
    logs = models.get_logs_after(after, limit) if after is not None else models.get_logs(limit=limit)

    wants_json = 'application/json' in (request.headers.get('accept') or '').lower() or response_format.lower() == 'json'
    if wants_json:
        payload = [
            {
                'id': log.id,
                'timestamp': utils.to_iso_datetime(log.timestamp),
                'context': log.context,
                'info': log.info
            }
            for log in logs
        ]
        return JSONResponse(payload)

    formatted_logs = '\n'.join([f"{log.timestamp} -- [{log.context}] -- {log.info}" for log in logs])
    response = Response(content=formatted_logs, media_type='text/plain')
    if logs:
        response.headers['X-Log-Last-Id'] = str(logs[-1].id)
    return response


@router.get('/server/battery')
def battery_view(
    all_data: Optional[str] = Query(None, alias='all'),
    from_date: Optional[str] = Query(None, alias='from'),
    to_date: Optional[str] = Query(None, alias='to')
) -> JSONResponse:
    """Fetch battery data from the client database and return it in JSON format."""
    from_dt = None
    to_dt = None
    if from_date:
        try:
            from_dt = datetime.strptime(from_date, '%Y-%m-%d')
        except ValueError:
            pass
    if to_date:
        try:
            to_dt = datetime.strptime(to_date, '%Y-%m-%d')
        except ValueError:
            pass

    if not all_data and not from_dt and not to_dt:
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        from_dt = today
        to_dt = today + timedelta(days=1)

    limit = None if all_data else 1000
    history = models.get_battery_history(limit=limit, from_date=from_dt, to_date=to_dt)

    response_data = [
        {
            'timestamp': utils.to_iso_datetime(entry.timestamp),
            'battery_voltage': entry.voltage,
            'rssi': entry.rssi
        }
        for entry in history
    ]
    response_data.sort(key=lambda x: x['timestamp'])

    return JSONResponse(response_data)


@router.get('/status')
def status_view(device_id: Optional[str] = Query(None, alias='device_id')) -> JSONResponse:
    """Retrieve the current status of the server and connected devices."""
    selected_device_id = device_id or state.DEFAULT_DEVICE_ID
    uptime_seconds = int(time() - state.start_time)
    uptime = str(timedelta(seconds=uptime_seconds))

    cpu_load = psutil.cpu_percent(interval=None)
    current_time = utils.to_iso_datetime(datetime.now(timezone.utc))

    metrics = state.get_client_metrics(selected_device_id)
    profile = state.ensure_device_profile(selected_device_id)
    refresh_interval = state.get_refresh_interval(selected_device_id)
    battery_voltage = metrics['battery_voltage']
    battery_state = utils.get_battery_state(battery_voltage)
    wifi_signal = metrics['rssi']
    wifi_signal_strength = utils.get_wifi_signal_strength(wifi_signal)

    battery_history = models.get_battery_history(limit=30)
    client_data_db = [
        {
            'timestamp': utils.to_iso_datetime(entry.timestamp),
            'battery_voltage': entry.voltage,
            'rssi': entry.rssi
        }
        for entry in battery_history
    ]
    client_data_db.sort(key=lambda x: x['timestamp'])

    device_state = state.get_device_state(selected_device_id)
    current_entry_hash = device_state.get('last_entry_hash')
    current_preview_url = device_state.get('current_preview_url') or f"/preview/{quote(selected_device_id)}"
    current_preview_token = device_state.get('current_preview_token')

    metrics_store = state.get_all_client_metrics()

    def _has_contact(device_id: str) -> bool:
        metrics_record = metrics_store.get(device_id) or {}
        last_contact = metrics_record.get('last_contact')
        return isinstance(last_contact, (int, float)) and last_contact > 0

    filtered_ids = []
    seen_ids = set()
    for known_id in state.known_device_ids(include_default=False):
        if _has_contact(known_id):
            filtered_ids.append(known_id)
            seen_ids.add(known_id)
    if selected_device_id and selected_device_id != state.DEFAULT_DEVICE_ID and selected_device_id not in seen_ids:
        filtered_ids.append(selected_device_id)
    devices_summary = [_serialize_device_payload(known_id) for known_id in filtered_ids]
    playlists_summary = state.list_playlist_targets()

    status_data = {
        'server': {
            'uptime': uptime,
            'cpu_load': cpu_load,
            'current_time': current_time
        },
        'client': {
            'device_id': selected_device_id,
            'friendly_name': profile.get('friendly_name') or selected_device_id,
            'battery_voltage': battery_voltage,
            'battery_voltage_max': config.BATTERY_MAX_VOLTAGE,
            'battery_voltage_min': config.BATTERY_MIN_VOLTAGE,
            'battery_state': battery_state,
            'wifi_signal': wifi_signal,
            'wifi_signal_strength': wifi_signal_strength,
            'refresh_time': refresh_interval,
            'last_contact': utils.to_iso_timestamp(metrics['last_contact']),
            'profile': {
                'refresh_interval': profile.get('refresh_interval'),
                'time_zone': profile.get('time_zone'),
                'last_seen': utils.to_iso_datetime(profile.get('last_seen')) if profile.get('last_seen') else None
            },
            'supports_grayscale': device_state.get('supports_grayscale'),
            'current_entry_hash': current_entry_hash,
            'current_plugin_id': device_state.get('last_entry_plugin'),
            'current_preview_url': current_preview_url,
            'current_preview_token': current_preview_token
        },
        'devices': devices_summary,
        'playlists': playlists_summary,
        'client_data_db': client_data_db
    }
    return JSONResponse(status_data)
