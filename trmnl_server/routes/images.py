"""Image-serving routes for TRMNL local server."""

from __future__ import annotations

from io import BytesIO

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response

from .. import config, models, utils
from ..services import state

router = APIRouter()
logger = config.logger

SCREEN_VARIANTS = {'screen.bmp', 'screen1.bmp'}
ORIGINAL_VARIANTS = {'original.bmp', 'original1.bmp'}
GRAYSCALE_VARIANTS = {'grayscale.png', 'grayscale1.png'}


def _device_context_for_image_request(request: Request) -> tuple[str, dict]:
    token = request.query_params.get('token')
    resolved = state.resolve_device_id_from_token(token)
    if resolved:
        return resolved, state.get_device_state(resolved)
    return state.get_device_state_from_request(request)


def _binary_response(image_blob: BytesIO, media_type: str) -> Response:
    payload = image_blob.getvalue()
    headers = {'Content-Length': str(len(payload))}
    return Response(content=payload, media_type=media_type, headers=headers)


def _client_host(request: Request) -> str:
    client = request.client
    return client.host if client else 'unknown'


def _serve_bmp_frame(request: Request, route: str) -> Response:
    device_id, device_state = _device_context_for_image_request(request)
    try:
        frame = state.get_next_bmp_frame(device_id, device_state)
    except state.RotationUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    client_host = _client_host(request)
    models.add_log_entry(
        f'Request received at {route}',
        f'serving image for IP: {client_host} device: {device_id}'
    )
    logger.info('[API] %s - serving image for IP: %s device: %s', route, client_host, device_id)
    return _binary_response(frame, 'image/bmp')


def _serve_original_image(request: Request) -> Response:
    _, device_state = _device_context_for_image_request(request)
    entry_idx = state.current_frame_entry_index(device_state)
    if entry_idx is None:
        raise HTTPException(status_code=404, detail='No original image available')
    png_bytes = state.get_rotation_png_bytes(entry_idx)
    return _binary_response(BytesIO(png_bytes), 'image/png')


def _serve_grayscale_frame(request: Request) -> Response:
    device_id, device_state = _device_context_for_image_request(request)
    try:
        png_blob = state.get_next_png_frame(device_id, device_state)
    except state.RotationUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _binary_response(png_blob, 'image/png')


@router.get('/image/{image_name}')
def serve_image_variant(request: Request, image_name: str) -> Response:
    """Serve BMP, PNG, or placeholder assets behind /image/* routes."""
    route = f'/image/{image_name}'
    if image_name in SCREEN_VARIANTS:
        return _serve_bmp_frame(request, route)
    if image_name in ORIGINAL_VARIANTS:
        return _serve_original_image(request)
    if image_name in GRAYSCALE_VARIANTS:
        return _serve_grayscale_frame(request)
    if image_name == 'dummy.bmp':
        return FileResponse(utils.asset_path('img', 'dummy.bmp'), media_type='image/bmp')
    raise HTTPException(status_code=404, detail='Unknown image path')


@router.get('/images/current.png')
def serve_current_png(request: Request) -> Response:
    """Convert the current BMP frame to PNG for browser viewing."""
    _, device_state = _device_context_for_image_request(request)
    entry_idx = state.current_frame_entry_index(device_state)
    if entry_idx is None:
        raise HTTPException(status_code=404, detail='No current image available')
    bmp_blob = BytesIO(state.get_rotation_bmp_bytes(entry_idx))
    png_bytes = utils.convert_bmp_bytes_to_png(bmp_blob)
    return _binary_response(png_bytes, 'image/png')


@router.get('/preview/{device_id}')
def serve_preview(device_id: str) -> Response:
    """Serve the last rotation PNG fetched by a device."""
    device_state = state.get_device_state(device_id or state.DEFAULT_DEVICE_ID)
    entry_idx = state.preview_frame_entry_index(device_state)
    if entry_idx is None:
        raise HTTPException(status_code=404, detail='No preview available')
    payload = state.get_rotation_png_bytes(entry_idx)
    response = _binary_response(BytesIO(payload), 'image/png')
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    return response
