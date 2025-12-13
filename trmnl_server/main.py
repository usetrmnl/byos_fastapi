#! /usr/bin/env python
"""FastAPI entrypoint and CLI tooling for the TRMNL local server."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Awaitable, Callable, Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles

from . import config, models, utils
from .routes import api_router, image_router, page_router
from .services import plugins, state

###################################################################################################

logger = config.logger
logger.info('[Main] Starting trmnlServer')

API_LOG_PATH_PREFIXES = ('/api',)
MAX_REQUEST_LOG_BODY = 2048
MAX_RESPONSE_LOG_BODY = 2048
BINARY_CONTENT_PREFIXES = (
    'application/octet-stream',
    'application/pdf',
    'application/zip',
    'image/',
    'audio/',
    'video/'
)

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
STATIC_MOUNT_PATHS: Optional[Tuple[str, str]] = None


def should_log_request(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in API_LOG_PATH_PREFIXES)


def format_request_body(body: bytes, limit: int = MAX_REQUEST_LOG_BODY) -> str:
    if not body:
        return '<empty>'
    body_text = body.decode('utf-8', errors='replace')
    if len(body_text) > limit:
        return f"{body_text[:limit]}...<truncated>"
    return body_text


def is_binary_content_type(content_type: str) -> bool:
    lowered = (content_type or '').lower()
    return any(lowered.startswith(prefix) for prefix in BINARY_CONTENT_PREFIXES)


def format_response_body(body: bytes, limit: int = MAX_RESPONSE_LOG_BODY) -> str:
    if not body:
        return '<empty>'
    text = body.decode('utf-8', errors='replace')
    if len(text) > limit:
        return f"{text[:limit]}...<truncated>"
    return text


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    logger.info('Running initial plugin refresh')
    await plugins.refresh_plugin_assets()
    logger.info('Starting plugin refresh workers')
    await plugins.start_plugin_refreshers()
    yield
    logger.info('Stopping plugin refresh workers')
    await plugins.stop_plugin_refreshers()


app = FastAPI(lifespan=lifespan)


@app.middleware('http')
async def log_api_request(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    log_this_request = should_log_request(request.url.path)
    if log_this_request:
        body_bytes = await request.body()
        logger.info(
            '[RequestDump] method=%s path=%s query=%s headers=%s body=%s',
            request.method,
            request.url.path,
            dict(request.query_params),
            dict(request.headers),
            format_request_body(body_bytes)
        )
    response = await call_next(request)
    if log_this_request:
        content_type = response.headers.get('content-type', '')
        if not is_binary_content_type(content_type):
            response_body_chunks = [chunk async for chunk in response.body_iterator]
            response_body = b''.join(response_body_chunks)
            logger.info(
                '[ResponseDump] path=%s status=%s content_type=%s headers=%s body=%s',
                request.url.path,
                response.status_code,
                content_type,
                dict(response.headers),
                format_response_body(response_body)
            )
            return Response(
                content=response_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
                background=response.background
            )
    return response
app.include_router(image_router)
app.include_router(api_router)
app.include_router(page_router)


def _parse_cli_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='TRMNL local server')
    parser.add_argument('workdir', nargs='?', help='Runtime working directory', default=None)
    parser.add_argument('--list-plugins', action='store_true', help='List registered plugins and exit')
    parser.add_argument('--run-plugin', metavar='PLUGIN', help='Run a single plugin and exit')
    parser.add_argument('--plugin-output', metavar='DIR', help='Override output directory when running a plugin')
    parser.add_argument(
        '--plugin-arg',
        action='append',
        default=[],
        metavar='KEY=VALUE',
        help='Additional keyword arguments for --run-plugin'
    )
    return parser.parse_args(argv)


def _parse_plugin_kwargs(pairs: List[str]) -> Dict[str, str]:
    kwargs: Dict[str, str] = {}
    for entry in pairs:
        if '=' not in entry:
            raise ValueError(f"Invalid plugin arg '{entry}'. Expected KEY=VALUE")
        key, value = entry.split('=', 1)
        kwargs[key] = value
    return kwargs


def _resolve_workdir(candidate: Optional[str]) -> str:
    if not candidate:
        return BASE_PATH
    if not os.path.isdir(candidate):
        print(f"Path {candidate} is not a directory. Using default path {BASE_PATH}.")
        return BASE_PATH
    return candidate


def _ensure_static_mounts() -> None:
    global STATIC_MOUNT_PATHS
    desired = (config.WEB_STATIC_DIR, config.WEB_GENERATED_DIR)
    if STATIC_MOUNT_PATHS == desired:
        return
    # Remove existing mounts if present so new directories are reflected
    app.router.routes = [
        route for route in app.router.routes
        if getattr(route, 'name', None) not in {'web-static', 'web-generated', 'generated-static'}
    ]
    app.mount('/web', StaticFiles(directory=config.WEB_STATIC_DIR), name='web-static')
    # Serve generated, volatile plugin outputs directly under /generated
    app.mount('/generated', StaticFiles(directory=config.WEB_GENERATED_DIR), name='generated-static')
    STATIC_MOUNT_PATHS = desired


def _enforce_runtime_directory_defaults(entries: Optional[Dict[str, str]] = None) -> None:
    desired_static = 'web'
    desired_assets = 'web'
    desired_generated = 'var/generated'
    if entries is None:
        entries = {}

    persisted_static = entries.get('static_root')
    if persisted_static and persisted_static != desired_static:
        logger.info(
            'Migrating static_root from %s to %s to align with reorganized web assets',
            persisted_static,
            desired_static
        )
        config.update_config('static_root', desired_static)
        models.save_config_entry('static_root', desired_static)

    persisted_assets = entries.get('assets_root')
    if persisted_assets and persisted_assets != desired_assets:
        logger.info(
            'Migrating assets_root from %s to %s to align with reorganized web assets',
            persisted_assets,
            desired_assets
        )
        config.update_config('assets_root', desired_assets)
        models.save_config_entry('assets_root', desired_assets)

    persisted_generated = entries.get('generated_root')
    if persisted_generated and persisted_generated != desired_generated:
        logger.info(
            'Migrating generated_root from %s to %s to keep volatile assets under var/',
            persisted_generated,
            desired_generated
        )
        config.update_config('generated_root', desired_generated)
        models.save_config_entry('generated_root', desired_generated)


def _prepare_runtime(current_dir: str) -> str:
    config.load_config(current_dir)
    os.makedirs(config.VAR_ROOT, exist_ok=True)
    os.makedirs(os.path.dirname(config.DATABASE_PATH), exist_ok=True)
    models.init_db()
    persisted_entries = models.load_config_entries()
    config.apply_persisted_config(persisted_entries)
    _enforce_runtime_directory_defaults(persisted_entries)
    state.initialize_rotation_playlists_from_storage()

    runtime_paths = {
        config.VAR_ROOT,
        os.path.dirname(config.DATABASE_PATH),
        config.LOGS_DIR,
        config.SSL_DIR,
        config.WEB_STATIC_DIR,
        config.WEB_GENERATED_DIR
    }
    for path in runtime_paths:
        os.makedirs(path, exist_ok=True)

    server_ip = utils.get_ip_address()
    server_scheme = config.SERVER_SCHEME
    server_base_url = f"{server_scheme}://{server_ip}:{config.SERVER_PORT}"
    state.set_server_base_url(server_base_url)
    logger.info(
        'Server will be running on IP: %s and port: %s (scheme: %s)',
        server_ip,
        config.SERVER_PORT,
        server_scheme
    )

    for path in (config.WEB_ROOT_DIR, config.WEB_STATIC_DIR, config.WEB_GENERATED_DIR, config.SSL_DIR):
        os.makedirs(path, exist_ok=True)
    _ensure_static_mounts()
    return server_ip


def _print_available_plugins() -> None:
    print('Registered plugins:')
    for name in plugins.list_available_plugins():
        print(f' - {name}')


def _run_plugin_command(plugin_name: str, output_dir: Optional[str], plugin_kwargs: Dict[str, str]) -> None:
    try:
        result = asyncio.run(
            plugins.run_single_plugin_by_name(
                plugin_name,
                output_dir=output_dir,
                plugin_kwargs=plugin_kwargs
            )
        )
    except ValueError as exc:
        logger.error('%s', exc)
        sys.exit(2)
    except RuntimeError as exc:
        logger.error('%s', exc)
        sys.exit(1)
    logger.info(
        'Plugin %s assets saved to %s and %s',
        plugin_name,
        result.monochrome_path,
        result.grayscale_path
    )


def _start_http_server(server_ip: str) -> None:
    with state.STATE_LOCK:
        state.get_device_state(state.DEFAULT_DEVICE_ID)['bmp_send_switch'] = True
    if config.ENABLE_SSL:
        cert_file = os.path.join(config.SSL_DIR, 'cert.pem')
        key_file = os.path.join(config.SSL_DIR, 'key.pem')

        if not os.path.exists(cert_file) or not os.path.exists(key_file):
            logger.debug('[Main] cert.pem and key.pem not found, generating new ones')
            os.system(
                f'openssl req -x509 -newkey rsa:4096 -keyout {key_file} -out {cert_file} '
                f'-days 365 -nodes '
                f'-subj "/C=US/ST=Georgia/L=Atlanta/O=trmnlServer/OU=webapp/CN={server_ip}"'
            )

        logger.debug('[Main] Starting the server with uvicorn and SSL')
        uvicorn.run(
            app,
            host='0.0.0.0',
            port=config.SERVER_PORT,
            ssl_keyfile=key_file,
            ssl_certfile=cert_file,
            log_level='info'
        )
    else:
        logger.debug('[Main] Starting the server without SSL')
        uvicorn.run(
            app,
            host='0.0.0.0',
            port=config.SERVER_PORT,
            log_level='info'
        )


_prepare_runtime(BASE_PATH)


def run() -> None:
    args = _parse_cli_args(sys.argv[1:])
    current_dir = _resolve_workdir(args.workdir)
    server_ip = _prepare_runtime(current_dir)

    if args.list_plugins:
        _print_available_plugins()
        return

    if args.run_plugin:
        try:
            plugin_kwargs = _parse_plugin_kwargs(args.plugin_arg)
        except ValueError as exc:
            logger.error('%s', exc)
            sys.exit(2)
        _run_plugin_command(args.run_plugin, args.plugin_output, plugin_kwargs)
        return

    _start_http_server(server_ip)


if __name__ == '__main__':
    run()
