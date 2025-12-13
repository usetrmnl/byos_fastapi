from __future__ import annotations

import logging
from os import environ, getcwd
from os.path import abspath, isdir, join
from sys import stdout

# Logging Configuration
LOG_LEVEL = environ.get('LOG_LEVEL', 'DEBUG').upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.DEBUG),
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=stdout
)
logger = logging.getLogger('trmnlServer')

# Pillow emits very noisy DEBUG logs (PNG chunk dumps). Keep them at INFO+.
logging.getLogger('PIL').setLevel(logging.INFO)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.INFO)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

logger.info('[Config] loading module')

_TRUE_VALUES = {'true', '1', 't', 'yes', 'on'}

IMAGE_PATH = 'images/screen.bmp'
REFRESH_TIME = 900
BATTERY_MAX_VOLTAGE = 4.1
BATTERY_MIN_VOLTAGE = 2.3
TIME_ZONE = 'UTC'
SERVER_PORT = 4567
ENABLE_SSL = False
SERVER_SCHEME = 'http'
SETUP_API_KEY = ''
SETUP_FRIENDLY_ID = 'trmnl-byod'
SETUP_MESSAGE = 'Configured'
DITHERING_MODE = 'none'
ASSETS_ROOT = 'web'
STATIC_ROOT = 'web'
GENERATED_ROOT = 'var/generated'

# Photographic plugin grading
#
# When enabled, photographic plugins apply additional histogram and shadow grading
# prior to tone-curve-aware quantization/dithering.
# When disabled, photographic plugins output raw grayscale (no extra grading),
# which makes it easier to reason about the tone curve/LUT calibration.
PHOTO_GRADING_ENABLED = True

# Calibration plugin control
#
# When disabled, calibration plugins are excluded from the plugin registry and
# no calibration assets are generated.
CALIBRATION_PLUGIN_ENABLED = False

# E-ink grayscale response compensation
#
# These settings allow quantization and dithering to operate in a non-linear
# "panel space" so the resulting grays are closer to perceptually uniform on
# real e-ink panels.
#
# - EINK_TONE_POINTS: optional anchor list "in:out" in 0-255, comma-separated
#   e.g. "0:0,32:40,128:160,255:255" (digital input -> observed panel output).
# - EINK_TONE_GAMMA: fallback forward gamma (digital -> panel) when points unset.
EINK_TONE_POINTS = '0:0,32:6,64:18,85:32,128:95,170:155,192:190,224:225,255:250'
EINK_TONE_GAMMA = 1.0

CONFIG_DIR = getcwd()
VAR_ROOT = join(CONFIG_DIR, 'var')
DATABASE_PATH = join(VAR_ROOT, 'db', 'trmnl.db')
LOGS_DIR = join(VAR_ROOT, 'logs')
SSL_DIR = join(VAR_ROOT, 'ssl')
WEB_ROOT_DIR = join(CONFIG_DIR, ASSETS_ROOT)
WEB_STATIC_DIR = join(CONFIG_DIR, STATIC_ROOT)
WEB_GENERATED_DIR = join(CONFIG_DIR, GENERATED_ROOT)

_ENV_OVERRIDES: set[str] = set()


def _env_str(name: str, default: str, config_key: str) -> str:
    value = environ.get(name)
    if value is None:
        return default
    _ENV_OVERRIDES.add(config_key)
    return value


def _env_bool(name: str, default: bool, config_key: str) -> bool:
    value = environ.get(name)
    if value is None:
        return default
    _ENV_OVERRIDES.add(config_key)
    return value.strip().lower() in _TRUE_VALUES


def _env_int(name: str, default: int, config_key: str) -> int:
    value = environ.get(name)
    if value is None:
        return default
    try:
        number = int(value)
        _ENV_OVERRIDES.add(config_key)
        return number
    except ValueError:
        logger.warning('[Config] Invalid int for %s: %s', name, value)
        return default


def _env_float(name: str, default: float, config_key: str) -> float:
    value = environ.get(name)
    if value is None:
        return default
    try:
        number = float(value)
        _ENV_OVERRIDES.add(config_key)
        return number
    except ValueError:
        logger.warning('[Config] Invalid float for %s: %s', name, value)
        return default


def _apply_environment_overrides() -> None:
    global IMAGE_PATH, REFRESH_TIME
    global BATTERY_MAX_VOLTAGE, BATTERY_MIN_VOLTAGE, TIME_ZONE
    global SERVER_PORT, ENABLE_SSL, SERVER_SCHEME
    global SETUP_API_KEY, SETUP_FRIENDLY_ID, SETUP_MESSAGE
    global DITHERING_MODE, ASSETS_ROOT, STATIC_ROOT, GENERATED_ROOT
    global EINK_TONE_POINTS, EINK_TONE_GAMMA
    global PHOTO_GRADING_ENABLED, CALIBRATION_PLUGIN_ENABLED
    _ENV_OVERRIDES.clear()

    default_eink_tone_points = EINK_TONE_POINTS
    default_eink_tone_gamma = EINK_TONE_GAMMA

    IMAGE_PATH = _env_str('IMAGE_PATH', 'images/screen.bmp', 'image_path')
    REFRESH_TIME = _env_int('REFRESH_TIME', 900, 'refresh_time')
    BATTERY_MAX_VOLTAGE = _env_float('BATTERY_MAX_VOLTAGE', 4.1, 'battery_max_voltage')
    BATTERY_MIN_VOLTAGE = _env_float('BATTERY_MIN_VOLTAGE', 2.3, 'battery_min_voltage')
    TIME_ZONE = _env_str('TIME_ZONE', 'UTC', 'time_zone')
    SERVER_PORT = _env_int('SERVER_PORT', 4567, 'server_port')
    ENABLE_SSL = _env_bool('ENABLE_SSL', False, 'enable_ssl')
    SETUP_API_KEY = _env_str('SETUP_API_KEY', '', 'setup_api_key')
    SETUP_FRIENDLY_ID = _env_str('SETUP_FRIENDLY_ID', 'trmnl-byod', 'setup_friendly_id')
    SETUP_MESSAGE = _env_str('SETUP_MESSAGE', 'Configured', 'setup_message')
    DITHERING_MODE = _env_str('DITHERING_MODE', 'none', 'dithering_mode')
    ASSETS_ROOT = _env_str('ASSETS_ROOT', 'web', 'assets_root')
    STATIC_ROOT = _env_str('STATIC_ROOT', 'web', 'static_root')
    GENERATED_ROOT = _env_str('GENERATED_ROOT', 'var/generated', 'generated_root')
    EINK_TONE_POINTS = _env_str('EINK_TONE_POINTS', default_eink_tone_points, 'eink_tone_points')
    EINK_TONE_GAMMA = _env_float('EINK_TONE_GAMMA', default_eink_tone_gamma, 'eink_tone_gamma')
    PHOTO_GRADING_ENABLED = _env_bool('PHOTO_GRADING_ENABLED', PHOTO_GRADING_ENABLED, 'photo_grading_enabled')
    CALIBRATION_PLUGIN_ENABLED = _env_bool('CALIBRATION_PLUGIN_ENABLED', CALIBRATION_PLUGIN_ENABLED, 'calibration_plugin_enabled')
    _refresh_server_scheme()


def _refresh_server_scheme() -> None:
    global SERVER_SCHEME
    SERVER_SCHEME = 'https' if ENABLE_SSL else 'http'


def _refresh_path_constants() -> None:
    global VAR_ROOT, DATABASE_PATH, LOGS_DIR, SSL_DIR
    global WEB_ROOT_DIR, WEB_STATIC_DIR, WEB_GENERATED_DIR
    VAR_ROOT = join(CONFIG_DIR, 'var')
    DATABASE_PATH = join(VAR_ROOT, 'db', 'trmnl.db')
    LOGS_DIR = join(VAR_ROOT, 'logs')
    SSL_DIR = join(VAR_ROOT, 'ssl')
    WEB_ROOT_DIR = join(CONFIG_DIR, ASSETS_ROOT)
    WEB_STATIC_DIR = join(CONFIG_DIR, STATIC_ROOT)
    WEB_GENERATED_DIR = join(CONFIG_DIR, GENERATED_ROOT)


def load_config(base_dir: str | None = None) -> None:
    """Apply environment overrides and update path constants for the provided base directory."""
    global CONFIG_DIR
    _apply_environment_overrides()
    if base_dir:
        candidate = abspath(base_dir)
        if not isdir(candidate):
            logger.warning('[Config] Provided base_dir %s is not a directory; using current working directory', base_dir)
            candidate = getcwd()
        CONFIG_DIR = candidate
    else:
        CONFIG_DIR = getcwd()
    _refresh_path_constants()


def _coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_VALUES
    return bool(value)


def update_config(key: str, value) -> None:
    """Update an in-memory configuration value."""
    global IMAGE_PATH, REFRESH_TIME
    global BATTERY_MAX_VOLTAGE, BATTERY_MIN_VOLTAGE, TIME_ZONE
    global SERVER_PORT, ENABLE_SSL, SETUP_API_KEY
    global SETUP_FRIENDLY_ID, SETUP_MESSAGE, DITHERING_MODE
    global ASSETS_ROOT, STATIC_ROOT, GENERATED_ROOT
    global EINK_TONE_POINTS, EINK_TONE_GAMMA
    global PHOTO_GRADING_ENABLED, CALIBRATION_PLUGIN_ENABLED

    logger.info('[Config] Updating %s to %s', key, value)

    if key == 'image_path':
        IMAGE_PATH = str(value)
    elif key == 'refresh_time':
        REFRESH_TIME = int(value)
    elif key == 'battery_max_voltage':
        BATTERY_MAX_VOLTAGE = float(value)
    elif key == 'battery_min_voltage':
        BATTERY_MIN_VOLTAGE = float(value)
    elif key == 'time_zone':
        TIME_ZONE = str(value)
    elif key == 'server_port':
        SERVER_PORT = int(value)
    elif key == 'enable_ssl':
        ENABLE_SSL = _coerce_bool(value)
        _refresh_server_scheme()
    elif key == 'setup_api_key':
        SETUP_API_KEY = str(value)
    elif key == 'setup_friendly_id':
        SETUP_FRIENDLY_ID = str(value)
    elif key == 'setup_message':
        SETUP_MESSAGE = str(value)
    elif key == 'dithering_mode':
        DITHERING_MODE = str(value)
    elif key == 'assets_root':
        ASSETS_ROOT = str(value)
        _refresh_path_constants()
    elif key == 'static_root':
        STATIC_ROOT = str(value)
        _refresh_path_constants()
    elif key == 'generated_root':
        GENERATED_ROOT = str(value)
        _refresh_path_constants()
    elif key == 'eink_tone_points':
        EINK_TONE_POINTS = str(value)
    elif key == 'eink_tone_gamma':
        EINK_TONE_GAMMA = float(value)
    elif key == 'photo_grading_enabled':
        PHOTO_GRADING_ENABLED = _coerce_bool(value)
    elif key == 'calibration_plugin_enabled':
        CALIBRATION_PLUGIN_ENABLED = _coerce_bool(value)
    else:
        logger.warning('[Config] Unknown config key: %s', key)


def apply_persisted_config(entries: dict[str, str]) -> None:
    """Apply database-backed configuration entries unless overridden by env vars."""
    for key, raw_value in entries.items():
        if key in _ENV_OVERRIDES:
            continue
        update_config(key, raw_value)


_apply_environment_overrides()
_refresh_path_constants()
