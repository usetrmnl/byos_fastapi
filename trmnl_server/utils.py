import os
import socket
from io import BytesIO
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import hashlib
import math
import httpx
import datetime
from PIL import Image, ImageDraw, ImageFont
from . import config

# Constants
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

AVAILABLE_DITHER_MODES: Tuple[str, ...] = (
    'none',
    'floyd-steinberg',
    'ordered-blue-noise',
    'perceptual',
    'multi-pass'
)

_DITHER_TRUE_VALUES = {'true', '1', 'yes', 'on'}
_DITHER_MODE_ALIASES = {
    'none': 'none',
    'off': 'none',
    '0': 'none',
    'false': 'none',
    'floyd': 'floyd-steinberg',
    'fs': 'floyd-steinberg',
    'floyd-steinberg': 'floyd-steinberg',
    'ordered': 'ordered-blue-noise',
    'blue-noise': 'ordered-blue-noise',
    'ordered-blue-noise': 'ordered-blue-noise',
    'perceptual': 'perceptual',
    'multi-pass': 'multi-pass',
    'multipass': 'multi-pass',
    'multi': 'multi-pass'
}

BLUE_NOISE_MATRIX: List[List[int]] = [
    [50, 47, 14, 52, 44, 8, 56, 61],
    [24, 23, 36, 37, 22, 12, 57, 34],
    [42, 38, 32, 33, 63, 27, 58, 3],
    [43, 31, 28, 11, 40, 17, 15, 20],
    [62, 39, 5, 16, 10, 60, 26, 48],
    [54, 35, 59, 45, 53, 9, 4, 46],
    [25, 19, 55, 29, 2, 1, 49, 21],
    [13, 41, 7, 6, 18, 0, 51, 30]
]
BLUE_NOISE_HEIGHT = len(BLUE_NOISE_MATRIX)
BLUE_NOISE_WIDTH = len(BLUE_NOISE_MATRIX[0])
BLUE_NOISE_AREA = BLUE_NOISE_HEIGHT * BLUE_NOISE_WIDTH


def _clamp_byte(value: int) -> int:
    return max(0, min(255, int(value)))


def _parse_tone_points(raw: str) -> List[Tuple[int, int]]:
    """Parse EINK_TONE_POINTS as comma-separated "in:out" byte pairs."""
    points: List[Tuple[int, int]] = []
    raw = (raw or '').strip()
    if not raw:
        return points

    for token in raw.split(','):
        token = token.strip()
        if not token:
            continue
        if ':' not in token:
            continue
        left, right = token.split(':', 1)
        left = left.strip()
        right = right.strip()
        if not left or not right:
            continue
        try:
            x = _clamp_byte(int(left))
            y = _clamp_byte(int(right))
        except ValueError:
            continue
        points.append((x, y))

    points = sorted(set(points), key=lambda pair: pair[0])
    return points


def _enforce_monotonic(values: List[int]) -> List[int]:
    last = 0
    for idx, value in enumerate(values):
        if idx == 0:
            last = _clamp_byte(value)
            values[idx] = last
            continue
        last = max(last, _clamp_byte(value))
        values[idx] = last
    return values


@lru_cache(maxsize=32)
def _tone_curve_forward_lut_cached(points_raw: str, gamma: float) -> List[int]:
    """Return 256-entry LUT mapping digital gray -> panel gray for the given settings."""
    points = _parse_tone_points(points_raw)
    if points:
        if points[0][0] != 0:
            points = [(0, 0)] + points
        if points[-1][0] != 255:
            points = points + [(255, 255)]

        lut: List[int] = [0] * 256
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if x1 <= x0:
                continue
            span = x1 - x0
            for x in range(x0, x1 + 1):
                if x < 0 or x > 255:
                    continue
                t = (x - x0) / float(span)
                lut[x] = _clamp_byte(int(round(y0 + (y1 - y0) * t)))
        return _enforce_monotonic(lut)

    if gamma and gamma != 1.0:
        lut = [_clamp_byte(int(round((math.pow(i / 255.0, gamma)) * 255))) for i in range(256)]
        return _enforce_monotonic(lut)

    return list(range(256))


def _tone_curve_forward_lut() -> List[int]:
    """Return 256-entry LUT mapping digital gray -> panel gray."""
    points_raw = str(getattr(config, 'EINK_TONE_POINTS', '') or '')
    gamma = float(getattr(config, 'EINK_TONE_GAMMA', 1.0) or 1.0)
    return _tone_curve_forward_lut_cached(points_raw, gamma)


@lru_cache(maxsize=32)
def _tone_curve_inverse_lut_cached(points_raw: str, gamma: float) -> List[int]:
    """Return 256-entry LUT mapping desired panel gray -> digital gray for the given settings."""
    forward = _tone_curve_forward_lut_cached(points_raw, gamma)
    inverse: List[int] = [0] * 256

    idx = 0
    for target in range(256):
        while idx < 255 and forward[idx] < target:
            idx += 1
        if idx == 0:
            inverse[target] = 0
            continue
        if forward[idx] == target or forward[idx] == forward[idx - 1]:
            inverse[target] = idx
            continue
        y0 = forward[idx - 1]
        y1 = forward[idx]
        t = (target - y0) / float(y1 - y0)
        inverse[target] = _clamp_byte(int(round((idx - 1) + t)))

    inverse[0] = 0
    inverse[255] = 255
    return _enforce_monotonic(inverse)


def _tone_curve_inverse_lut() -> List[int]:
    """Return 256-entry LUT mapping desired panel gray -> digital gray."""
    points_raw = str(getattr(config, 'EINK_TONE_POINTS', '') or '')
    gamma = float(getattr(config, 'EINK_TONE_GAMMA', 1.0) or 1.0)
    return _tone_curve_inverse_lut_cached(points_raw, gamma)


def _tone_curve_enabled() -> bool:
    points = (getattr(config, 'EINK_TONE_POINTS', '') or '').strip()
    gamma = float(getattr(config, 'EINK_TONE_GAMMA', 1.0) or 1.0)
    return bool(points) or (gamma != 1.0)


def _palette_levels_digital(levels: int) -> List[int]:
    """Return list of palette gray values in digital space (0-255)."""
    if levels <= 1:
        return [0]
    inverse = _tone_curve_inverse_lut()

    if not _tone_curve_enabled():
        step_values_panel = [int(round(index * 255 / (levels - 1))) for index in range(levels)]
    else:
        forward = _tone_curve_forward_lut()
        panel_min = int(forward[0])
        panel_max = int(forward[255])
        if panel_max <= panel_min:
            panel_min = 0
            panel_max = 255
        step_values_panel = [
            int(round(panel_min + (index * (panel_max - panel_min) / (levels - 1))))
            for index in range(levels)
        ]
    values = [inverse[v] for v in step_values_panel]

    for idx in range(1, len(values)):
        if values[idx] <= values[idx - 1]:
            values[idx] = min(255, values[idx - 1] + 1)
    if len(set(values)) != len(values):
        values = [int(round(index * 255 / (levels - 1))) for index in range(levels)]
    return values


def get_effective_grayscale_palette_levels(levels: int = 4) -> Tuple[List[int], List[int]]:
    """Return (panel_levels, digital_levels) used for grayscale palette generation.

    - panel_levels are the target levels in "panel space" (0-255) that the palette
      aims to represent.
    - digital_levels are the pixel values (0-255) that will be written into the
      generated grayscale PNG/BMP assets.
    """
    if levels <= 1:
        return ([0], [0])

    if not _tone_curve_enabled():
        levels_minus = levels - 1
        panel_levels = [int(round(index * 255 / levels_minus)) for index in range(levels)]
        return (panel_levels, panel_levels)

    forward = _tone_curve_forward_lut()
    panel_min = int(forward[0])
    panel_max = int(forward[255])
    if panel_max <= panel_min:
        panel_min = 0
        panel_max = 255

    levels_minus = levels - 1
    panel_levels = [
        int(round(panel_min + (index * (panel_max - panel_min) / levels_minus)))
        for index in range(levels)
    ]
    return (panel_levels, _palette_levels_digital(levels))


def get_assets_root() -> Path:
    """Return the absolute path to the configured assets directory."""
    return Path(config.WEB_ROOT_DIR)


def asset_path(*parts: str) -> Path:
    """Build a path inside the configured assets directory."""
    return static_asset_path(*parts)


def get_static_assets_root() -> Path:
    """Return the absolute path to the static assets directory."""
    return Path(config.WEB_STATIC_DIR)


def get_generated_assets_root() -> Path:
    """Return the absolute path to the generated assets directory."""
    return Path(config.WEB_GENERATED_DIR)


def static_asset_path(*parts: str) -> Path:
    """Build a path inside the static assets directory."""
    return get_static_assets_root().joinpath(*parts)


def generated_asset_path(*parts: str) -> Path:
    """Build a path inside the generated assets directory."""
    return get_generated_assets_root().joinpath(*parts)


def _default_font_candidates() -> Tuple[str, ...]:
    return (
        asset_path('fonts/ttf/static/SpaceGrotesk-Medium.ttf').as_posix(),
        asset_path('fonts/ttf/static/SpaceGrotesk-Regular.ttf').as_posix()
    )


def get_available_dither_modes() -> Tuple[str, ...]:
    """Return the supported dithering mode names."""
    return AVAILABLE_DITHER_MODES


def path_to_web_url(path: str, prefix: str = '/web') -> Optional[str]:
    """Convert a filesystem path into the correct static or generated URL."""
    candidate = Path(path)
    try:
        candidate = candidate.resolve()
    except FileNotFoundError:
        candidate = candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)

    lookups: Tuple[Tuple[Path, str], ...] = (
        (get_static_assets_root(), prefix),
        (get_generated_assets_root(), '/generated'),
        (get_assets_root(), prefix)
    )
    for root, base in lookups:
        try:
            relative = candidate.relative_to(root)
            return f"{base}/{relative.as_posix()}"
        except ValueError:
            continue
    return None


def resolve_dither_mode(mode: Optional[str]) -> str:
    """Normalize a user-specified dithering mode to a canonical value."""
    if not mode:
        return config.DITHERING_MODE

    lowered = mode.strip().lower()
    if lowered in _DITHER_TRUE_VALUES:
        return config.DITHERING_MODE
    if lowered in _DITHER_MODE_ALIASES:
        return _DITHER_MODE_ALIASES[lowered]

    config.logger.warning("[dither] Unknown mode '%s', falling back to %s", mode, config.DITHERING_MODE)
    return config.DITHERING_MODE


def load_font(size: int, candidates: Optional[Sequence[str]] = None) -> ImageFont.ImageFont:
    """Load the first available font from the candidate list or fall back to default."""
    font_candidates = candidates or _default_font_candidates()
    for candidate in font_candidates:
        candidate_path = Path(candidate)
        if not candidate_path.is_absolute():
            candidate_path = PROJECT_ROOT / candidate_path
        if candidate_path.exists():
            try:
                return ImageFont.truetype(candidate_path.as_posix(), size)
            except OSError as exc:
                config.logger.warning("[font] failed to load %s: %s", candidate_path, exc)
    config.logger.warning("[font] falling back to default font")
    return ImageFont.load_default()

def get_ip_address() -> str:
    """
    Get the local IP address of the machine.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.254.254.254', 1))
        ip = s.getsockname()[0]
    except (IndexError, KeyError):
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def get_battery_state(battery_voltage: float) -> float:
    """
    Calculate the battery state based on the given battery voltage.
    """
    if float(battery_voltage) > 4.6: # is charging
        return 255
    
    try:
        battery_state = round(
            (
                (float(battery_voltage) - config.BATTERY_MIN_VOLTAGE) /
                (
                    config.BATTERY_MAX_VOLTAGE -
                    config.BATTERY_MIN_VOLTAGE
                )) * 100, 1
            )
    except ZeroDivisionError:
        battery_state = 0

    if battery_state > 100:
        battery_state = 100
    elif battery_state < 0:
        battery_state = 0
    return battery_state

def get_wifi_signal_strength(rssi: int) -> int:
    """
    Calculate the WiFi signal strength quality based on the RSSI value.
    """
    if rssi <= -100:
        quality = 0
    elif rssi >= -50:
        quality = 100
    else:
        quality = 2 * (rssi + 100)
    return quality

def load_image(image_path: str) -> BytesIO:
    """
    Load an image from a local file path or a URL.
    """
    if image_path.startswith('http://') or image_path.startswith('https://'):
        response = httpx.get(image_path, timeout=10)
        response.raise_for_status()
        return BytesIO(response.content)
    with open(image_path, 'rb') as image_file:
        return BytesIO(image_file.read())


def ensure_image_mode(image: Union[Image.Image, BytesIO, bytes, bytearray], mode: str) -> Image.Image:
    """Return the image in the requested mode, converting only if needed."""
    if isinstance(image, (bytes, bytearray)):
        image = BytesIO(image)
    if isinstance(image, BytesIO):
        image.seek(0)
        image = Image.open(image)
    if image.mode == mode:
        return image
    return image.convert(mode)

def get_no_image() -> BytesIO:
    """
    Create a blank image with a white background and overlay text indicating no image is available,
    along with the current date and time. The image is saved in BMP format and returned as
    a BytesIO object.
    """
    # Create a blank image with white background
    img = Image.new('1', (800, 480), color=1)  # '1' mode for 1-bit pixels, black and white

    # Initialize ImageDraw
    d = ImageDraw.Draw(img)

    # Load font
    text_font = load_font(24, (
        asset_path('DejaVuSans.ttf').as_posix(),
        "DejaVuSans.ttf"
    ))

    # Define text position and content
    text = "No image available"
    date_time = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    text = f"{text}\n{date_time}"
    text_bbox = d.textbbox((0, 0), text, font=text_font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    text_position = ((img.width - text_width) // 2, (img.height - text_height) // 2)

    # Draw text on the image
    d.text(text_position, text, fill=0, font=text_font)  # fill=0 for black

    # Save the image to a BytesIO object
    img_io = BytesIO()
    img.save(img_io, format="BMP")
    img_io.seek(0)
    return img_io

def convert_bmp_bytes_to_png(bmp_bytes: BytesIO) -> BytesIO:
    """Convert a BMP image stored in memory to PNG format.

    The input BytesIO position is reset to the beginning before reading and the
    returned BytesIO is positioned at the start of the PNG data.
    """
    bmp_bytes.seek(0)
    image = Image.open(bmp_bytes)
    png_io = BytesIO()
    image.save(png_io, format="PNG")
    png_io.seek(0)
    return png_io


def convert_to_grayscale_levels(image: Image.Image, levels: int = 4) -> Image.Image:
    """Convert an image to discrete grayscale levels without dithering."""
    if levels < 2:
        return image.convert('L')

    gray = image.convert('L')
    levels_minus = max(levels - 1, 1)

    if not _tone_curve_enabled():
        step = 255 / levels_minus

        def _quantize(value: int) -> int:
            return int(round(value / step) * step)

        return gray.point(_quantize)

    forward = _tone_curve_forward_lut()
    panel_levels, digital_levels = get_effective_grayscale_palette_levels(levels)
    panel_min = int(panel_levels[0]) if panel_levels else 0
    panel_max = int(panel_levels[-1]) if panel_levels else 255
    panel_min_n = panel_min / 255.0
    panel_max_n = panel_max / 255.0
    span_n = panel_max_n - panel_min_n

    remap: List[int] = []
    if span_n <= 0.0:
        remap = list(range(256))
        return gray.point(remap)

    for value in range(256):
        panel_value_n = forward[value] / 255.0
        scaled = (panel_value_n - panel_min_n) / span_n
        scaled = max(0.0, min(1.0, scaled))
        level = int(round(scaled * levels_minus))
        level = max(0, min(level, levels_minus))
        remap.append(int(digital_levels[level]))

    return gray.point(remap)


def _build_grayscale_palette(levels: int) -> Tuple[List[int], List[int]]:
    step_values = _palette_levels_digital(levels)
    if step_values:
        step_values[0] = 0
        step_values[-1] = 255
    palette: List[int] = []
    for value in step_values:
        palette.extend([value, value, value])
    palette.extend([0] * (768 - len(palette)))

    remap_table: List[int] = []
    value_to_index = {value: idx for idx, value in enumerate(step_values)}
    if not _tone_curve_enabled():
        for value in range(256):
            target = value
            closest = min(step_values, key=lambda candidate, t=target: abs(candidate - t))
            remap_table.append(value_to_index[closest])
        return palette, remap_table

    forward = _tone_curve_forward_lut()
    palette_panel = [forward[value] for value in step_values]
    for value in range(256):
        panel_value = forward[value]
        closest_index = min(
            range(len(palette_panel)),
            key=lambda idx, pv=panel_value: abs(palette_panel[idx] - pv)
        )
        remap_table.append(closest_index)
    return palette, remap_table


def _ordered_blue_noise_dither(image: Image.Image, levels: int) -> Image.Image:
    gray = image.convert('L')
    width, height = gray.size
    result = Image.new('L', gray.size)
    src = gray.load()
    dst = result.load()
    levels_minus = max(levels - 1, 1)

    if not _tone_curve_enabled():
        for y in range(height):
            for x in range(width):
                threshold = BLUE_NOISE_MATRIX[y % BLUE_NOISE_HEIGHT][x % BLUE_NOISE_WIDTH] / BLUE_NOISE_AREA
                value = src[x, y] / 255.0
                scaled = value * levels_minus
                base_level = math.floor(scaled)
                frac = scaled - base_level
                level = base_level
                if frac > threshold and level < levels_minus:
                    level += 1
                dst[x, y] = int(round((level / levels_minus) * 255)) if levels_minus else 0
        return result

    forward = _tone_curve_forward_lut()
    panel_levels, digital_levels = get_effective_grayscale_palette_levels(levels)
    panel_min = int(panel_levels[0]) if panel_levels else 0
    panel_max = int(panel_levels[-1]) if panel_levels else 255
    panel_min_n = panel_min / 255.0
    panel_max_n = panel_max / 255.0
    span_n = panel_max_n - panel_min_n
    if span_n <= 0.0:
        return gray

    for y in range(height):
        for x in range(width):
            threshold = BLUE_NOISE_MATRIX[y % BLUE_NOISE_HEIGHT][x % BLUE_NOISE_WIDTH] / BLUE_NOISE_AREA
            value = forward[int(src[x, y])] / 255.0
            scaled = (value - panel_min_n) / span_n
            scaled = max(0.0, min(1.0, scaled)) * levels_minus
            base_level = math.floor(scaled)
            frac = scaled - base_level
            level = base_level
            if frac > threshold and level < levels_minus:
                level += 1
            level = max(0, min(int(level), levels_minus))
            dst[x, y] = int(digital_levels[level])
    return result


def _error_diffusion_dither(image: Image.Image, levels: int, gamma: Optional[float] = None) -> Image.Image:
    gray = image.convert('L')
    width, height = gray.size
    levels_minus = max(levels - 1, 1)
    work: List[List[float]] = []

    if _tone_curve_enabled():
        forward = _tone_curve_forward_lut()
        panel_levels, digital_levels = get_effective_grayscale_palette_levels(levels)
        panel_min = int(panel_levels[0]) if panel_levels else 0
        panel_max = int(panel_levels[-1]) if panel_levels else 255
        panel_min_n = panel_min / 255.0
        panel_max_n = panel_max / 255.0
        span_n = panel_max_n - panel_min_n
        if span_n <= 0.0:
            return image.convert('L')

        use_gamma = bool(gamma) and float(gamma) != 1.0
        for y in range(height):
            row: List[float] = []
            for x in range(width):
                value = forward[int(gray.getpixel((x, y)))] / 255.0
                scaled = (value - panel_min_n) / span_n
                scaled = max(0.0, min(1.0, scaled))
                if use_gamma:
                    scaled = math.pow(scaled, float(gamma))
                row.append(scaled)
            work.append(row)
        result = Image.new('L', gray.size)
        dst = result.load()
        kernel = (
            (1, 0, 7 / 16),
            (-1, 1, 3 / 16),
            (0, 1, 5 / 16),
            (1, 1, 1 / 16)
        )

        for y in range(height):
            for x in range(width):
                value = work[y][x]
                scaled = max(0.0, min(1.0, value))
                level = int(round(scaled * levels_minus))
                level = max(0, min(level, levels_minus))

                if levels_minus:
                    linear_level = level / levels_minus
                else:
                    linear_level = 0.0
                quantized = math.pow(linear_level, float(gamma)) if use_gamma else linear_level
                dst[x, y] = int(digital_levels[level])
                error = value - quantized

                for dx, dy, weight in kernel:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        work[ny][nx] += error * weight
                        work[ny][nx] = max(0.0, min(1.0, work[ny][nx]))
        return result

    inv_gamma = 1.0 / gamma if gamma else None

    for y in range(height):
        row = []
        for x in range(width):
            linear = gray.getpixel((x, y)) / 255.0
            row.append(math.pow(linear, gamma) if gamma else linear)
        work.append(row)

    result = Image.new('L', gray.size)
    dst = result.load()
    kernel = (
        (1, 0, 7 / 16),
        (-1, 1, 3 / 16),
        (0, 1, 5 / 16),
        (1, 1, 1 / 16)
    )

    for y in range(height):
        for x in range(width):
            value = work[y][x]
            scaled = value * levels_minus
            level = round(scaled)
            level = max(0, min(level, levels_minus))
            quantized = level / levels_minus if levels_minus else 0.0
            linear_value = math.pow(quantized, inv_gamma) if inv_gamma else quantized
            dst[x, y] = int(round(linear_value * 255))
            error = value - quantized

            for dx, dy, weight in kernel:
                nx = x + dx
                ny = y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    work[ny][nx] += error * weight
                    work[ny][nx] = max(0.0, min(1.0, work[ny][nx]))

    return result


def _multi_pass_dither(image: Image.Image, levels: int) -> Image.Image:
    higher_levels = min(levels * 2, 16)
    ordered = _ordered_blue_noise_dither(image, higher_levels)
    return _error_diffusion_dither(ordered, levels)


def apply_dithering(
    image: Image.Image,
    levels: int,
    mode: Optional[str] = None
) -> Image.Image:
    """Apply the configured dithering mode to the provided image."""
    normalized = resolve_dither_mode(mode)

    if normalized == 'none':
        return convert_to_grayscale_levels(image, levels)
    if normalized == 'ordered-blue-noise':
        return _ordered_blue_noise_dither(image, levels)
    if normalized == 'perceptual':
        return _error_diffusion_dither(image, levels, gamma=2.2)
    if normalized == 'multi-pass':
        return _multi_pass_dither(image, levels)
    # Default to Floyd-Steinberg
    return _error_diffusion_dither(image, levels)


def ensure_monochrome_bmp(image_blob: BytesIO, dither_mode: Optional[str] = None) -> BytesIO:
    """Convert arbitrary bytes into a TRMNL-compatible 1-bit, palette BMP."""
    image_blob.seek(0)
    image = Image.open(image_blob)

    if image.size != (800, 480):
        image = image.resize((800, 480), resample=Image.Resampling.NEAREST)

    dithered = apply_dithering(image, levels=2, mode=dither_mode)
    mono = dithered.point(lambda value: 255 if value >= 128 else 0, mode='1')

    mono_io = BytesIO()
    mono.save(mono_io, format='BMP')
    mono_io.seek(0)
    image_blob.seek(0)

    bmp_bytes = bytearray(mono_io.getvalue())
    # force 2 color table entries (black, white)
    bmp_bytes[46:50] = (2).to_bytes(4, 'little')
    bmp_bytes[54:62] = bytes([0, 0, 0, 0, 255, 255, 255, 0])

    corrected = BytesIO(bmp_bytes)
    corrected.seek(0)
    return corrected


def save_display_assets(
    image: Image.Image,
    output_dir: str,
    basename: str,
    dither_mode: Optional[str] = None,
    grayscale_levels: Optional[int] = 4
) -> Tuple[str, str]:
    """Persist monochrome BMP plus grayscale PNG assets with optional dithering."""
    os.makedirs(output_dir, exist_ok=True)
    safe_name = Path(basename).stem or 'plugin_output'

    if grayscale_levels is None:
        grayscale = image.convert('L') if image.mode != 'L' else image
    else:
        levels = int(grayscale_levels)
        if dither_mode:
            grayscale = apply_dithering(image, levels=levels, mode=dither_mode)
        else:
            grayscale = convert_to_grayscale_levels(image, levels=levels)
    png_path = os.path.abspath(os.path.join(output_dir, f"{safe_name}.png"))
    grayscale.save(png_path, format='PNG')

    bmp_path = os.path.abspath(os.path.join(output_dir, f"{safe_name}.bmp"))
    if grayscale_levels is None:
        mono = grayscale.point(lambda value: 255 if value >= 128 else 0, mode='1')
        mono_io = BytesIO()
        mono.save(mono_io, format='BMP')
        mono_io.seek(0)
        bmp_bytes = bytearray(mono_io.getvalue())
        bmp_bytes[46:50] = (2).to_bytes(4, 'little')
        bmp_bytes[54:62] = bytes([0, 0, 0, 0, 255, 255, 255, 0])
        with open(bmp_path, 'wb') as file:
            file.write(bytes(bmp_bytes))
    else:
        buffer = BytesIO()
        grayscale.save(buffer, format='PNG')
        buffer.seek(0)
        mono_blob = ensure_monochrome_bmp(buffer, dither_mode=dither_mode)
        with open(bmp_path, 'wb') as file:
            file.write(mono_blob.getvalue())

    return bmp_path, png_path


def generate_image_token(
    image_blob: BytesIO,
    length: int = 16,
    salt: Optional[str] = None
) -> str:
    """Return a short digest that identifies the current image payload."""
    position = image_blob.tell()
    image_blob.seek(0)
    hasher = hashlib.sha256()
    hasher.update(image_blob.read())
    if salt:
        hasher.update(salt.encode('utf-8'))
    digest = hasher.hexdigest()
    image_blob.seek(position)
    return digest[:length]


def generate_grayscale_png(image_blob: BytesIO, levels: int = 4) -> BytesIO:
    """Convert the provided image into a palette PNG with limited grayscale levels."""
    if levels < 2:
        raise ValueError("levels must be >= 2 for grayscale rendering")

    position = image_blob.tell()
    image_blob.seek(0)
    image = Image.open(image_blob)
    quantized = convert_to_grayscale_levels(image, levels)
    palette, remap_table = _build_grayscale_palette(levels)

    indexed = quantized.point(remap_table, 'P')
    indexed.putpalette(palette)

    png_io = BytesIO()
    indexed.save(png_io, format='PNG', optimize=True)
    png_io.seek(0)
    image_blob.seek(position)
    return png_io


def generate_dithered_grayscale_png(
    image_blob: BytesIO,
    levels: int = 4,
    mode: Optional[str] = None
) -> BytesIO:
    """Create a multi-tone PNG using the selected dithering strategy."""
    if levels < 2:
        raise ValueError("levels must be >= 2 for grayscale rendering")

    position = image_blob.tell()
    image_blob.seek(0)
    image = Image.open(image_blob)
    dithered = apply_dithering(image, levels, mode=mode)
    palette, remap_table = _build_grayscale_palette(levels)

    indexed = dithered.point(remap_table, 'P')
    indexed.putpalette(palette)

    png_io = BytesIO()
    indexed.save(png_io, format='PNG', optimize=True)
    png_io.seek(0)
    image_blob.seek(position)
    return png_io


def parse_semver(version: Optional[str]) -> Tuple[int, int, int]:
    """Parse a semantic version string into a numeric tuple."""
    if not version:
        return (0, 0, 0)
    parts = version.split('.')
    numbers = []
    for part in parts[:3]:
        digits = ''.join(ch for ch in part if ch.isdigit())
        if digits:
            numbers.append(int(digits))
        else:
            numbers.append(0)
    while len(numbers) < 3:
        numbers.append(0)
    return numbers[0], numbers[1], numbers[2]


def firmware_supports_grayscale(
    version: Optional[str],
    minimum: Tuple[int, int, int] = (1, 6, 0)
) -> bool:
    """Return True if the firmware version meets the minimum required for grayscale."""
    parsed = parse_semver(version)
    return parsed >= minimum


def to_iso_datetime(value: Optional[datetime.datetime]) -> str:
    """Return an ISO-8601 representation for datetimes or POSIX timestamps."""
    if value is None:
        return ''
    if isinstance(value, (int, float)):
        if value <= 0:
            return ''
        value = datetime.datetime.fromtimestamp(value, datetime.timezone.utc)
    trimmed = value.replace(microsecond=0)
    return trimmed.isoformat()


def to_iso_timestamp(timestamp: Optional[float]) -> str:
    """Return an ISO-8601 string for a POSIX timestamp in seconds."""
    if timestamp is None or timestamp <= 0:
        return ''
    dt = datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc)
    return to_iso_datetime(dt)

