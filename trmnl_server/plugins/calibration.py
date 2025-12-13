from __future__ import annotations

from os import makedirs
from typing import Optional, Sequence, Tuple

from PIL import Image, ImageDraw

from .base import PluginBase, PluginOutput
from .. import config
from ..utils import get_available_dither_modes, get_effective_grayscale_palette_levels, save_display_assets


def _draw_labeled_patch(
    draw: ImageDraw.ImageDraw,
    *,
    box: Tuple[int, int, int, int],
    fill: Tuple[int, int, int],
    label: str,
    font
) -> None:
    draw.rectangle(box, fill=fill)
    x0, y0, x1, _y1 = box
    draw.rectangle((x0, y0, x1, y0 + 20), fill=(255, 255, 255))
    draw.text((x0 + 4, y0 + 2), label, fill=(0, 0, 0), font=font)


def _render_calibration_canvas(
    size: Tuple[int, int],
    *,
    title: str,
    font_title,
    font_small,
    strip_panel_levels: Optional[Sequence[int]] = None,
    strip_digital_levels: Optional[Sequence[int]] = None
) -> Image.Image:
    width, _height = size
    image = Image.new('RGB', size, color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    draw.text((18, 14), title, fill=(0, 0, 0), font=font_title)

    # Grayscale gradient ramp
    ramp_top = 60
    ramp_height = 40
    for x in range(width):
        v = int(round((x / max(width - 1, 1)) * 255))
        draw.line((x, ramp_top, x, ramp_top + ramp_height), fill=(v, v, v))
    draw.rectangle((0, ramp_top, width - 1, ramp_top + ramp_height), outline=(0, 0, 0))

    # Tick marks every 32 values
    for v in range(0, 256, 32):
        x = int(round((v / 255.0) * (width - 1)))
        draw.line((x, ramp_top + ramp_height + 2, x, ramp_top + ramp_height + 12), fill=(0, 0, 0))
        draw.text((x + 2, ramp_top + ramp_height + 10), str(v), fill=(0, 0, 0), font=font_small)

    # Solid reference strip using the *effective* palette levels.
    # This makes it easy to see what the server is actually sending after tone-curve
    # compensation (often not 0/85/170/255 in digital space).
    strip_top = 122
    strip_height = 20
    strip_x0 = 18
    strip_x1 = width - 18
    strip_gap = 8
    block_w = int((strip_x1 - strip_x0 - (3 * strip_gap)) / 4)

    if strip_panel_levels is None or strip_digital_levels is None:
        panel_levels, digital_levels = get_effective_grayscale_palette_levels(4)
    else:
        panel_levels = list(strip_panel_levels)
        digital_levels = list(strip_digital_levels)

    for idx, value in enumerate(digital_levels[:4]):
        x0 = strip_x0 + idx * (block_w + strip_gap)
        x1 = x0 + block_w
        box = (x0, strip_top, x1, strip_top + strip_height)
        draw.rectangle(box, fill=(value, value, value), outline=(0, 0, 0))
        text_color = (255, 255, 255) if value < 96 else (0, 0, 0)
        panel_value = panel_levels[idx] if idx < len(panel_levels) else value
        draw.text((x0 + 4, strip_top + 2), f"{value}/{panel_value}", fill=text_color, font=font_small)

    # Color patches (helps visualize how RGB collapses to grayscale)
    patches_top = 150
    patch_w = 190
    patch_h = 78
    gap = 10
    colors: Sequence[Tuple[str, Tuple[int, int, int]]] = (
        ("RED", (255, 0, 0)),
        ("GREEN", (0, 255, 0)),
        ("BLUE", (0, 0, 255)),
        ("CYAN", (0, 255, 255)),
        ("MAGENTA", (255, 0, 255)),
        ("YELLOW", (255, 255, 0)),
        ("ORANGE", (255, 165, 0)),
        ("PURPLE", (128, 0, 128)),
    )

    for idx, (label, color) in enumerate(colors):
        col = idx % 4
        row = idx // 4
        x0 = 18 + col * (patch_w + gap)
        y0 = patches_top + row * (patch_h + gap)
        x1 = x0 + patch_w
        y1 = y0 + patch_h
        _draw_labeled_patch(draw, box=(x0, y0, x1, y1), fill=color, label=label, font=font_small)
        draw.rectangle((x0, y0, x1, y1), outline=(0, 0, 0))

    # Fine detail patterns
    patterns_top = 330
    pattern_h = 140
    pattern_w = 250

    def box_at(index: int) -> Tuple[int, int, int, int]:
        x0 = 18 + index * (pattern_w + 10)
        y0 = patterns_top
        return (x0, y0, x0 + pattern_w, y0 + pattern_h)

    # 1px vertical lines
    x0, y0, x1, y1 = box_at(0)
    draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255), outline=(0, 0, 0))
    draw.text((x0 + 4, y0 + 2), "1px vertical", fill=(0, 0, 0), font=font_small)
    for x in range(x0 + 8, x1 - 8):
        if (x - (x0 + 8)) % 2 == 0:
            draw.line((x, y0 + 24, x, y1 - 8), fill=(0, 0, 0))

    # 1px diagonal
    x0, y0, x1, y1 = box_at(1)
    draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255), outline=(0, 0, 0))
    draw.text((x0 + 4, y0 + 2), "diagonal", fill=(0, 0, 0), font=font_small)
    for i in range(0, min(x1 - x0, y1 - y0) - 40):
        if i % 2 == 0:
            draw.point((x0 + 20 + i, y0 + 30 + i), fill=(0, 0, 0))

    # Checkerboard 2x2
    x0, y0, x1, y1 = box_at(2)
    draw.rectangle((x0, y0, x1, y1), fill=(255, 255, 255), outline=(0, 0, 0))
    draw.text((x0 + 4, y0 + 2), "checker 2x2", fill=(0, 0, 0), font=font_small)
    cell = 6
    for yy in range(y0 + 26, y1 - 8, cell):
        for xx in range(x0 + 8, x1 - 8, cell):
            if (((xx - (x0 + 8)) // cell) + ((yy - (y0 + 26)) // cell)) % 2 == 0:
                draw.rectangle((xx, yy, xx + cell - 1, yy + cell - 1), fill=(0, 0, 0))

    return image


class CalibrationPlugin(PluginBase):
    AUTO_REGISTER: bool = config.CALIBRATION_PLUGIN_ENABLED
    DISPLAY_NAME: str = 'Calibration'
    BASENAME: str = 'calibration'
    OUTPUT_SUBDIR: Optional[str] = 'calibration'
    REGISTRY_ORDER: int = 5
    VARIANT_ROOT_NAME: str = 'calibration'
    PRIMARY_DITHER_MODE: str = 'floyd-steinberg'
    PRIMARY_VARIANT: str = ''

    def prepare_image(self, image: Image.Image) -> Image.Image:
        """Calibration output must bypass any plugin-specific grading/tweaks."""
        return image.convert('L') if image.mode != 'L' else image

    def apply_adjustments(self, image: Image.Image) -> Image.Image:
        """No-op: calibration images should remain ungraded."""
        return image

    async def run(self, **kwargs) -> Optional[PluginOutput]:
        output_dir = str(kwargs.get('output_dir') or '').strip() or 'web'
        makedirs(output_dir, exist_ok=True)

        font_title = self.load_font(34)
        font_small = self.load_font(14)

        primary_variant = (self.PRIMARY_VARIANT or '').strip().lower() or self.PRIMARY_DITHER_MODE

        plugin_label = self.get_display_name()
        title_prefix = f"TRMNL {plugin_label}" if plugin_label else "TRMNL Calibration"

        if primary_variant == 'unquantized':
            title = f"{title_prefix} — unquantized (raw, no tone curve/dither)"
        elif primary_variant == 'none':
            title = f"{title_prefix} — none (tone curve, no dithering)"
        else:
            title = f"{title_prefix} — {primary_variant} (tone curve + dithering)"

        base = _render_calibration_canvas(
            (800, 480),
            title=title,
            font_title=font_title,
            font_small=font_small,
            strip_panel_levels=[0, 85, 170, 255] if primary_variant == 'unquantized' else None,
            strip_digital_levels=[0, 85, 170, 255] if primary_variant == 'unquantized' else None
        )

        if primary_variant == 'unquantized':
            basename = f"{self.VARIANT_ROOT_NAME}_unquantized"
            bmp_path, png_path = save_display_assets(
                base,
                output_dir,
                basename,
                dither_mode='none',
                grayscale_levels=None
            )
            return PluginOutput(monochrome_path=bmp_path, grayscale_path=png_path)

        basename = f"{self.VARIANT_ROOT_NAME}_{primary_variant}".replace('-', '_')
        bmp_path, png_path = save_display_assets(
            base,
            output_dir,
            basename,
            dither_mode=primary_variant
        )
        return PluginOutput(monochrome_path=bmp_path, grayscale_path=png_path)


class CalibrationNonePlugin(CalibrationPlugin):
    """Calibration plugin variant that returns the non-dithered image for rotation."""

    DISPLAY_NAME: str = 'Calibration (No Dither)'
    BASENAME: str = 'calibration_none'
    OUTPUT_SUBDIR: Optional[str] = 'calibration'
    REGISTRY_ORDER: int = 6
    PRIMARY_DITHER_MODE: str = 'none'


class CalibrationUnquantizedPlugin(CalibrationPlugin):
    """Calibration plugin variant that returns the unquantized (8-bit) image for rotation."""

    DISPLAY_NAME: str = 'Calibration (Unquantized)'
    BASENAME: str = 'calibration_unquantized'
    OUTPUT_SUBDIR: Optional[str] = 'calibration'
    REGISTRY_ORDER: int = 7
    PRIMARY_VARIANT: str = 'unquantized'


class CalibrationPerceptualPlugin(CalibrationPlugin):
    """Calibration plugin variant that returns the unquantized (8-bit) image for rotation."""

    DISPLAY_NAME: str = 'Calibration (Perceptual)'
    BASENAME: str = 'calibration_perceptual'
    OUTPUT_SUBDIR: Optional[str] = 'calibration'
    REGISTRY_ORDER: int = 7
    PRIMARY_VARIANT: str = 'perceptual'
    PRIMARY_DITHER_MODE: str = 'perceptual'
