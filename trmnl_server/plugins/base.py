from abc import ABC, abstractmethod
import asyncio
import logging
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Sequence

from PIL import Image, ImageOps, ImageEnhance, ImageFont, ImageDraw

from .. import config
from ..utils import save_display_assets, load_font as utils_load_font

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PluginOutput:
    """Paths to the generated monochrome BMP and grayscale PNG assets."""

    monochrome_path: str
    grayscale_path: str


@dataclass(frozen=True)
class ChartBounds:
    """Normalized rectangle describing the drawable chart area."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass(frozen=True)
class AxisScale:
    """Normalized Y-axis scaling parameters."""

    axis_min: float
    axis_max: float
    step: float

    @property
    def span(self) -> float:
        return max(self.axis_max - self.axis_min, 1.0)


class PluginBase(ABC):
    """Abstract base class for image-producing plugins with optional adjustments."""

    BASENAME: str = 'plugin'
    OUTPUT_SUBDIR: Optional[str] = None
    SET_PRIMARY: bool = False
    AUTO_REGISTER: bool = True
    REFRESH_INTERVAL: Optional[int] = None
    REGISTRY_ORDER: int = 100

    def __init__(self):
        self.name = self.__class__.__name__

    def get_display_name(self) -> str:
        display_attr = getattr(self, 'DISPLAY_NAME', None)
        return str(display_attr) if display_attr else self.name

    @abstractmethod
    async def run(self, **kwargs) -> Optional[PluginOutput]:
        """Execute the plugin logic asynchronously."""
        raise NotImplementedError

    def get_adjustment_settings(self) -> Tuple[bool, float, float]:
        """Return (apply_contrast, gamma_value, contrast_cutoff)."""
        return (False, 1.0, 0.0)

    def get_content_ttl(self) -> int:
        """Number of seconds this plugin's output remains fresh."""
        return 900

    def apply_adjustments(self, image: Image.Image) -> Image.Image:
        """Apply optional contrast and gamma adjustments according to plugin settings."""
        apply_contrast, gamma_value, contrast_cutoff = self.get_adjustment_settings()
        adjusted = image

        if apply_contrast:
            adjusted = ImageOps.autocontrast(adjusted, cutoff=contrast_cutoff)

        if gamma_value and gamma_value != 1.0:
            inv_gamma = 1.0 / gamma_value
            adjusted = adjusted.point(
                lambda value: max(0, min(255, int(round((value / 255.0) ** inv_gamma * 255))))
            )

        return adjusted

    @staticmethod
    def lift_black_point(image: Image.Image, offset: int = 16) -> Image.Image:
        """Raise the black point to recover detail in deep shadows."""
        offset = max(0, min(offset, 64))
        lut = [min(255, value + offset) for value in range(256)]
        return image.point(lut)

    @staticmethod
    def boost_shadows(image: Image.Image, pivot: int = 180, shadow_gamma: float = 0.7) -> Image.Image:
        """Brighten tonal values below the pivot using a gamma curve."""
        pivot = max(1, min(pivot, 254))
        pivot_norm = pivot / 255.0
        lut = []
        for value in range(256):
            normalized = value / 255.0
            if normalized < pivot_norm:
                ratio = normalized / pivot_norm
                remapped = (ratio ** shadow_gamma) * pivot_norm
            else:
                remapped = normalized
            lut.append(int(round(remapped * 255)))
        return image.point(lut)

    def apply_eink_grading(
        self,
        image: Image.Image,
        *,
        shadow_pivot: int = 180,
        shadow_gamma: float = 0.65,
        brightness: float = 1.1,
        contrast_cutoff: float = 0.05
    ) -> Image.Image:
        """Apply a shadow lift, brightness tweak, and autocontrast pass."""
        lifted = self.boost_shadows(image, pivot=shadow_pivot, shadow_gamma=shadow_gamma)
        brightened = ImageEnhance.Brightness(lifted).enhance(brightness)
        return ImageOps.autocontrast(brightened, cutoff=contrast_cutoff)

    def prepare_image(self, image: Image.Image) -> Image.Image:
        """Convert plugin output to grayscale and apply the configured adjustments."""
        grayscale = image.convert('L') if image.mode != 'L' else image
        return self.apply_adjustments(grayscale)

    def save_assets(
        self,
        image: Image.Image,
        output_dir: str,
        basename: str,
        dither_mode: Optional[str] = None
    ) -> PluginOutput:
        """Apply uniform processing and persist BMP/PNG outputs for the plugin."""
        prepared = self.prepare_image(image)
        bmp_path, png_path = save_display_assets(
            prepared,
            output_dir,
            basename,
            dither_mode=dither_mode
        )
        return PluginOutput(monochrome_path=bmp_path, grayscale_path=png_path)

    @staticmethod
    def load_font(size: int, fallback_paths: Optional[Tuple[str, ...]] = None) -> ImageFont.ImageFont:
        """Attempt to load a font from several candidate paths, falling back gracefully."""
        return utils_load_font(size, fallback_paths)


class ChartPlugin(PluginBase):
    """Base class for numeric time-series charts with smooth curves and shared styling."""

    SERIES_LABEL: str = "Series"
    BASENAME: str = "chart"
    CANVAS_SIZE: Tuple[int, int] = (800, 480)
    MARGIN_X: int = 70
    MARGIN_Y: int = 80
    GRID_Y_STEPS: int = 5
    GRID_X_LABELS: int = 8
    CURVE_SAMPLES: int = 12
    CURVE_SMOOTHING: float = 2.5
    TITLE_FONT_SIZE: int = 32
    AXIS_FONT_SIZE: int = 16
    VALUE_FONT_SIZE: int = 20
    CAPTION_FONT_SIZE: int = 14
    GRID_COLOR: int = 210
    AXIS_COLOR: int = 120
    CURVE_COLOR: int = 0
    MAX_MARKER_RADIUS: int = 8
    CHART_STYLE: str = "area"
    AREA_GRADIENT_TOP: int = 170
    AREA_GRADIENT_BOTTOM: int = 255
    DITHER_MODE: Optional[str] = 'floyd-steinberg'
    CAPTION_TEXT: Optional[str] = None

    def get_content_ttl(self) -> int:
        return 1800  # default 30 minutes

    async def run(self, **kwargs) -> Optional[PluginOutput]:
        output_dir = kwargs.get('output_dir', 'web')
        os.makedirs(output_dir, exist_ok=True)

        dataset = await self._fetch_series()
        if not dataset:
            logger.warning("%s feed returned no datapoints", self.__class__.__name__)
            return None

        chart = await asyncio.to_thread(self._render_chart, dataset)
        output = await asyncio.to_thread(
            self.save_assets,
            chart,
            output_dir,
            self.BASENAME,
            dither_mode=self.DITHER_MODE
        )
        logger.info(
            "%s assets saved to %s and %s",
            self.__class__.__name__,
            output.monochrome_path,
            output.grayscale_path
        )
        return output

    @abstractmethod
    async def _fetch_series(self) -> Sequence[Tuple[str, int]]:
        ...

    def _render_chart(self, dataset: Sequence[Tuple[str, int]]) -> Image.Image:
        image, draw = self._create_canvas()
        bounds = self._chart_bounds()
        fonts = {
            'title': self.load_font(self.TITLE_FONT_SIZE),
            'axis': self.load_font(self.AXIS_FONT_SIZE),
            'value': self.load_font(self.VALUE_FONT_SIZE),
            'caption': self.load_font(self.CAPTION_FONT_SIZE)
        }

        labels: List[str] = [name for name, _ in dataset]
        values: List[int] = [int(value) for _, value in dataset]
        if not values:
            return image

        stats = self._compute_value_stats(values)
        axis_scale = self._calculate_axis_scale(stats[0], stats[1])
        points = self._map_points(values, bounds, axis_scale)
        smooth_points = self._smooth_points(points)

        self._draw_title(draw, fonts['title'])
        self._draw_axes(draw, bounds)
        self._draw_grid(draw, bounds, axis_scale)
        self._draw_max_band(draw, bounds, axis_scale, stats[1])
        self._draw_area_fill(image, smooth_points, bounds)
        self._draw_curve(draw, smooth_points)
        self._draw_y_labels(draw, bounds, fonts['axis'], axis_scale)
        self._draw_x_labels(draw, bounds, fonts['axis'], labels)
        self._draw_max_marker(draw, points, values, fonts['value'])
        self._draw_legend(draw, bounds, fonts['axis'], stats, sum(values))
        self._draw_caption(draw, bounds, fonts['caption'])

        return image

    def _create_canvas(self) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
        image = Image.new('L', self.CANVAS_SIZE, color=255)
        return image, ImageDraw.Draw(image)

    def _chart_bounds(self) -> ChartBounds:
        width, height = self.CANVAS_SIZE
        return ChartBounds(self.MARGIN_X, self.MARGIN_Y, width - self.MARGIN_X, height - 60)

    def _draw_title(self, draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> None:
        title = self.SERIES_LABEL
        bbox = draw.textbbox((0, 0), title, font=font)
        width, _ = self.CANVAS_SIZE
        draw.text(((width - (bbox[2] - bbox[0])) / 2, 20), title, fill=0, font=font)

    def _draw_axes(self, draw: ImageDraw.ImageDraw, bounds: ChartBounds) -> None:
        draw.line([(bounds.x0, bounds.y0), (bounds.x0, bounds.y1)], fill=self.AXIS_COLOR, width=2)
        draw.line([(bounds.x0, bounds.y1), (bounds.x1, bounds.y1)], fill=self.AXIS_COLOR, width=2)

    def _draw_grid(self, draw: ImageDraw.ImageDraw, bounds: ChartBounds, axis_scale: AxisScale) -> None:
        tick_values = self._generate_tick_values(axis_scale)
        for value in tick_values:
            if value in (axis_scale.axis_min, axis_scale.axis_max):
                continue
            ratio = (axis_scale.axis_max - value) / axis_scale.span
            gy = bounds.y0 + ratio * bounds.height
            draw.line([(bounds.x0, gy), (bounds.x1, gy)], fill=self.GRID_COLOR, width=1)
        for step in range(1, self.GRID_X_LABELS):
            gx = bounds.x0 + (step * bounds.width / self.GRID_X_LABELS)
            draw.line([(gx, bounds.y0), (gx, bounds.y1)], fill=self.GRID_COLOR, width=1)

    @staticmethod
    def _compute_value_stats(values: Sequence[int]) -> Tuple[int, int, float]:
        if not values:
            return (0, 1, 1.0)
        min_val = min(values)
        max_val = max(values)
        if min_val == max_val:
            max_val += 1
        span = float(max_val - min_val)
        return (min_val, max_val, span)

    def _map_points(
        self,
        values: Sequence[int],
        bounds: ChartBounds,
        axis_scale: AxisScale
    ) -> List[Tuple[float, float]]:
        sample_count = max(1, len(values) - 1)
        points: List[Tuple[float, float]] = []
        for idx, value in enumerate(values):
            px = bounds.x0 + (idx / sample_count) * bounds.width if sample_count else bounds.x0
            normalized = (value - axis_scale.axis_min) / axis_scale.span if axis_scale.span else 0.0
            py = bounds.y1 - normalized * bounds.height
            points.append((px, py))
        return points

    def _smooth_points(self, points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if len(points) < 2 or self.CURVE_SAMPLES <= 0:
            return list(points)

        xs = [px for px, _ in points]
        ys = [py for _, py in points]
        deltas: List[float] = []
        for idx in range(len(points) - 1):
            dx = xs[idx + 1] - xs[idx]
            if dx <= 0:
                return list(points)
            deltas.append((ys[idx + 1] - ys[idx]) / dx)

        slopes = self._compute_monotone_slopes(xs, deltas)
        smooth: List[Tuple[float, float]] = [points[0]]
        for idx in range(len(points) - 1):
            smooth.extend(
                self._hermite_segment_samples(
                    xs[idx],
                    ys[idx],
                    xs[idx + 1],
                    ys[idx + 1],
                    slopes[idx],
                    slopes[idx + 1]
                )
            )
            smooth.append(points[idx + 1])
        return smooth

    def _compute_monotone_slopes(self, xs: Sequence[float], deltas: Sequence[float]) -> List[float]:
        count = len(xs)
        slopes = [0.0] * count
        if count < 2:
            return slopes

        slopes[0] = self._scale_and_clamp_slope(None, deltas[0], deltas[0])
        slopes[-1] = self._scale_and_clamp_slope(deltas[-1], None, deltas[-1])
        for idx in range(1, count - 1):
            prev = deltas[idx - 1]
            curr = deltas[idx]
            if prev == 0 or curr == 0 or prev * curr < 0:
                slopes[idx] = 0.0
                continue
            dx_prev = xs[idx] - xs[idx - 1]
            dx_next = xs[idx + 1] - xs[idx]
            w1 = 2 * dx_next + dx_prev
            w2 = dx_next + 2 * dx_prev
            raw_slope = (w1 + w2) / (w1 / prev + w2 / curr)
            slopes[idx] = self._scale_and_clamp_slope(prev, curr, raw_slope)
        return slopes

    def _scale_and_clamp_slope(
        self,
        prev_delta: Optional[float],
        next_delta: Optional[float],
        slope: float
    ) -> float:
        if slope == 0.0:
            return 0.0
        scaled = slope * self.CURVE_SMOOTHING
        limits: List[float] = []
        if prev_delta not in (None, 0.0):
            limits.append(3.0 * abs(prev_delta))
        if next_delta not in (None, 0.0):
            limits.append(3.0 * abs(next_delta))
        if not limits:
            return 0.0
        limit = min(limits)
        magnitude = min(abs(scaled), limit)
        return math.copysign(magnitude, scaled)

    def _hermite_segment_samples(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        m0: float,
        m1: float
    ) -> List[Tuple[float, float]]:
        segment_points: List[Tuple[float, float]] = []
        span = x1 - x0
        if span <= 0:
            return segment_points

        for step in range(1, self.CURVE_SAMPLES + 1):
            t = step / (self.CURVE_SAMPLES + 1)
            t2 = t * t
            t3 = t2 * t
            h00 = 2 * t3 - 3 * t2 + 1
            h10 = t3 - 2 * t2 + t
            h01 = -2 * t3 + 3 * t2
            h11 = t3 - t2
            x = x0 + t * span
            y = (
                h00 * y0 +
                h10 * span * m0 +
                h01 * y1 +
                h11 * span * m1
            )
            segment_points.append((x, y))

        return segment_points

    def _is_area_chart(self) -> bool:
        return self.CHART_STYLE.lower() == 'area'

    def _draw_area_fill(
        self,
        image: Image.Image,
        points: Sequence[Tuple[float, float]],
        bounds: ChartBounds
    ) -> None:
        if not self._is_area_chart() or len(points) < 2:
            return
        polygon = self._build_area_polygon(points, bounds)
        area_layer = Image.new('L', self.CANVAS_SIZE, color=self.AREA_GRADIENT_TOP)
        gradient_patch = self._build_area_gradient(bounds)
        area_layer.paste(
            gradient_patch,
            (int(round(bounds.x0)), int(round(bounds.y0)))
        )
        mask = Image.new('L', self.CANVAS_SIZE, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.polygon(self._round_points(polygon), fill=255)
        image.paste(area_layer, mask=mask)

    def _build_area_gradient(self, bounds: ChartBounds) -> Image.Image:
        width = max(1, int(math.ceil(bounds.width)))
        height = max(1, int(math.ceil(bounds.height)))
        top = max(0, min(255, self.AREA_GRADIENT_TOP))
        bottom = max(0, min(255, self.AREA_GRADIENT_BOTTOM))
        column = Image.new('L', (1, height), color=top)
        for y in range(height):
            ratio = y / max(1, height - 1)
            value = int(round(top + (bottom - top) * ratio))
            column.putpixel((0, y), value)
        return column.resize((width, height))

    @staticmethod
    def _build_area_polygon(
        points: Sequence[Tuple[float, float]],
        bounds: ChartBounds
    ) -> List[Tuple[float, float]]:
        polygon: List[Tuple[float, float]] = [(bounds.x0, bounds.y1)]
        polygon.extend(points)
        polygon.append((points[-1][0], bounds.y1))
        return polygon

    @staticmethod
    def _round_points(points: Sequence[Tuple[float, float]]) -> List[Tuple[int, int]]:
        return [(int(round(px)), int(round(py))) for px, py in points]

    def _draw_curve(self, draw: ImageDraw.ImageDraw, points: Sequence[Tuple[float, float]]) -> None:
        if len(points) < 2:
            return
        draw.line(points, fill=self.CURVE_COLOR, width=3)

    def _draw_y_labels(
        self,
        draw: ImageDraw.ImageDraw,
        bounds: ChartBounds,
        font: ImageFont.ImageFont,
        axis_scale: AxisScale
    ) -> None:
        for value in self._generate_tick_values(axis_scale):
            ratio = (axis_scale.axis_max - value) / axis_scale.span
            yy = bounds.y0 + ratio * bounds.height
            label = f"{int(round(value)):,}"
            bbox = draw.textbbox((0, 0), label, font=font)
            draw.line([(bounds.x0 - 6, yy), (bounds.x0, yy)], fill=self.AXIS_COLOR, width=1)
            draw.text((bounds.x0 - bbox[2] - 12, yy - (bbox[3] - bbox[1]) / 2), label, fill=self.AXIS_COLOR, font=font)

    def _draw_x_labels(
        self,
        draw: ImageDraw.ImageDraw,
        bounds: ChartBounds,
        font: ImageFont.ImageFont,
        labels: Sequence[str]
    ) -> None:
        if not labels:
            return
        step = max(1, len(labels) // self.GRID_X_LABELS)
        for idx in range(0, len(labels), step):
            label = labels[idx]
            bbox = draw.textbbox((0, 0), label, font=font)
            px = bounds.x0 + (idx / max(1, len(labels) - 1)) * bounds.width
            draw.text((px - (bbox[2] - bbox[0]) / 2, bounds.y1 + 8), label, fill=0, font=font)

    def _draw_max_band(
        self,
        draw: ImageDraw.ImageDraw,
        bounds: ChartBounds,
        axis_scale: AxisScale,
        max_value: int
    ) -> None:
        if not axis_scale.span:
            return
        ratio = (axis_scale.axis_max - max_value) / axis_scale.span
        yy = bounds.y0 + ratio * bounds.height
        draw.line([(bounds.x0, yy), (bounds.x1, yy)], fill=self.AXIS_COLOR, width=1)

    def _draw_max_marker(
        self,
        draw: ImageDraw.ImageDraw,
        points: Sequence[Tuple[float, float]],
        values: Sequence[int],
        font: ImageFont.ImageFont
    ) -> None:
        if not points or not values:
            return
        max_idx = max(range(len(values)), key=lambda idx: values[idx])
        px, py = points[max_idx]
        draw.ellipse(
            (
                px - self.MAX_MARKER_RADIUS,
                py - self.MAX_MARKER_RADIUS,
                px + self.MAX_MARKER_RADIUS,
                py + self.MAX_MARKER_RADIUS
            ),
            outline=self.AXIS_COLOR,
            width=2,
            fill=255
        )
        inner_radius = max(2, self.MAX_MARKER_RADIUS // 3)
        draw.ellipse(
            (px - inner_radius, py - inner_radius, px + inner_radius, py + inner_radius),
            fill=self.CURVE_COLOR
        )
        label = f"{values[max_idx]:,}"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        offset = 12 if px < (self.CANVAS_SIZE[0] - 120) else -text_width - 12
        draw.text((px + offset, py - (bbox[3] - bbox[1]) / 2), label, fill=0, font=font)

    def _draw_legend(
        self,
        draw: ImageDraw.ImageDraw,
        bounds: ChartBounds,
        font: ImageFont.ImageFont,
        stats: Tuple[int, int, float],
        total: int
    ) -> None:
        min_val, max_val, _ = stats
        legend = f"min {min_val:,} · max {max_val:,} · total {total:,}"
        draw.text((bounds.x0, bounds.y1 + 40), legend, fill=self.AXIS_COLOR, font=font)

    def _draw_caption(
        self,
        draw: ImageDraw.ImageDraw,
        bounds: ChartBounds,
        font: ImageFont.ImageFont
    ) -> None:
        caption = getattr(self, 'CAPTION_TEXT', None)
        if not caption:
            return
        bbox = draw.textbbox((0, 0), caption, font=font)
        width = bbox[2] - bbox[0]
        x = bounds.x1 - width
        y = bounds.y1 + 40
        draw.text((x, y), caption, fill=self.AXIS_COLOR, font=font)

    def _calculate_axis_scale(self, min_value: int, max_value: int, ticks: int = 5) -> AxisScale:
        if max_value == min_value:
            max_value += 1
        span = max_value - min_value
        nice_steps = (1, 2, 2.5, 5, 10)
        base_power = max(math.floor(math.log10(max(max_value, 1))) - 1, 0)
        base_unit = max(10 ** base_power, 1)
        step = nice_steps[-1] * base_unit
        desired_ticks = max(ticks, self.GRID_Y_STEPS)
        span = span if span > 0 else step

        for candidate in nice_steps:
            candidate_step = int(math.ceil(candidate * base_unit))
            ticks_needed = math.ceil(span / candidate_step)
            if ticks_needed <= desired_ticks + 2:
                step = candidate_step
                break

        axis_min = math.floor(min_value / step) * step
        axis_max = math.ceil(max_value / step) * step
        if axis_max == axis_min:
            axis_max = axis_min + step
        return AxisScale(axis_min, axis_max, step)

    def _generate_tick_values(self, axis_scale: AxisScale) -> List[float]:
        if axis_scale.step <= 0:
            return [axis_scale.axis_min, axis_scale.axis_max]
        ticks: List[float] = []
        current = axis_scale.axis_min
        while current <= axis_scale.axis_max + 1e-6:
            ticks.append(current)
            current += axis_scale.step
        if ticks[-1] != axis_scale.axis_max:
            ticks.append(axis_scale.axis_max)
        return ticks


class PhotographicPlugin(PluginBase):
    """Base class for photograph-oriented plugins with enhanced grading."""

    def get_adjustment_settings(self) -> Tuple[bool, float, float]:
        return (True, 1.2, 0.05)

    def apply_adjustments(self, image: Image.Image) -> Image.Image:
        if not bool(getattr(config, 'PHOTO_GRADING_ENABLED', True)):
            return image

        adjusted = super().apply_adjustments(image)
        return self.apply_eink_grading(
            adjusted,
            shadow_pivot=180,
            shadow_gamma=0.65,
            brightness=1.1,
            contrast_cutoff=0.05
        )
