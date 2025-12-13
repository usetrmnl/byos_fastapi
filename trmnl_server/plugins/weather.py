import asyncio
from typing import Any, Dict, List

import httpx
from PIL import Image, ImageDraw

from .base import PluginBase, PluginOutput
import logging
from .. import config

logger = logging.getLogger(__name__)

class WeatherPlugin(PluginBase):
    """
    Plugin to fetch weather data and generate a Braun-inspired minimalist visualization.
    Uses only black and white with clean lines and excellent typography.
    """
    
    BASENAME = "weather"
    OUTPUT_SUBDIR = "weather"
    SET_PRIMARY = True
    REGISTRY_ORDER = 10
    DISPLAY_NAME = "Weather"

    def __init__(self):
        super().__init__()
        # Glyphs temporarily disabled — keep keys for compatibility
        self.icons = {
            'sun': '',
            'cloud': '',
            'rain': '',
            'wind': '',
            'thermometer': '',
            'droplet': '',
        }

    async def run(self, **kwargs):
        """
        Fetch weather and generate Braun-inspired black and white image.
        kwargs can contain 'latitude' and 'longitude'.
        """
        lat = kwargs.get('latitude', 38.7223) # Default: Lisbon, Portugal
        lon = kwargs.get('longitude', -9.1393)
        output_dir = kwargs.get('output_dir', config.ASSETS_ROOT)
        
        logger.info(f"Running WeatherPlugin for lat={lat}, lon={lon}")
        
        try:
            data = await self._fetch_weather_data(lat, lon)
            
            current_weather = data.get('current_weather', {})
            current_temp = current_weather.get('temperature', 'N/A')
            current_windspeed = current_weather.get('windspeed', 'N/A')
            
            hourly = data.get('hourly', {})
            times = hourly.get('time', [])[:24] # Next 24 hours
            temps = hourly.get('temperature_2m', [])[:24]
            precip = hourly.get('precipitation', [])[:24]
            
            if not times or not temps:
                logger.error("No weather data found")
                return
            
            # Calculate temperature range
            min_temp = min(temps)
            max_temp = max(temps)
            temp_range = max_temp - min_temp if max_temp != min_temp else 1
            
            # Calculate precipitation range
            max_precip = max(precip) if precip else 0
            
            image = await asyncio.to_thread(
                self._render_weather_panel,
                current_temp,
                current_windspeed,
                times,
                temps,
                precip,
                min_temp,
                max_temp,
                max_precip
            )
            output = await asyncio.to_thread(self.save_assets, image, output_dir, 'weather')
            logger.info(
                "Weather plugin execution complete. Assets saved to %s and %s",
                output.monochrome_path,
                output.grayscale_path
            )
            return output

        except Exception as e:
            logger.error(f"Error running WeatherPlugin: {e}")
            raise

    async def _fetch_weather_data(self, lat: float, lon: float) -> Dict[str, Any]:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}&hourly=temperature_2m,precipitation&current_weather=true"
        )
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.json()

    def _render_weather_panel(
        self,
        current_temp: Any,
        current_windspeed: Any,
        times: List[str],
        temps: List[float],
        precip: List[float],
        min_temp: float,
        max_temp: float,
        max_precip: float
    ) -> Image.Image:
        img = Image.new('RGB', (800, 480), color='white')
        draw = ImageDraw.Draw(img)

        title_font = self.load_font(32)
        temp_font = self.load_font(96)
        label_font = self.load_font(18)
        small_font = self.load_font(14)
        tick_font = self.load_font(11)
        # Use the project's default font for icons (Space Grotesk)
        icon_font = self.load_font(24)

        margin = 30
        y_pos = margin

        # Title (glyphs removed) — align at left margin
        draw.text((margin, y_pos), "WEATHER", fill='black', font=title_font)
        y_pos += 50

        draw.line([(margin, y_pos), (800 - margin, y_pos)], fill='black', width=1)
        y_pos += 25

        draw.text((margin, y_pos), f"{current_temp}°", fill='black', font=temp_font)

        wind_icon_x = margin + 180
        # Wind glyph removed; draw only the text aligned where the icon used to be
        draw.text((wind_icon_x, y_pos + 73), f"{current_windspeed} km/h",
              fill='black', font=small_font)
        y_pos += 120

        draw.line([(margin, y_pos), (800 - margin, y_pos)], fill='black', width=1)
        y_pos += 25

        chart_y = y_pos
        chart_height = 200
        chart_width = 350

        chart1_x = margin
        draw.text((chart1_x, chart_y - 20), "TEMPERATURE", fill='black', font=label_font)
        self._draw_braun_bar_chart(
            draw, chart1_x, chart_y, chart_width, chart_height,
            temps[::2], times[::2], min_temp, max_temp,
            tick_font
        )

        chart2_x = margin + chart_width + 40
        # Droplet glyph removed; align label at chart start
        draw.text((chart2_x, chart_y - 20), "PRECIPITATION", fill='black', font=label_font)
        self._draw_braun_bar_chart(
            draw, chart2_x, chart_y, chart_width, chart_height,
            precip[::2], times[::2], 0, max_precip if max_precip > 0 else 1,
            tick_font
        )

        return img
    
    def _draw_braun_bar_chart(self, draw, x, y, width, height, values, labels, min_val, max_val, font):
        """
        Draw a Braun-inspired minimalist bar chart.
        - Thin, clean lines
        - No fills, only outlines
        - Minimal labels
        - Grid lines for readability
        """
        # Draw thin axes
        draw.line([(x, y + height), (x + width, y + height)], fill='black', width=1)
        draw.line([(x, y), (x, y + height)], fill='black', width=1)
        
        # Draw subtle horizontal grid lines (Braun style)
        for i in range(1, 4):
            grid_y = y + (i * height / 4)
            draw.line([(x, grid_y), (x + width, grid_y)], fill='#CCCCCC', width=1)
        
        # Calculate bar width
        num_bars = len(values)
        if num_bars == 0:
            return
        
        bar_spacing = 3
        bar_width = (width - (num_bars + 1) * bar_spacing) / num_bars
        
        # Draw bars - Braun style: outlined rectangles, no fill
        val_range = max_val - min_val if max_val != min_val else 1
        for i, val in enumerate(values):
            bar_x = x + bar_spacing + i * (bar_width + bar_spacing)
            bar_height = ((val - min_val) / val_range) * height
            bar_y = y + height - bar_height
            
            # Draw outlined bar (no fill for Braun aesthetic)
            if bar_height > 2:  # Only draw if visible
                draw.rectangle(
                    [(bar_x, bar_y), (bar_x + bar_width, y + height)],
                    fill='black',  # Solid black bars for contrast
                    outline='black',
                    width=1
                )
        
        # Draw minimal x-axis labels (every 4th)
        for i in range(0, len(labels), 4):
            if i < len(labels):
                label_x = x + bar_spacing + i * (bar_width + bar_spacing)
                time_str = labels[i].split('T')[1][:2]  # Just hour
                bbox = draw.textbbox((0, 0), time_str, font=font)
                text_width = bbox[2] - bbox[0]
                draw.text((label_x - text_width/2 + bar_width/2, y + height + 5), 
                         time_str, fill='black', font=font)
        
        # Draw minimal y-axis scale (3 ticks) - closer to axis
        for i in range(3):
            tick_val = min_val + (val_range * i / 2)
            tick_y = y + height - (i / 2) * height
            # Small tick mark
            draw.line([(x - 3, tick_y), (x, tick_y)], fill='black', width=1)
            # Value label - positioned closer to axis
            label_text = f"{tick_val:.0f}"
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            draw.text((x - text_width - 8, tick_y - 6), label_text, fill='black', font=font)


