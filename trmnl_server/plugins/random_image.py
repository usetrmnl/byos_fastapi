import asyncio
import logging
import os
import random
from pathlib import Path
from typing import Optional, Sequence, Tuple

from PIL import Image, ImageOps

from .base import PhotographicPlugin, PluginOutput

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_ROOT = Path(os.environ.get('HOME', '~')).expanduser() / 'Pictures' / 'Samurai Jack'
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}


class RandomImagePlugin(PhotographicPlugin):
    """Select a random image from disk, adapt it, and add it to the rotation."""

    BASENAME = "random_image"
    OUTPUT_SUBDIR = "random"
    REGISTRY_ORDER = 50
    REFRESH_INTERVAL = 1800
    DISPLAY_NAME = "Random Image"

    def get_content_ttl(self) -> int:
        return 3600  # 60 minutes

    async def run(self, **kwargs) -> Optional[PluginOutput]:
        target_size = kwargs.get('target_size', (800, 480))
        output_dir = kwargs.get('output_dir', 'web')
        source_root = Path(kwargs.get('image_root', DEFAULT_IMAGE_ROOT))
        return await asyncio.to_thread(
            self._run_sync,
            target_size,
            output_dir,
            source_root
        )

    def _run_sync(
        self,
        target_size: Sequence[int],
        output_dir: str,
        source_root: Path
    ) -> Optional[PluginOutput]:
        os.makedirs(output_dir, exist_ok=True)

        if not source_root.exists():
            logger.warning("Random image root %s does not exist", source_root)
            return None

        image_path = self._pick_random_image(source_root)
        if not image_path:
            logger.warning("No images found under %s", source_root)
            return None

        adapted = self._prepare_image(image_path, target_size)
        output = self.save_assets(
            adapted,
            output_dir,
            'random_image',
            dither_mode='floyd-steinberg'
        )
        logger.info(
            "Random image assets saved to %s and %s",
            output.monochrome_path,
            output.grayscale_path
        )
        return output

    def _pick_random_image(self, root: Path) -> Optional[Path]:
        candidates = [
            path for path in root.rglob('*')
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if not candidates:
            return None
        return random.choice(candidates)

    def _prepare_image(self, image_path: Path, target_size: Sequence[int]) -> Image.Image:
        with Image.open(image_path) as img:
            rgb_image = img.convert('RGB')
            fitted = ImageOps.fit(
                rgb_image,
                target_size,
                method=Image.Resampling.LANCZOS,
                bleed=0.0,
                centering=(0.5, 0.5)
            )
            return fitted.convert('L')
