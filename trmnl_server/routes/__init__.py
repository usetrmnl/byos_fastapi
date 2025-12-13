"""FastAPI routers for the TRMNL local server."""

from .api import router as api_router
from .images import router as image_router
from .pages import router as page_router

__all__ = ['api_router', 'image_router', 'page_router']
