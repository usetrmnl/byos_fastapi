"""Page routes for TRMNL local server."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from .. import utils

router = APIRouter()


@router.get('/')
def index() -> HTMLResponse:
    """Serve the main HTML dashboard."""
    index_path = utils.asset_path('index.html')
    with open(index_path, 'r', encoding='utf-8') as file:
        return HTMLResponse(file.read())
