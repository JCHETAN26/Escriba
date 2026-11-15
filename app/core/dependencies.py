"""Shared FastAPI dependencies.

Dependencies take ``HTTPConnection`` so they work on both HTTP and WebSocket routes.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from starlette.requests import HTTPConnection

from app.core.config import Settings


def get_settings(connection: HTTPConnection) -> Settings:
    settings: Settings = connection.app.state.settings
    return settings


SettingsDep = Annotated[Settings, Depends(get_settings)]
