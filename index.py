"""Vercel entrypoint for the Escriba demo."""

import os

from sqlalchemy.engine import make_url

from app.core.config import Settings
from app.main import create_app


def hosted_settings() -> Settings:
    if os.environ.get("ESCRIBA_DATABASE_URL"):
        return Settings()

    # Neon supplies a direct URL and can also supply a pooler URL. Prefer the
    # direct endpoint because asyncpg already pools connections in Escriba.
    database_url = os.environ.get("DATABASE_URL_UNPOOLED") or os.environ.get("DATABASE_URL")
    if not database_url:
        return Settings()

    url = make_url(database_url).set(drivername="postgresql+asyncpg")
    url = url.difference_update_query(["sslmode", "channel_binding"])
    url = url.update_query_dict({"ssl": "require"})
    return Settings(database_url=url.render_as_string(hide_password=False))


app = create_app(hosted_settings())
