"""Async SQLAlchemy 2.0 engine, session factory, and declarative base.

The engine and session factory are created inside the FastAPI lifespan and stored on
``app.state``; nothing here holds module-level connection state. The same code runs
against ``sqlite+aiosqlite`` locally and ``postgresql+asyncpg`` in production.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import Depends
from sqlalchemy import MetaData, event
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from starlette.requests import HTTPConnection

from app.core.config import Settings

# Deterministic constraint names so future Alembic autogenerate diffs are stable
# across SQLite and PostgreSQL.
NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def _enable_sqlite_foreign_keys(dbapi_connection: Any, _connection_record: Any) -> None:
    # SQLite ignores FOREIGN KEY / ON DELETE CASCADE unless enabled per connection.
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def build_engine(settings: Settings) -> AsyncEngine:
    """Create an async engine tuned for the configured backend."""
    url = make_url(settings.database_url)
    is_sqlite = url.get_backend_name() == "sqlite"

    engine_kwargs: dict[str, Any] = {"echo": settings.database_echo, "future": True}
    if not is_sqlite:
        engine_kwargs.update(
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_pre_ping=True,
            pool_recycle=1800,
        )

    engine = create_async_engine(url, **engine_kwargs)
    if is_sqlite:
        event.listen(engine.sync_engine, "connect", _enable_sqlite_foreign_keys)
    return engine


def build_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Session factory. ``expire_on_commit=False`` keeps ORM objects usable post-commit."""
    return async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)


async def create_schema(engine: AsyncEngine) -> None:
    """Create all tables. Development convenience only; production uses migrations."""
    # Import for side effects so every model is registered on Base.metadata.
    import app.models  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db_session(connection: HTTPConnection) -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a request-scoped session.

    Services own commit boundaries; any exception escaping the request rolls back.
    """
    sessionmaker: async_sessionmaker[AsyncSession] = connection.app.state.sessionmaker
    async with sessionmaker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


DbSession = Annotated[AsyncSession, Depends(get_db_session)]
