"""Escriba application factory.

Run with: ``uvicorn app.main:create_app --factory --reload``
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from openai import AsyncOpenAI

from app.core.config import Settings
from app.core.database import build_engine, build_sessionmaker, create_schema
from app.routers import audio, encounter, intake, orchestrator, outreach, workspace

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    engine = build_engine(settings)
    app.state.engine = engine
    app.state.sessionmaker = build_sessionmaker(engine)

    if settings.auto_create_schema:
        await create_schema(engine)
        logger.info("Database schema ensured (auto_create_schema=true)")

    openai_client: AsyncOpenAI | None = None
    if settings.openai_api_key is not None:
        openai_client = AsyncOpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            base_url=str(settings.openai_base_url) if settings.openai_base_url else None,
            max_retries=settings.openai_max_retries,
            timeout=settings.guardrail_timeout_seconds,
        )
    else:
        logger.warning("ESCRIBA_OPENAI_API_KEY not set: clinical guardrail is DISABLED")
    app.state.openai_client = openai_client

    try:
        yield
    finally:
        if openai_client is not None:
            await openai_client.close()
        await engine.dispose()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    is_production = settings.environment == "production"
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        summary="Agentic clinical scribe: Autonomous Planning, Human Approval.",
        lifespan=lifespan,
        docs_url=None if is_production else "/docs",
        redoc_url=None if is_production else "/redoc",
        openapi_url=None if is_production else "/openapi.json",
    )
    app.state.settings = settings
    app.include_router(workspace.router)
    if settings.demo_mode:
        app.include_router(intake.router)
        app.include_router(audio.router)
        app.include_router(encounter.router)
        app.include_router(orchestrator.router)
        app.include_router(outreach.router)

    @app.get("/healthz", tags=["ops"], summary="Liveness probe")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app
