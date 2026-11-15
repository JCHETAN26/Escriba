"""Application configuration, loaded from environment variables (prefix ``ESCRIBA_``)."""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, AnyHttpUrl, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SYNTHESIS_MODEL = "gpt-5.1"


class Settings(BaseSettings):
    """Typed runtime settings. Instantiated once per app in ``create_app`` (no module globals)."""

    model_config = SettingsConfigDict(
        env_prefix="ESCRIBA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    app_name: str = "Escriba"
    environment: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    database_url: str = "sqlite+aiosqlite:///./escriba.db"
    database_echo: bool = False
    database_pool_size: int = Field(default=10, ge=1)
    database_max_overflow: int = Field(default=20, ge=0)
    auto_create_schema: bool = True
    demo_mode: bool = False

    twilio_account_sid: str | None = None
    twilio_auth_token: SecretStr | None = None
    twilio_from_number: str | None = None
    twilio_validate_signature: bool = True
    public_base_url: AnyHttpUrl | None = None

    default_provider_id: str = Field(default="provider-sandbox-001", min_length=1, max_length=64)

    # --- LLM guardrail (OpenAI) ---
    openai_api_key: SecretStr | None = None
    openai_base_url: AnyHttpUrl | None = None  # Azure OpenAI / gateway / proxy
    openai_max_retries: int = Field(default=1, ge=0, le=5)  # real-time path: fail fast
    guardrail_model: str = Field(default="gpt-4o-mini", min_length=1)
    guardrail_temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)
    guardrail_timeout_seconds: float = Field(default=8.0, gt=0, le=60)
    guardrail_max_concurrency: int = Field(default=4, ge=1, le=32)
    guardrail_max_pending: int = Field(default=32, ge=1, le=1024)
    guardrail_context_window: int = Field(default=6, ge=0, le=50)

    # --- Encounter audio stream ---
    stream_api_token: SecretStr | None = None
    stream_max_message_chars: int = Field(default=512_000, ge=1_024)
    stream_drain_timeout_seconds: float = Field(default=10.0, ge=0, le=120)
    mock_transcription_latency_ms: int = Field(default=400, ge=0, le=10_000)
    transcription_model: str = Field(default="gpt-4o-transcribe-diarize", min_length=1)
    transcription_timeout_seconds: float = Field(default=120.0, gt=0, le=300)
    spanish_visits_note_language: Literal["en", "es"] = Field(
        default="es",
        validation_alias=AliasChoices(
            "SPANISH_VISITS_NOTE_LANGUAGE", "ESCRIBA_SPANISH_VISITS_NOTE_LANGUAGE"
        ),
    )

    # --- Clinical NLP Synthesis ---
    synthesis_model: str = Field(default=DEFAULT_SYNTHESIS_MODEL, min_length=1)
    synthesis_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    synthesis_timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    synthesis_max_transcript_chars: int = Field(default=100_000, ge=1_000)

    @property
    def guardrail_configured(self) -> bool:
        return self.openai_api_key is not None

    @model_validator(mode="after")
    def _enforce_production_safety(self) -> Settings:
        if self.demo_mode and self.environment != "development":
            raise ValueError("Demo mode is available only in development.")
        if self.twilio_validate_signature and self.twilio_auth_token is None:
            raise ValueError(
                "ESCRIBA_TWILIO_AUTH_TOKEN is required when signature validation is enabled. "
                "Set ESCRIBA_TWILIO_VALIDATE_SIGNATURE=false only for local development."
            )
        if self.environment == "production":
            if not self.twilio_validate_signature:
                raise ValueError("Twilio signature validation cannot be disabled in production.")
            if self.database_url.startswith("sqlite"):
                raise ValueError("SQLite is not a supported production database.")
            if self.auto_create_schema:
                raise ValueError("auto_create_schema must be false in production; use migrations.")
            if self.openai_api_key is None:
                raise ValueError(
                    "The clinical guardrail requires ESCRIBA_OPENAI_API_KEY in production."
                )
            if self.stream_api_token is None:
                raise ValueError("ESCRIBA_STREAM_API_TOKEN is required in production.")
        return self
