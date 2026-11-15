"""Pydantic v2 schemas for the pre-visit intake flow."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from app.schemas.clinical import Allergy, Medication, dedupe_allergies, dedupe_medications

E164PhoneNumber = Annotated[str, StringConstraints(pattern=r"^\+[1-9]\d{1,14}$")]
TwilioMessageSid = Annotated[str, StringConstraints(pattern=r"^(SM|MM)[0-9a-fA-F]{32}$")]
TwilioAccountSid = Annotated[str, StringConstraints(pattern=r"^AC[0-9a-fA-F]{32}$")]


class TwilioInboundSMS(BaseModel):
    """Twilio Messaging webhook payload (application/x-www-form-urlencoded).

    Field names follow Twilio's PascalCase via aliases. Unknown fields are ignored
    because Twilio adds parameters over time and must not break ingestion.
    """

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )

    message_sid: TwilioMessageSid = Field(alias="MessageSid")
    account_sid: TwilioAccountSid = Field(alias="AccountSid")
    messaging_service_sid: str | None = Field(default=None, alias="MessagingServiceSid")
    from_number: E164PhoneNumber = Field(alias="From")
    to_number: E164PhoneNumber = Field(alias="To")
    body: str = Field(default="", alias="Body", max_length=1600)
    num_media: int = Field(default=0, alias="NumMedia", ge=0, le=10)
    num_segments: int = Field(default=1, alias="NumSegments", ge=1)
    sms_status: str | None = Field(default=None, alias="SmsStatus")
    api_version: str | None = Field(default=None, alias="ApiVersion")
    from_city: str | None = Field(default=None, alias="FromCity")
    from_state: str | None = Field(default=None, alias="FromState")
    from_zip: str | None = Field(default=None, alias="FromZip")
    from_country: str | None = Field(default=None, alias="FromCountry")


class IntakeExtractionSchema(BaseModel):
    """Structured output target for the intake LLM extraction step.

    ``extra="forbid"`` (here and on ``Medication``/``Allergy``) makes the generated JSON
    schema closed, so hallucinated keys fail validation instead of being silently dropped.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    chief_complaint: str = Field(
        min_length=1,
        max_length=500,
        description="The patient's primary reason for the visit, in their own words.",
    )
    symptom_duration: str = Field(
        min_length=1,
        max_length=200,
        description="How long the chief complaint has been present, e.g. '3 days'. "
        "Use 'unknown' if the patient did not say.",
    )
    reported_medications: list[Medication] = Field(
        default_factory=list,
        max_length=50,
        description="Medications the patient reports currently taking. Empty list if none.",
    )
    reported_allergies: list[Allergy] = Field(
        default_factory=list,
        max_length=50,
        description="Allergies the patient reports. Empty list if none reported or the "
        "patient reports no known allergies.",
    )

    @field_validator("reported_medications")
    @classmethod
    def _dedupe_medications(cls, values: list[Medication]) -> list[Medication]:
        return dedupe_medications(values)

    @field_validator("reported_allergies")
    @classmethod
    def _dedupe_allergies(cls, values: list[Allergy]) -> list[Allergy]:
        return dedupe_allergies(values)
