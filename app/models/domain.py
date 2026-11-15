"""Core relational domain models.

Portability notes:
* ``Uuid`` maps to native UUID on PostgreSQL and CHAR(32) on SQLite.
* ``JSONType`` is JSONB on PostgreSQL (indexable, GIN-capable) and JSON elsewhere.
* Enums are stored as constrained VARCHAR (``native_enum=False``) so adding a status
  is a code change rather than an ``ALTER TYPE`` migration.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from typing import Any, TypedDict, cast

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    String,
    Text,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, validates

from app.core.database import Base
from app.schemas.clinical import normalize_allergy_records, normalize_medication_records

JSONType = JSON().with_variant(JSONB(), "postgresql")
WORKSPACE_PROVIDER_ID = "local-scribe-workspace"


class MedicationRecord(TypedDict):
    """JSON(B) element shape for ``Patient.active_medications``
    (see ``schemas.clinical.Medication``).
    """

    name: str
    dosage: str | None
    frequency: str | None


class AllergyRecord(TypedDict):
    """JSON(B) element shape for ``Patient.allergies`` (see ``schemas.clinical.Allergy``)."""

    allergen: str
    reaction: str | None


def utcnow() -> datetime:
    return datetime.now(UTC)


class EncounterStatus(enum.StrEnum):
    SCHEDULED = "scheduled"
    INTAKE_IN_PROGRESS = "intake_in_progress"
    INTAKE_COMPLETE = "intake_complete"
    IN_PROGRESS = "in_progress"
    DOCUMENTATION_PENDING = "documentation_pending"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class MessageDirection(enum.StrEnum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class DocumentStatus(enum.StrEnum):
    DRAFT = "draft"
    REVIEW_PENDING = "review_pending"
    APPROVED = "approved"


def _str_enum(enum_cls: type[enum.StrEnum], name: str) -> Enum:
    return Enum(
        enum_cls,
        name=name,
        native_enum=False,
        length=32,
        validate_strings=True,
        create_constraint=True,
        values_callable=lambda members: [m.value for m in members],
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        server_default=func.now(),
        onupdate=utcnow,
        nullable=False,
    )


class Patient(TimestampMixin, Base):
    __tablename__ = "patients"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    # E.164 contact number used to route inbound SMS to the patient record.
    phone_number: Mapped[str | None] = mapped_column(String(16), unique=True, index=True)
    demographics: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    active_medications: Mapped[list[MedicationRecord]] = mapped_column(
        JSONType, default=list, nullable=False
    )
    allergies: Mapped[list[AllergyRecord]] = mapped_column(JSONType, default=list, nullable=False)
    chronic_conditions: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONType, default=list, nullable=False
    )

    encounters: Mapped[list[Encounter]] = relationship(
        back_populates="patient",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Encounter.scheduled_time",
    )

    @validates("active_medications")
    def _validate_active_medications(self, _key: str, value: object) -> list[MedicationRecord]:
        # Every write path goes through the canonical schema, so JSONB shape cannot drift.
        # Note: assign a new list; in-place mutation of the JSON list is not tracked.
        return cast("list[MedicationRecord]", normalize_medication_records(value))

    @validates("allergies")
    def _validate_allergies(self, _key: str, value: object) -> list[AllergyRecord]:
        return cast("list[AllergyRecord]", normalize_allergy_records(value))

    def __repr__(self) -> str:  # Avoid PHI in logs/tracebacks.
        return f"<Patient id={self.id}>"


class Encounter(TimestampMixin, Base):
    __tablename__ = "encounters"
    __table_args__ = (
        Index("ix_encounters_patient_status", "patient_id", "status"),
        Index("ix_encounters_provider_schedule", "provider_id", "scheduled_time"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    patient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), index=True, nullable=False
    )
    # External identity-provider subject; no Provider table until the auth phase.
    provider_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    scheduled_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    status: Mapped[EncounterStatus] = mapped_column(
        _str_enum(EncounterStatus, "encounter_status"),
        default=EncounterStatus.SCHEDULED,
        nullable=False,
        index=True,
    )
    audio_file_url: Mapped[str | None] = mapped_column(String(2048))
    transcript: Mapped[str | None] = mapped_column(Text)
    triage_alert: Mapped[bool] = mapped_column(default=False, nullable=False)
    triage_notes: Mapped[str | None] = mapped_column(Text)

    patient: Mapped[Patient] = relationship(back_populates="encounters")
    intake_messages: Mapped[list[IntakeChat]] = relationship(
        back_populates="encounter",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="IntakeChat.timestamp",
    )
    clinical_documents: Mapped[list[ClinicalDocument]] = relationship(
        back_populates="encounter",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ClinicalDocument.created_at",
    )

    def __repr__(self) -> str:
        return f"<Encounter id={self.id} status={self.status}>"


class RecordingTimeline(Base):
    """Audio offsets stored separately so existing visits remain readable."""

    __tablename__ = "recording_timelines"

    encounter_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("encounters.id", ondelete="CASCADE"), primary_key=True
    )
    segments: Mapped[list[dict[str, Any]]] = mapped_column(JSONType, nullable=False)


class IntakeChat(Base):
    __tablename__ = "intake_chats"
    __table_args__ = (Index("ix_intake_chats_encounter_timestamp", "encounter_id", "timestamp"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    encounter_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("encounters.id", ondelete="CASCADE"), index=True, nullable=False
    )
    message_direction: Mapped[MessageDirection] = mapped_column(
        _str_enum(MessageDirection, "message_direction"), nullable=False
    )
    message_body: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now(), nullable=False
    )
    # Twilio MessageSid for inbound messages; unique constraint makes webhook retries idempotent.
    twilio_message_sid: Mapped[str | None] = mapped_column(String(64), unique=True)

    encounter: Mapped[Encounter] = relationship(back_populates="intake_messages")

    def __repr__(self) -> str:
        return f"<IntakeChat id={self.id} direction={self.message_direction}>"


class ClinicalDocument(TimestampMixin, Base):
    __tablename__ = "clinical_documents"
    __table_args__ = (Index("ix_clinical_documents_encounter_status", "encounter_id", "status"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    encounter_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("encounters.id", ondelete="CASCADE"), index=True, nullable=False
    )
    generated_soap: Mapped[dict[str, Any]] = mapped_column(JSONType, default=dict, nullable=False)
    staged_orders: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONType, default=list, nullable=False
    )
    visit_summary_for_patient: Mapped[str | None] = mapped_column(Text)
    status: Mapped[DocumentStatus] = mapped_column(
        _str_enum(DocumentStatus, "document_status"),
        default=DocumentStatus.DRAFT,
        nullable=False,
    )
    # Human-approval audit trail.
    approved_by: Mapped[str | None] = mapped_column(String(64))
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    encounter: Mapped[Encounter] = relationship(back_populates="clinical_documents")

    def __repr__(self) -> str:
        return f"<ClinicalDocument id={self.id} status={self.status}>"


class DraftVerification(Base):
    """Verification trace for the current saved revision of a clinical document."""

    __tablename__ = "draft_verifications"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("clinical_documents.id", ondelete="CASCADE"), primary_key=True
    )
    revision: Mapped[str] = mapped_column(String(40))
    report: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
