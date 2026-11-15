"""Pydantic v2 schemas for formulary verification and 1-click order orchestration."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.domain import DocumentStatus, EncounterStatus
from app.schemas.clinical import StagedOrder


class FormularyVerificationResult(BaseModel):
    """External formulary coverage and prior authorization check result."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    is_covered: bool = Field(
        description="Whether the drug is covered under the patient's active pharmacy benefit plan."
    )
    requires_prior_auth: bool = Field(
        description="Whether the prescription requires prior authorization before dispensing."
    )
    generic_alternative: str | None = Field(
        default=None,
        description="Recommended bioequivalent or therapeutic generic alternative, if non-covered.",
    )
    estimated_copay_usd: float | None = Field(
        default=None,
        ge=0.0,
        description="Estimated out-of-pocket patient copay in USD under benefit tier.",
    )


class EnrichedOrder(StagedOrder):
    """Staged order enriched with real-time formulary verification for clinician review."""

    model_config = ConfigDict(extra="ignore", frozen=True, str_strip_whitespace=True)

    order_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for order tracking and 1-click approval.",
    )
    formulary_verification: FormularyVerificationResult | None = Field(
        default=None,
        description="Formulary coverage and pricing verification details for prescriptions.",
    )


class OrderApprovalRequest(BaseModel):
    """Payload submitted by clinician during 1-click review and sign-off."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    approved_order_ids: list[str] = Field(
        default_factory=list,
        description="Specific order IDs approved. If empty, all staged orders are approved.",
    )
    clinician_id: str = Field(
        min_length=1,
        max_length=64,
        description="Authenticated clinician ID authorizing the orders and clinical document.",
    )
    approval_notes: str | None = Field(
        default=None,
        max_length=1000,
        description="Optional clinical review notes or sign-off remarks.",
    )


class OrderApprovalResponse(BaseModel):
    """Response returned upon successful 1-click document and order approval."""

    model_config = ConfigDict(extra="ignore")

    document_id: uuid.UUID
    encounter_id: uuid.UUID
    status: DocumentStatus
    approved_by: str
    approved_at: datetime
    approved_orders_count: int
    encounter_status: EncounterStatus
