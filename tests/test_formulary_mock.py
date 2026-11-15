"""Unit tests for the FormularyService mock and insurance benefit verification logic."""

from __future__ import annotations

import pytest

from app.schemas.orchestrator import FormularyVerificationResult
from app.services.formulary_mock import FormularyService


@pytest.fixture
def service() -> FormularyService:
    return FormularyService()


@pytest.mark.parametrize(
    ("drug_name", "expected_generic", "expected_copay"),
    [
        ("Ozempic 1 mg/dose subcutaneous pen", "Metformin 500mg oral tablet", 950.00),
        ("ozempic", "Metformin 500mg oral tablet", 950.00),
        ("Vyvanse 30mg oral capsule", "Dextroamphetamine-Amphetamine (Adderall generic)", 320.00),
        ("VYVANSE", "Dextroamphetamine-Amphetamine (Adderall generic)", 320.00),
        ("Synthroid 50 mcg tablet", "Levothyroxine sodium oral tablet", 85.00),
        ("synthroid", "Levothyroxine sodium oral tablet", 85.00),
    ],
)
async def test_flagged_brand_drugs_require_prior_auth_and_offer_generic(
    service: FormularyService,
    drug_name: str,
    expected_generic: str,
    expected_copay: float,
) -> None:
    result = await service.check_coverage(drug_name, patient_id="pt-123")

    assert isinstance(result, FormularyVerificationResult)
    assert result.is_covered is False
    assert result.requires_prior_auth is True
    assert result.generic_alternative == expected_generic
    assert result.estimated_copay_usd == expected_copay


@pytest.mark.parametrize(
    "drug_name",
    [
        "Amoxicillin 500mg oral capsule",
        "Lisinopril 10mg tablet",
        "Metformin 500mg tablet",
        "Augmentin 875-125mg BID",
        "Atorvastatin 20mg tablet",
    ],
)
async def test_standard_drugs_default_to_covered(
    service: FormularyService, drug_name: str
) -> None:
    result = await service.check_coverage(drug_name, patient_id="pt-456")

    assert isinstance(result, FormularyVerificationResult)
    assert result.is_covered is True
    assert result.requires_prior_auth is False
    assert result.generic_alternative is None
    assert result.estimated_copay_usd == 15.00
