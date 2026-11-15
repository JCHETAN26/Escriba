"""Mock external integration service simulating real-time e-prescribing formulary checks."""

from __future__ import annotations

import logging
from typing import Any

from app.schemas.orchestrator import FormularyVerificationResult

logger = logging.getLogger(__name__)

# Brand-name drugs flagged for non-coverage / prior authorization in hackathon demo
REJECTED_BRAND_DRUGS: dict[str, dict[str, Any]] = {
    "ozempic": {
        "generic_alternative": "Metformin 500mg oral tablet",
        "estimated_copay_usd": 950.00,
    },
    "vyvanse": {
        "generic_alternative": "Dextroamphetamine-Amphetamine (Adderall generic)",
        "estimated_copay_usd": 320.00,
    },
    "synthroid": {
        "generic_alternative": "Levothyroxine sodium oral tablet",
        "estimated_copay_usd": 85.00,
    },
}

DEFAULT_COVERED_COPAY_USD = 15.00


class FormularyService:
    """Simulates real-time e-prescribing network (e.g. Surescripts, Change Healthcare) checks.

    Structured so it can be swapped for an async HTTP client (e.g. httpx.AsyncClient)
    in production environments.
    """

    def __init__(self, http_client: Any | None = None) -> None:
        self._http_client = http_client

    async def check_coverage(
        self, medication_name: str, patient_id: str
    ) -> FormularyVerificationResult:
        """Evaluate pharmacy benefit coverage, prior auth requirements, and copay tier.

        Hackathon Demo:
        Flags known high-cost brand drugs ("Ozempic", "Vyvanse", "Synthroid") as non-covered
        with prior auth required and suggests cheaper bioequivalent generic alternatives.
        All other medications default to covered tier-1/tier-2 status.
        """
        med_normalized = medication_name.lower().strip()

        for brand_key, config in REJECTED_BRAND_DRUGS.items():
            if brand_key in med_normalized:
                logger.info(
                    "Formulary check: intercepted non-covered brand '%s' for patient=%s",
                    medication_name,
                    patient_id,
                )
                return FormularyVerificationResult(
                    is_covered=False,
                    requires_prior_auth=True,
                    generic_alternative=config["generic_alternative"],
                    estimated_copay_usd=config["estimated_copay_usd"],
                )

        logger.debug(
            "Formulary check: '%s' is covered for patient=%s", medication_name, patient_id
        )
        return FormularyVerificationResult(
            is_covered=True,
            requires_prior_auth=False,
            generic_alternative=None,
            estimated_copay_usd=DEFAULT_COVERED_COPAY_USD,
        )
