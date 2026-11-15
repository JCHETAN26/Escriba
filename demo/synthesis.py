"""Run synthetic encounters through the Escriba synthesis endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.main import create_app
from app.models.domain import ClinicalDocument, Encounter, EncounterStatus, Patient, utcnow
from app.routers.encounter import SynthesizeResponse
from tests.fakes import FakeOpenAI

CASES = Path(__file__).with_name("cases.json")
FLAGS = re.IGNORECASE
SOAP_SECTIONS = ("subjective", "objective", "assessment", "plan")
CaseResult = tuple[dict[str, Any], SynthesizeResponse | None, list[str]]


def normalized_quote(text: str) -> str:
    punctuation = " \"'\u201c\u201d\u2018\u2019.!?"
    quote = " ".join(text.split()).strip(punctuation)
    return re.sub(r"^clinician\s*:\s*", "", quote, flags=FLAGS).strip(punctuation)


def check_output(case: dict[str, Any], result: SynthesizeResponse) -> list[str]:
    """Check targeted facts and order grounding. Empty findings mean checks passed."""
    sections = result.soap_note.model_dump()
    sections["summary"] = result.visit_summary_for_patient
    sections["order description"] = "\n".join(order.description for order in result.staged_orders)
    sections["order rationale"] = "\n".join(order.icd10_rationale for order in result.staged_orders)
    sections["all"] = (
        "\n".join(sections.values())
        + "\n"
        + "\n".join(order.model_dump_json() for order in result.staged_orders)
    )

    findings = []
    for section, patterns in case["required"].items():
        for pattern in patterns:
            if not re.search(pattern, sections[section], FLAGS):
                findings.append(f"Missing expected wording in {section}: `{pattern}`")
    for section, patterns in case.get("forbidden", {}).items():
        for pattern in patterns:
            if re.search(pattern, sections[section], FLAGS):
                findings.append(f"Unexpected wording in {section}: `{pattern}`")

    remaining = list(result.staged_orders)
    clinician_lines = [
        normalized_quote(line) for line in case["transcript"] if line.startswith("Clinician: ")
    ]
    for expected in case["orders"]:
        supporting_text = normalized_quote(expected["support"])
        for order in remaining:
            if order.order_type != expected["type"] or not all(
                re.search(pattern, order.description, FLAGS) for pattern in expected["description"]
            ):
                continue
            quote = normalized_quote(order.transcript_timestamp_reference)
            if quote and any(
                quote in line and supporting_text in quote for line in clinician_lines
            ):
                remaining.remove(order)
                for pattern in expected.get("details", []):
                    if not re.search(pattern, order.description, FLAGS):
                        findings.append(
                            f"Missing required detail in {order.order_type} description: "
                            f"`{pattern}`"
                        )
                break
        else:
            findings.append(
                f"Missing matching {expected['type']} with a verbatim clinician quote "
                f"supporting this action: {expected['support']}"
            )
    for order in remaining:
        findings.append(f"Unexpected or ungrounded {order.order_type}: {order.description}")
    return findings


async def run_cases(
    settings: Settings, cases: list[dict[str, Any]], *, live: bool
) -> list[CaseResult]:
    app = create_app(settings)
    results: list[CaseResult] = []
    async with (
        app.router.lifespan_context(app),
        AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://demo",
        ) as client,
    ):
        fake = None if live else FakeOpenAI()
        if fake is not None:
            app.state.openai_client = fake

        for case in cases:
            transcript = "\n".join(case["transcript"])
            async with app.state.sessionmaker() as session:
                encounter = Encounter(
                    patient=Patient(),
                    provider_id="demo-provider",
                    scheduled_time=utcnow(),
                    status=EncounterStatus.IN_PROGRESS,
                )
                session.add(encounter)
                await session.commit()
                encounter_id = encounter.id
            if fake is not None:
                fake.completions.responses.append(case["fixture"])

            response = await client.post(
                f"/encounter/{encounter_id}/synthesize", json={"transcript": transcript}
            )
            if response.status_code != 200:
                results.append((case, None, [f"API returned HTTP {response.status_code}."]))
                continue

            result = SynthesizeResponse.model_validate(response.json())
            findings = check_output(case, result)
            async with app.state.sessionmaker() as session:
                doc = await session.get(ClinicalDocument, result.document_id)
                if (
                    doc is None
                    or doc.encounter_id != encounter_id
                    or doc.status != "review_pending"
                    or doc.generated_soap != result.soap_note.model_dump(mode="json")
                    or doc.staged_orders
                    != [order.model_dump(mode="json") for order in result.staged_orders]
                    or doc.visit_summary_for_patient != result.visit_summary_for_patient
                ):
                    findings.append("The database did not preserve the returned review draft.")
            results.append((case, result, findings))
    return results


def render_report(
    results: list[CaseResult],
    *,
    model: str | None,
) -> str:
    mode = f"**Live OpenAI model** `{model}`" if model else "**Offline mode** with scripted replies"
    passed = sum(1 for _, result, findings in results if result is not None and not findings)
    lines = [
        "# Escriba synthesis demo",
        "",
        mode,
        "",
        f"**{passed} of {len(results)} cases passed the automated checks.**",
        "",
        "The runner sends synthetic transcripts through the synthesis API and compares the saved "
        "draft with the response. Offline mode uses hand-authored reference outputs to exercise "
        "the API, database, and reporting. Live mode scores model-generated output.",
        "",
        "For clinical review, compare each output with the transcript for omissions, added facts, "
        "negation, uncertainty, order rationale, and patient-friendly wording. The automated "
        "checks use selected phrases and verbatim order quotes. Acceptable paraphrases can fail, "
        "and errors outside the checked facts can pass.",
    ]
    for case, result, findings in results:
        outcome = "ERROR" if result is None else "REVIEW" if findings else "PASS"
        lines.extend(["", f"## {case['title']}", "", f"**{outcome}**", ""])
        lines.extend(f"- {finding}" for finding in findings)
        if findings:
            lines.append("")
        lines.extend(["### Synthetic transcript", ""])
        lines.append("\n>\n".join(f"> {line}" for line in case["transcript"]))
        if result is None:
            continue

        lines.extend(["", "### SOAP note", ""])
        for section in SOAP_SECTIONS:
            lines.extend(
                [f"**{section.capitalize()}**", "", getattr(result.soap_note, section), ""]
            )
        lines.extend(["### Patient summary", "", result.visit_summary_for_patient, ""])
        lines.extend(["### Proposed orders", ""])
        if not result.staged_orders:
            lines.append("None.")
        for order in result.staged_orders:
            lines.append(
                f"- **{order.order_type.capitalize()}** — {order.description} "
                f"({order.icd10_rationale})"
            )
            lines.append(f"  - Transcript evidence: “{order.transcript_timestamp_reference}”")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Use the configured OpenAI model.")
    parser.add_argument("--output", type=Path, help="Path for the Markdown report.")
    args = parser.parse_args()

    cases = json.loads(CASES.read_text(encoding="utf-8"))
    with TemporaryDirectory(prefix="escriba-demo-") as tmp:
        overrides: dict[str, Any] = {
            "environment": "development",
            "demo_mode": True,
            "database_url": f"sqlite+aiosqlite:///{Path(tmp) / 'demo.db'}",
            "auto_create_schema": True,
            "twilio_validate_signature": False,
        }
        if args.live:
            settings = Settings(**overrides)
            if settings.openai_api_key is None:
                parser.error("--live requires ESCRIBA_OPENAI_API_KEY in the environment or .env")
        else:
            settings = Settings(_env_file=None, openai_api_key=None, **overrides)
        results = asyncio.run(run_cases(settings, cases, live=args.live))

    output = args.output or Path("demo/live-report.md" if args.live else "demo/demo-report.md")
    output.write_text(
        render_report(results, model=settings.synthesis_model if args.live else None),
        encoding="utf-8",
    )
    output = output.resolve()
    passed = sum(1 for _, result, findings in results if result is not None and not findings)
    print(f"{passed}/{len(results)} cases passed. Report saved to {output}")
    return int(passed != len(results))


if __name__ == "__main__":
    raise SystemExit(main())
