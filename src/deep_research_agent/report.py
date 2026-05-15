"""Report rendering and persistence helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import Evidence, ResearchReport


def slugify(text: str, *, default: str = "research-report") -> str:
    """Create a filesystem-safe slug."""

    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug[:80] or default


def render_markdown_report(report: ResearchReport, *, slug: str | None = None) -> str:
    """Render a citation-preserving Markdown report."""

    report_slug = slug or slugify(report.question)
    lines = [
        "# Deep Research Report",
        "",
        f"Slug: `{report_slug}`",
        f"Generated: `{report.generated_at.isoformat()}`",
        "",
        "## Question",
        "",
        report.question,
        "",
        "## Answer",
        "",
        report.answer,
        "",
        "## Evidence",
        "",
    ]
    if report.evidence:
        for idx, item in enumerate(report.evidence, start=1):
            lines.extend(_render_evidence_item(idx, item))
    else:
        lines.append("No evidence collected.")
    lines.extend(["", "## Citations", ""])
    if report.citations:
        for citation in report.citations:
            lines.append(f"- {citation}")
    else:
        lines.append("No citations.")
    return "\n".join(lines).rstrip() + "\n"


def report_to_json_dict(report: ResearchReport, *, slug: str | None = None) -> dict[str, Any]:
    """Convert a report to a JSON-serializable object."""

    return {
        "slug": slug or slugify(report.question),
        "question": report.question,
        "answer": report.answer,
        "citations": list(report.citations),
        "generated_at": report.generated_at.isoformat(),
        "evidence": [
            {
                "step_id": item.step_id,
                "title": item.title,
                "url": item.url,
                "quote": item.quote,
                "confidence": item.confidence,
            }
            for item in report.evidence
        ],
    }


def write_report_files(report: ResearchReport, output_dir: str | Path, *, slug: str | None = None) -> tuple[Path, Path]:
    """Write Markdown and JSON report files and return their paths."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_slug = slug or slugify(report.question)
    markdown_path = out / f"{report_slug}.md"
    json_path = out / f"{report_slug}.json"
    markdown_path.write_text(render_markdown_report(report, slug=report_slug))
    json_path.write_text(json.dumps(report_to_json_dict(report, slug=report_slug), indent=2, sort_keys=True) + "\n")
    return markdown_path, json_path


def _render_evidence_item(idx: int, item: Evidence) -> list[str]:
    return [
        f"{idx}. **{item.title}**",
        f"   - Step: `{item.step_id}`",
        f"   - Quote: {item.quote} [Source: {item.url}]",
        f"   - Confidence: `{item.confidence:.2f}`",
        "",
    ]
