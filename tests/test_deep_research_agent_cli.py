"""CLI/report-writer tests for Deep Research Agent v0."""

from __future__ import annotations

import json
from pathlib import Path

from deep_research_agent.cli import main
from deep_research_agent.file_adapters import FileMemoryStore, JsonFileSearchProvider
from deep_research_agent.report import render_markdown_report


def test_json_file_search_provider_exact_and_default_queries(tmp_path: Path) -> None:
    fixture = tmp_path / "search.json"
    fixture.write_text(
        json.dumps(
            {
                "queries": {
                    "specific query": [
                        {
                            "title": "Specific",
                            "url": "https://example.com/specific",
                            "snippet": "Specific evidence.",
                        }
                    ],
                    "*": [
                        {
                            "title": "Fallback",
                            "url": "https://example.com/fallback",
                            "snippet": "Fallback evidence.",
                        }
                    ],
                }
            }
        )
    )

    provider = JsonFileSearchProvider(fixture)

    assert provider.search("specific query")[0].url == "https://example.com/specific"
    assert provider.search("unknown query")[0].url == "https://example.com/fallback"


def test_file_memory_store_recalls_and_persists_reports(tmp_path: Path) -> None:
    memory_path = tmp_path / "memory.jsonl"
    memory_path.write_text(json.dumps({"fact": "Deep Research Agent needs citations."}) + "\n")
    store = FileMemoryStore(memory_path)

    recalled = store.recall("research citations")
    assert recalled == ["Deep Research Agent needs citations."]

    fixture = tmp_path / "search.json"
    fixture.write_text(
        json.dumps(
            {
                "queries": {
                    "*": [
                        {
                            "title": "Evidence",
                            "url": "https://example.com/evidence",
                            "snippet": "Evidence quote.",
                        }
                    ]
                }
            }
        )
    )

    exit_code = main(
        [
            "--question",
            "Deep Research Agent citation persistence",
            "--search-fixture",
            str(fixture),
            "--memory-file",
            str(memory_path),
            "--output-dir",
            str(tmp_path / "reports"),
            "--slug",
            "citation-test",
        ]
    )

    assert exit_code == 0
    persisted_lines = memory_path.read_text().splitlines()
    assert len(persisted_lines) == 2
    assert "citation-test" in (tmp_path / "reports" / "citation-test.md").read_text()
    assert json.loads((tmp_path / "reports" / "citation-test.json").read_text())["question"]


def test_render_markdown_report_contains_inline_sources(tmp_path: Path) -> None:
    fixture = tmp_path / "search.json"
    fixture.write_text(
        json.dumps(
            {
                "queries": {
                    "*": [
                        {
                            "title": "Source One",
                            "url": "https://example.com/one",
                            "snippet": "Quote one.",
                        }
                    ]
                }
            }
        )
    )
    output_dir = tmp_path / "reports"

    assert main(
        [
            "--question",
            "How should reports cite sources?",
            "--search-fixture",
            str(fixture),
            "--output-dir",
            str(output_dir),
            "--slug",
            "sources",
        ]
    ) == 0

    markdown = (output_dir / "sources.md").read_text()
    assert "# Deep Research Report" in markdown
    assert "[Source: https://example.com/one]" in markdown
    assert "## Evidence" in markdown

    data = json.loads((output_dir / "sources.json").read_text())
    assert data["citations"] == ["https://example.com/one"]


def test_render_markdown_report_escapes_empty_evidence() -> None:
    from deep_research_agent.models import ResearchReport

    report = ResearchReport(question="No evidence", answer="I couldn't find enough evidence to answer this.", evidence=())
    markdown = render_markdown_report(report, slug="no-evidence")

    assert "I couldn't find enough evidence" in markdown
    assert "## Evidence" in markdown
    assert "No evidence collected." in markdown
