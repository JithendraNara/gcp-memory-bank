"""File-backed adapters for Deep Research Agent v0."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .adapters import report_to_memory_fact
from .models import ResearchReport, SearchHit


class JsonFileSearchProvider:
    """SearchProvider backed by a JSON fixture file.

    Supported fixture shape:

    ```json
    {
      "queries": {
        "exact query": [{"title": "...", "url": "...", "snippet": "..."}],
        "*": [{"title": "fallback", "url": "..."}]
      }
    }
    ```
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._data = self._load()

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        queries = self._data.get("queries", {})
        raw_hits = queries.get(query) or queries.get("*") or []
        return [self._to_hit(item) for item in raw_hits[:limit]]

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(f"search fixture not found: {self.path}")
        loaded = json.loads(self.path.read_text())
        if not isinstance(loaded, dict):
            raise ValueError("search fixture root must be an object")
        return loaded

    @staticmethod
    def _to_hit(item: dict[str, Any]) -> SearchHit:
        if not isinstance(item, dict):
            raise ValueError("search fixture hit must be an object")
        return SearchHit(
            title=str(item.get("title") or item.get("url") or "Untitled"),
            url=str(item.get("url") or ""),
            snippet=str(item.get("snippet") or ""),
            published_at=item.get("published_at"),
            metadata=dict(item.get("metadata") or {}),
        )


class FileMemoryStore:
    """JSONL-backed MemoryStore for local reports and deterministic demos."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def recall(self, query: str, *, top_k: int = 5) -> list[str]:
        if not self.path.exists():
            return []
        terms = {word.lower().strip(".,:;!?()[]{}") for word in query.split() if len(word) > 2}
        scored: list[tuple[int, str]] = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            fact = self._line_to_fact(line)
            score = sum(1 for term in terms if term in fact.lower())
            if score:
                scored.append((score, fact))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [fact for _, fact in scored[:top_k]]

    def store_report(self, report: ResearchReport) -> None:
        record = {
            "type": "deep_research_report",
            "question": report.question,
            "fact": report_to_memory_fact(report),
            "citations": list(report.citations),
            "generated_at": report.generated_at.isoformat(),
        }
        with self.path.open("a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    @staticmethod
    def _line_to_fact(line: str) -> str:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            return line.strip()
        if isinstance(parsed, dict):
            return str(parsed.get("fact") or parsed.get("answer") or parsed.get("question") or "")
        return str(parsed)
