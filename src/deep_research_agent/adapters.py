"""Optional adapters for wiring Deep Research Agent to host runtimes."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from .models import ResearchReport, SearchHit

RecallFn = Callable[[str, int], list[str]]
StoreFn = Callable[[ResearchReport], None]
SearchFn = Callable[[str, int], Any]
ProfileFn = Callable[[], Any]


class CallableSearchProvider:
    """SearchProvider adapter backed by a host-provided callable.

    The callable can return common search shapes from Hermes tools, MiniMax web
    search, lightweight HTTP clients, or test fakes.  Normalization happens here
    so the core agent only handles `SearchHit` objects.
    """

    def __init__(self, *, search_fn: SearchFn) -> None:
        self._search_fn = search_fn

    def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        raw = self._search_fn(query, limit)
        return normalize_search_result(raw)[:limit]


class CallableMemoryStore:
    """MemoryStore adapter backed by host-provided callables.

    This keeps the core package independent from Hermes tool injection.  A
    Hermes runtime can pass thin wrappers around `memory_search` and
    `memory_store`; tests can pass simple Python functions.
    """

    def __init__(self, *, recall_fn: RecallFn, store_fn: StoreFn | None = None) -> None:
        self._recall_fn = recall_fn
        self._store_fn = store_fn

    def recall(self, query: str, *, top_k: int = 5) -> list[str]:
        return list(self._recall_fn(query, top_k))

    def store_report(self, report: ResearchReport) -> None:
        if self._store_fn is not None:
            self._store_fn(report)


def normalize_search_result(result: Any) -> list[SearchHit]:
    """Normalize common live-search result shapes into SearchHit objects."""

    if result is None:
        return []
    if isinstance(result, SearchHit):
        return [result]
    if isinstance(result, str):
        stripped = result.strip()
        return [SearchHit(title=stripped, url=stripped, snippet="")] if stripped else []
    if isinstance(result, dict):
        for key in ("organic", "results", "items", "hits"):
            if key in result:
                return normalize_search_result(result[key])
        hit = _dict_to_search_hit(result)
        return [hit] if hit is not None else []
    if isinstance(result, Iterable):
        hits: list[SearchHit] = []
        for item in result:
            hits.extend(normalize_search_result(item))
        return hits
    return []


def _dict_to_search_hit(item: dict[str, Any]) -> SearchHit | None:
    url = item.get("url") or item.get("link") or item.get("href")
    if not url:
        return None
    title = item.get("title") or item.get("name") or url
    snippet = item.get("snippet") or item.get("content") or item.get("text") or item.get("description") or ""
    published_at = item.get("published_at") or item.get("date") or item.get("published")
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return SearchHit(
        title=str(title),
        url=str(url),
        snippet=str(snippet),
        published_at=str(published_at) if published_at is not None else None,
        metadata=dict(metadata),
    )


def report_to_memory_fact(report: ResearchReport, *, max_chars: int = 2000) -> str:
    """Convert a report into one durable memory fact string."""

    sources = ", ".join(report.citations)
    fact = f"Deep Research report for '{report.question}': {report.answer}"
    if sources:
        fact += f" Sources: {sources}"
    return fact[:max_chars]


def normalize_profile_result(result: Any) -> list[str]:
    """Normalize GCP Memory Bank structured profile results into memory facts.

    Expected live shape from the Hermes `memory_profiles` tool:
    `{ "profiles": { "hermes-profile": { "profile": {...} } } }`.
    The formatter is deliberately generic so tests and alternate runtimes can
    pass a bare profile dict, a list of profile records, or nested values.
    """

    profile = _extract_profile_payload(result)
    if not isinstance(profile, dict):
        return []
    return [f"Profile {key}: {value}" for key, value in _flatten_profile(profile)]


def _extract_profile_payload(result: Any) -> Any:
    if result is None:
        return None
    if isinstance(result, dict):
        if "profiles" in result:
            profiles = result.get("profiles")
            if isinstance(profiles, dict):
                for candidate in profiles.values():
                    payload = _extract_profile_payload(candidate)
                    if payload:
                        return payload
            return None
        if "profile" in result:
            return result.get("profile")
        return result
    if isinstance(result, list):
        merged: dict[str, Any] = {}
        for item in result:
            payload = _extract_profile_payload(item)
            if isinstance(payload, dict):
                merged.update(payload)
        return merged
    return None


def _flatten_profile(profile: dict[str, Any], *, prefix: str = "") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key, value in profile.items():
        if value in (None, "", [], {}):
            continue
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.extend(_flatten_profile(value, prefix=path))
        elif isinstance(value, (list, tuple, set)):
            rows.append((path, "; ".join(str(item) for item in value if item not in (None, ""))))
        else:
            rows.append((path, str(value)))
    return rows

def normalize_memory_search_result(result: Any) -> list[str]:
    """Normalize common Hermes/GCP Memory Bank search result shapes."""

    if result is None:
        return []
    if isinstance(result, str):
        return [line.strip("- ") for line in result.splitlines() if line.strip()]
    if isinstance(result, list):
        normalized: list[str] = []
        for item in result:
            if isinstance(item, str):
                normalized.append(item)
            elif isinstance(item, dict):
                fact = item.get("fact") or item.get("text") or item.get("content")
                if fact:
                    normalized.append(str(fact))
        return normalized
    if isinstance(result, dict):
        for key in ("memories", "results", "facts"):
            if key in result:
                return normalize_memory_search_result(result[key])
        fact = result.get("fact") or result.get("text") or result.get("content")
        return [str(fact)] if fact else []
    return [str(result)]
