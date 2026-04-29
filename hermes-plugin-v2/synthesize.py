"""REAL memory synthesis using Gemini.

v1's ``memory_synthesize`` was fake — just ``" ".join(facts) + "."``. This
module calls the configured ``synthesis_model`` (default gemini-2.5-flash)
to produce an actual narrative summary grounded in retrieved facts.

Falls back to the v1 join behaviour if the SDK isn't available so the tool
never hard-fails.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_PROMPT_TEMPLATE = (
    "You are a careful summarizer. Compose a short factual narrative "
    "(2-5 sentences) that answers the user's question by drawing only on "
    "the provided memories. Do not invent facts. Cite the most relevant "
    "facts inline as plain prose. If the memories don't answer the "
    "question, say so plainly.\n\n"
    "Question: {query}\n\n"
    "Memories (most relevant first):\n{memories}\n\n"
    "Narrative summary:"
)


def synthesize_memories(
    *,
    project: str,
    location: str,
    model: str,
    query: str,
    memories: List[Dict[str, Any]],
    max_chars: int = 2200,
) -> str:
    """Return a narrative answer or the join-fallback if the SDK is missing."""
    if not memories:
        return ""
    facts = [str(m.get("fact") or "").strip() for m in memories if m.get("fact")]
    facts = [f for f in facts if f]
    if not facts:
        return ""

    try:
        return _gemini_synthesize(
            project=project,
            location=location,
            model=model,
            query=query,
            facts=facts,
            max_chars=max_chars,
        )
    except Exception as e:
        logger.info(
            "gcp-memory-bank: real synthesis unavailable (%s) — using join fallback.", e,
        )
        return _join_fallback(facts, max_chars=max_chars)


def _gemini_synthesize(
    *,
    project: str,
    location: str,
    model: str,
    query: str,
    facts: List[str],
    max_chars: int,
) -> str:
    from google import genai  # type: ignore
    from google.genai import types as genai_types  # type: ignore

    client = genai.Client(vertexai=True, project=project, location=location)
    facts_block = "\n".join(f"- {f}" for f in facts[:30])
    prompt = _PROMPT_TEMPLATE.format(query=query.strip(), memories=facts_block)
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=800,
            response_mime_type="text/plain",
        ),
    )
    text = (getattr(resp, "text", None) or "").strip()
    if not text:
        # Some genai responses pack text inside candidates[0].content.parts.
        try:
            cand = resp.candidates[0]
            text = cand.content.parts[0].text or ""
        except Exception:
            text = ""
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…"
    return text


def _join_fallback(facts: List[str], *, max_chars: int) -> str:
    out = " ".join(f if f.endswith((".", "!", "?")) else f + "." for f in facts)
    if len(out) > max_chars:
        out = out[:max_chars].rstrip() + "…"
    return out
