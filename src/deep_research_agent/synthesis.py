"""Built-in synthesis strategies for Deep Research Agent v0."""

from __future__ import annotations

from collections.abc import Callable

from .models import Evidence, ResearchPlan

SynthesizeFn = Callable[[str], str]


class ExtractiveSynthesizer:
    """Deterministic citation-preserving fallback synthesizer.

    Provider-backed synthesizers can later replace this with Gemini/Vertex or
    Gemini Deep Research output, but this one is intentionally pure Python so
    CI can prove the orchestration path without network calls.
    """

    def synthesize(
        self,
        *,
        question: str,
        plan: ResearchPlan,
        evidence: list[Evidence],
        memories: list[str],
    ) -> str:
        if not evidence:
            return "I couldn't find enough evidence to answer this."

        lines = [f"Question: {question}", "", "Answer:"]
        if memories:
            lines.append(f"Relevant memory: {memories[0]}")

        for item in evidence:
            quote = item.quote.strip().rstrip(".")
            lines.append(f"- {quote}. {item.citation()}")

        lines.append("")
        lines.append("Research steps covered: " + ", ".join(step.id for step in plan.steps))
        return "\n".join(lines)


class CitationPreservingSynthesizer:
    """Provider-backed synthesizer wrapper that preserves evidence citations.

    The wrapped callable receives a complete prompt and returns plain text.  This
    adapter does not trust the provider to keep every citation, so it appends a
    compact missing-source line when needed.
    """

    def __init__(self, *, synthesize_fn: SynthesizeFn) -> None:
        self._synthesize_fn = synthesize_fn

    def synthesize(
        self,
        *,
        question: str,
        plan: ResearchPlan,
        evidence: list[Evidence],
        memories: list[str],
    ) -> str:
        prompt = build_citation_prompt(question=question, plan=plan, evidence=evidence, memories=memories)
        answer = self._synthesize_fn(prompt).strip()
        missing = [item.url for item in evidence if item.url and f"[Source: {item.url}]" not in answer]
        if missing:
            answer = answer.rstrip() + "\n\nAdditional sources checked: " + " ".join(
                f"[Source: {url}]" for url in dict.fromkeys(missing)
            )
        return answer


def build_citation_prompt(
    *,
    question: str,
    plan: ResearchPlan,
    evidence: list[Evidence],
    memories: list[str],
) -> str:
    """Build a strict citation-preservation prompt for external synthesizers."""

    lines = [
        "Answer the research question using only the evidence below.",
        "Every factual sentence must include an inline [Source: URL] citation.",
        f"Question: {question}",
        "",
        "Research steps: " + ", ".join(step.id for step in plan.steps),
    ]
    if memories:
        lines.extend(["", "Relevant durable memories:"])
        lines.extend(f"- {memory}" for memory in memories)
    lines.extend(["", "Evidence:"])
    for item in evidence:
        lines.append(f"- {item.title}: {item.quote} [Source: {item.url}]")
    return "\n".join(lines)
