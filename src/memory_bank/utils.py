"""
Utilities: Jinja templates, formatters, scope builders, retry helpers.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional

from jinja2 import Template

from memory_bank.models import Memory, MemoryScope, RetrievedMemory


DEFAULT_MEMORY_PROMPT_TEMPLATE = Template(
    """
<memories>
Here is what I remember about you:
{% for mem in memories %}
- {{ mem.fact }}
{% endfor %}
</memories>
""".strip()
)

PROFILE_PROMPT_TEMPLATE = Template(
    """
<user_profile>
{% for key, value in profile.items() %}
{{ key|title }}: {{ value }}
{% endfor %}
</user_profile>
""".strip()
)


def format_memories_for_prompt(
    memories: List[Memory],
    template: Optional[Template] = None,
) -> str:
    """Render memories into a system-prompt injectable string."""
    if not memories:
        return ""
    tmpl = template or DEFAULT_MEMORY_PROMPT_TEMPLATE
    return tmpl.render(memories=memories)


def format_profile_for_prompt(
    profile: Dict[str, Any],
    template: Optional[Template] = None,
) -> str:
    """Render a structured profile for system prompt injection."""
    if not profile:
        return ""
    tmpl = template or PROFILE_PROMPT_TEMPLATE
    return tmpl.render(profile=profile)


def build_scope(
    user_id: Optional[str] = None,
    agent: Optional[str] = None,
    project: Optional[str] = None,
    session_id: Optional[str] = None,
    team: Optional[str] = None,
    **extra: str,
) -> MemoryScope:
    """Convenience builder for MemoryScope."""
    kwargs: Dict[str, Any] = {
        "user_id": user_id,
        "agent": agent,
        "project": project,
        "session_id": session_id,
        "team": team,
    }
    kwargs.update(extra)
    return MemoryScope(**{k: v for k, v in kwargs.items() if v is not None})


def scope_to_filter(scope: MemoryScope) -> str:
    """Convert a MemoryScope to an EBNF filter string for purge operations."""
    parts = [f'scope.{k}="{v}"' for k, v in scope.to_dict().items()]
    return " AND ".join(parts)


def format_retrieved_for_context(
    retrieved: List[RetrievedMemory],
    include_distance: bool = False,
) -> str:
    """Format retrieved memories into a bullet list."""
    lines: List[str] = []
    for r in retrieved:
        line = f"- {r.memory.fact}"
        if include_distance and r.distance is not None:
            line += f" (distance: {r.distance:.3f})"
        lines.append(line)
    return "\n".join(lines)


def memory_to_dict(memory: Memory) -> Dict[str, Any]:
    """Serialize a Memory to a plain dict (for JSON, logging, etc.)."""
    return {
        "name": memory.name,
        "fact": memory.fact,
        "scope": memory.scope.to_dict(),
        "create_time": memory.create_time.isoformat() if memory.create_time else None,
        "update_time": memory.update_time.isoformat() if memory.update_time else None,
        "topics": memory.topics,
        "metadata": memory.metadata.model_dump() if memory.metadata else None,
    }
