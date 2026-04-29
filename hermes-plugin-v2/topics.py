"""Memory topic catalog with REAL few-shot examples for the unified plugin.

Pulled from v1's working ``_build_engine_config`` (Fort Wayne, Gemini, etc.)
plus three Hermes-flavored custom topics borrowed from v1's
HERMES_MEMORY_CONFIG (technical_decisions, project_context, corrected_mistakes).

The schema shape ``{"managed_memory_topic": {"managed_topic_enum": ...}}`` is
what the live SDK actually accepts — verified against engine
``4938048007586185216``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

MANAGED_TOPICS = [
    "USER_PERSONAL_INFO",
    "USER_PREFERENCES",
    "KEY_CONVERSATION_DETAILS",
    "EXPLICIT_INSTRUCTIONS",
]

DEFAULT_CUSTOM_TOPICS: List[Dict[str, Any]] = [
    {
        "label": "TECHNICAL_DECISIONS",
        "description": (
            "Technology choices the user has committed to (frameworks, "
            "databases, languages, models). Capture decision + short rationale."
        ),
    },
    {
        "label": "PROJECT_CONTEXT",
        "description": (
            "Active project state — what the user is currently building, "
            "current blockers, and the next concrete step."
        ),
    },
    {
        "label": "CORRECTED_MISTAKES",
        "description": (
            "Mistakes the agent made that the user explicitly corrected. "
            "Persist the wrong assumption + the right answer so future "
            "turns don't repeat the error."
        ),
    },
]


# Few-shot examples in the EXACT shape the live SDK accepts.
# Source: v1 hermes-plugin/__init__.py lines 624-707, verified working.
DEFAULT_FEW_SHOT_EXAMPLES: List[Dict[str, Any]] = [
    {
        "conversationSource": {
            "events": [
                {"content": {"role": "model",
                             "parts": [{"text": "Hey! What can I help you with today?"}]}},
                {"content": {"role": "user",
                             "parts": [{"text": "I just moved to Fort Wayne, Indiana. Still getting used to the area."}]}},
            ]
        },
        "generatedMemories": [{"fact": "The user lives in Fort Wayne, Indiana."}],
    },
    {
        "conversationSource": {
            "events": [
                {"content": {"role": "model",
                             "parts": [{"text": "Want me to use the latest Gemini model for this task?"}]}},
                {"content": {"role": "user",
                             "parts": [{"text": "Yes, always use the newest available model. I don't care about cost."}]}},
            ]
        },
        "generatedMemories": [
            {"fact": "The user prefers using the newest available AI models regardless of cost."},
        ],
    },
    {
        "conversationSource": {
            "events": [
                {"content": {"role": "model",
                             "parts": [{"text": "Got it. Any other preferences I should know about?"}]}},
                {"content": {"role": "user",
                             "parts": [{"text": "Remember to always follow official docs first. Don't make up your own rules."}]}},
            ]
        },
        "generatedMemories": [
            {"fact": "The user explicitly instructed the agent to always follow official documentation first and not invent its own rules."},
        ],
    },
    {
        # Negative example — model should NOT generate a memory from a thank-you.
        "conversationSource": {
            "events": [
                {"content": {"role": "model",
                             "parts": [{"text": "Here's the weather forecast for Fort Wayne: sunny, 72°F."}]}},
                {"content": {"role": "user",
                             "parts": [{"text": "Thanks, that's helpful."}]}},
            ]
        },
        "generatedMemories": [],
    },
    {
        # TECHNICAL_DECISIONS positive example.
        "conversationSource": {
            "events": [
                {"content": {"role": "user",
                             "parts": [{"text": "Let's go with Postgres for the new service — pgvector for the embeddings."}]}},
                {"content": {"role": "model",
                             "parts": [{"text": "Got it, Postgres + pgvector."}]}},
            ]
        },
        "generatedMemories": [
            {"fact": "Decided to use Postgres with pgvector for the new service's embedding storage."},
        ],
    },
]


def build_memory_bank_config(
    *,
    project_id: str,
    generation_model: str = "gemini-3.1-pro-preview",
    embedding_model: str = "gemini-embedding-001",
    create_ttl_days: int = 365,
    generate_created_ttl_days: int = 365,
    revision_ttl_days: int = 365,
    custom_topics: Optional[List[Dict[str, Any]]] = None,
    few_shot_examples_enabled: bool = True,
    consolidation_revisions_per_candidate: int = 5,
    enable_third_person_memories: bool = False,
    disable_memory_revisions: bool = False,
) -> Dict[str, Any]:
    """Idempotent ``memory_bank_config`` payload.

    Returns the full block to nest under ``context_spec.memory_bank_config``.
    """
    custom = custom_topics if custom_topics is not None else DEFAULT_CUSTOM_TOPICS
    cfg: Dict[str, Any] = {
        "generation_config": {
            "model": (
                f"projects/{project_id}/locations/global/publishers/google/models/"
                f"{generation_model}"
            ),
        },
        "similarity_search_config": {
            "embedding_model": (
                f"projects/{project_id}/locations/global/publishers/google/models/"
                f"{embedding_model}"
            ),
        },
        # NOTE: revision_ttl is a per-request field on GenerateMemories.config,
        # NOT an instance-config field. The SDK's AgentEngineMemoryBankConfig
        # rejects it. We accept the kwarg for API parity but don't emit it.
        "ttl_config": {
            "granular_ttl_config": {
                "create_ttl": f"{int(create_ttl_days) * 86400}s",
                "generate_created_ttl": f"{int(generate_created_ttl_days) * 86400}s",
            },
        },
        "disable_memory_revisions": bool(disable_memory_revisions),
    }

    topic_entries: List[Dict[str, Any]] = [
        {"managed_memory_topic": {"managed_topic_enum": t}} for t in MANAGED_TOPICS
    ]
    for ct in custom:
        topic_entries.append({
            "custom_memory_topic": {
                "label": ct["label"],
                "description": ct.get("description", ""),
            },
        })

    customization: Dict[str, Any] = {
        "memory_topics": topic_entries,
        "consolidation_config": {
            "revisions_per_candidate_count": int(consolidation_revisions_per_candidate),
        },
        "enable_third_person_memories": bool(enable_third_person_memories),
    }
    if few_shot_examples_enabled:
        customization["generate_memories_examples"] = list(DEFAULT_FEW_SHOT_EXAMPLES)
    cfg["customization_configs"] = [customization]
    return cfg


def resolve_allowed_topics(allowed: Optional[List[str]]) -> Optional[List[Dict[str, Any]]]:
    if not allowed:
        return None
    out: List[Dict[str, Any]] = []
    for label in allowed:
        if label in MANAGED_TOPICS:
            out.append({"managed_memory_topic": {"managed_topic_enum": label}})
        else:
            out.append({"custom_memory_topic": {"label": label}})
    return out
