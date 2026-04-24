"""
Configuration for Memory Bank instances.

Defines topics, TTL, profile schemas, and consolidation strategies.
"""

from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from memory_bank.models import CustomTopicConfig, ManagedTopicEnum, MemoryProfileSchema


class MemoryTopic(BaseModel):
    """A memory topic configuration entry."""

    managed_memory_topic: Optional[ManagedTopicEnum] = None
    custom_memory_topic: Optional[CustomTopicConfig] = None

    @field_validator("custom_memory_topic", mode="before")
    @classmethod
    def _coerce_dict(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return CustomTopicConfig(**v)
        return v

    def to_api_dict(self) -> Dict[str, Any]:
        if self.managed_memory_topic:
            return {"managed_memory_topic": {"managed_topic_enum": self.managed_memory_topic.value}}
        if self.custom_memory_topic:
            return {
                "custom_memory_topic": {
                    "label": self.custom_memory_topic.label,
                    "description": self.custom_memory_topic.description,
                }
            }
        raise ValueError("MemoryTopic must have either managed or custom topic")


class MemoryTTL(BaseModel):
    """Time-to-live for automatic memory expiration."""

    seconds: int = Field(default=7776000, description="Default: 90 days")

    @field_validator("seconds")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("TTL seconds must be non-negative")
        return v


class MetadataMergeStrategy(str):
    """How metadata interacts during consolidation."""

    MERGE = "MERGE"
    OVERWRITE = "OVERWRITE"
    REQUIRE_EXACT_MATCH = "REQUIRE_EXACT_MATCH"


class MemoryBankConfig(BaseModel):
    """
    Complete configuration for a Memory Bank instance.

    Passed to agent_engines.create() or agent_engines.update().
    """

    memory_topics: List[MemoryTopic] = Field(
        default_factory=lambda: [
            MemoryTopic(managed_memory_topic=ManagedTopicEnum.USER_PERSONAL_INFO),
            MemoryTopic(managed_memory_topic=ManagedTopicEnum.USER_PREFERENCES),
            MemoryTopic(managed_memory_topic=ManagedTopicEnum.KEY_CONVERSATION_DETAILS),
            MemoryTopic(managed_memory_topic=ManagedTopicEnum.EXPLICIT_INSTRUCTIONS),
        ],
        description="Topics that guide extraction",
    )
    memory_ttl: MemoryTTL = Field(
        default_factory=MemoryTTL,
        description="Automatic expiration for memories",
    )
    revisions_per_candidate_count: int = Field(
        default=3,
        ge=1,
        le=10,
        description="How many historical revisions to consider during consolidation",
    )
    memory_profile_schemas: Optional[List[MemoryProfileSchema]] = Field(
        default=None,
        description="Structured profile schemas (Preview)",
    )
    perspective: str = Field(
        default="first_person",
        description="first_person ('I prefer...') or third_person ('User prefers...')",
    )

    def to_api_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "memory_bank_config": {
                "memory_topics": [t.to_api_dict() for t in self.memory_topics],
                "memory_ttl": {"seconds": self.memory_ttl.seconds},
                "revisions_per_candidate_count": self.revisions_per_candidate_count,
            }
        }
        if self.memory_profile_schemas:
            result["memory_bank_config"]["memory_profile_schemas"] = [
                s.schema for s in self.memory_profile_schemas
            ]
        return result


# Pre-built configurations for common use cases

HERMES_MEMORY_CONFIG = MemoryBankConfig(
    memory_topics=[
        MemoryTopic(managed_memory_topic=ManagedTopicEnum.USER_PERSONAL_INFO),
        MemoryTopic(managed_memory_topic=ManagedTopicEnum.USER_PREFERENCES),
        MemoryTopic(managed_memory_topic=ManagedTopicEnum.KEY_CONVERSATION_DETAILS),
        MemoryTopic(managed_memory_topic=ManagedTopicEnum.EXPLICIT_INSTRUCTIONS),
        MemoryTopic(
            custom_memory_topic=CustomTopicConfig(
                label="technical_decisions",
                description=(
                    "Technical decisions made during projects: architecture choices, "
                    "tool selections, configuration decisions, and their rationale."
                ),
            )
        ),
        MemoryTopic(
            custom_memory_topic=CustomTopicConfig(
                label="project_context",
                description=(
                    "Active projects, their status, repositories, deployment targets, "
                    "and associated resources."
                ),
            )
        ),
        MemoryTopic(
            custom_memory_topic=CustomTopicConfig(
                label="corrected_mistakes",
                description=(
                    "Mistakes the agent made and corrections provided by the user. "
                    "Include what went wrong and the correct approach."
                ),
            )
        ),
    ],
    memory_ttl=MemoryTTL(seconds=7776000),  # 90 days
    revisions_per_candidate_count=5,
)

OPENCLAW_MEMORY_CONFIG = MemoryBankConfig(
    memory_topics=[
        MemoryTopic(managed_memory_topic=ManagedTopicEnum.USER_PREFERENCES),
        MemoryTopic(managed_memory_topic=ManagedTopicEnum.EXPLICIT_INSTRUCTIONS),
        MemoryTopic(
            custom_memory_topic=CustomTopicConfig(
                label="model_preferences",
                description="User's preferred models, providers, and fallback chains.",
            )
        ),
        MemoryTopic(
            custom_memory_topic=CustomTopicConfig(
                label="gateway_rules",
                description=(
                    "Routing rules, rate limits, caching preferences, and provider "
                    "configurations for the OpenClaw gateway."
                ),
            )
        ),
    ],
    memory_ttl=MemoryTTL(seconds=2592000),  # 30 days
    revisions_per_candidate_count=3,
)
