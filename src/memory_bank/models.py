"""
Core Pydantic models for Memory Bank types.

Mirrors the official Google Cloud API surface with strict typing,
validation, and ergonomic Pythonic interfaces.
"""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConsolidationAction(str, Enum):
    """Actions Memory Bank can take during consolidation."""

    CREATED = "CREATED"
    UPDATED = "UPDATED"
    DELETED = "DELETED"
    NOOP = "NOOP"


class ManagedTopicEnum(str, Enum):
    """Built-in memory topics managed by Google."""

    USER_PERSONAL_INFO = "USER_PERSONAL_INFO"
    USER_PREFERENCES = "USER_PREFERENCES"
    KEY_CONVERSATION_DETAILS = "KEY_CONVERSATION_DETAILS"
    EXPLICIT_INSTRUCTIONS = "EXPLICIT_INSTRUCTIONS"


class MemoryScope(BaseModel):
    """
    Identity isolation scope for memories.

    Up to 5 key-value pairs. Only memories with the EXACT same scope
    are considered for consolidation or retrieval together.
    """

    model_config = ConfigDict(extra="allow")

    user_id: Optional[str] = Field(default=None, description="Primary user identifier")
    agent: Optional[str] = Field(default=None, description="Agent name")
    project: Optional[str] = Field(default=None, description="Project context")
    session_id: Optional[str] = Field(default=None, description="Session context")
    team: Optional[str] = Field(default=None, description="Team context")

    @field_validator("user_id", "agent", "project", "session_id", "team", mode="before")
    @classmethod
    def _reject_wildcards(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and "*" in v:
            raise ValueError("Scope values cannot contain '*' characters")
        return v

    def to_dict(self) -> Dict[str, str]:
        """Serialize to API-compatible dict, excluding None values."""
        return {k: v for k, v in self.model_dump().items() if v is not None}

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.to_dict().items())))


class MemoryMetadataValue(BaseModel):
    """Single metadata value — typed for API compatibility."""

    string_value: Optional[str] = None
    double_value: Optional[float] = None
    bool_value: Optional[bool] = None
    timestamp_value: Optional[datetime.datetime] = None


class MemoryMetadata(BaseModel):
    """Arbitrary metadata attached to a memory."""

    model_config = ConfigDict(extra="allow")

    def to_api_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, val in self.model_dump(exclude_none=True).items():
            if isinstance(val, (str, float, bool, datetime.datetime)):
                if isinstance(val, str):
                    result[key] = {"string_value": val}
                elif isinstance(val, float):
                    result[key] = {"double_value": val}
                elif isinstance(val, bool):
                    result[key] = {"bool_value": val}
                elif isinstance(val, datetime.datetime):
                    result[key] = {"timestamp_value": val}
            else:
                result[key] = val
        return result


class Memory(BaseModel):
    """A single Memory Bank memory resource."""

    name: str = Field(description="Fully-qualified resource name")
    fact: str = Field(description="Consolidated memory content")
    scope: MemoryScope = Field(description="Identity isolation scope")
    create_time: Optional[datetime.datetime] = None
    update_time: Optional[datetime.datetime] = None
    expire_time: Optional[datetime.datetime] = None
    topics: Optional[List[str]] = Field(default=None, description="Associated memory topics")
    metadata: Optional[MemoryMetadata] = None
    memory_type: Optional[Literal["STRUCTURED_PROFILE", "FACT"]] = Field(
        default="FACT", alias="memoryType"
    )
    structured_content: Optional[Dict[str, Any]] = Field(default=None, alias="structuredContent")

    model_config = ConfigDict(populate_by_name=True)


class GeneratedMemory(BaseModel):
    """Result of a single memory generation operation."""

    memory: Memory
    action: ConsolidationAction


class SimilaritySearchParams(BaseModel):
    """Parameters for embedding-based similarity retrieval."""

    search_query: str = Field(description="Query text to embed and compare")
    top_k: int = Field(default=3, ge=1, le=50, description="Number of results to return")


class RetrievedMemory(BaseModel):
    """A memory returned from retrieval with similarity distance."""

    memory: Memory
    distance: Optional[float] = Field(
        default=None,
        description="Euclidean distance (lower = more similar)",
    )


class MemoryRevision(BaseModel):
    """Immutable snapshot of a memory at a specific point in time."""

    name: str
    fact: str
    create_time: Optional[datetime.datetime] = None
    expire_time: Optional[datetime.datetime] = None
    extracted_memories: List[IntermediateExtractedMemory] = Field(default_factory=list)


class IntermediateExtractedMemory(BaseModel):
    """Raw extraction output before consolidation."""

    fact: str


class MemoryFilter(BaseModel):
    """
    Criteria for purging or listing memories.

    Supports EBNF syntax for system fields and DNF for metadata.
    """

    filter_string: Optional[str] = Field(
        default=None,
        description="EBNF filter on system fields: fact, create_time, update_time, topics",
    )
    filter_groups: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Metadata filter groups (DNF logic)",
    )


class SessionEvent(BaseModel):
    """A single event within an Agent Platform Session."""

    role: Literal["user", "model", "tool"] = Field(description="Event author")
    content: str = Field(description="Event text content")
    timestamp: Optional[datetime.datetime] = None
    invocation_id: Optional[str] = None
    tool_name: Optional[str] = None


class IngestEvent(BaseModel):
    """Event for the IngestEvents streaming API (Preview)."""

    event_id: str = Field(description="Unique event ID for deduplication")
    role: Literal["user", "model"]
    content: str
    timestamp: Optional[datetime.datetime] = None


class CustomTopicConfig(BaseModel):
    """User-defined memory topic for extraction guidance."""

    label: str = Field(description="Unique topic identifier")
    description: str = Field(
        description="Instructions for what to extract (passed to LLM prompt)"
    )
    few_shot_examples: Optional[List[str]] = Field(
        default=None,
        description="Example extractions to guide the LLM",
    )


class MemoryProfileSchema(BaseModel):
    """
    Static JSON schema for structured memory profiles (Preview).

    Profiles are optimized for low-latency retrieval — curation happens
    at write-time so reads are instant.
    """

    schema_id: str = Field(description="Unique schema identifier")
    schema: Dict[str, Any] = Field(description="JSON Schema or Pydantic model dict")
    description: Optional[str] = None
