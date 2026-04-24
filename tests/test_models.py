"""Unit tests for Pydantic models."""

import pytest
from memory_bank.models import (
    ConsolidationAction,
    ManagedTopicEnum,
    MemoryScope,
    MemoryMetadata,
    SimilaritySearchParams,
)


def test_memory_scope_basic():
    scope = MemoryScope(user_id="123", agent="hermes")
    assert scope.to_dict() == {"user_id": "123", "agent": "hermes"}


def test_memory_scope_rejects_wildcards():
    with pytest.raises(ValueError):
        MemoryScope(user_id="12*3")


def test_memory_scope_hashable():
    a = MemoryScope(user_id="123")
    b = MemoryScope(user_id="123")
    assert hash(a) == hash(b)


def test_similarity_search_params_validation():
    with pytest.raises(ValueError):
        SimilaritySearchParams(search_query="test", top_k=0)

    with pytest.raises(ValueError):
        SimilaritySearchParams(search_query="test", top_k=100)


def test_consolidation_action_enum():
    assert ConsolidationAction.CREATED.value == "CREATED"
    assert ConsolidationAction.UPDATED.value == "UPDATED"


def test_managed_topic_enum():
    assert ManagedTopicEnum.USER_PERSONAL_INFO.value == "USER_PERSONAL_INFO"
