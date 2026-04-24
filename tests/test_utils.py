"""Unit tests for utilities."""

from memory_bank.models import Memory, MemoryScope
from memory_bank.utils import (
    build_scope,
    format_memories_for_prompt,
    format_retrieved_for_context,
    memory_to_dict,
    scope_to_filter,
)


def test_build_scope():
    scope = build_scope(user_id="123", project="alpha")
    assert scope.user_id == "123"
    assert scope.project == "alpha"
    assert scope.agent is None


def test_scope_to_filter():
    scope = build_scope(user_id="123", agent="hermes")
    f = scope_to_filter(scope)
    assert 'scope.user_id="123"' in f
    assert 'scope.agent="hermes"' in f
    assert "AND" in f


def test_format_memories_for_prompt():
    memories = [
        Memory(name="m1", fact="Likes Python", scope=MemoryScope()),
        Memory(name="m2", fact="Uses VS Code", scope=MemoryScope()),
    ]
    prompt = format_memories_for_prompt(memories)
    assert "Likes Python" in prompt
    assert "Uses VS Code" in prompt


def test_format_memories_empty():
    assert format_memories_for_prompt([]) == ""


def test_memory_to_dict():
    m = Memory(
        name="projects/.../memories/1",
        fact="Test fact",
        scope=MemoryScope(user_id="123"),
    )
    d = memory_to_dict(m)
    assert d["fact"] == "Test fact"
    assert d["scope"]["user_id"] == "123"
