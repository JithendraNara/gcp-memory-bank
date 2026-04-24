"""Pytest configuration."""

import pytest


@pytest.fixture
def sample_scope():
    from memory_bank.models import MemoryScope
    return MemoryScope(user_id="test-user", agent="hermes")
