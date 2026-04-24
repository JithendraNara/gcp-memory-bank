"""
gcp-memory-bank: Production-grade SDK for Google Memory Bank.

Provides async-first, type-safe access to the Gemini Enterprise Agent Platform
Memory Bank with scoped isolation, structured profiles, and comprehensive
retrieval strategies.
"""

from memory_bank.client import MemoryBankClient
from memory_bank.config import (
    CustomTopicConfig,
    ManagedTopicEnum,
    MemoryBankConfig,
    MemoryProfileSchema,
    MemoryTopic,
)
from memory_bank.models import (
    ConsolidationAction,
    Memory,
    MemoryFilter,
    MemoryRevision,
    MemoryScope,
    RetrievedMemory,
    SimilaritySearchParams,
)
from memory_bank.bridge import BaseMemoryBridge, HermesBridgeExample, OpenClawBridgeExample

__version__ = "0.1.0"
__all__ = [
    "MemoryBankClient",
    "MemoryBankConfig",
    "MemoryTopic",
    "CustomTopicConfig",
    "ManagedTopicEnum",
    "MemoryProfileSchema",
    "Memory",
    "MemoryScope",
    "MemoryRevision",
    "RetrievedMemory",
    "ConsolidationAction",
    "SimilaritySearchParams",
    "MemoryFilter",
    "BaseMemoryBridge",
    "HermesBridgeExample",
    "OpenClawBridgeExample",
]
