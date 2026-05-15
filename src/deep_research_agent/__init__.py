"""Deep Research Agent v0 scaffold.

A small, dependency-light research orchestration core designed to sit on top of
GCP Memory Bank and any search/fetch/synthesis backend.  The v0 package keeps
interfaces explicit and testable so provider-specific adapters can be added
without coupling the planner to one API surface.
"""

from .adapters import CallableMemoryStore, CallableSearchProvider, normalize_memory_search_result, normalize_profile_result, normalize_search_result, report_to_memory_fact
from .agent import DeepResearchAgent
from .file_adapters import FileMemoryStore, JsonFileSearchProvider
from .interfaces import MemoryStore, SearchProvider, Synthesizer
from .memory import ProfileAugmentedMemoryStore
from .models import Evidence, ResearchPlan, ResearchReport, ResearchStep, SearchHit
from .promotion import MemoryPromotionPolicy, PromotingMemoryStore
from .report import render_markdown_report, report_to_json_dict, slugify, write_report_files
from .synthesis import CitationPreservingSynthesizer

__all__ = [
    "CallableMemoryStore",
    "CallableSearchProvider",
    "CitationPreservingSynthesizer",
    "DeepResearchAgent",
    "Evidence",
    "FileMemoryStore",
    "JsonFileSearchProvider",
    "MemoryPromotionPolicy",
    "MemoryStore",
    "ProfileAugmentedMemoryStore",
    "PromotingMemoryStore",
    "ResearchPlan",
    "ResearchReport",
    "ResearchStep",
    "SearchHit",
    "SearchProvider",
    "Synthesizer",
    "normalize_memory_search_result",
    "normalize_profile_result",
    "normalize_search_result",
    "render_markdown_report",
    "report_to_json_dict",
    "report_to_memory_fact",
    "slugify",
    "write_report_files",
]
