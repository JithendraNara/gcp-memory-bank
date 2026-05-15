"""Command-line entry point for Deep Research Agent v0."""

from __future__ import annotations

import argparse
from pathlib import Path

from .agent import DeepResearchAgent
from .file_adapters import FileMemoryStore, JsonFileSearchProvider
from .memory import NullMemoryStore
from .report import slugify, write_report_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Deep Research Agent v0")
    parser.add_argument("--question", required=True, help="Research question to answer")
    parser.add_argument(
        "--search-fixture",
        required=True,
        help="JSON search fixture path for the v0/offline SearchProvider",
    )
    parser.add_argument("--memory-file", help="Optional JSONL memory file for recall/persistence")
    parser.add_argument("--output-dir", default="reports/deep-research", help="Directory for .md/.json reports")
    parser.add_argument("--slug", help="Optional report slug")
    parser.add_argument("--no-persist", action="store_true", help="Do not persist report to memory file")
    parser.add_argument("--search-limit", type=int, default=3, help="Search hits per research step")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    slug = args.slug or slugify(args.question)
    memory_store = FileMemoryStore(args.memory_file) if args.memory_file else NullMemoryStore()
    agent = DeepResearchAgent(
        search_provider=JsonFileSearchProvider(args.search_fixture),
        memory_store=memory_store,
        search_limit_per_step=args.search_limit,
    )
    report = agent.run(args.question, persist=not args.no_persist)
    markdown_path, json_path = write_report_files(report, Path(args.output_dir), slug=slug)
    print(f"markdown={markdown_path}")
    print(f"json={json_path}")
    return 0


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
