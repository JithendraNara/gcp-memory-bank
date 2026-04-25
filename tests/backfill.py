"""Backfill tool: Migrate existing Hermes conversations into GCP Memory Bank.

Scans ~/.hermes/state.db for historical user messages, extracts structured facts,
and stores them in GCP Memory Bank with deduplication.

Run: python backfill.py [--dry-run] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, "/Users/jithendranara/.hermes/hermes-agent")

from plugins.memory import load_memory_provider


HERMES_DB = Path.home() / ".hermes" / "state.db"


def fetch_user_messages(db_path: Path, limit: Optional[int] = None) -> List[str]:
    """Fetch unique user messages from Hermes message history."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    sql = """
        SELECT DISTINCT content FROM messages
        WHERE role = 'user' AND content IS NOT NULL AND LENGTH(content) > 5
        ORDER BY timestamp DESC
    """
    if limit:
        sql += f" LIMIT {limit}"

    cursor.execute(sql)
    rows = [row[0] for row in cursor.fetchall()]
    conn.close()
    return rows


def backfill(
    dry_run: bool = False,
    limit: Optional[int] = None,
    user_id: str = "hermes-user",
    app_name: str = "hermes",
    structured_only: bool = False,
) -> Dict[str, Any]:
    """Run the backfill process."""
    print(f"Loading GCP Memory Bank provider...")
    provider = load_memory_provider("gcp-memory-bank")
    provider.initialize(
        session_id=f"backfill-{int(time.time())}",
        user_id=user_id,
        agent_identity=app_name,
    )

    print(f"Fetching user messages from {HERMES_DB}...")
    messages = fetch_user_messages(HERMES_DB, limit=limit)
    print(f"Found {len(messages)} unique user messages.")

    # Import the fact extractor from the plugin
    sys.path.insert(0, str(Path.home() / ".hermes" / "plugins" / "gcp-memory-bank"))
    from __init__ import _extract_fact, _memory_hash

    extracted: set[str] = set()
    raw_fallbacks: set[str] = set()
    for msg in messages:
        fact = _extract_fact(msg, "")
        if fact:
            extracted.add(fact)
        else:
            # Fallback: store messages that contain personal-topic keywords
            lower = msg.lower()
            # Skip system notes, image descriptions, code, URLs-only
            skip_prefixes = (
                "[system note:", "[the user sent an image", "http", "https://",
                "/users/", "> ##", "```", "curl ", "jithendranara@",
            )
            if any(msg.strip().lower().startswith(p) for p in skip_prefixes):
                continue
            if any(p in lower for p in skip_prefixes):
                continue
            # Skip if mostly non-personal
            if len(msg) > 500:
                continue
            if any(kw in lower for kw in [
                "like", "love", "enjoy", "prefer", "live", "work", "name",
                "from", "interested", "learning", "want", "have", "dog", "cat",
                "pet", "family", "wife", "husband", "kids", "house", "apartment",
                "city", "travel", "visit", "plan", "goal", "project", "health",
                "allergic", "diet", "vegetarian", "vegan",
            ]):
                # Sanitize and truncate
                clean = msg.replace("\n", " ").strip()[:300]
                if len(clean) > 10:
                    raw_fallbacks.add(f"User mentioned: {clean}")

    print(f"Extracted {len(extracted)} structured facts.")
    print(f"Found {len(raw_fallbacks)} raw fallback candidates.")

    if dry_run:
        print("\n--- DRY RUN — would store the following facts ---")
        for fact in sorted(extracted):
            print(f"  \u2022 {fact}")
        for fact in sorted(raw_fallbacks):
            print(f"  \u2022 {fact[:100]}")
        return {"dry_run": True, "messages_scanned": len(messages), "facts_extracted": len(extracted), "raw_fallbacks": len(raw_fallbacks), "stored": 0}

    stored = 0
    skipped = 0
    errors = 0

    print(f"\nStoring facts in GCP Memory Bank (engine: {provider._engine_id})...")
    facts_to_store = sorted(extracted)
    if not structured_only:
        facts_to_store = sorted(extracted | raw_fallbacks)
    for fact in facts_to_store:
        try:
            result = provider.handle_tool_call("memory_store", {"fact": fact})
            data = json.loads(result)
            if data.get("deduplicated"):
                skipped += 1
                print(f"  SKIP (dup): {fact[:80]}")
            else:
                stored += 1
                print(f"  STORED: {fact[:80]}")
        except Exception as e:
            errors += 1
            print(f"  ERROR: {fact[:80]} | {e}")

    print(f"\nDone. Stored: {stored} | Skipped (dup): {skipped} | Errors: {errors}")
    return {
        "messages_scanned": len(messages),
        "facts_extracted": len(extracted),
        "raw_fallbacks": len(raw_fallbacks),
        "stored": stored,
        "skipped": skipped,
        "errors": errors,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill Hermes memories into GCP Memory Bank")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be stored without storing")
    parser.add_argument("--limit", type=int, default=None, help="Max messages to scan")
    parser.add_argument("--user-id", default="jithendra", help="User ID for scope")
    parser.add_argument("--app-name", default="hermes", help="App name for scope")
    parser.add_argument("--structured-only", action="store_true", help="Only store structured facts, skip raw fallbacks")
    args = parser.parse_args()

    result = backfill(
        dry_run=args.dry_run,
        limit=args.limit,
        user_id=args.user_id,
        app_name=args.app_name,
        structured_only=args.structured_only,
    )
    print(f"\nSummary: {json.dumps(result, indent=2)}")
