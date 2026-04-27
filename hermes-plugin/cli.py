"""CLI commands for the GCP Memory Bank plugin.

Appears under: hermes gcp-memory-bank <subcommand>
Only active when gcp-memory-bank is the configured memory provider.
"""

import json
import sys
from pathlib import Path


def _load_provider_config():
    """Read the provider's JSON config from HERMES_HOME."""
    try:
        from hermes_constants import get_hermes_home
        config_path = get_hermes_home() / "gcp-memory-bank.json"
        if config_path.exists():
            return json.loads(config_path.read_text())
    except Exception:
        pass
    return {}


def _get_provider_instance():
    """Import and instantiate the provider (for CLI use)."""
    import os
    plugin_dir = Path(__file__).parent
    sys.path.insert(0, str(plugin_dir))
    from __init__ import GcpMemoryBankProvider
    provider = GcpMemoryBankProvider()
    provider.initialize(session_id="cli-cmd", user_id="cli", agent_identity="hermes", platform="cli")
    return provider


def _cmd_status(args):
    """Show provider status and configuration."""
    cfg = _load_provider_config()
    print("GCP Memory Bank Status")
    print("=" * 50)
    print(f"  Project:     {cfg.get('project_id', 'NOT SET')}")
    print(f"  Location:    {cfg.get('location', 'NOT SET')}")
    print(f"  Engine ID:   {cfg.get('engine_id', 'NOT PROVISIONED')}")
    print(f"  User ID:     {cfg.get('user_id', 'default (from platform)')}")
    print(f"  App Name:    {cfg.get('app_name', 'hermes')}")
    print()
    # Check availability
    import os
    sys.path.insert(0, str(Path(__file__).parent))
    from __init__ import GcpMemoryBankProvider
    provider = GcpMemoryBankProvider()
    avail = provider.is_available()
    print(f"  Available:   {'✅ Yes' if avail else '❌ No'}")
    if not avail:
        print("  Reason:      Missing google-cloud-aiplatform / vertexai, or no GCP credentials.")
    print()


def _cmd_config(args):
    """Show raw provider config."""
    cfg = _load_provider_config()
    print(json.dumps(cfg, indent=2))


def _cmd_search(args):
    """Search memories by query."""
    query_parts = getattr(args, "query", None)
    query = " ".join(query_parts) if query_parts else ""
    if not query:
        print("Usage: hermes gcp-memory-bank search '<query>'")
        sys.exit(1)
    provider = _get_provider_instance()
    result = provider.handle_tool_call("memory_search", {"query": query, "top_k": 10})
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        print(result)
        return
    if "error" in data:
        print(f"Error: {data['error']}")
        return
    results = data.get("results", [])
    if not results:
        print("No memories found.")
        return
    print(f"Found {len(results)} memory(s):")
    print()
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r.get('fact', 'N/A')}")
        print(f"      distance: {r.get('distance', 'N/A')}")
        print()


def _cmd_profile(args):
    """Show all stored memories (profile)."""
    provider = _get_provider_instance()
    result = provider.handle_tool_call("memory_profile", {})
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        print(result)
        return
    if "error" in data:
        print(f"Error: {data['error']}")
        return
    memories = data.get("result", "")
    count = data.get("count", 0)
    if not memories or memories == "No memories stored yet.":
        print("No memories stored yet.")
        return
    print(f"Memory Profile ({count} memories):")
    print("=" * 50)
    print(memories)


def _cmd_revisions(args):
    """List revisions for a memory."""
    memory_name = getattr(args, "memory_name", None)
    if not memory_name:
        print("Usage: hermes gcp-memory-bank revisions <memory_name> [--filter 'labels.verified=true']")
        sys.exit(1)
    provider = _get_provider_instance()
    payload = {"memory_name": memory_name}
    flt = getattr(args, "filter", None)
    if flt:
        payload["filter"] = flt
    result = provider.handle_tool_call("memory_revisions", payload)
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        print(result)
        return
    if "error" in data:
        print(f"Error: {data['error']}")
        return
    revisions = data.get("revisions", [])
    if not revisions:
        print(data.get("result", "No revisions found."))
        return
    print(f"Found {len(revisions)} revision(s):")
    for i, rev in enumerate(revisions, 1):
        rev_id = rev.get("name", "").split("/")[-1]
        print(f"  [{i}] {rev_id}")
        print(f"      fact: {rev.get('fact', 'N/A')}")
        print(f"      created: {rev.get('create_time', 'N/A')}")
        if rev.get("labels"):
            print(f"      labels: {rev['labels']}")
        print()


def _cmd_revision_get(args):
    """Get a specific revision."""
    revision_name = getattr(args, "revision_name", None)
    if not revision_name:
        print("Usage: hermes gcp-memory-bank revision-get <revision_name>")
        sys.exit(1)
    provider = _get_provider_instance()
    result = provider.handle_tool_call("memory_revision_get", {"revision_name": revision_name})
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        print(result)
        return
    if "error" in data:
        print(f"Error: {data['error']}")
        return
    rev = data.get("revision", {})
    print(f"Revision: {rev.get('name', 'N/A')}")
    print(f"Fact: {rev.get('fact', 'N/A')}")
    print(f"Created: {rev.get('create_time', 'N/A')}")
    if rev.get("expire_time"):
        print(f"Expires: {rev['expire_time']}")
    if rev.get("labels"):
        print(f"Labels: {rev['labels']}")


def _cmd_rollback(args):
    """Rollback a memory to a previous revision."""
    memory_name = getattr(args, "memory_name", None)
    target_id = getattr(args, "target_revision_id", None)
    if not memory_name or not target_id:
        print("Usage: hermes gcp-memory-bank rollback <memory_name> <target_revision_id>")
        sys.exit(1)
    provider = _get_provider_instance()
    result = provider.handle_tool_call("memory_rollback", {
        "memory_name": memory_name,
        "target_revision_id": target_id,
    })
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        print(result)
        return
    if "error" in data:
        print(f"Error: {data['error']}")
        return
    print(data.get("result", "Done."))


def _cmd_get(args):
    """Get a single memory by name."""
    memory_name = getattr(args, "memory_name", None)
    if not memory_name:
        print("Usage: hermes gcp-memory-bank get <memory_name>")
        sys.exit(1)
    provider = _get_provider_instance()
    result = provider.handle_tool_call("memory_get", {"memory_name": memory_name})
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        print(result)
        return
    if "error" in data:
        print(f"Error: {data['error']}")
        return
    mem = data.get("memory", {})
    print(f"Name:    {mem.get('name', 'N/A')}")
    print(f"Fact:    {mem.get('fact', 'N/A')}")
    print(f"Scope:   {mem.get('scope', {})}")
    if mem.get("create_time"):
        print(f"Created: {mem['create_time']}")
    if mem.get("expire_time"):
        print(f"Expires: {mem['expire_time']}")
    if mem.get("labels"):
        print(f"Labels:  {mem['labels']}")


def _cmd_delete(args):
    """Delete a single memory by name."""
    memory_name = getattr(args, "memory_name", None)
    if not memory_name:
        print("Usage: hermes gcp-memory-bank delete <memory_name>")
        sys.exit(1)
    provider = _get_provider_instance()
    result = provider.handle_tool_call("memory_delete", {"memory_name": memory_name})
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        print(result)
        return
    if "error" in data:
        print(f"Error: {data['error']}")
        return
    print(data.get("result", "Done."))


def _cmd_ingest(args):
    """Ingest events via streaming API."""
    texts = getattr(args, "texts", None)
    if not texts:
        print("Usage: hermes gcp-memory-bank ingest 'event1' 'event2' ... [--stream-id ID]")
        sys.exit(1)
    events = [{"text": t} for t in texts]
    payload = {"events": events}
    sid = getattr(args, "stream_id", None)
    if sid:
        payload["stream_id"] = sid
    provider = _get_provider_instance()
    result = provider.handle_tool_call("memory_ingest", payload)
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        print(result)
        return
    if "error" in data:
        print(f"Error: {data['error']}")
        return
    print(data.get("result", "Done."))
    print(f"Events: {data.get('event_count', 0)}")


def _cmd_purge(args):
    """Delete memories matching an optional filter. Requires --force."""
    if not getattr(args, "force", False):
        print("This will DELETE memories matching the filter (or all for the current user scope).")
        print("Run again with --force to confirm.")
        sys.exit(1)
    provider = _get_provider_instance()
    payload = {}
    user_filter = getattr(args, "filter", "")
    if user_filter:
        payload["filter"] = user_filter
    result = provider.handle_tool_call("memory_purge", payload)
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        print(result)
        return
    if "error" in data:
        print(f"Error: {data['error']}")
        return
    print(data.get("result", "Done."))
    if "filter" in data:
        print(f"Filter: {data['filter']}")


def _cmd_store(args):
    """Store a fact directly from CLI."""
    fact_parts = getattr(args, "fact", None)
    fact = " ".join(fact_parts) if fact_parts else ""
    if not fact:
        print("Usage: hermes gcp-memory-bank store 'Your fact here'")
        sys.exit(1)
    provider = _get_provider_instance()
    result = provider.handle_tool_call("memory_store", {"fact": fact})
    try:
        data = json.loads(result)
    except json.JSONDecodeError:
        print(result)
        return
    if "error" in data:
        print(f"Error: {data['error']}")
        return
    print(f"Stored: {data.get('name', 'N/A')}")


def _dispatch(args):
    """Route to the correct subcommand handler."""
    sub = getattr(args, "gcp_memory_bank_cmd", None)
    handlers = {
        "status": _cmd_status,
        "config": _cmd_config,
        "search": _cmd_search,
        "profile": _cmd_profile,
        "get": _cmd_get,
        "delete": _cmd_delete,
        "revisions": _cmd_revisions,
        "revision-get": _cmd_revision_get,
        "rollback": _cmd_rollback,
        "ingest": _cmd_ingest,
        "purge": _cmd_purge,
        "store": _cmd_store,
    }
    handler = handlers.get(sub)
    if handler:
        handler(args)
    else:
        print("Usage: hermes gcp-memory-bank <status|config|search|profile|get|delete|revisions|revision-get|rollback|ingest|store|purge>")
        sys.exit(1)


def register_cli(subparser) -> None:
    """Build the `hermes gcp-memory-bank` argparse tree.

    Called by discover_plugin_cli_commands() at argparse setup time.
    """
    subs = subparser.add_subparsers(dest="gcp_memory_bank_cmd")

    subs.add_parser("status", help="Show provider status and config")
    subs.add_parser("config", help="Show raw provider config JSON")

    search_parser = subs.add_parser("search", help="Search memories by query")
    search_parser.add_argument("query", nargs="+", help="Search query string")

    subs.add_parser("profile", help="Show all stored memories")

    get_parser = subs.add_parser("get", help="Fetch a specific memory by name")
    get_parser.add_argument("memory_name", help="Full memory resource name")

    del_parser = subs.add_parser("delete", help="Delete a specific memory by name")
    del_parser.add_argument("memory_name", help="Full memory resource name")

    rev_parser = subs.add_parser("revisions", help="List revision history for a memory")
    rev_parser.add_argument("memory_name", help="Full memory resource name")
    rev_parser.add_argument("--filter", help="EBNF filter string (e.g. labels.verified='true')")

    rev_get_parser = subs.add_parser("revision-get", help="Retrieve a specific revision")
    rev_get_parser.add_argument("revision_name", help="Full revision resource name")

    rollback_parser = subs.add_parser("rollback", help="Rollback a memory to a previous revision")
    rollback_parser.add_argument("memory_name", help="Full memory resource name")
    rollback_parser.add_argument("target_revision_id", help="Revision ID to restore to")

    ingest_parser = subs.add_parser("ingest", help="Stream events via ingest_events API")
    ingest_parser.add_argument("texts", nargs="+", help="Event text strings")
    ingest_parser.add_argument("--stream-id", help="Optional stream/session identifier")

    store_parser = subs.add_parser("store", help="Store a fact from CLI")
    store_parser.add_argument("fact", nargs="+", help="Fact text to store")

    purge_parser = subs.add_parser("purge", help="Delete memories matching filter (requires --force)")
    purge_parser.add_argument("--force", action="store_true", help="Confirm deletion")
    purge_parser.add_argument("--filter", type=str, default="", help="Optional EBNF filter string (e.g. scope.user_id=\"jithendra\")")

    subparser.set_defaults(func=_dispatch)
