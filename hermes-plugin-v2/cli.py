"""CLI subcommands for gcp-memory-bank v2.

    hermes gcp-memory-bank status
    hermes gcp-memory-bank doctor
    hermes gcp-memory-bank scope             [--set k=tmpl ...]
    hermes gcp-memory-bank scope-migrate     [--from-user X --to-user Y]
    hermes gcp-memory-bank instance describe / create / update-config
    hermes gcp-memory-bank topics list
    hermes gcp-memory-bank revisions list MEMORY [--label k=v]
    hermes gcp-memory-bank revisions get MEMORY REVISION_ID
    hermes gcp-memory-bank rollback MEMORY REVISION_ID
    hermes gcp-memory-bank purge --filter EXPR [--force]
    hermes gcp-memory-bank sessions list / describe / delete / replay
    hermes gcp-memory-bank iam check
    hermes gcp-memory-bank audit            (NEW: scope drift + leaked session report)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _hermes_home() -> str:
    return os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")


def _build_client():
    from .client import MemoryBankClient
    from .config import load_config

    cfg = load_config(_hermes_home())
    if not cfg.engine_id:
        print("error: engine_id not configured. Run `hermes memory setup`.",
              file=sys.stderr)
        return None, None
    if not cfg.project:
        print("error: GOOGLE_CLOUD_PROJECT not set.", file=sys.stderr)
        return None, None
    client = MemoryBankClient(
        project=cfg.project,
        location=cfg.location,
        engine_id=cfg.engine_id,
    )
    return client, cfg


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, default=str))


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def _cmd_status(args: argparse.Namespace) -> int:
    from .config import load_config
    cfg = load_config(_hermes_home())
    print("Provider: gcp-memory-bank v2")
    print(f"Project:  {cfg.project or '(unset)'}")
    print(f"Location: {cfg.location}")
    print(f"Engine:   {cfg.engine_id or '(unset)'}")
    print(f"Scope:    {cfg.scope_keys}")
    print(f"Recall:   mode={cfg.recall_mode}  budget={cfg.raw.get('recall_budget')}  detail={cfg.raw.get('recall_detail')}")
    print(f"Sessions: {'enabled' if cfg.raw.get('use_gcp_sessions') else 'disabled'}")
    print(f"Mid-session generate: every {cfg.raw.get('generate_every_n_turns')} turns")
    print(f"Models:   gen={cfg.raw.get('generation_model')}  emb={cfg.raw.get('embedding_model')}  syn={cfg.raw.get('synthesis_model')}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    from .config import load_config
    cfg = load_config(_hermes_home())
    issues: List[str] = []
    ok: List[str] = []

    try:
        import google.auth  # type: ignore
        creds, project = google.auth.default()
        ok.append(f"ADC ok (project={project or 'n/a'})")
    except Exception as e:
        issues.append(f"ADC missing — run `gcloud auth application-default login` ({e})")

    try:
        import vertexai  # noqa: F401
        ok.append("vertexai SDK importable")
    except Exception as e:
        issues.append(f"google-cloud-aiplatform not installed: {e}")

    if not cfg.project:
        issues.append("GOOGLE_CLOUD_PROJECT unset and no `project_id` in config.")
    if not cfg.engine_id:
        issues.append("engine_id unset — run `hermes gcp-memory-bank instance create`.")

    if cfg.project and cfg.engine_id:
        try:
            client, _ = _build_client()
            if client:
                eng = client.get_engine()
                eng_name = (
                    getattr(eng, "name", None)
                    or getattr(getattr(eng, "api_resource", None), "name", "?")
                )
                ok.append(f"Engine reachable: {eng_name}")
                # Memory + session counts.
                try:
                    mems = client.list_memories(page_size=1)
                    ok.append(f"List memories: returned {len(mems)} sample(s).")
                except Exception as e:
                    issues.append(f"list_memories failed: {e}")
                try:
                    sess = client.list_sessions()
                    ok.append(f"List sessions: {len(sess)} session(s).")
                except Exception as e:
                    issues.append(f"list_sessions failed: {e}")
        except Exception as e:
            issues.append(f"Engine not reachable: {e}")

    # Check that hindsight isn't simultaneously configured.
    try:
        cfg_yaml = Path(_hermes_home()) / "config.yaml"
        if cfg_yaml.exists():
            content = cfg_yaml.read_text()
            if "hindsight" in content and "gcp-memory-bank" in content:
                issues.append(
                    "Both hindsight and gcp-memory-bank appear in config.yaml — "
                    "Hermes only allows one external memory provider."
                )
    except Exception:
        pass

    issues.append(
        "note: ListMemories / PurgeMemories ignore IAM Conditions on memoryScope. "
        "Bind aiplatform.memoryViewer carefully if multi-tenant."
    )

    for line in ok:
        print(f"[ok]   {line}")
    for line in issues:
        print(f"[warn] {line}")
    return 0 if not any("missing" in s or "not installed" in s for s in issues) else 1


def _cmd_scope(args: argparse.Namespace) -> int:
    from .config import load_config, save_config_file
    cfg = load_config(_hermes_home())
    if args.set_pairs:
        scope_keys: List[str] = []
        scope_template: Dict[str, str] = {}
        for pair in args.set_pairs:
            if "=" not in pair:
                print(f"error: --set expects key=template, got {pair!r}", file=sys.stderr)
                return 2
            k, v = pair.split("=", 1)
            scope_keys.append(k.strip())
            scope_template[k.strip()] = v
        save_config_file({"scope_keys": scope_keys, "scope_template": scope_template},
                         _hermes_home())
        print(f"Saved scope_keys={scope_keys}")
        return 0
    print(f"scope_keys     = {cfg.scope_keys}")
    print(f"scope_template = {cfg.raw.get('scope_template')}")
    return 0


def _cmd_scope_migrate(args: argparse.Namespace) -> int:
    """Re-key memories under from_user → to_user.

    Workaround for the v1 issue where 3 different user_ids ended up in the
    engine. Reads each memory, recreates it under the new scope, deletes the
    old one. NOT atomic — make sure no Hermes process is writing during.
    """
    client, cfg = _build_client()
    if client is None:
        return 1
    if not args.from_user or not args.to_user:
        print("error: --from-user and --to-user are required.", file=sys.stderr)
        return 2
    flt = f'scope.user_id="{args.from_user}" AND scope.app_name="{args.app or cfg.app_name}"'
    print(f"Filter: {flt}")
    if not args.force:
        memories = client.list_memories(filter_expr=flt, page_size=500)
        print(f"Dry run: {len(memories)} memories would be migrated. Re-run with --force.")
        for m in memories[:10]:
            print(f"  - {m.get('name', '?').split('/')[-1]}: {(m.get('fact') or '')[:80]}")
        if len(memories) > 10:
            print(f"  ... +{len(memories) - 10} more")
        return 0
    moved = 0
    failed = 0
    memories = client.list_memories(filter_expr=flt, page_size=500)
    for m in memories:
        try:
            new_scope = dict(m.get("scope") or {})
            new_scope["user_id"] = args.to_user
            client.create_memory(scope=new_scope, fact=m.get("fact", ""),
                                 revision_labels={"migrated_from": args.from_user})
            client.delete_memory(m.get("name"))
            moved += 1
        except Exception as e:
            failed += 1
            print(f"  fail {m.get('name')}: {e}", file=sys.stderr)
    print(f"Migrated {moved} memories. {failed} failures.")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    """Surface the runtime-audit findings against the live engine."""
    client, cfg = _build_client()
    if client is None:
        return 1
    print("== gcp-memory-bank audit ==\n")

    # Distinct user_ids in memories — exposes scope drift.
    print("# Scope drift")
    memories = client.list_memories(page_size=500)
    by_scope: Dict[tuple, int] = {}
    for m in memories:
        scope = m.get("scope") or {}
        key = tuple(sorted((k, str(v)) for k, v in scope.items()))
        by_scope[key] = by_scope.get(key, 0) + 1
    if len(by_scope) > 1:
        print(f"  WARN: {len(by_scope)} distinct scopes found — memories are sharded:")
    for scope, n in sorted(by_scope.items(), key=lambda x: -x[1]):
        print(f"    ({n:>4}) {dict(scope)}")

    # Sessions — open vs ended.
    print("\n# Sessions")
    sessions = client.list_sessions()
    print(f"  Total sessions: {len(sessions)}")
    if len(sessions) > 20:
        print(f"  WARN: {len(sessions)} sessions on this engine — possible leak.")

    # Topic distribution (rough).
    print("\n# Topic distribution (top 10 by topic label)")
    topic_counts: Dict[str, int] = {}
    for m in memories:
        topics = m.get("topics") or []
        for t in topics:
            label = ""
            if isinstance(t, dict):
                mt = t.get("managed_memory_topic")
                if isinstance(mt, dict):
                    label = mt.get("managed_topic_enum") or ""
                elif isinstance(mt, str):
                    label = mt
                else:
                    cust = t.get("custom_memory_topic") or {}
                    label = cust.get("label") or ""
            if label:
                topic_counts[label] = topic_counts.get(label, 0) + 1
    for label, n in sorted(topic_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    ({n:>4}) {label}")

    print("\nRun `hermes gcp-memory-bank scope-migrate --from-user X --to-user Y` to merge.")
    return 0


def _cmd_instance_describe(args: argparse.Namespace) -> int:
    client, _ = _build_client()
    if client is None:
        return 1
    eng = client.get_engine()
    _print_json(_to_dict(eng))
    return 0


def _cmd_instance_create(args: argparse.Namespace) -> int:
    client, cfg = _build_client()
    if client is None:
        return 1
    from .topics import build_memory_bank_config
    body = build_memory_bank_config(
        project_id=cfg.project,
        generation_model=str(cfg.raw.get("generation_model")),
        embedding_model=str(cfg.raw.get("embedding_model")),
        create_ttl_days=int(cfg.raw.get("create_ttl_days", 365)),
        generate_created_ttl_days=int(cfg.raw.get("generate_created_ttl_days", 365)),
        revision_ttl_days=int(cfg.raw.get("revision_ttl_days", 365)),
    )
    op = client.create_engine(memory_bank_config=body)
    _print_json(_to_dict(op))
    return 0


def _cmd_instance_update_config(args: argparse.Namespace) -> int:
    client, cfg = _build_client()
    if client is None:
        return 1
    from .topics import build_memory_bank_config
    body = build_memory_bank_config(
        project_id=cfg.project,
        generation_model=str(cfg.raw.get("generation_model")),
        embedding_model=str(cfg.raw.get("embedding_model")),
        create_ttl_days=int(cfg.raw.get("create_ttl_days", 365)),
        generate_created_ttl_days=int(cfg.raw.get("generate_created_ttl_days", 365)),
        revision_ttl_days=int(cfg.raw.get("revision_ttl_days", 365)),
        custom_topics=cfg.raw.get("custom_topics"),
        few_shot_examples_enabled=bool(cfg.raw.get("few_shot_examples_enabled", True)),
        consolidation_revisions_per_candidate=int(cfg.raw.get("consolidation_revisions_per_candidate", 5)),
        enable_third_person_memories=bool(cfg.raw.get("enable_third_person_memories", False)),
        disable_memory_revisions=bool(cfg.raw.get("disable_memory_revisions", False)),
    )
    op = client.update_engine_config(body)
    _print_json(_to_dict(op))
    return 0


def _cmd_topics_list(args: argparse.Namespace) -> int:
    from .topics import DEFAULT_CUSTOM_TOPICS, MANAGED_TOPICS
    print("Managed:")
    for t in MANAGED_TOPICS:
        print(f"  - {t}")
    print("Custom (default):")
    for t in DEFAULT_CUSTOM_TOPICS:
        print(f"  - {t['label']}: {t.get('description', '')}")
    return 0


def _cmd_revisions_list(args: argparse.Namespace) -> int:
    client, _ = _build_client()
    if client is None:
        return 1
    label_filter = " AND ".join(f"labels.{p}" for p in (args.label or [])) or None
    revs = client.list_revisions(args.memory_id, label_filter=label_filter)
    _print_json(revs)
    return 0


def _cmd_revisions_get(args: argparse.Namespace) -> int:
    client, _ = _build_client()
    if client is None:
        return 1
    name = (
        args.revision_id
        if args.revision_id.startswith("projects/")
        else f"{args.memory_id}/revisions/{args.revision_id}"
    )
    rev = client.get_revision(name)
    _print_json(rev or {})
    return 0


def _cmd_rollback(args: argparse.Namespace) -> int:
    client, _ = _build_client()
    if client is None:
        return 1
    client.rollback(args.memory_id, args.target_revision_id)
    print(f"rolled back {args.memory_id} -> {args.target_revision_id}")
    return 0


def _cmd_purge(args: argparse.Namespace) -> int:
    client, _ = _build_client()
    if client is None:
        return 1
    result = client.purge(filter_expr=args.filter, force=args.force,
                          wait=args.force)
    _print_json(_to_dict(result))
    return 0


def _cmd_sessions_list(args: argparse.Namespace) -> int:
    client, _ = _build_client()
    if client is None:
        return 1
    sess = client.list_sessions(user_id=args.user)
    _print_json(sess)
    return 0


def _cmd_sessions_delete(args: argparse.Namespace) -> int:
    client, _ = _build_client()
    if client is None:
        return 1
    client.delete_session(args.session_name)
    print(f"deleted {args.session_name}")
    return 0


def _cmd_iam_check(args: argparse.Namespace) -> int:
    print("IAM Conditions: api.getAttribute('aiplatform.googleapis.com/memoryScope', {})")
    print("CAVEAT: ListMemories and PurgeMemories ignore Conditions.")
    print("Bind aiplatform.memoryViewer / memoryEditor / memoryUser carefully.")
    return 0


def _to_dict(obj: Any) -> Any:
    from .client import _to_dict as _td
    return _td(obj)


def _dispatch(args: argparse.Namespace) -> int:
    handler = getattr(args, "_handler", None)
    if handler is None:
        print("Usage: hermes gcp-memory-bank <subcommand>", file=sys.stderr)
        return 2
    return int(handler(args) or 0)


def register_cli(subparser: argparse.ArgumentParser) -> None:
    sub = subparser.add_subparsers(dest="gmb_command")

    p = sub.add_parser("status"); p.set_defaults(_handler=_cmd_status)
    p = sub.add_parser("doctor"); p.set_defaults(_handler=_cmd_doctor)
    p = sub.add_parser("audit"); p.set_defaults(_handler=_cmd_audit)

    p = sub.add_parser("scope")
    p.add_argument("--set", dest="set_pairs", nargs="+", default=[],
                   metavar="KEY=TEMPLATE")
    p.set_defaults(_handler=_cmd_scope)

    p = sub.add_parser("scope-migrate",
                       help="Re-key memories from one user_id to another.")
    p.add_argument("--from-user", required=True)
    p.add_argument("--to-user", required=True)
    p.add_argument("--app", default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(_handler=_cmd_scope_migrate)

    inst = sub.add_parser("instance")
    isub = inst.add_subparsers(dest="instance_command")
    p = isub.add_parser("describe"); p.set_defaults(_handler=_cmd_instance_describe)
    p = isub.add_parser("create"); p.set_defaults(_handler=_cmd_instance_create)
    p = isub.add_parser("update-config"); p.set_defaults(_handler=_cmd_instance_update_config)

    topics = sub.add_parser("topics")
    tsub = topics.add_subparsers(dest="topics_command")
    p = tsub.add_parser("list"); p.set_defaults(_handler=_cmd_topics_list)

    rev = sub.add_parser("revisions")
    rsub = rev.add_subparsers(dest="revisions_command")
    p = rsub.add_parser("list"); p.add_argument("memory_id"); p.add_argument("--label", action="append", default=[]); p.set_defaults(_handler=_cmd_revisions_list)
    p = rsub.add_parser("get"); p.add_argument("memory_id"); p.add_argument("revision_id"); p.set_defaults(_handler=_cmd_revisions_get)

    p = sub.add_parser("rollback"); p.add_argument("memory_id"); p.add_argument("target_revision_id"); p.set_defaults(_handler=_cmd_rollback)

    p = sub.add_parser("purge"); p.add_argument("--filter", required=True); p.add_argument("--force", action="store_true")
    p.set_defaults(_handler=_cmd_purge)

    sess = sub.add_parser("sessions")
    ssub = sess.add_subparsers(dest="sessions_command")
    p = ssub.add_parser("list"); p.add_argument("--user", default=None); p.set_defaults(_handler=_cmd_sessions_list)
    p = ssub.add_parser("delete"); p.add_argument("session_name"); p.set_defaults(_handler=_cmd_sessions_delete)

    iam = sub.add_parser("iam"); iamsub = iam.add_subparsers(dest="iam_command")
    p = iamsub.add_parser("check"); p.set_defaults(_handler=_cmd_iam_check)

    subparser.set_defaults(func=_dispatch)
