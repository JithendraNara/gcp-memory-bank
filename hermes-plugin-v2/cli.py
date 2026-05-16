"""CLI subcommands for gcp-memory-bank v2.

    hermes gcp-memory-bank status
    hermes gcp-memory-bank doctor
    hermes gcp-memory-bank scope             [--set k=tmpl ...]
    hermes gcp-memory-bank config path / show / set / unset
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
from typing import Any, Dict, Iterable, List, Optional


_profile_override: Optional[str] = None


def _hermes_home() -> str:
    if _profile_override:
        return _resolve_profile_home(_profile_override)
    return os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")


def _resolve_profile_home(profile_name: str) -> str:
    try:
        from hermes_cli.profiles import resolve_profile_env
        return resolve_profile_env(profile_name)
    except ImportError:
        canon = str(profile_name or "").strip().lower()
        if canon in ("", "default"):
            return str(Path.home() / ".hermes")
        return str(Path.home() / ".hermes" / "profiles" / canon)


def _active_profile_name() -> str:
    if _profile_override:
        return _profile_override
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name()
    except Exception:
        return "custom" if os.environ.get("HERMES_HOME") else "default"


def _config_file_path(hermes_home: Optional[str] = None) -> Path:
    return Path(hermes_home or _hermes_home()) / CONFIG_FILENAME


def _read_config_file(hermes_home: Optional[str] = None) -> Dict[str, Any]:
    path = _config_file_path(hermes_home)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_config_file(values: Dict[str, Any], hermes_home: Optional[str] = None) -> None:
    path = _config_file_path(hermes_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __package__:
    from .client import MemoryBankClient, _to_dict as _client_to_dict
    from .config import CONFIG_FILENAME, load_config, save_config_file
    from .retrieval import is_pollution
    from .topics import DEFAULT_CUSTOM_TOPICS, MANAGED_TOPICS, build_memory_bank_config
else:  # pragma: no cover - pytest / flat import compatibility
    from client import MemoryBankClient, _to_dict as _client_to_dict
    from config import CONFIG_FILENAME, load_config, save_config_file
    from retrieval import is_pollution
    from topics import DEFAULT_CUSTOM_TOPICS, MANAGED_TOPICS, build_memory_bank_config


def _build_client():
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


def _redact_config(obj: Any) -> Any:
    if isinstance(obj, dict):
        redacted = {}
        for key, value in obj.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in ("secret", "password", "api_key", "apikey")):
                redacted[key] = "***"
            else:
                redacted[key] = _redact_config(value)
        return redacted
    if isinstance(obj, list):
        return [_redact_config(item) for item in obj]
    return obj


def _coerce_config_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _config_key_parts(key: str) -> List[str]:
    parts = [part.strip() for part in str(key or "").split(".") if part.strip()]
    if not parts:
        raise ValueError("config key cannot be empty")
    return parts


def _set_config_key(data: Dict[str, Any], key: str, value: Any) -> None:
    parts = _config_key_parts(key)
    target: Dict[str, Any] = data
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            child = {}
            target[part] = child
        target = child
    target[parts[-1]] = value


def _unset_config_key(data: Dict[str, Any], key: str) -> bool:
    parts = _config_key_parts(key)
    target: Dict[str, Any] = data
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            return False
        target = child
    return target.pop(parts[-1], None) is not None


def _iter_profile_homes() -> Iterable[tuple[str, str]]:
    try:
        from hermes_cli.profiles import list_profiles
        for info in list_profiles():
            yield str(info.name), str(info.path)
        return
    except Exception:
        pass
    yield _active_profile_name(), _hermes_home()


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
def _cmd_status(args: argparse.Namespace) -> int:
    if getattr(args, "all", False):
        active = _active_profile_name()
        for name, home in _iter_profile_homes():
            raw = _read_config_file(home)
            cfg = load_config(home)
            marker = "*" if name == active else "-"
            print(f"{marker} {name}")
            print(f"  Home:    {home}")
            print(f"  Config:  {_config_file_path(home)} ({'present' if raw else 'missing'})")
            print(f"  Project: {cfg.project or '(unset)'}")
            print(f"  Engine:  {cfg.engine_id or '(unset)'}")
        return 0

    cfg = load_config(_hermes_home())
    print("Provider: gcp-memory-bank v2")
    print(f"Profile:  {_active_profile_name()}")
    print(f"Config:   {_config_file_path()}")
    print(f"Project:  {cfg.project or '(unset)'}")
    print(f"Location: {cfg.location}")
    print(f"Engine:   {cfg.engine_id or '(unset)'}")
    print(f"Scope:    {cfg.scope_keys}")
    print(f"Recall:   mode={cfg.recall_mode}  budget={cfg.raw.get('recall_budget')}  detail={cfg.raw.get('recall_detail')}")
    print(f"Sessions: {'enabled' if cfg.raw.get('use_gcp_sessions') else 'disabled'}")
    print(f"Mid-session generate: every {cfg.raw.get('generate_every_n_turns')} turns")
    print(f"Models:   gen={cfg.raw.get('generation_model')}  emb={cfg.raw.get('embedding_model')}  syn={cfg.raw.get('synthesis_model')}")
    return 0


def _cmd_config_path(args: argparse.Namespace) -> int:
    print(_config_file_path())
    return 0


def _cmd_config_show(args: argparse.Namespace) -> int:
    data = load_config(_hermes_home()).raw if args.effective else _read_config_file()
    if not args.no_redact:
        data = _redact_config(data)
    _print_json(data)
    return 0


def _cmd_config_set(args: argparse.Namespace) -> int:
    key = args.key
    value = args.value
    if value is None and "=" in key:
        key, value = key.split("=", 1)
    if value is None:
        print("error: config set expects KEY VALUE or KEY=VALUE", file=sys.stderr)
        return 2
    data = _read_config_file()
    _set_config_key(data, key, _coerce_config_value(value))
    _write_config_file(data)
    print(f"saved {key} in {_config_file_path()}")
    return 0


def _cmd_config_unset(args: argparse.Namespace) -> int:
    data = _read_config_file()
    removed = _unset_config_key(data, args.key)
    _write_config_file(data)
    print(f"{'removed' if removed else 'missing'} {args.key} in {_config_file_path()}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
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


def _cmd_sessions_clean(args: argparse.Namespace) -> int:
    """Delete all sessions on the engine EXCEPT the one currently persisted
    for cross-process reuse. Recovers from the v1 leak (we found 40 sessions
    on engine YOUR_ENGINE_ID on 2026-04-29)."""
    client, _ = _build_client()
    if client is None:
        return 1
    keep: set = set()
    persist_dir = Path(_hermes_home()) / ".gmb-sessions"
    if persist_dir.is_dir():
        for f in persist_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                name = data.get("session_name") or ""
                if name:
                    keep.add(name)
            except Exception:
                pass
    if not keep and not args.force:
        print("error: no persisted session(s) to keep — re-run with --force to "
              "delete ALL sessions on the engine.", file=sys.stderr)
        return 2
    sessions = client.list_sessions(user_id=args.user)
    to_delete = []
    for s in sessions:
        name = s.get("name") if isinstance(s, dict) else getattr(s, "name", "")
        if name and name not in keep:
            to_delete.append(name)
    if not to_delete:
        print(f"Nothing to clean — {len(sessions)} session(s) all match the persisted set.")
        return 0
    print(f"Will keep {len(keep)} persisted session(s):")
    for k in keep:
        print(f"  KEEP  {k.split('/')[-1]}")
    print(f"\nWill delete {len(to_delete)} session(s):")
    for n in to_delete[:20]:
        print(f"  DROP  {n.split('/')[-1]}")
    if len(to_delete) > 20:
        print(f"  ... +{len(to_delete) - 20} more")
    if not args.force:
        print(f"\nDry run. Re-run with --force to actually delete.")
        return 0
    deleted, failed = 0, 0
    for n in to_delete:
        try:
            client.delete_session(n)
            deleted += 1
        except Exception as e:
            failed += 1
            print(f"  fail {n}: {e}", file=sys.stderr)
    print(f"\nDeleted {deleted}, failed {failed}.")
    return 0


def _cmd_clean_pollution(args: argparse.Namespace) -> int:
    """Delete memories that match the pollution patterns we've added to
    the live ingest filter. Recovers from the polluted writes between
    plugin install and pollution_filter landing."""
    client, _ = _build_client()
    if client is None:
        return 1
    polluted = []
    for m in client.list_memories(page_size=500):
        fact = m.get("fact") or ""
        if is_pollution(fact):
            polluted.append(m)
    print(f"Found {len(polluted)} polluted memories.")
    for m in polluted[:10]:
        name = m.get("name", "?")
        print(f"  [{name.split('/')[-1]}] {(m.get('fact') or '')[:80]}")
    if len(polluted) > 10:
        print(f"  ... +{len(polluted) - 10} more")
    if not args.force:
        print("\nDry run. Re-run with --force to delete.")
        return 0
    deleted, failed = 0, 0
    for m in polluted:
        try:
            client.delete_memory(m["name"])
            deleted += 1
        except Exception as e:
            failed += 1
            print(f"  fail {m['name']}: {e}", file=sys.stderr)
    print(f"\nDeleted {deleted}, failed {failed}.")
    return 0


def _cmd_iam_check(args: argparse.Namespace) -> int:
    print("IAM Conditions: api.getAttribute('aiplatform.googleapis.com/memoryScope', {})")
    print("CAVEAT: ListMemories and PurgeMemories ignore Conditions.")
    print("Bind aiplatform.memoryViewer / memoryEditor / memoryUser carefully.")
    return 0


def _to_dict(obj: Any) -> Any:
    return _client_to_dict(obj)


def _dispatch(args: argparse.Namespace) -> int:
    global _profile_override
    _profile_override = getattr(args, "target_profile", None)
    handler = getattr(args, "_handler", None)
    if handler is None:
        print("Usage: hermes gcp-memory-bank <subcommand>", file=sys.stderr)
        return 2
    try:
        return int(handler(args) or 0)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> None:
    rc = _dispatch(args)
    if rc:
        raise SystemExit(rc)


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--target-profile", metavar="NAME", dest="target_profile",
        help="Target a specific Hermes profile's GCP Memory Bank config",
    )
    sub = subparser.add_subparsers(dest="gmb_command")

    p = sub.add_parser("status")
    p.add_argument("--all", action="store_true", help="Show config overview across all profiles")
    p.set_defaults(_handler=_cmd_status)
    p = sub.add_parser("doctor"); p.set_defaults(_handler=_cmd_doctor)
    p = sub.add_parser("audit"); p.set_defaults(_handler=_cmd_audit)

    cfg = sub.add_parser("config", help="Read or write profile-local gcp-memory-bank.json")
    csub = cfg.add_subparsers(dest="config_command")
    p = csub.add_parser("path", help="Print the active config path")
    p.set_defaults(_handler=_cmd_config_path)
    p = csub.add_parser("show", help="Show profile-local config JSON")
    p.add_argument("--effective", action="store_true", help="Include defaults and env overrides")
    p.add_argument("--no-redact", action="store_true", help="Do not redact secret-like keys")
    p.set_defaults(_handler=_cmd_config_show)
    p = csub.add_parser("set", help="Set a config value: KEY VALUE or KEY=VALUE")
    p.add_argument("key")
    p.add_argument("value", nargs="?")
    p.set_defaults(_handler=_cmd_config_set)
    p = csub.add_parser("unset", help="Remove a config key")
    p.add_argument("key")
    p.set_defaults(_handler=_cmd_config_unset)

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
    p = ssub.add_parser("clean", help="Delete all sessions except the persisted one(s).")
    p.add_argument("--user", default=None)
    p.add_argument("--force", action="store_true")
    p.set_defaults(_handler=_cmd_sessions_clean)

    p = sub.add_parser("clean-pollution",
                       help="Delete memories matching the pollution-filter patterns.")
    p.add_argument("--force", action="store_true")
    p.set_defaults(_handler=_cmd_clean_pollution)

    iam = sub.add_parser("iam"); iamsub = iam.add_subparsers(dest="iam_command")
    p = iamsub.add_parser("check"); p.set_defaults(_handler=_cmd_iam_check)

    subparser.set_defaults(func=_run)
