#!/usr/bin/env python3
"""Lora Atlas command line.

    python run.py init "E:/SD/webui-forge/models/Lora"
    python run.py serve
    python run.py scan
    python run.py sync
    python run.py stats
    python run.py search "rin tohsaka"
    python run.py duplicates
    python run.py backup export
    python run.py backup restore <file>
    python run.py doctor

Every subcommand shares the same code path as the web UI — the CLI is a second
front end, not a second implementation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The Windows console defaults to cp1252, which mangles the box drawing and
# dashes used below. Opt into UTF-8 where the stream supports it.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from atlas import config, db, events, hashing, query, tasks, thumbnails   # noqa: E402
from atlas.jobs import QUEUE                                             # noqa: E402


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} PB"


def _echo_events() -> None:
    """Mirror the SSE log stream to stdout so CLI runs are not silent."""
    import threading

    subscription = events.BROKER.subscribe()

    def pump() -> None:
        while True:
            message = subscription.get()
            if message is None:
                return
            if message["kind"] == "log":
                data = message["data"]
                print(f"  {data['line']}")
            elif message["kind"] == "job":
                job = message["data"].get("job") or {}
                if job.get("status") == "running" and job.get("total"):
                    pct = job.get("percent", 0)
                    print(f"\r  {job['label']}: {pct}% "
                          f"({job['current']}/{job['total']}) {job.get('detail','')[:48]:<48}",
                          end="", flush=True)
                elif job.get("status") in ("done", "error", "cancelled"):
                    print(f"\r  {job['label']}: {job.get('message') or job['status']:<70}")

    threading.Thread(target=pump, daemon=True).start()


def run_job(action: str, **kwargs) -> int:
    """Run one registered action synchronously and report the outcome."""
    import time

    _echo_events()
    QUEUE.start()
    job = tasks.submit(action, **kwargs)
    while job.status in ("pending", "running"):
        time.sleep(0.1)
    QUEUE.stop()
    if job.status == "error":
        print(f"\nFailed: {job.error}", file=sys.stderr)
        return 1
    print(f"\n{job.label}: {job.message}")
    return 0


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_init(args) -> int:
    root = Path(args.path).expanduser()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        return 2
    config.set_many({"library_root": str(root)})
    conn = db.connect()
    print(f"Library root set to {root}")
    print(f"Database at {config.db_path()}")
    print("\nNext: python run.py sync")
    return 0


def cmd_serve(args) -> int:
    from atlas import server
    server.serve(port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_scan(args) -> int:
    return run_job("scan")


def cmd_sync(args) -> int:
    return run_job("quick-refresh" if args.quick else "full-sync")


def cmd_hash(args) -> int:
    return run_job("hash")


def cmd_thumbs(args) -> int:
    return run_job("thumbs")


def cmd_stats(args) -> int:
    conn = db.connect()
    s = query.stats(conn)
    rows = [
        ("Models", s["total"]),
        ("Total size", human(s["total_bytes"])),
        ("Missing metadata", s["missing_metadata"]),
        ("Partial metadata", s["partial_metadata"]),
        ("Missing previews", s["missing_preview"]),
        ("Unhashed", s["unhashed"]),
        ("Favourites", s["favorites"]),
        ("Rated", s["rated"]),
        ("Collections", s["collections"]),
        ("Duplicate groups", s["duplicate_groups"]),
        ("Missing files", s["missing_files"]),
    ]
    width = max(len(label) for label, _ in rows)
    print(f"\n{config.APP_NAME} — {config.library_root()}\n")
    for label, value in rows:
        print(f"  {label:<{width}}  {value}")
    thumb_files, thumb_bytes = thumbnails.cache_size()
    print(f"  {'Thumbnail cache':<{width}}  {thumb_files} files, {human(thumb_bytes)}")
    print()
    return 0


def cmd_search(args) -> int:
    conn = db.connect()
    result = query.list_models(conn, {"q": args.term, "page_size": args.limit})
    print(f"\n{result['total']} match(es); showing {len(result['items'])}\n")
    for item in result["items"]:
        flags = "".join([
            "★" if item["favorite"] else " ",
            "!" if item["meta_state"] != "complete" else " ",
        ])
        print(f"  {flags} {item['name'][:48]:<48} {item['base_model'][:14]:<14} {item['filename']}")
    print()
    return 0


def cmd_duplicates(args) -> int:
    conn = db.connect()
    pending = hashing.pending_count(conn)
    if pending:
        print(f"{pending} file(s) unhashed — run `python run.py hash` for complete results.\n")
    groups = hashing.find_duplicates(conn)
    if not groups:
        print("No duplicates found.")
        return 0
    total = sum(g["reclaimable"] for g in groups)
    print(f"\n{len(groups)} duplicate group(s), {human(total)} reclaimable\n")
    for group in groups[: args.limit]:
        print(f"  {human(group['size'])}  ({len(group['files'])} copies)")
        for entry in group["files"]:
            print(f"      {entry['path']}")
    print()
    return 0


def cmd_backup(args) -> int:
    from atlas import backup as backup_mod

    conn = db.connect()
    root = config.library_root()
    if args.action == "export":
        target = Path(args.path) if args.path else (
            config.state_dir() / "backups" / backup_mod.default_name(args.favourites)
        )
        _echo_events()
        QUEUE.start()
        job = tasks.submit_export(target=str(target), favourites_only=args.favourites)
        import time
        while job.status in ("pending", "running"):
            time.sleep(0.1)
        QUEUE.stop()
        if job.status == "error":
            print(f"Failed: {job.error}", file=sys.stderr)
            return 1
        print(f"\n{job.message}")
        return 0

    if args.action == "inspect":
        info = backup_mod.inspect(Path(args.path))
        for key, value in info.items():
            print(f"  {key:<16} {value}")
        return 0

    if args.action == "restore":
        if not args.path:
            print("restore needs a path", file=sys.stderr)
            return 2
        _echo_events()
        QUEUE.start()
        job = tasks.submit_restore(archive=args.path)
        import time
        while job.status in ("pending", "running"):
            time.sleep(0.1)
        QUEUE.stop()
        if job.status == "error":
            print(f"Failed: {job.error}", file=sys.stderr)
            return 1
        print(f"\n{job.message}")
        return 0
    return 2


def cmd_organise(args) -> int:
    from atlas import organizer

    conn = db.connect()
    root = config.library_root()
    plan = organizer.plan_collection_layout(
        conn, root, group_versions=not args.no_group)
    summary = plan.summary()

    print(f"\nPlan: {summary['moves']} move(s), "
          f"{summary['already_correct']} already correct, "
          f"{summary['files_affected']} file(s) affected\n")
    for move in plan.real_moves[:25]:
        print(f"  {move.from_dir or '<root>'}")
        print(f"    -> {move.to_dir}   (+{len(move.assets)} asset(s), {move.reason})")
    if len(plan.real_moves) > 25:
        print(f"  ... and {len(plan.real_moves) - 25} more")
    if plan.orphans:
        print(f"\n  {len(plan.orphans)} leftover file(s); see `run.py cleanup`")

    if args.preview:
        print("\nNothing changed. Drop --preview to apply.\n")
        return 0
    if not plan.real_moves:
        print("\nNothing to do.\n")
        return 0

    print()
    confirm = input("Apply these moves? [y/N] ").strip().lower()
    if not confirm.startswith("y"):
        print("Cancelled.\n")
        return 0
    return run_job("organise")


def cmd_cleanup(args) -> int:
    from atlas import organizer

    conn = db.connect()
    root = config.library_root()
    orphans = organizer.find_orphans(conn, root)
    if not orphans:
        print("\nNo leftover files.\n")
    else:
        print(f"\n{len(orphans)} leftover file(s) with no matching model:\n")
        for rel in orphans[:40]:
            print(f"  {rel}")
        if len(orphans) > 40:
            print(f"  ... and {len(orphans) - 40} more")
    if not args.delete:
        if orphans:
            print("\nNothing deleted. Pass --delete to remove them.\n")
        return 0
    stats = organizer.delete_orphans(root, orphans)
    removed = organizer.prune_empty_dirs(root)
    print(f"\nDeleted {stats['deleted']} file(s), {human(stats['bytes'])}, "
          f"removed {removed} empty folder(s).\n")
    return 0


def cmd_folders(args) -> int:
    """Analyse, preview, or adopt the existing folder hierarchy."""
    from atlas import folders

    conn = db.connect()
    report = folders.analyse(conn)

    if args.action == "analyse":
        print(f"\nFolder structure — {config.library_root()}\n")
        print(f"  Models                {report['total_models']}")
        print(f"  Loose at root         {report['models_at_root']}")
        print(f"  Deepest nesting       {report['max_depth']} level(s)")
        print(f"  Distinct leaf folders {report['distinct_leaf_folders']}")
        print(f"  Single-model folders  {report['single_model_folders']}")
        print(f"  Recommended depth     {report['recommended_depth']}")
        print("\n  Models per nesting depth:")
        for depth, count in report["depth_histogram"].items():
            print(f"    depth {depth}: {count}")
        print("\n  Per level, if adopted:")
        for level in report["levels"]:
            print(f"    depth {level['depth']}: {level['usable_folders']} collection(s), "
                  f"{level['models_covered']} model(s)")
        print("\n  Largest folders:")
        for name, count in report["largest_folders"][:10]:
            print(f"    {count:>5}  {name}")
        print()
        return 0

    depth = args.depth or report["recommended_depth"]
    preview = folders.preview_import(conn, max_depth=depth, min_models=args.min_models)

    if args.action == "preview":
        print(f"\nWould create/update {preview['total_collections']} collection(s) "
              f"at depth {depth}, {preview['total_assignments']} assignment(s):\n")
        for entry in preview["collections"][:40]:
            mark = "update" if entry["exists"] else "new   "
            print(f"  [{mark}] {entry['models']:>5}  {entry['name']}")
        if preview["total_collections"] > 40:
            print(f"  ... and {preview['total_collections'] - 40} more")
        print("\n  Nothing changed. Run `folders adopt` to apply.\n")
        return 0

    print(f"\nAdopting folder structure at depth {depth}. No files will be moved.\n")
    stats = folders.import_as_collections(
        conn, None, max_depth=depth, min_models=args.min_models, prune=args.prune)
    for key, value in stats.items():
        print(f"  {key:<22} {value}")
    print()
    return 0


def cmd_doctor(args) -> int:
    """Check the environment and report anything that will bite later."""
    import sqlite3

    ok = True
    print(f"\n{config.APP_NAME} {config.VERSION} — environment check\n")
    print(f"  Python              {sys.version.split()[0]}")
    print(f"  SQLite              {sqlite3.sqlite_version}")

    probe = sqlite3.connect(":memory:")
    has_fts = bool(probe.execute(
        "SELECT 1 FROM pragma_compile_options WHERE compile_options LIKE '%FTS5%'").fetchone())
    probe.close()
    print(f"  FTS5                {'yes' if has_fts else 'NO — search falls back to LIKE'}")
    if not has_fts:
        ok = False

    print(f"  Pillow              {'yes' if thumbnails.available() else 'NO — thumbnails disabled'}")

    try:
        root = config.library_root()
        print(f"  Library root        {root}")
        if not root.is_dir():
            print("                      ^ does not exist")
            ok = False
        conn = db.connect()
        print(f"  Database            {config.db_path()}")
        journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        print(f"  Journal mode        {journal}")
        s = query.stats(conn)
        print(f"  Indexed models      {s['total']}")
    except config.ConfigError as exc:
        print(f"  Library root        NOT SET — {exc}")
        ok = False

    from atlas import providers
    for provider in providers.presentations():
        source = provider["key_source"]
        note = {"env": "from environment", "file": "from atlas.json (prefer an env var)",
                "unset": "not configured"}[source]
        print(f"  {provider['display_name']:<18}  {note}")

    print(f"\n  {'All good.' if ok else 'Issues found above.'}\n")
    return 0 if ok else 1


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py", description=f"{config.APP_NAME} {config.VERSION}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="set the library root")
    p.add_argument("path")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("serve", help="start the web UI")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("scan", help="reconcile the database with the filesystem")
    p.set_defaults(func=cmd_scan)

    p = sub.add_parser("sync", help="scan, read sidecars, resolve metadata, fetch previews")
    p.add_argument("--quick", action="store_true", help="offline stages only")
    p.set_defaults(func=cmd_sync)

    p = sub.add_parser("hash", help="hash every unhashed model file")
    p.set_defaults(func=cmd_hash)

    p = sub.add_parser("thumbs", help="pre-generate thumbnails")
    p.set_defaults(func=cmd_thumbs)

    p = sub.add_parser("stats", help="library health summary")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("search", help="full-text search")
    p.add_argument("term")
    p.add_argument("--limit", type=int, default=25)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("duplicates", help="list duplicate model files")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_duplicates)

    p = sub.add_parser("backup", help="export / inspect / restore a backup")
    p.add_argument("action", choices=["export", "inspect", "restore"])
    p.add_argument("path", nargs="?")
    p.add_argument("--favourites", action="store_true")
    p.set_defaults(func=cmd_backup)

    p = sub.add_parser("setup", help="run the first-time setup wizard again")
    p.set_defaults(func=lambda a: __import__(
        "atlas.onboarding", fromlist=["onboarding"]).run_wizard())

    p = sub.add_parser("organise", help="move collection members into collection folders")
    p.add_argument("--preview", action="store_true", help="show the plan and stop")
    p.add_argument("--no-group-versions", action="store_true", dest="no_group")
    p.set_defaults(func=cmd_organise)

    p = sub.add_parser("cleanup", help="find leftover sidecars and empty folders")
    p.add_argument("--delete", action="store_true", help="actually remove them")
    p.set_defaults(func=cmd_cleanup)

    p = sub.add_parser("folders", help="analyse / preview / adopt your folder hierarchy")
    p.add_argument("action", choices=["analyse", "preview", "adopt"])
    p.add_argument("--depth", type=int, default=0, help="0 = use the recommended depth")
    p.add_argument("--min-models", type=int, default=1, dest="min_models")
    p.add_argument("--prune", action="store_true",
                   help="also remove members that no longer match the folder tree")
    p.set_defaults(func=cmd_folders)

    p = sub.add_parser("doctor", help="check the environment")
    p.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    raw = sys.argv[1:] if argv is None else argv

    # Bare `python run.py` is the beginner path: set up if needed, then serve.
    # Nobody should have to read documentation to open their own library.
    if not raw:
        from atlas import onboarding

        if not onboarding.is_configured():
            onboarding.run_wizard()
        return cmd_serve(argparse.Namespace(port=None, no_browser=False))

    args = build_parser().parse_args(raw)
    try:
        return args.func(args)
    except config.ConfigError as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
