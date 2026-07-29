"""Runnable background operations, and the pipelines that compose them.

Each entry in :data:`ACTIONS` is a factory returning a job body. The bodies are
deliberately thin — the work lives in scanner/hashing/enrich/thumbnails — so
that a pipeline like ``full_sync`` is legible as a sequence of stages rather
than a wall of orchestration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from . import (backup, config, db, enrich, events, folders, hashing, organizer,
               query, scanner, sidecar, thumbnails)
from .jobs import JobHandle, QUEUE


def _ctx() -> tuple[Any, Path]:
    return db.connect(), config.library_root()


def _publish_stats() -> None:
    try:
        conn, _ = _ctx()
        events.publish("stats", query.stats(conn))
    except Exception:                                       # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Individual stages
# ---------------------------------------------------------------------------

def scan(handle: JobHandle) -> dict[str, Any]:
    conn, root = _ctx()
    handle.progress(0, "Walking the library...")

    # The walker discovers the true total as it goes, so the bar rescales
    # rather than sitting at an invented 100%.
    seen_total = 0

    def on_progress(current: int, total: int, name: str) -> None:
        nonlocal seen_total
        if total != seen_total:
            seen_total = total
            handle.set_total(total)
        handle.progress(current, detail=name)

    result = scanner.sync(conn, root, progress=on_progress)
    summary = (
        f"{result.added} new, {result.modified} changed, "
        f"{result.missing} missing, {result.unchanged} unchanged "
        f"in {result.duration:.1f}s"
    )
    handle.progress(message=summary)
    handle.log(summary, "ok")
    _publish_stats()
    return result.__dict__


def ingest(handle: JobHandle) -> dict[str, Any]:
    conn, root = _ctx()
    handle.progress(0, "Reading sidecars...")
    stats = enrich.ingest_sidecars(conn, root, handle)
    handle.progress(message=(
        f"Described {stats['described']} of {stats['scanned']} "
        f"({stats['repaired']} needed repair)"
    ))
    _publish_stats()
    return stats


def resolve(handle: JobHandle, force: bool = False) -> dict[str, Any]:
    conn, root = _ctx()
    handle.progress(0, "Resolving metadata from providers...")
    stats = enrich.resolve_online(conn, root, handle, force=force)
    handle.progress(message=(
        f"Resolved {stats['resolved']}, not found {stats['not_found']}, "
        f"errors {stats['errors']}"
    ))
    _publish_stats()
    return stats


def previews(handle: JobHandle) -> dict[str, Any]:
    conn, root = _ctx()
    handle.progress(0, "Downloading missing previews...")
    stats = enrich.fetch_previews(conn, root, handle)
    handle.progress(message=f"Downloaded {stats['downloaded']}, skipped {stats['skipped']}")
    _publish_stats()
    return stats


def hash_all(handle: JobHandle) -> dict[str, Any]:
    conn, root = _ctx()
    pending = hashing.pending_count(conn)
    gb = hashing.pending_bytes(conn) / (1024 ** 3)
    handle.progress(0, f"Hashing {pending} file(s), {gb:.1f} GB")
    handle.set_total(pending)
    done = hashing.hash_pending(
        conn, root, progress=lambda c, t, name: handle.progress(c, detail=name),
        cancel=lambda: handle.cancelled,
    )
    handle.progress(message=f"Hashed {done} file(s)")
    _publish_stats()
    return {"hashed": done}


def thumbs(handle: JobHandle) -> dict[str, Any]:
    conn, root = _ctx()
    handle.progress(0, "Generating thumbnails...")
    stats = thumbnails.warm(conn, root, handle)
    handle.progress(message=(
        f"Generated {stats['generated']}, cached {stats['skipped']}, failed {stats['failed']}"
    ))
    return stats


def duplicates(handle: JobHandle) -> dict[str, Any]:
    conn, root = _ctx()
    pending = hashing.pending_count(conn)
    if pending:
        handle.progress(0, f"Hashing {pending} file(s) first...")
        handle.set_total(pending)
        hashing.hash_pending(
            conn, root, progress=lambda c, t, n: handle.progress(c, detail=n),
            cancel=lambda: handle.cancelled,
        )
    groups = hashing.find_duplicates(conn)
    reclaimable = sum(g["reclaimable"] for g in groups)
    handle.progress(message=(
        f"{len(groups)} duplicate group(s), {reclaimable / (1024**3):.2f} GB reclaimable"
    ))
    events.publish("duplicates", {"groups": groups[:200]})
    return {"groups": len(groups), "reclaimable": reclaimable}


def rebuild_gallery(handle: JobHandle) -> dict[str, Any]:
    """Rebuild everything the gallery reads from, in one pass.

    Offline and non-destructive: rescan the filesystem, re-read every sidecar,
    regenerate thumbnails, and rebuild the search index. This is the "make the
    gallery reflect reality again" button when things look stale.
    """
    conn, root = _ctx()
    out: dict[str, Any] = {}
    steps = [
        ("Scanning files", lambda: scanner.sync(conn, root)),
        ("Re-reading sidecars", lambda: enrich.ingest_sidecars(conn, root, handle, only_unknown=False)),
        ("Rebuilding search index", lambda: _rebuild_fts(conn)),
        ("Generating thumbnails", lambda: thumbnails.warm(conn, root, handle)),
    ]
    for i, (label, fn) in enumerate(steps, 1):
        handle.check_cancelled()
        handle.log(f"[{i}/{len(steps)}] {label}", "info")
        handle.progress(message=f"[{i}/{len(steps)}] {label}")
        out[label] = fn()
    handle.progress(message="Gallery rebuilt")
    events.publish("collections", {})
    _publish_stats()
    return out


def _rebuild_fts(conn) -> dict[str, Any]:
    with db.transaction(conn):
        n = db.fts_rebuild_all(conn)
    return {"indexed": n}


def file_loose_models(handle: JobHandle) -> dict[str, Any]:
    """Give every model sitting loose at the library root its own folder.

    A model is "loose" when its path has no slash in it — nothing has filed it
    yet. Each one moves into a folder named after the model, taking its sidecar
    and previews with it.

    Callers must re-scan first. The move only carries assets the index knows
    about, so metadata written moments earlier by a resolve or preview fetch
    would otherwise be left behind at the old location.
    """
    conn, root = _ctx()
    stats: dict[str, Any] = {"moved_to_folders": 0, "failed_moves": 0}
    if config.get("dry_run"):
        return stats

    loose = conn.execute(
        "SELECT f.id, f.path, f.filename, m.name, m.base_model, m.creator, "
        "m.provider, m.provider_model_id "
        "FROM model_file f LEFT JOIN model m ON m.file_id=f.id "
        "WHERE f.present=1 AND instr(f.path,'/')=0"
    ).fetchall()
    if not loose:
        return stats

    moved = failed = 0
    handle.set_total(len(loose))
    for row in loose:
        handle.check_cancelled()
        folder = organizer.folder_name_for(row)
        assets = organizer._assets_for(conn, root, row["id"], row["path"])
        plan = organizer.Plan(moves=[organizer.Move(
            file_id=row["id"], from_dir="", to_dir=folder,
            assets=assets, reason="onboard loose file")])
        # No handle: a single-move plan must not reset the outer progress
        # bar. Error detail comes back in the stats instead.
        result = organizer.apply_plan(conn, root, plan, handle=None,
                                      remove_empty=False)
        if result.get("errors"):
            failed += 1
            why = "; ".join(result.get("error_messages") or []) or "unknown error"
            handle.log(f"could not file {row['filename']} into “{folder}”: {why}", "error")
        else:
            moved += result.get("moved", 0)
            handle.log(
                f"filed {row['filename']} -> {folder}/"
                + (f" (+{len(assets)} asset file(s))" if assets else " (no metadata yet)"),
                "ok")
        handle.advance(detail=row["filename"])

    stats["empty_dirs_removed"] = organizer.prune_empty_dirs(root)
    scanner.sync(conn, root)
    stats["moved_to_folders"] = moved
    stats["failed_moves"] = failed
    return stats


def process_new(handle: JobHandle) -> dict[str, Any]:
    """Onboard loose / bare model files.

    Finds models that have no metadata yet (typically freshly-dropped
    ``.safetensors`` with no sidecar), downloads their metadata and a preview,
    and — for any sitting loose at the library root — moves them into a folder
    named for the model so they are ready to organise from the gallery.
    Nothing already organised is touched.
    """
    conn, root = _ctx()

    # Scan FIRST. Files dropped since the last scan are not in the index yet,
    # so counting or enriching before this would skip the very models this job
    # exists to onboard — they would end up filed but blank.
    handle.progress(0, "Scanning for new files...")
    found = scanner.sync(conn, root)
    handle.log(f"scan: {found.added} new, {found.modified} changed, "
               f"{found.unchanged} unchanged", "info")

    # Bare = present, no metadata row or 'missing' metadata.
    bare = conn.execute(
        "SELECT f.id, f.path, f.filename FROM model_file f "
        "LEFT JOIN model m ON m.file_id=f.id "
        "WHERE f.present=1 AND COALESCE(m.meta_state,'missing')='missing'"
    ).fetchall()
    handle.log(f"{len(bare)} model(s) need metadata", "info")
    for row in bare[:25]:
        handle.log(f"  • {row['filename']}", "info")
    if len(bare) > 25:
        handle.log(f"  … and {len(bare) - 25} more", "info")

    stats = {"found": len(bare)}
    # 1. Read any sidecars they do have, then resolve the rest online.
    stats["ingested"] = enrich.ingest_sidecars(conn, root, handle)["described"]
    stats["resolve"] = enrich.resolve_online(conn, root, handle)
    stats["previews"] = enrich.fetch_previews(conn, root, handle)

    # 2. Re-scan before moving. Resolve/preview just wrote new .json and image
    #    files, and the move only carries assets the index knows about — without
    #    this the freshly-downloaded metadata gets stranded at the old location.
    handle.progress(message="Indexing downloaded metadata...")
    scanner.sync(conn, root)

    # 3. Move loose root-level models into their own folder.
    filed = file_loose_models(handle)
    stats.update(filed)
    moved, failed = filed["moved_to_folders"], filed["failed_moves"]
    resolved = (stats.get("resolve") or {}).get("resolved", 0)
    handle.progress(message=(
        f"{stats['found']} new model(s): {resolved} got metadata, "
        f"{moved} filed into folders"
        + (f", {failed} failed" if failed else "")))
    _publish_stats()
    return stats


def adopt_folders(handle: JobHandle, max_depth: int = 0, min_models: int = 1,
                  skip_noise: bool = True, prune: bool = False) -> dict[str, Any]:
    """Turn the existing folder tree into collections. Moves nothing."""
    conn, _ = _ctx()
    report = folders.analyse(conn)
    depth = int(max_depth or report["recommended_depth"])
    handle.progress(0, f"Adopting folder structure at depth {depth}")
    handle.log(
        f"{report['total_models']} models, max depth {report['max_depth']}, "
        f"{report['distinct_leaf_folders']} leaf folders; using depth {depth}", "info")
    stats = folders.import_as_collections(
        conn, handle, max_depth=depth, min_models=min_models,
        skip_noise=skip_noise, prune=prune)
    stats["depth_used"] = depth
    events.publish("collections", {})
    _publish_stats()
    return stats


def organise_collections(handle: JobHandle, collection_id: int | None = None,
                         group_versions: bool = True) -> dict[str, Any]:
    """Move each collection's members into a folder named for the collection."""
    conn, root = _ctx()
    handle.progress(0, "Planning moves...")
    plan = organizer.plan_collection_layout(
        conn, root, collection_id=collection_id, group_versions=group_versions)
    summary = plan.summary()
    handle.log(
        f"{summary['moves']} move(s), {summary['already_correct']} already correct, "
        f"{summary['files_affected']} file(s) affected", "info")
    stats = organizer.apply_plan(conn, root, plan, handle)
    scanner.sync(conn, root)                     # re-anchor paths after the move

    # A move can strand sidecars whose model went elsewhere. Detecting that
    # *after* the move is the only way to catch it, so it is part of the job
    # rather than something the user has to remember to run.
    leftovers = organizer.find_orphans(conn, root)
    stats["leftovers"] = len(leftovers)
    if leftovers:
        handle.log(f"{len(leftovers)} leftover file(s) stranded by the move; "
                   f"run Cleanup to remove them.", "warn")
    stats["dirs_removed"] += organizer.prune_empty_dirs(root) if not stats["dry_run"] else 0

    handle.progress(message=(
        f"Moved {stats['moved']} model(s) ({stats['files_moved']} files), "
        f"removed {stats['dirs_removed']} empty folder(s)"
        + (f", {len(leftovers)} leftover(s) found" if leftovers else "")
        + (" [dry run]" if stats["dry_run"] else "")))
    _publish_stats()
    return {**stats, **summary}


def group_versions(handle: JobHandle) -> dict[str, Any]:
    """Put every version of a model into one folder, leaving layout otherwise alone."""
    conn, root = _ctx()
    handle.progress(0, "Finding version groups...")
    plan = organizer.plan_version_grouping(conn, root)
    if not plan.real_moves:
        handle.progress(message="Every version group is already together")
        return {"moved": 0, "groups": 0}
    stats = organizer.apply_plan(conn, root, plan, handle)
    scanner.sync(conn, root)
    handle.progress(message=f"Consolidated {stats['moved']} file(s) into version folders")
    _publish_stats()
    return stats


def cleanup(handle: JobHandle, delete: bool = False) -> dict[str, Any]:
    """Tidy leftovers: legacy .atlas.json files, orphan sidecars, empty folders.

    The legacy files are always retired (they are this tool's own dead output
    and hold nothing the rich sidecar lacks). Orphaned sidecars from other
    tools are only reported unless ``delete`` is set, because those may be
    files the user still wants.
    """
    conn, root = _ctx()
    dry = bool(config.get("dry_run"))

    # 1. Retire the second metadata file an earlier version of this tool wrote.
    handle.progress(0, "Checking for superseded metadata files...")
    legacy = sorted(root.rglob(f"*{sidecar.LEGACY_ATLAS_SUFFIX}"))
    retired = 0
    carried_any = 0
    for path in legacy:
        handle.check_cancelled()
        model_stem = str(path)[: -len(sidecar.LEGACY_ATLAS_SUFFIX)]
        summary = sidecar.retire_legacy_atlas(Path(model_stem), dry_run=dry)
        if summary:
            retired += 1
            if summary["carried"]:
                carried_any += 1
                handle.log(f"merged {', '.join(summary['carried'])} from "
                           f"{path.name} before removing it", "info")
    if legacy:
        handle.log(f"{'would retire' if dry else 'retired'} {retired} superseded "
                   f".atlas.json file(s); {carried_any} had data worth keeping", "ok")

    # 2. Orphaned sidecars/previews with no model beside them.
    handle.progress(message="Scanning for leftovers...")
    orphans = organizer.find_orphans(conn, root)
    result: dict[str, Any] = {"legacy_retired": retired,
                              "orphans_found": len(orphans),
                              "orphans": orphans[:200]}
    if delete and orphans:
        result.update(organizer.delete_orphans(root, orphans, handle))
    removed = organizer.prune_empty_dirs(root) if not dry else 0
    result["empty_dirs_removed"] = removed

    if legacy and not dry:
        scanner.sync(conn, root)             # drop the removed files from the index

    handle.progress(message=(
        f"{retired} superseded metadata file(s) removed, "
        f"{len(orphans)} orphan file(s)"
        + (f", {result.get('deleted', 0)} deleted" if delete else " (kept)")
        + f", {removed} empty folder(s) removed"))
    events.publish("cleanup", result)
    _publish_stats()
    return result


def rebuild_search(handle: JobHandle) -> dict[str, Any]:
    conn, _ = _ctx()
    handle.progress(0, "Rebuilding the search index...")
    with db.transaction(conn):
        count = db.fts_rebuild_all(conn)
    handle.progress(message=f"Reindexed {count} model(s)")
    return {"indexed": count}


def verify_hashes(handle: JobHandle) -> dict[str, Any]:
    """Recompute hashes that were taken on trust from a sidecar."""
    conn, root = _ctx()
    rows = conn.execute(
        "SELECT id, path FROM model_file WHERE present=1 AND hash_state='sidecar'"
    ).fetchall()
    handle.set_total(len(rows))
    handle.progress(0, f"Verifying {len(rows)} sidecar-sourced hash(es)")
    mismatched = 0
    for row in rows:
        handle.check_cancelled()
        stored = conn.execute(
            "SELECT sha256 FROM model_file WHERE id=?", (row["id"],)).fetchone()["sha256"]
        actual = hashing.hash_file(root / row["path"], cancel=lambda: handle.cancelled)
        with db.transaction(conn):
            if actual:
                conn.execute(
                    "UPDATE model_file SET sha256=?, hash_state='done' WHERE id=?",
                    (actual, row["id"]),
                )
                if stored and actual != stored:
                    mismatched += 1
                    handle.log(f"hash mismatch: {row['path']}", "warn")
            else:
                conn.execute("UPDATE model_file SET hash_state='error' WHERE id=?", (row["id"],))
        handle.advance(detail=row["path"].rsplit("/", 1)[-1])
    handle.progress(message=f"Verified {len(rows)}, {mismatched} mismatch(es)")
    return {"verified": len(rows), "mismatched": mismatched}


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

def full_sync(handle: JobHandle) -> dict[str, Any]:
    """The one button most people should ever need.

    Ordered cheapest-first so that a cancel partway through still leaves the
    library strictly better off than before.
    """
    def file_new(h: JobHandle) -> dict[str, Any]:
        # Resolve and previews just wrote sidecars and images next to models
        # that may still be loose at the root. Re-index first or the move
        # carries the model and leaves its brand-new metadata behind.
        scanner.sync(*_ctx())
        return file_loose_models(h)

    out: dict[str, Any] = {}
    stages: list[tuple[str, Callable[[JobHandle], Any]]] = [
        ("Scanning files", scan),
        ("Reading sidecars", ingest),
        ("Resolving metadata", resolve),
        ("Downloading previews", previews),
        ("Filing loose models", file_new),
        ("Generating thumbnails", thumbs),
    ]
    for index, (label, fn) in enumerate(stages, start=1):
        handle.check_cancelled()
        handle.log(f"[{index}/{len(stages)}] {label}", "info")
        handle.progress(message=f"[{index}/{len(stages)}] {label}")
        out[fn.__name__] = fn(handle)
    handle.progress(message="Sync complete")
    return out


def quick_refresh(handle: JobHandle) -> dict[str, Any]:
    """Scan + sidecars only. No network, so it is safe to run constantly."""
    out = {"scan": scan(handle), "ingest": ingest(handle)}
    handle.progress(message="Refresh complete")
    return out


def export_backup(handle: JobHandle, target: str = "", favourites_only: bool = False,
                  include_previews: bool = True) -> dict[str, Any]:
    conn, root = _ctx()
    path = Path(target) if target else (
        backup.backup_folder() / backup.default_name(favourites_only))
    result = backup.export(conn, root, path, handle,
                           include_previews=include_previews,
                           favourites_only=favourites_only)
    handle.progress(message=f"Exported {result['models']} model(s) to {Path(result['path']).name}")
    handle.log(f"Backup written: {result['path']}", "ok")
    return result


def restore_backup(handle: JobHandle, archive: str = "", restore_metadata: bool = True,
                   restore_previews: bool = True,
                   download_missing: bool = False) -> dict[str, Any]:
    conn, root = _ctx()
    path = Path(archive)
    if not path.is_file():
        raise FileNotFoundError(f"Backup not found: {archive}")
    stats = backup.restore(conn, root, path, handle,
                           restore_metadata=restore_metadata,
                           restore_previews=restore_previews,
                           download_missing=download_missing)
    if download_missing:
        scanner.sync(conn, root)
    handle.progress(message=(
        f"Matched {stats['matched']}, unmatched {stats['unmatched']}, "
        f"user data {stats['user_data']}"
    ))
    _publish_stats()
    return stats


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Each action carries three pieces of text:
#   label — the button
#   desc  — the one line printed under it on the card
#   help  — the longer plain-English tooltip, written for someone who has never
#           met the words "hash", "sidecar" or "index"
ACTIONS: dict[str, dict[str, Any]] = {
    "full-sync": {
        "label": "Update Everything", "fn": full_sync, "priority": "bulk",
        "group": "core", "primary": True,
        "desc": "Find new files, get their info, file them into folders.",
        "help": "The one button that does the lot: look for new or changed models, "
                "read any info files already on disk, download missing details and "
                "preview pictures, give any model still sitting loose its own folder, "
                "then make thumbnails. Safe to run any time — your own ratings, notes "
                "and trigger words are never overwritten."},
    "quick-refresh": {
        "label": "Quick Refresh", "fn": quick_refresh, "priority": "interactive",
        "group": "core",
        "desc": "Look for new or changed files. No downloads.",
        "help": "A fast, offline check: notices models you've added, moved or removed "
                "and re-reads the info files sitting next to them. Never touches the "
                "internet, so it's quick and safe to run as often as you like."},
    "process-new": {
        "label": "Process New Files", "fn": process_new, "priority": "bulk",
        "group": "core",
        "desc": "Set up models you just dropped in.",
        "help": "For models you've just added: finds them, downloads their details and "
                "a preview picture, and tidies loose files sitting at the top of your "
                "library into their own folders. Models you've already organised are "
                "left exactly where they are."},
    "rebuild-gallery": {
        "label": "Rebuild Gallery", "fn": rebuild_gallery, "priority": "normal",
        "group": "core",
        "desc": "Redo the gallery from what's on disk. No downloads.",
        "help": "Use this if the gallery looks out of date or wrong. Re-reads every "
                "file and info file on disk, regenerates thumbnails and rebuilds "
                "search. Offline and non-destructive — it only rebuilds what the "
                "gallery displays, never your files."},

    "scan": {
        "label": "Scan Library", "fn": scan, "priority": "interactive",
        "group": "metadata",
        "desc": "Check which model files exist right now.",
        "help": "Walks your folders and notes what's new, what changed and what's "
                "gone missing. Reads nothing else and downloads nothing."},
    "ingest": {
        "label": "Import Metadata Files", "fn": ingest, "priority": "normal",
        "group": "metadata",
        "desc": "Read info files already saved next to your models.",
        "help": "Many models come with a companion file — the same name as the model "
                "but ending .json — holding its description, trigger words and so on. "
                "This reads those into the gallery. Completely offline."},
    "resolve": {
        "label": "Download Metadata", "fn": resolve, "priority": "bulk",
        "group": "metadata",
        "desc": "Fetch details for models that don't have any.",
        "help": "Looks up models that are still missing information on CivitAI and "
                "Arcenciel and downloads their name, description, trigger words and "
                "tags. Anything you typed yourself is kept."},
    "resolve-force": {
        "label": "Refresh All Metadata", "fn": lambda h: resolve(h, force=True),
        "priority": "bulk", "group": "metadata",
        "desc": "Re-download details for every model.",
        "help": "Fetches fresh information for your whole library, not just the models "
                "that are missing it — useful after the site updates a description or "
                "trigger list. Your own ratings, notes and trigger words are never "
                "replaced."},
    "previews": {
        "label": "Download Previews", "fn": previews, "priority": "bulk",
        "group": "metadata",
        "desc": "Get a preview picture for models with none.",
        "help": "Downloads one example image for every model that currently shows a "
                "blank card in the gallery."},

    "adopt-folders": {
        "label": "Adopt Folder Structure", "fn": adopt_folders, "priority": "normal",
        "group": "organize",
        "desc": "Turn your existing folders into collections. Moves nothing.",
        "help": "If you've already sorted models into folders, this reads that layout "
                "and mirrors it as collections you can filter by. Nothing on disk is "
                "moved or renamed — it only learns from the structure you built."},
    "organise": {
        "label": "Apply Collection Folders", "fn": organise_collections, "priority": "normal",
        "group": "organize",
        "desc": "Move each collection's models into a matching folder.",
        "help": "The reverse of the button above: takes your collections and makes them "
                "real folders on disk, moving each model — with its info file and "
                "preview — into the folder named after its collection."},
    "group-versions": {
        "label": "Group Model Versions", "fn": group_versions, "priority": "normal",
        "group": "organize",
        "desc": "Put every version of a model in one folder.",
        "help": "When you have v1 and v2 of the same model in different places, this "
                "brings them together into a single folder so versions of one model "
                "live side by side."},
    "cleanup": {
        "label": "Clean Up Leftovers", "fn": cleanup, "priority": "normal",
        "group": "organize",
        "desc": "Remove stale files left behind by older versions and moves.",
        "help": "Deletes the extra .atlas.json files an earlier version of this app "
                "created, and reports info files or pictures whose model is no longer "
                "beside them, plus any empty folders. Those leftovers are only listed, "
                "never deleted without you asking."},

    "hash": {
        "label": "Fingerprint All Files", "fn": hash_all, "priority": "bulk",
        "group": "maintenance",
        "desc": "Give every model a content fingerprint.",
        "help": "A fingerprint is calculated from a file's actual contents, so it stays "
                "the same if you rename or move the file and changes if the file does. "
                "It's how duplicates are spotted, how models are identified online, and "
                "how a backup finds its way home after you reorganise. Reads every byte "
                "of every model, so it can take a while on a big library."},
    "verify-hashes": {
        "label": "Verify File Integrity", "fn": verify_hashes, "priority": "bulk",
        "group": "maintenance",
        "desc": "Double-check fingerprints that were taken on trust.",
        "help": "Some fingerprints aren't calculated — they're simply believed, because "
                "a model's info file claimed one. This re-calculates those from the real "
                "file and tells you about any that disagree, which usually means the "
                "download was interrupted or the file was replaced."},
    "thumbs": {
        "label": "Generate Thumbnails", "fn": thumbs, "priority": "normal",
        "group": "maintenance",
        "desc": "Pre-make small preview images for smooth scrolling.",
        "help": "Preview pictures are often several megabytes each. This makes tiny "
                "copies up front so the gallery scrolls smoothly instead of loading "
                "full-size images as you go."},
    "duplicates": {
        "label": "Find Duplicates", "fn": duplicates, "priority": "normal",
        "group": "maintenance",
        "desc": "Find identical models and how much space they waste.",
        "help": "Compares fingerprints to find models that are byte-for-byte identical "
                "even under different names, and reports how much disk space you'd get "
                "back. It only reports — nothing is deleted."},
    "rebuild-search": {
        "label": "Rebuild Search Index", "fn": rebuild_search, "priority": "normal",
        "group": "maintenance",
        "desc": "Rebuild the search database.",
        "help": "Recreates the behind-the-scenes list that makes the search box fast. "
                "Use it if searching misses things you know are there."},
}

# Human-readable group headings for the dashboard, in display order.
ACTION_GROUPS = [
    ("core",        "Core workflow"),
    ("metadata",    "Metadata & downloads"),
    ("organize",    "Organization"),
    ("maintenance", "Maintenance & diagnostics"),
]


def submit(action: str, **kwargs: Any):
    """Enqueue a registered action. Extra kwargs are bound into the job body."""
    spec = ACTIONS.get(action)
    if spec is None:
        raise KeyError(action)
    fn = spec["fn"]
    body = (lambda handle: fn(handle, **kwargs)) if kwargs else fn
    return QUEUE.submit(action, spec["label"], body, priority=spec["priority"])


def submit_export(**kwargs: Any):
    return QUEUE.submit(
        "export-backup", "Export Backup",
        lambda handle: export_backup(handle, **kwargs),
        priority="normal", coalesce=False,
    )


def submit_restore(**kwargs: Any):
    return QUEUE.submit(
        "restore-backup", "Restore Backup",
        lambda handle: restore_backup(handle, **kwargs),
        priority="interactive", coalesce=False,
    )
