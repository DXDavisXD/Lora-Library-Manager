"""Portable library backup.

A backup is a zip containing a manifest plus (optionally) preview images. Model
weights are never included — a 1500-LoRA library is measured in hundreds of
gigabytes, and the files are re-downloadable from their provider. What is *not*
re-downloadable is the part this captures: ratings, notes, personal triggers,
collections and which provider version each file actually is.

Entries are keyed by SHA-256, so a restore reattaches correctly even if every
file has been renamed or moved since the export.
"""

from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path
from typing import Any

from . import config, db, hashing, sidecar, thumbnails
from .jobs import JobHandle

FORMAT = "llmb/2"
EXTENSION = ".llmb"

# Previews are stored as thumbnails, not originals. A 1500-model library holds
# several GB of preview PNGs; the same set as ~20 KB WebP is a few tens of MB,
# which is what keeps a whole-library backup a fraction of the library size.
BACKUP_THUMB_SIZE = 384


def export(conn, root: Path, target: Path, handle: JobHandle | None = None,
           include_previews: bool = True, favourites_only: bool = False) -> dict[str, Any]:
    sql = """
        SELECT f.id, f.path, f.filename, f.size, f.sha256, f.mtime,
               m.provider, m.provider_model_id, m.provider_version_id,
               m.name, m.version_name, m.creator, m.base_model, m.description,
               m.trained_words, m.tags, m.preferred_weight, m.negative_prompt, m.nsfw,
               u.rating, u.favorite, u.notes, u.personal_triggers, u.tags AS user_tags,
               u.weight, u.negative_prompt AS user_negative, u.use_count, u.last_used
        FROM model_file f
        LEFT JOIN model m     ON m.file_id = f.id
        LEFT JOIN user_data u ON u.file_id = f.id
        WHERE f.present = 1
    """
    if favourites_only:
        sql += " AND COALESCE(u.favorite,0) = 1"
    rows = conn.execute(sql).fetchall()

    if handle:
        handle.set_total(len(rows) + 1)
        handle.progress(0, f"Exporting {len(rows)} model(s)")

    # A backup keyed by hash is worthless without hashes.
    unhashed = [r["id"] for r in rows if not r["sha256"]]
    if unhashed:
        if handle:
            handle.log(f"Hashing {len(unhashed)} file(s) to key the backup...", "info")
        for file_id in unhashed:
            if handle:
                handle.check_cancelled()
            hashing.ensure_hash(conn, root, file_id)
        rows = conn.execute(sql).fetchall()

    entries: list[dict[str, Any]] = []
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")

    previews_written = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for row in rows:
            if handle:
                handle.check_cancelled()
            entry = {
                "sha256": row["sha256"] or "",
                "path": row["path"],
                "filename": row["filename"],
                "size": row["size"],
                "provider": row["provider"] or "",
                "provider_model_id": row["provider_model_id"] or "",
                "provider_version_id": row["provider_version_id"] or "",
                "name": row["name"] or "",
                "version_name": row["version_name"] or "",
                "creator": row["creator"] or "",
                "base_model": row["base_model"] or "",
                # Carried so a rebuilt library is still described even if the
                # provider is unreachable at restore time.
                "description": row["description"] or "",
                "trained_words": db.jload(row["trained_words"]),
                "tags": db.jload(row["tags"]),
                "preferred_weight": row["preferred_weight"],
                "negative_prompt": row["negative_prompt"] or "",
                "nsfw": bool(row["nsfw"]),
                # Everything below is user-authored and cannot be re-downloaded.
                # This is the part the backup exists to protect.
                "user": {
                    "rating": row["rating"],
                    "favorite": bool(row["favorite"]),
                    "notes": row["notes"] or "",
                    "personal_triggers": db.jload(row["personal_triggers"]),
                    "tags": db.jload(row["user_tags"]),
                    "weight": row["weight"],
                    "negative_prompt": row["user_negative"] or "",
                    "use_count": row["use_count"] or 0,
                    "last_used": row["last_used"],
                },
                "collections": [
                    r["name"] for r in conn.execute(
                        "SELECT c.name FROM collection c JOIN collection_member cm "
                        "ON cm.collection_id=c.id WHERE cm.file_id=? ORDER BY c.name",
                        (row["id"],),
                    )
                ],
                "previews": [],
            }

            if include_previews and row["sha256"]:
                for i, prow in enumerate(conn.execute(
                    "SELECT path FROM preview WHERE file_id=? ORDER BY is_primary DESC, path "
                    "LIMIT 2", (row["id"],)
                )):
                    src = root / prow["path"]
                    if not src.is_file():
                        continue
                    # Prefer a thumbnail: same picture, ~2% of the bytes.
                    thumb = thumbnails.get_or_create(src, BACKUP_THUMB_SIZE)
                    payload = thumb if thumb is not None else src
                    arc = f"previews/{row['sha256'][:2]}/{row['sha256']}_{i}{payload.suffix.lower()}"
                    try:
                        zf.write(payload, arc)
                        entry["previews"].append(arc)
                        previews_written += 1
                    except OSError:
                        pass

            entries.append(entry)
            if handle:
                handle.advance(detail=row["filename"])

        collections = [
            {"name": r["name"], "color": r["color"], "sort_order": r["sort_order"]}
            for r in conn.execute("SELECT name, color, sort_order FROM collection ORDER BY name")
        ]
        manifest = {
            "format": FORMAT,
            "app_version": config.VERSION,
            "created_at": time.time(),
            "library_root": str(root),
            "model_count": len(entries),
            "preview_count": previews_written,
            "favourites_only": favourites_only,
            "collections": collections,
            "models": entries,
        }
        zf.writestr("manifest.json", json.dumps(manifest, indent=1, ensure_ascii=False))

    tmp.replace(target)
    if handle:
        handle.advance()
        handle.progress(message=f"Wrote {target.name}")

    return {
        "path": str(target),
        "models": len(entries),
        "previews": previews_written,
        "bytes": target.stat().st_size,
    }


def _download_missing(conn, root: Path, models: list[dict[str, Any]],
                      handle: JobHandle | None) -> int:
    """Re-download model files the library no longer has.

    The archive deliberately carries no weights — they are large and always
    re-obtainable — so a rebuild uses the provider/model/version ids recorded
    for each entry. Files already on disk are skipped, so this is safe to run
    over a partially-intact library.
    """
    from . import providers

    if config.get("dry_run"):
        if handle:
            handle.log("dry run: skipping model downloads", "warn")
        return 0

    have = {r["sha256"] for r in conn.execute(
        "SELECT sha256 FROM model_file WHERE present=1 AND sha256 IS NOT NULL")}

    todo = []
    for entry in models:
        rel = entry.get("path") or ""
        if not rel:
            continue
        if (entry.get("sha256") or "").lower() in have:
            continue
        if (root / rel).is_file():
            continue
        todo.append(entry)

    if not todo:
        if handle:
            handle.log("every model in the backup is already present", "ok")
        return 0

    total_mb = sum(e.get("size", 0) for e in todo) / (1024 ** 2)
    if handle:
        handle.log(f"{len(todo)} model file(s) missing, ~{total_mb:.0f} MB to download",
                   "info")
        handle.set_total(len(todo))

    fetched = 0
    for index, entry in enumerate(todo, start=1):
        if handle:
            handle.check_cancelled()
        name = entry.get("name") or entry.get("filename") or "?"
        provider = providers.get(entry.get("provider") or "")
        if provider is None or not entry.get("provider_version_id"):
            if handle:
                handle.log(f"cannot re-download {name}: no provider ids recorded", "warn")
                handle.advance()
            continue

        target = root / entry["path"]
        url = provider.download_url(entry.get("provider_model_id") or None,
                                    str(entry["provider_version_id"]))
        if handle:
            handle.progress(index - 1,
                            message=f"Downloading {index}/{len(todo)}",
                            detail=f"{name[:60]} ({entry.get('size', 0) / 1024**2:.0f} MB)")
        try:
            ok = _stream_download(url, target, provider,
                                  expect_sha=(entry.get("sha256") or ""),
                                  handle=handle)
        except Exception as exc:                            # noqa: BLE001
            ok = False
            if handle:
                handle.log(f"download failed for {name}: {type(exc).__name__}: {exc}",
                           "error")
        if ok:
            fetched += 1
            if handle:
                handle.log(f"downloaded {entry.get('filename')}", "ok")
        elif handle:
            handle.log(f"could not download {name} — the model may have been "
                       f"removed from {provider.display_name}, or needs an API key",
                       "warn")
        if handle:
            handle.advance()

    if handle:
        handle.log(f"re-downloaded {fetched} of {len(todo)} missing model(s)",
                   "ok" if fetched else "warn")
    return fetched


def _stream_download(url: str, target: Path, provider: Any, expect_sha: str = "",
                     handle: JobHandle | None = None) -> bool:
    """Stream a model file to disk, verifying its hash before committing."""
    import hashlib
    import urllib.error
    import urllib.request

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    digest = hashlib.sha256()

    request = urllib.request.Request(
        url, headers=provider.download_headers(), method="GET")
    try:
        provider.limiter.acquire()
        with urllib.request.urlopen(request, timeout=120) as response:
            # A provider that wants auth often answers with an HTML login page
            # instead of an error status; refuse anything that is not binary.
            ctype = (response.headers.get("Content-Type") or "").lower()
            if "text/html" in ctype:
                tmp.unlink(missing_ok=True)
                return False
            with open(tmp, "wb") as fh:
                while True:
                    if handle and handle.cancelled:
                        tmp.unlink(missing_ok=True)
                        return False
                    chunk = response.read(1024 * 512)
                    if not chunk:
                        break
                    fh.write(chunk)
                    digest.update(chunk)
    except (urllib.error.URLError, OSError, TimeoutError):
        tmp.unlink(missing_ok=True)
        return False

    if expect_sha and digest.hexdigest().lower() != expect_sha.lower():
        # Wrong bytes: better no file than a silently corrupt one.
        tmp.unlink(missing_ok=True)
        if handle:
            handle.log(f"hash mismatch on {target.name} — discarded", "error")
        return False

    tmp.replace(target)
    return True


def inspect(archive: Path) -> dict[str, Any]:
    """Read a backup's manifest header without extracting anything."""
    with zipfile.ZipFile(archive) as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
    return {
        "format": manifest.get("format"),
        "app_version": manifest.get("app_version"),
        "created_at": manifest.get("created_at"),
        "library_root": manifest.get("library_root"),
        "model_count": manifest.get("model_count", len(manifest.get("models", []))),
        "preview_count": manifest.get("preview_count", 0),
        "favourites_only": manifest.get("favourites_only", False),
        "collections": [c["name"] for c in manifest.get("collections", [])],
    }


def restore(conn, root: Path, archive: Path, handle: JobHandle | None = None,
            restore_metadata: bool = True, restore_previews: bool = True,
            download_missing: bool = False) -> dict[str, Any]:
    """Reattach backed-up state to the current library, matching on SHA-256.

    With ``download_missing`` the backup becomes a full rebuild: any entry with
    no local file is re-downloaded from its provider using the ids recorded in
    the manifest, placed at its original path, and then matched normally. Model
    weights are never stored in the archive — only the pointers needed to fetch
    them again.
    """
    from . import enrich

    with zipfile.ZipFile(archive) as zf:
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        models = manifest.get("models", [])

        if handle:
            handle.set_total(len(models))
            handle.progress(0, f"Restoring {len(models)} entry(s)")
            handle.log(f"backup: {len(models)} model(s), created "
                       f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(manifest.get('created_at', 0)))}",
                       "info")

        # Optionally re-fetch the model files themselves before matching.
        if download_missing:
            fetched = _download_missing(conn, root, models, handle)
            if fetched:
                from . import scanner
                scanner.sync(conn, root)

        # Hash anything unhashed, otherwise matching is impossible.
        missing = [r["id"] for r in conn.execute(
            "SELECT id FROM model_file WHERE present=1 AND sha256 IS NULL")]
        if missing:
            if handle:
                handle.log(f"Hashing {len(missing)} local file(s) to match the backup...", "info")
            for file_id in missing:
                if handle:
                    handle.check_cancelled()
                hashing.ensure_hash(conn, root, file_id)

        by_hash = {
            r["sha256"]: r["id"]
            for r in conn.execute(
                "SELECT id, sha256 FROM model_file WHERE present=1 AND sha256 IS NOT NULL")
        }
        # Fallbacks for models whose hash could not be computed (unreadable file,
        # or a restore into a library that has not finished hashing).
        by_path = {r["path"]: r["id"] for r in conn.execute(
            "SELECT id, path FROM model_file WHERE present=1")}
        by_name = {}
        for r in conn.execute("SELECT id, filename FROM model_file WHERE present=1"):
            by_name.setdefault(r["filename"].lower(), r["id"])

        # Say up front what we are matching against. A restore into an empty
        # library reports "matched 0" and that looks like a failure unless the
        # reason is stated plainly.
        if handle:
            live = len(by_path)
            handle.log(f"library currently holds {live} model file(s) to match against",
                       "info" if live else "warn")
            if not live:
                handle.log("nothing on disk to attach this backup to — re-run with "
                           "“Re-download missing models” to rebuild from scratch",
                           "warn")

        stats = {"matched": 0, "matched_by_hash": 0, "matched_by_path": 0,
                 "matched_by_name": 0, "unmatched": 0, "user_data": 0,
                 "metadata": 0, "previews": 0, "collections": 0,
                 "sidecars_written": 0, "user_files_written": 0,
                 "missing_models": []}

        collection_ids: dict[str, int] = {}
        with db.transaction(conn):
            for coll in manifest.get("collections", []):
                conn.execute(
                    "INSERT OR IGNORE INTO collection(name, color, sort_order, created_at) "
                    "VALUES (?,?,?,?)",
                    (coll["name"], coll.get("color"), coll.get("sort_order", 0), db.now()),
                )
            for r in conn.execute("SELECT id, name FROM collection"):
                collection_ids[r["name"]] = r["id"]
            stats["collections"] = len(collection_ids)

        for entry in models:
            if handle:
                handle.check_cancelled()
            sha = (entry.get("sha256") or "").lower()
            file_id = by_hash.get(sha)
            if file_id is not None:
                stats["matched_by_hash"] += 1
            else:
                file_id = by_path.get(entry.get("path", ""))
                if file_id is not None:
                    stats["matched_by_path"] += 1
                else:
                    file_id = by_name.get(str(entry.get("filename", "")).lower())
                    if file_id is not None:
                        stats["matched_by_name"] += 1

            if file_id is None:
                stats["unmatched"] += 1
                # Record enough to re-download and re-place the model later.
                stats["missing_models"].append({
                    "name": entry.get("name") or entry.get("filename"),
                    "filename": entry.get("filename"),
                    "path": entry.get("path"),
                    "provider": entry.get("provider"),
                    "model_id": entry.get("provider_model_id"),
                    "version_id": entry.get("provider_version_id"),
                    "sha256": sha,
                })
                # Park the user data so it reattaches if the file shows up later.
                if sha and any(entry.get("user", {}).values()):
                    with db.transaction(conn):
                        conn.execute(
                            "INSERT INTO user_data_orphan(sha256, payload, orphaned_at) "
                            "VALUES (?,?,?) ON CONFLICT(sha256) DO UPDATE SET payload=excluded.payload",
                            (sha, db.jdump(entry["user"]), db.now()),
                        )
                if handle:
                    handle.advance()
                continue

            stats["matched"] += 1
            user = entry.get("user") or {}
            with db.transaction(conn):
                # Union rather than replace: a restore must never be able to
                # remove a tag or trigger the user has added since the export.
                held = conn.execute(
                    "SELECT personal_triggers, tags FROM user_data WHERE file_id=?",
                    (file_id,)).fetchone()
                merged_triggers = db.jload(held["personal_triggers"]) if held else []
                for word in user.get("personal_triggers") or []:
                    if word.casefold() not in {t.casefold() for t in merged_triggers}:
                        merged_triggers.append(word)
                merged_tags = db.jload(held["tags"]) if held else []
                for tag in user.get("tags") or []:
                    if tag.casefold() not in {t.casefold() for t in merged_tags}:
                        merged_tags.append(tag)

                conn.execute(
                    "INSERT INTO user_data(file_id, rating, favorite, notes, personal_triggers, "
                    "tags, weight, negative_prompt, use_count, last_used, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(file_id) DO UPDATE SET "
                    "rating=COALESCE(excluded.rating, user_data.rating), "
                    "favorite=MAX(excluded.favorite, user_data.favorite), "
                    "notes=CASE WHEN excluded.notes != '' THEN excluded.notes ELSE user_data.notes END, "
                    "personal_triggers=excluded.personal_triggers, "
                    "tags=excluded.tags, "
                    "weight=COALESCE(excluded.weight, user_data.weight), "
                    "negative_prompt=CASE WHEN excluded.negative_prompt != '' "
                    "  THEN excluded.negative_prompt ELSE user_data.negative_prompt END, "
                    "use_count=MAX(excluded.use_count, user_data.use_count), "
                    "updated_at=excluded.updated_at",
                    (file_id, user.get("rating"), int(bool(user.get("favorite"))),
                     user.get("notes") or "", db.jdump(merged_triggers),
                     db.jdump(merged_tags), user.get("weight"),
                     user.get("negative_prompt") or "", int(user.get("use_count") or 0),
                     user.get("last_used"), db.now()),
                )
                stats["user_data"] += 1

                for name in entry.get("collections", []):
                    cid = collection_ids.get(name)
                    if cid:
                        conn.execute(
                            "INSERT OR IGNORE INTO collection_member(collection_id, file_id) "
                            "VALUES (?,?)", (cid, file_id),
                        )

                if restore_metadata:
                    enrich.upsert_model(conn, file_id, {
                        "provider": entry.get("provider", ""),
                        "provider_model_id": entry.get("provider_model_id", ""),
                        "provider_version_id": entry.get("provider_version_id", ""),
                        "name": entry.get("name", ""),
                        "version_name": entry.get("version_name", ""),
                        "creator": entry.get("creator", ""),
                        "base_model": entry.get("base_model", ""),
                        "trained_words": entry.get("trained_words", []),
                        "tags": entry.get("tags", []),
                        "preferred_weight": entry.get("preferred_weight"),
                        "negative_prompt": entry.get("negative_prompt", ""),
                        "nsfw": entry.get("nsfw", False),
                        "sha256": sha,
                    }, "backup")
                    stats["metadata"] += 1

            # Write the on-disk files too, not just database rows. A rebuilt
            # library must be self-describing — readable by Forge/A1111 and
            # re-importable — without a second metadata download.
            if not config.get("dry_run"):
                row = conn.execute(
                    "SELECT path FROM model_file WHERE id=?", (file_id,)).fetchone()
                if row:
                    model_path = root / row["path"]
                    try:
                        if restore_metadata:
                            sidecar.write_metadata_sidecar(model_path, {
                                "provider": entry.get("provider", ""),
                                "provider_model_id": entry.get("provider_model_id", ""),
                                "provider_version_id": entry.get("provider_version_id", ""),
                                "name": entry.get("name", ""),
                                "version_name": entry.get("version_name", ""),
                                "creator": entry.get("creator", ""),
                                "base_model": entry.get("base_model", ""),
                                "description": entry.get("description", ""),
                                "trained_words": entry.get("trained_words", []),
                                "tags": entry.get("tags", []),
                                "preferred_weight": entry.get("preferred_weight"),
                                "negative_prompt": entry.get("negative_prompt", ""),
                                "sha256": sha,
                            })
                            stats["sidecars_written"] += 1
                        sidecar.write_legacy_user_data(model_path, {
                            "rating": user.get("rating"),
                            "notes": user.get("notes") or "",
                            "personal_triggers": user.get("personal_triggers") or [],
                            "tags": user.get("tags") or [],
                            "weight": user.get("weight"),
                            "negative_prompt": user.get("negative_prompt") or "",
                            "favorite": bool(user.get("favorite")),
                            "collections": entry.get("collections", []),
                        })
                        stats["user_files_written"] += 1
                    except OSError:
                        pass

            if restore_previews and entry.get("previews"):
                row = conn.execute(
                    "SELECT path FROM model_file WHERE id=?", (file_id,)).fetchone()
                model_path = root / row["path"]
                if not list(conn.execute(
                        "SELECT 1 FROM preview WHERE file_id=?", (file_id,))):
                    arc = entry["previews"][0]
                    try:
                        suffix = Path(arc).suffix or ".png"
                        dest = model_path.with_suffix("").with_suffix(suffix)
                        if not dest.exists() and not config.get("dry_run"):
                            dest.write_bytes(zf.read(arc))
                            with db.transaction(conn):
                                conn.execute(
                                    "INSERT OR IGNORE INTO preview(file_id, path, is_primary) "
                                    "VALUES (?,?,1)",
                                    (file_id, dest.relative_to(root).as_posix()),
                                )
                            stats["previews"] += 1
                    except (KeyError, OSError):
                        pass

            with db.transaction(conn):
                db.fts_replace(conn, file_id)
            if handle:
                handle.advance(detail=entry.get("filename", ""))

    if handle:
        handle.log(
            f"matched {stats['matched']} "
            f"(hash {stats['matched_by_hash']}, path {stats['matched_by_path']}, "
            f"name {stats['matched_by_name']}), unmatched {stats['unmatched']}", "info")
        handle.log(
            f"restored user data on {stats['user_data']}, metadata on "
            f"{stats['metadata']}, previews {stats['previews']}, "
            f"collections {stats['collections']}", "ok")
        # Name what could not be attached, so "unmatched N" is actionable.
        for miss in stats["missing_models"][:20]:
            handle.log(f"  not in library: {miss.get('name') or miss.get('filename')}",
                       "warn")
        if len(stats["missing_models"]) > 20:
            handle.log(f"  … and {len(stats['missing_models']) - 20} more", "warn")
        if stats["missing_models"]:
            handle.log("their ratings/notes are parked and reattach automatically "
                       "if those models come back", "info")

    return stats


def backup_folder() -> Path:
    """Where backups are written.

    Defaults to ``<library>/.atlas/backups`` but can be pointed anywhere —
    an external drive is a much better home for a backup than the folder it
    is backing up.
    """
    configured = str(config.get("backup_folder") or "").strip().strip('"').strip("'")
    folder = Path(configured).expanduser() if configured else (
        config.state_dir() / "backups")
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def default_name(favourites_only: bool = False) -> str:
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    tag = "favourites" if favourites_only else "library"
    return f"atlas_{tag}_{stamp}{EXTENSION}"
