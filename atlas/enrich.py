"""Turning files into described models.

Three stages, cheapest first — the ordering is the whole point:

1. ``ingest_sidecars``  — free. Reads what is already on disk.
2. ``resolve_online``   — costs a hash plus an API call, so it only runs for
   files stage 1 could not describe.
3. ``fetch_previews``   — costs a download, only for models with no image.

Running stage 1 across the library before stage 2 typically removes most of the
API traffic outright, because a library that has been through any WebUI already
has sidecars for much of it.
"""

from __future__ import annotations

import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from . import config, db, hashing, providers, sidecar
from .jobs import JobHandle
from .providers.base import ModelInfo, ProviderError


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def upsert_model(conn, file_id: int, fields: dict[str, Any], source: str) -> None:
    """Write normalised metadata for a file, preserving anything better we hold.

    A partial refetch must not blank out fields it did not supply, so empty
    incoming values fall back to what is already stored.
    """
    existing = conn.execute("SELECT * FROM model WHERE file_id=?", (file_id,)).fetchone()

    def pick(key: str, default: Any = "") -> Any:
        value = fields.get(key)
        if value in (None, "", [], {}):
            return existing[key] if existing is not None else default
        return value

    trained = fields.get("trained_words") or []
    tags = fields.get("tags") or []
    if existing is not None:
        if not trained:
            trained = db.jload(existing["trained_words"])
        if not tags:
            tags = db.jload(existing["tags"])

    state = sidecar.completeness({**fields, "trained_words": trained})

    conn.execute(
        """
        INSERT INTO model (file_id, provider, provider_model_id, provider_version_id,
                           name, version_name, creator, base_model, description,
                           description_html, trained_words, tags, preferred_weight,
                           negative_prompt, nsfw, meta_state, fetched_at, source)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(file_id) DO UPDATE SET
            provider=excluded.provider,
            provider_model_id=excluded.provider_model_id,
            provider_version_id=excluded.provider_version_id,
            name=excluded.name, version_name=excluded.version_name,
            creator=excluded.creator, base_model=excluded.base_model,
            description=excluded.description,
            description_html=excluded.description_html,
            trained_words=excluded.trained_words,
            tags=excluded.tags, preferred_weight=excluded.preferred_weight,
            negative_prompt=excluded.negative_prompt, nsfw=excluded.nsfw,
            meta_state=excluded.meta_state, fetched_at=excluded.fetched_at,
            source=excluded.source
        """,
        (
            file_id, pick("provider"), pick("provider_model_id"), pick("provider_version_id"),
            pick("name"), pick("version_name"), pick("creator"), pick("base_model"),
            pick("description"), pick("description_html"),
            db.jdump(trained), db.jdump(tags),
            fields.get("preferred_weight") if fields.get("preferred_weight") is not None
            else (existing["preferred_weight"] if existing is not None else None),
            pick("negative_prompt"), int(bool(fields.get("nsfw"))),
            state, db.now(), source,
        ),
    )
    db.fts_replace(conn, file_id)


def info_to_fields(info: ModelInfo) -> dict[str, Any]:
    return {
        "provider": info.provider,
        "provider_model_id": info.provider_model_id,
        "provider_version_id": info.provider_version_id,
        "name": info.name,
        "version_name": info.version_name,
        "creator": info.creator,
        "base_model": info.base_model,
        "description": info.description,
        "description_html": info.description_html,
        "trained_words": info.trained_words,
        "tags": info.tags,
        "preferred_weight": info.preferred_weight,
        "negative_prompt": info.negative_prompt,
        "nsfw": info.nsfw,
        "sha256": info.file_sha256,
    }


# ---------------------------------------------------------------------------
# Stage 1 — sidecars
# ---------------------------------------------------------------------------

def ingest_sidecars(conn, root: Path, handle: JobHandle | None = None,
                    only_unknown: bool = True) -> dict[str, int]:
    # Only files the scanner actually saw a sidecar next to can be described
    # here, so the filesystem is never re-probed and models with no sidecar
    # cost nothing at all.
    sql = ("SELECT id, path, filename, sidecars FROM model_file "
           "WHERE present=1 AND sidecars != '[]'")
    if only_unknown:
        sql += (" AND (id NOT IN (SELECT file_id FROM model) "
                "OR id IN (SELECT file_id FROM model WHERE meta_state != 'complete'))")
    rows = conn.execute(sql).fetchall()

    stats = {"scanned": 0, "described": 0, "repaired": 0, "adopted": 0}
    if handle:
        handle.set_total(len(rows))

    pending: list[tuple[int, dict[str, Any]]] = []
    for row in rows:
        if handle:
            handle.check_cancelled()
        model_path = root / row["path"]
        known = [root / rel for rel in db.jload(row["sidecars"])]
        raw, notes, parsed = sidecar.read_all(model_path, known=known)
        # Lift anything hand-curated into user data before provider results can
        # touch the model row. This is the user's own accumulated work.
        if _adopt_user_authored(conn, row["id"], model_path, parsed):
            stats["adopted"] += 1
        stats["scanned"] += 1
        if notes:
            stats["repaired"] += 1
            if handle:
                for note in notes:
                    handle.log(f"sidecar: {row['filename']} — {note}", "warn")
        if raw:
            pending.append((row["id"], sidecar.normalise(raw, row["filename"])))
            stats["described"] += 1
        if handle:
            handle.advance(detail=row["filename"])
        if len(pending) >= 200:
            _flush_fields(conn, pending, "sidecar")
    _flush_fields(conn, pending, "sidecar")
    return stats


def _adopt_user_authored(conn, file_id: int, model_path: Path,
                         parsed: list[tuple[Path, dict[str, Any]]] | None) -> bool:
    """Copy hand-curated WebUI fields into ``user_data``.

    Trigger words and notes typed by a person are the one thing a library
    cannot regenerate, so they are moved out of the refreshable ``model`` row
    and into user data, which nothing but the user ever writes.

    Existing Atlas-side values always win — this fills gaps, it never
    overwrites. Returns True if anything was adopted.
    """
    authored = sidecar.read_user_authored(model_path, parsed=parsed)
    legacy = sidecar.read_legacy_user_data(model_path, parsed=parsed)

    incoming_triggers = list(authored["triggers"])
    incoming_notes = authored["notes"]
    incoming_weight = authored["weight"]
    incoming_rating = None
    incoming_tags: list[str] = []
    incoming_negative = authored.get("negative", "")
    incoming_favorite = False
    collections: list[str] = []

    if legacy is not None:
        # The previous manager's own file. Its values are the user's, so they
        # rank above anything inferred from a WebUI sidecar.
        incoming_triggers = legacy["personal_triggers"] + incoming_triggers
        incoming_notes = legacy["notes"] or incoming_notes
        incoming_weight = legacy["weight"] if legacy["weight"] is not None else incoming_weight
        incoming_rating = legacy["rating"]
        incoming_tags = legacy["tags"]
        incoming_negative = legacy["negative_prompt"] or incoming_negative
        incoming_favorite = legacy["favorite"]
        collections = legacy["collections"]

    if not any([incoming_triggers, incoming_notes, incoming_tags, collections,
                incoming_rating, incoming_favorite]) and incoming_weight is None:
        return False

    current = conn.execute(
        "SELECT notes, personal_triggers, tags, weight, rating, favorite, negative_prompt "
        "FROM user_data WHERE file_id=?", (file_id,)
    ).fetchone()

    def keep(column: str, incoming, empty=("", None, [])):
        """Existing Atlas-side values always win; adoption only fills gaps."""
        held = current[column] if current is not None else None
        if column in ("personal_triggers", "tags"):
            held = db.jload(held)
        if held not in empty and held is not None and held != []:
            return held
        return incoming

    merged_triggers = db.jload(current["personal_triggers"]) if current else []
    for word in incoming_triggers:
        if word.casefold() not in {t.casefold() for t in merged_triggers}:
            merged_triggers.append(word)

    merged_tags = db.jload(current["tags"]) if current else []
    for tag in incoming_tags:
        if tag.casefold() not in {t.casefold() for t in merged_tags}:
            merged_tags.append(tag)

    values = (
        keep("notes", incoming_notes),
        db.jdump(merged_triggers),
        db.jdump(merged_tags),
        keep("weight", incoming_weight),
        keep("rating", incoming_rating),
        1 if (current["favorite"] if current else 0) or incoming_favorite else 0,
        keep("negative_prompt", incoming_negative),
        db.now(),
    )
    conn.execute(
        "INSERT INTO user_data(file_id, notes, personal_triggers, tags, weight, rating, "
        "favorite, negative_prompt, updated_at) VALUES (?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(file_id) DO UPDATE SET notes=?, personal_triggers=?, tags=?, "
        "weight=?, rating=?, favorite=?, negative_prompt=?, updated_at=?",
        (file_id, *values, *values),
    )

    # Collections recorded by the previous manager, e.g. "Anime / Kiss X Sis".
    for name in collections:
        row = conn.execute("SELECT id FROM collection WHERE name=?", (name,)).fetchone()
        if row is None:
            cur = conn.execute(
                "INSERT INTO collection(name, sort_order, created_at) VALUES (?,?,?)",
                (name, name.count(" / "), db.now()))
            collection_id = int(cur.lastrowid)
        else:
            collection_id = row["id"]
        conn.execute(
            "INSERT OR IGNORE INTO collection_member(collection_id, file_id) VALUES (?,?)",
            (collection_id, file_id))
    return True


def _flush_fields(conn, pending: list[tuple[int, dict[str, Any]]], source: str) -> None:
    if not pending:
        return
    with db.transaction(conn):
        for file_id, fields in pending:
            upsert_model(conn, file_id, fields, source)
            sha = fields.get("sha256")
            if sha:
                # A sidecar hash is worth trusting enough to skip re-reading
                # gigabytes, but mark it so a verify pass can recompute later.
                conn.execute(
                    "UPDATE model_file SET sha256=COALESCE(sha256, ?), "
                    "hash_state=CASE WHEN sha256 IS NULL THEN 'sidecar' ELSE hash_state END "
                    "WHERE id=?",
                    (sha, file_id),
                )
    pending.clear()


# ---------------------------------------------------------------------------
# Stage 2 — providers
# ---------------------------------------------------------------------------

def needs_resolution(conn, force: bool = False) -> list[Any]:
    sql = """
        SELECT f.id, f.path, f.filename, f.sha256, f.hash_state,
               m.provider, m.provider_model_id, m.provider_version_id, m.meta_state
        FROM model_file f
        LEFT JOIN model m ON m.file_id = f.id
        WHERE f.present = 1
    """
    if not force:
        sql += " AND (m.meta_state IS NULL OR m.meta_state != 'complete')"
    sql += " ORDER BY f.size ASC"
    return conn.execute(sql).fetchall()


def resolve_online(conn, root: Path, handle: JobHandle | None = None,
                   force: bool = False) -> dict[str, int]:
    """Resolve metadata from providers for files sidecars could not describe."""
    rows = needs_resolution(conn, force)
    stats = {"attempted": 0, "resolved": 0, "not_found": 0, "errors": 0, "hashed": 0}
    if handle:
        handle.set_total(len(rows))
        handle.progress(0, f"Resolving {len(rows)} model(s)")

    if not any(config.secret_source(p.name) != "unset" for p in providers.all_providers()):
        if handle:
            # Public lookups work unauthenticated, so this is information, not a
            # failure — saying "no key" as a warning made successful runs look broken.
            handle.log("No API key set — using public lookups (fine for most "
                       "models; a key helps with rate limits and restricted ones).",
                       "info")

    for row in rows:
        if handle:
            handle.check_cancelled()
        stats["attempted"] += 1
        info: ModelInfo | None = None

        try:
            # Prefer an ID lookup: it is one request and needs no hashing.
            if row["provider"] and row["provider_version_id"]:
                provider = providers.get(row["provider"])
                if provider:
                    info = provider.fetch_by_id(
                        row["provider_model_id"] or None, row["provider_version_id"]
                    )

            tried: list[str] = []
            if info is None:
                sha = row["sha256"]
                if not sha:
                    if handle:
                        handle.progress(detail=f"hashing {row['filename']}")
                    sha = hashing.ensure_hash(conn, root, row["id"])
                    if sha:
                        stats["hashed"] += 1
                if sha:
                    for provider in providers.hash_lookup_order():
                        tried.append(provider.display_name)
                        info = provider.fetch_by_hash(sha)
                        if info:
                            break

            if info is None:
                stats["not_found"] += 1
                # Say *why* nothing was found, per model. Without this a model
                # that silently gets no metadata looks like a broken run.
                if handle:
                    if not row["sha256"] and not tried:
                        reason = "could not hash the file (unreadable?)"
                    elif tried:
                        reason = (f"no match on {' or '.join(tried)} "
                                  f"— not published there, or it is a private/local model")
                    else:
                        reason = "no provider ids and no hash to look up"
                    handle.log(f"no metadata for {row['filename']}: {reason}", "warn")
            else:
                fields = info_to_fields(info)
                with db.transaction(conn):
                    upsert_model(conn, row["id"], fields, "provider")
                _write_sidecars(root / row["path"], fields, info.raw)
                stats["resolved"] += 1
                if handle:
                    handle.log(
                        f"{row['filename']} -> {info.name or '(unnamed)'}"
                        f" [{info.provider}]", "ok")

        except ProviderError as exc:
            stats["errors"] += 1
            if handle:
                handle.log(f"{row['filename']}: {exc}", "error")
            if exc.status in (401, 403):
                if handle:
                    handle.log("Stopping: credentials rejected.", "error")
                break
        except Exception as exc:                            # noqa: BLE001
            stats["errors"] += 1
            if handle:
                handle.log(f"{row['filename']}: {type(exc).__name__}: {exc}", "error")

        if handle:
            handle.advance(detail=row["filename"])

    return stats


def _write_sidecars(model_path: Path, fields: dict[str, Any],
                    raw: dict[str, Any] | None = None) -> None:
    if config.get("dry_run"):
        return
    try:
        # One rich metadata file per model. The separate .atlas.json is gone.
        sidecar.write_metadata_sidecar(model_path, fields, raw)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Stage 3 — previews
# ---------------------------------------------------------------------------

def fetch_previews(conn, root: Path, handle: JobHandle | None = None,
                   limit: int | None = None) -> dict[str, int]:
    """Download one preview image for models that have none on disk."""
    sql = """
        SELECT f.id, f.path, f.filename, m.provider, m.provider_model_id, m.provider_version_id
        FROM model_file f
        JOIN model m ON m.file_id = f.id
        WHERE f.present = 1
          AND m.provider_version_id IS NOT NULL AND m.provider_version_id != ''
          AND NOT EXISTS (SELECT 1 FROM preview p WHERE p.file_id = f.id)
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql).fetchall()

    stats = {"attempted": 0, "downloaded": 0, "skipped": 0, "errors": 0}
    if handle:
        handle.set_total(len(rows))

    for row in rows:
        if handle:
            handle.check_cancelled()
        stats["attempted"] += 1
        provider = providers.get(row["provider"])
        if provider is None:
            stats["skipped"] += 1
            if handle:
                handle.advance()
            continue
        try:
            info = provider.fetch_by_id(row["provider_model_id"], row["provider_version_id"])
            candidates = _image_candidates(info)
            if not candidates:
                stats["skipped"] += 1
            elif config.get("dry_run"):
                stats["skipped"] += 1
            else:
                model_path = root / row["path"]
                saved = ""
                for url in candidates:
                    suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
                    if suffix not in config.IMAGE_EXTENSIONS:
                        suffix = ".png"
                    target = model_path.with_suffix("").with_suffix(suffix)
                    if not _download(url, target, provider):
                        continue
                    # The URL said image and the provider said image, but a
                    # mislabelled entry would leave a video sitting under a
                    # .png name — present on disk, blank in the gallery. Cheap
                    # enough to confirm from the bytes before recording it.
                    if not _looks_like_image(target):
                        target.unlink(missing_ok=True)
                        if handle:
                            handle.log(f"preview {row['filename']}: "
                                       f"not an image, trying the next one", "warn")
                        continue
                    saved = target.relative_to(root).as_posix()
                    break
                if saved:
                    with db.transaction(conn):
                        conn.execute(
                            "INSERT OR IGNORE INTO preview(file_id, path, is_primary) VALUES (?,?,1)",
                            (row["id"], saved),
                        )
                    stats["downloaded"] += 1
                else:
                    stats["errors"] += 1
        except Exception as exc:                            # noqa: BLE001
            stats["errors"] += 1
            if handle:
                handle.log(f"preview {row['filename']}: {exc}", "warn")
        if handle:
            handle.advance(detail=row["filename"])
    return stats


# Providers list a version's showcase media in one array, and on CivitAI the
# first entry is very often an animation — 12 of this author's own models are
# like that. Taking entry zero downloaded an mp4, and since CivitAI image URLs
# carry no extension it was written out as .png: a file on disk, a blank card
# in the gallery. These are the containers seen in the wild for those entries.
_VIDEO_SUFFIXES = {".mp4", ".webm", ".mov", ".m4v", ".avi", ".mkv"}

# Leading bytes of the formats a browser will actually paint. GIF counts: it
# animates, but it is still an <img>.
_IMAGE_MAGIC = (
    b"\x89PNG\r\n\x1a\n",           # png
    b"\xff\xd8\xff",                # jpeg
    b"GIF87a", b"GIF89a",           # gif
    b"BM",                          # bmp
)


def _image_candidates(info: Any) -> list[str]:
    """Preview URLs worth trying, best first, videos removed.

    Order is the provider's own, which puts the showcase image first — so the
    "next image" is simply the next entry once the videos are dropped.
    """
    if not info or not info.preview_urls:
        return []
    meta = {str(m.get("url")): m for m in (info.preview_meta or [])
            if isinstance(m, dict) and m.get("url")}
    out: list[str] = []
    for url in info.preview_urls:
        url = str(url)
        # CivitAI labels each entry; Arcenciel does not, hence the suffix check
        # as well. Either signal is enough to skip it.
        if str((meta.get(url) or {}).get("type") or "").lower() == "video":
            continue
        if Path(urllib.parse.urlparse(url).path).suffix.lower() in _VIDEO_SUFFIXES:
            continue
        out.append(url)
    return out


def _looks_like_image(path: Path) -> bool:
    """Do the first bytes match a still-image format a browser can render?"""
    try:
        head = path.open("rb").read(16)
    except OSError:
        return False
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return True
    return any(head.startswith(magic) for magic in _IMAGE_MAGIC)


def _download(url: str, target: Path, provider: Any) -> bool:
    """Stream a URL to disk atomically. Returns success."""
    tmp = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": providers.base.USER_AGENT}, method="GET"
    )
    try:
        provider.limiter.acquire()
        with urllib.request.urlopen(request, timeout=float(config.get("request_timeout") or 30)) as response:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as fh:
                shutil.copyfileobj(response, fh, length=1024 * 256)
        tmp.replace(target)
        return True
    except (urllib.error.URLError, OSError, TimeoutError):
        tmp.unlink(missing_ok=True)
        return False
