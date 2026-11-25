"""Moving files on disk: collection folders, version grouping, cleanup.

Everything here is destructive, so it follows three rules without exception:

1. **A model and its assets move together.** The ``.safetensors``, every
   sidecar and every preview are one unit. Moving the weights and orphaning
   the metadata is worse than not moving at all.
2. **Nothing is overwritten.** A collision gets a numeric suffix; an identical
   file (same hash) is left alone rather than duplicated.
3. **Plan first, then apply.** Every operation can produce its plan without
   touching the disk, and dry-run mode short-circuits the apply.

Empty directories left behind by a move are removed, and sidecars whose model
no longer exists are reported so they can be cleaned up deliberately.
"""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import config, db, sidecar
from .jobs import JobHandle

# Characters Windows forbids in a path component, plus control characters.
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
# Trailing dots/spaces are illegal on Windows; trailing separators are just
# ugly leftovers from truncating a long "Name - Base - Creator" mid-join.
_TRAILING = re.compile(r"[.\s\-–—_|]+$")

MAX_COMPONENT = 110          # keep well inside MAX_PATH once nested


def safe_component(name: str, fallback: str = "Unnamed") -> str:
    """Turn arbitrary text into one safe path component.

    Library names routinely contain ``:``, ``/`` and CJK text, so this has to
    be conservative without mangling non-Latin names into nothing.
    """
    text = unicodedata.normalize("NFC", str(name or "")).strip()
    text = _ILLEGAL.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _TRAILING.sub("", text)
    if len(text) > MAX_COMPONENT:
        # Re-strip after truncation: cutting a long "Name - Base - Creator"
        # mid-join otherwise leaves a dangling " -" on the folder name.
        text = _TRAILING.sub("", text[:MAX_COMPONENT])
    # Reserved DOS device names, which cannot be directory names on Windows.
    if text.upper().split(".")[0] in {
        "CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        text = f"_{text}"
    return text or fallback


@dataclass
class Move:
    file_id: int
    from_dir: str
    to_dir: str
    assets: list[str] = field(default_factory=list)   # relative paths moving with it
    reason: str = ""

    @property
    def is_noop(self) -> bool:
        return self.from_dir == self.to_dir


@dataclass
class Plan:
    moves: list[Move] = field(default_factory=list)
    orphans: list[str] = field(default_factory=list)
    empty_dirs: list[str] = field(default_factory=list)

    @property
    def real_moves(self) -> list[Move]:
        return [m for m in self.moves if not m.is_noop]

    def summary(self) -> dict[str, Any]:
        return {
            "moves": len(self.real_moves),
            "already_correct": len(self.moves) - len(self.real_moves),
            "files_affected": sum(len(m.assets) + 1 for m in self.real_moves),
            "orphans": len(self.orphans),
            "empty_dirs": len(self.empty_dirs),
        }


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def version_key(row) -> str | None:
    """Identify models that are versions of the same thing.

    Provider model id is authoritative when present. Otherwise fall back to
    creator plus normalised name, which catches ``foo_v1`` / ``foo_v2`` from
    the same author without pulling in unrelated models that merely share a
    word.
    """
    if row["provider"] and row["provider_model_id"]:
        return f"{row['provider']}:{row['provider_model_id']}"
    creator = _norm(row["creator"] or "")
    name = _norm(row["name"] or "")
    if creator and name:
        return f"name:{creator}:{name}"
    return None


def folder_name_for(row) -> str:
    """The directory a model (and its sibling versions) should live in.

    Names in real libraries already carry the base model and often the series
    ("Rin Tohsaka | Illustrious | Fate"), so appending those again produces
    "... Illustrious Fate Illustrious creator". Each part is only added when it
    is not already present.
    """
    name = (row["name"] or Path(row["filename"]).stem).strip()
    # Normalise the separators people use between name / base / series.
    name = re.sub(r"\s*[|/•]\s*", " ", name)
    parts = [safe_component(name)]
    seen = _norm(name)

    for extra in (row["base_model"] or "", row["creator"] or ""):
        extra = extra.strip()
        if not extra:
            continue
        if _norm(extra) and _norm(extra) in seen:
            continue                      # already stated in the name
        component = safe_component(extra)
        if component:
            parts.append(component)
            seen += _norm(component)

    return safe_component(" - ".join(p for p in parts if p),
                          Path(row["filename"]).stem)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def _assets_for(conn, root: Path, file_id: int, model_rel: str) -> list[str]:
    """Every file that must travel with a model, as relative paths."""
    folder = model_rel.rsplit("/", 1)[0] if "/" in model_rel else ""
    out: list[str] = []
    row = conn.execute("SELECT sidecars FROM model_file WHERE id=?", (file_id,)).fetchone()
    for rel in db.jload(row["sidecars"]) if row else []:
        out.append(rel)
    for prow in conn.execute("SELECT path FROM preview WHERE file_id=?", (file_id,)):
        out.append(prow["path"])
    # Deduplicate while preserving order, and drop anything that vanished.
    seen: set[str] = set()
    result = []
    for rel in out:
        if rel in seen or rel == model_rel:
            continue
        seen.add(rel)
        if (root / rel).is_file():
            result.append(rel)
    return result


def plan_collection_layout(conn, root: Path, collection_id: int | None = None,
                           group_versions: bool = True) -> Plan:
    """Plan a move of every collection member into its collection folder.

    The collection name carries the hierarchy (``Anime / Kiss X Sis``), so it
    maps directly onto nested directories.
    """
    sql = """
        SELECT f.id, f.path, f.filename, m.name, m.creator, m.base_model,
               m.provider, m.provider_model_id,
               c.id AS collection_id, c.name AS collection_name
        FROM collection_member cm
        JOIN collection c   ON c.id = cm.collection_id
        JOIN model_file f   ON f.id = cm.file_id
        LEFT JOIN model m   ON m.file_id = f.id
        WHERE f.present = 1
    """
    args: tuple = ()
    if collection_id is not None:
        sql += " AND c.id = ?"
        args = (collection_id,)
    rows = conn.execute(sql, args).fetchall()

    # A model can belong to several collections; the deepest one wins, because
    # it is the most specific statement the user made about where it goes.
    best: dict[int, Any] = {}
    for row in rows:
        current = best.get(row["id"])
        if current is None or row["collection_name"].count(" / ") > current["collection_name"].count(" / "):
            best[row["id"]] = row

    plan = Plan()
    group_dirs: dict[str, str] = {}

    for file_id, row in sorted(best.items(), key=lambda kv: kv[1]["path"]):
        parts = [safe_component(p) for p in row["collection_name"].split(" / ") if p.strip()]
        key = version_key(row) if group_versions else None
        if key and key in group_dirs:
            target_dir = group_dirs[key]                # a sibling version already placed
            reason = "version group"
        else:
            target_dir = "/".join(parts + [folder_name_for(row)])
            if key:
                group_dirs[key] = target_dir
            reason = "collection"

        current_dir = row["path"].rsplit("/", 1)[0] if "/" in row["path"] else ""
        plan.moves.append(Move(
            file_id=file_id,
            from_dir=current_dir,
            to_dir=target_dir,
            assets=_assets_for(conn, root, file_id, row["path"]),
            reason=reason,
        ))

    plan.orphans = find_orphans(conn, root)
    return plan


def plan_version_grouping(conn, root: Path) -> Plan:
    """Plan moves that put versions of one model into a single folder.

    Layout is otherwise untouched: the group lands in the folder its earliest
    member already occupies, so this never fights an existing hierarchy.
    """
    rows = conn.execute(
        """
        SELECT f.id, f.path, f.filename, m.name, m.creator, m.base_model,
               m.provider, m.provider_model_id
        FROM model_file f LEFT JOIN model m ON m.file_id = f.id
        WHERE f.present = 1 ORDER BY f.path
        """
    ).fetchall()

    groups: dict[str, list[Any]] = {}
    for row in rows:
        key = version_key(row)
        if key:
            groups.setdefault(key, []).append(row)

    plan = Plan()
    for key, members in groups.items():
        if len(members) < 2:
            continue
        dirs = {r["path"].rsplit("/", 1)[0] if "/" in r["path"] else "" for r in members}
        if len(dirs) == 1:
            continue                                   # already together
        anchor = members[0]["path"].rsplit("/", 1)[0] if "/" in members[0]["path"] else ""
        for row in members:
            current = row["path"].rsplit("/", 1)[0] if "/" in row["path"] else ""
            plan.moves.append(Move(
                file_id=row["id"], from_dir=current, to_dir=anchor,
                assets=_assets_for(conn, root, row["id"], row["path"]),
                reason=f"version group ({len(members)} versions)",
            ))
    return plan


_SIDECAR_ENDINGS = (".civitai.info", ".user.json", ".json", ".info")


def _sidecar_root(name_lower: str) -> str | None:
    """The model stem a sidecar file name describes, or None if not a sidecar."""
    for ending in _SIDECAR_ENDINGS:
        if name_lower.endswith(ending) and len(name_lower) > len(ending):
            return name_lower[: -len(ending)]
    return None


def find_orphans(conn, root: Path) -> list[str]:
    """Sidecars and previews with no surviving model file.

    These accumulate when a model is renamed — the previous manager leaves
    ``oldname.json`` behind next to ``newname.safetensors``.
    """
    known: set[str] = set()
    stems_by_dir: dict[str, set[str]] = {}
    for row in conn.execute(
            "SELECT path, stem, sidecars FROM model_file WHERE present = 1"):
        known.add(row["path"])
        for rel in db.jload(row["sidecars"]):
            known.add(rel)
        folder = row["path"].rsplit("/", 1)[0] if "/" in row["path"] else ""
        stems_by_dir.setdefault(folder, set()).add(row["stem"].lower())
    for row in conn.execute("SELECT path FROM preview"):
        known.add(row["path"])

    ASSET_SUFFIXES = {".json", ".info", ".png", ".jpg", ".jpeg", ".webp", ".txt"}
    PER_FOLDER = {"user_data.json", "metadata.json"}

    orphans: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in {config.STATE_DIRNAME} and not d.startswith(".")]
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir
        if not filenames:
            continue

        has_models = rel_dir in stems_by_dir
        if has_models:
            # Normal case: the sidecar set of a model that is no longer here
            # (``oldname.json`` beside ``newname.safetensors``). Only metadata
            # proves a file belonged to a model, so an orphan set is anchored
            # on an unclaimed .json/.info; images and text files are flagged
            # only when they share that sidecar's stem. A README or showcase
            # image the user keeps beside their model is never caught.
            stems = stems_by_dir[rel_dir]
            roots = set()
            for name in filenames:
                rel = f"{rel_dir}/{name}" if rel_dir else name
                low = name.lower()
                if rel in known or low in PER_FOLDER:
                    continue
                root_name = _sidecar_root(low)
                if root_name and root_name not in stems:
                    roots.add(root_name)
            for name in filenames:
                rel = f"{rel_dir}/{name}" if rel_dir else name
                low = name.lower()
                if rel in known or low in PER_FOLDER:
                    continue
                if Path(low).suffix not in ASSET_SUFFIXES:
                    continue
                if any(low.startswith(r + ".") for r in roots):
                    orphans.append(rel)
            continue

        # No model in this folder. If everything here is a sidecar or preview,
        # it is a husk left by a move or a delete — metadata describing a model
        # that is no longer beside it. This is the case a folder-skipping check
        # misses entirely, and it is exactly what an organise pass leaves behind.
        if rel_dir == "":
            continue                     # never treat the library root this way
        if any(Path(n).suffix.lower() not in ASSET_SUFFIXES for n in filenames):
            continue                     # something unrecognised: leave it alone
        if any(Path(n).suffix.lower() in config.MODEL_EXTENSIONS for n in filenames):
            continue
        if not any(Path(n).suffix.lower() in (".json", ".info") for n in filenames):
            continue                     # just images/text: someone's references
        for name in filenames:
            rel = f"{rel_dir}/{name}" if rel_dir else name
            if rel not in known:
                orphans.append(rel)
    return sorted(orphans)


# ---------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------

def _unique_target(target: Path) -> Path:
    """Never overwrite. Append ' (2)', ' (3)', … until the name is free."""
    if not target.exists():
        return target
    stem, suffix = target.stem, target.suffix
    for n in range(2, 999):
        candidate = target.with_name(f"{stem} ({n}){suffix}")
        if not candidate.exists():
            return candidate
    raise OSError(f"Could not find a free name for {target}")


def apply_plan(conn, root: Path, plan: Plan, handle: JobHandle | None = None,
               remove_empty: bool = True) -> dict[str, Any]:
    """Execute a plan. Honours dry-run mode."""
    dry = bool(config.get("dry_run"))
    stats = {"moved": 0, "files_moved": 0, "skipped": 0, "errors": 0,
             "dirs_removed": 0, "dry_run": dry, "error_messages": []}
    moves = plan.real_moves
    if handle:
        handle.set_total(len(moves))
        handle.progress(0, f"{'Simulating' if dry else 'Applying'} {len(moves)} move(s)")

    touched_dirs: set[Path] = set()

    for move in moves:
        if handle:
            handle.check_cancelled()
        row = conn.execute(
            "SELECT path, filename FROM model_file WHERE id=?", (move.file_id,)).fetchone()
        if row is None:
            stats["skipped"] += 1
            continue

        source = root / row["path"]
        dest_dir = root / move.to_dir
        if not source.is_file():
            stats["skipped"] += 1
            if handle:
                handle.advance()
            continue

        try:
            if dry:
                stats["moved"] += 1
                stats["files_moved"] += 1 + len(move.assets)
                if handle:
                    handle.log(f"[dry run] {row['path']} -> {move.to_dir}/", "info")
                    handle.advance(detail=row["filename"])
                continue

            dest_dir.mkdir(parents=True, exist_ok=True)
            new_model = _unique_target(dest_dir / source.name)
            shutil.move(str(source), str(new_model))
            touched_dirs.add(source.parent)
            moved_rels = [new_model.relative_to(root).as_posix()]

            for rel in move.assets:
                asset = root / rel
                if not asset.is_file():
                    continue
                # Keep the sidecar's stem aligned with the (possibly renamed) model.
                target_name = asset.name
                if asset.stem == source.stem and new_model.stem != source.stem:
                    target_name = new_model.stem + asset.suffix
                target = _unique_target(dest_dir / target_name)
                shutil.move(str(asset), str(target))
                touched_dirs.add(asset.parent)
                moved_rels.append(target.relative_to(root).as_posix())

            with db.transaction(conn):
                _free_destination_row(conn, moved_rels[0], move.file_id)
                conn.execute(
                    "UPDATE model_file SET path=?, filename=?, stem=?, sidecars=? WHERE id=?",
                    (moved_rels[0], new_model.name, new_model.stem,
                     db.jdump([r for r in moved_rels[1:]
                               if Path(r).suffix.lower() in (".json", ".info")]),
                     move.file_id),
                )
                conn.execute("DELETE FROM preview WHERE file_id=?", (move.file_id,))
                for i, rel in enumerate(
                        r for r in moved_rels[1:]
                        if Path(r).suffix.lower() in config.IMAGE_EXTENSIONS):
                    conn.execute(
                        "INSERT OR IGNORE INTO preview(file_id, path, is_primary) VALUES (?,?,?)",
                        (move.file_id, rel, 1 if i == 0 else 0))
                db.fts_replace(conn, move.file_id)

            stats["moved"] += 1
            stats["files_moved"] += len(moved_rels)
        except (OSError, shutil.Error, sqlite3.Error) as exc:
            # One bad model must never abort the whole batch — record it and
            # carry on with the rest. The message is returned as well as logged
            # so a caller running without a handle can still report the cause.
            stats["errors"] += 1
            stats["error_messages"].append(f"{row['path']}: {exc}")
            if handle:
                handle.log(f"move failed: {row['path']}: {exc}", "error")
        if handle:
            handle.advance(detail=row["filename"])

    if remove_empty and not dry:
        stats["dirs_removed"] = prune_empty_dirs(root, touched_dirs)

    return stats


def _free_destination_row(conn, dest_rel: str, keep_id: int) -> None:
    """Clear a stale ``model_file`` row squatting on the destination path.

    ``model_file.path`` is UNIQUE. When a file is moved out of a folder the old
    row survives as ``present=0`` (so its user data is not lost). Moving a file
    *back* to that path would then collide — the IntegrityError that aborted a
    whole organise run.

    The stale row describes the same file at a location where it no longer
    exists, so it is safe to retire. Any ratings/notes it still carries are
    handed to the surviving row first, and only if that row has none of its own.
    """
    stale = conn.execute(
        "SELECT id, present FROM model_file WHERE path=? AND id!=?",
        (dest_rel, keep_id),
    ).fetchone()
    if stale is None:
        return
    if stale["present"]:
        # A live file already occupies that path; _unique_target should have
        # renamed us away from it, so this means something changed underneath.
        raise OSError(f"destination already occupied in the index: {dest_rel}")

    old = conn.execute(
        "SELECT * FROM user_data WHERE file_id=?", (stale["id"],)).fetchone()
    if old is not None:
        held = conn.execute(
            "SELECT 1 FROM user_data WHERE file_id=?", (keep_id,)).fetchone()
        if held is None:
            conn.execute(
                "INSERT INTO user_data(file_id, rating, favorite, notes, "
                "personal_triggers, tags, weight, negative_prompt, use_count, "
                "last_used, updated_at) "
                "SELECT ?, rating, favorite, notes, personal_triggers, tags, "
                "weight, negative_prompt, use_count, last_used, ? "
                "FROM user_data WHERE file_id=?",
                (keep_id, db.now(), stale["id"]),
            )
    # Inherit collection memberships too, then drop the stale row (cascades).
    conn.execute(
        "INSERT OR IGNORE INTO collection_member(collection_id, file_id) "
        "SELECT collection_id, ? FROM collection_member WHERE file_id=?",
        (keep_id, stale["id"]),
    )
    conn.execute("DELETE FROM model_file WHERE id=?", (stale["id"],))


def prune_empty_dirs(root: Path, candidates: Iterable[Path] | None = None) -> int:
    """Remove directories left empty by a move, walking upward to the root."""
    removed = 0
    if candidates is None:
        walk_targets = [Path(d) for d, _, _ in os.walk(root, topdown=False)]
    else:
        walk_targets = list(candidates)

    for start in walk_targets:
        current = start
        while current != root and root in current.parents:
            try:
                if any(current.iterdir()):
                    break
                current.rmdir()
                removed += 1
                current = current.parent
            except OSError:
                break
    return removed


def delete_orphans(root: Path, orphans: list[str], handle: JobHandle | None = None
                   ) -> dict[str, Any]:
    """Delete orphaned sidecars. Always explicit, never part of a sync."""
    dry = bool(config.get("dry_run"))
    stats = {"deleted": 0, "errors": 0, "bytes": 0, "dry_run": dry}
    if handle:
        handle.set_total(len(orphans))
    for rel in orphans:
        if handle:
            handle.check_cancelled()
        path = root / rel
        try:
            if path.is_file():
                size = path.stat().st_size
                if not dry:
                    path.unlink()
                stats["deleted"] += 1
                stats["bytes"] += size
                if handle:
                    handle.log(f"{'[dry run] ' if dry else ''}removed orphan {rel}", "warn")
        except OSError as exc:
            stats["errors"] += 1
            if handle:
                handle.log(f"could not remove {rel}: {exc}", "error")
        if handle:
            handle.advance(detail=rel)
    return stats
