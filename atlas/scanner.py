"""Filesystem discovery.

Design notes for large libraries:

* One ``os.scandir`` per directory, and siblings are matched in memory. The
  original walked the tree once per concern (models, then previews, then
  sidecars); at 1500 models across a few hundred folders that is three full
  traversals where one will do.
* ``DirEntry.stat()`` reuses the data the OS already returned with the
  directory listing on Windows, so size/mtime cost no extra syscall.
* Change detection uses (size, mtime) only. Content hashing is expensive —
  1500 LoRAs is comfortably 300 GB — so it is deferred to `hashing.py` and
  only performed for files that actually need a provider lookup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from . import config, db

Progress = Callable[[int, int, str], None]

# Directories we never descend into.
SKIP_DIRS = {config.STATE_DIRNAME, ".git", "__pycache__", "$RECYCLE.BIN",
             "System Volume Information", ".trash", ".thumbnails"}

# Suffixes some UIs append to preview images.
PREVIEW_HINTS = ("", ".preview", ".thumb", ".example")


@dataclass
class FoundFile:
    rel_path: str
    size: int
    mtime: float
    previews: list[str] = field(default_factory=list)
    sidecars: list[str] = field(default_factory=list)

    @property
    def filename(self) -> str:
        return self.rel_path.rsplit("/", 1)[-1]

    @property
    def stem(self) -> str:
        return self.filename.rsplit(".", 1)[0]

    @property
    def ext(self) -> str:
        return "." + self.filename.rsplit(".", 1)[-1] if "." in self.filename else ""


@dataclass
class ScanResult:
    added: int = 0
    modified: int = 0
    missing: int = 0
    unchanged: int = 0
    previews_linked: int = 0
    duration: float = 0.0

    @property
    def changed(self) -> int:
        return self.added + self.modified + self.missing


def _rel(root: Path, p: Path) -> str:
    return p.relative_to(root).as_posix()


def walk(root: Path, follow_symlinks: bool = False) -> Iterator[FoundFile]:
    """Yield every model file under *root*, with its sibling assets attached."""
    stack: list[Path] = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except (PermissionError, FileNotFoundError, OSError):
            continue

        models: list[tuple[os.DirEntry, str]] = []
        images: list[str] = []
        jsons: list[str] = []

        for entry in entries:
            name = entry.name
            try:
                if entry.is_dir(follow_symlinks=follow_symlinks):
                    if name not in SKIP_DIRS and not name.startswith("."):
                        stack.append(Path(entry.path))
                    continue
                if not entry.is_file(follow_symlinks=follow_symlinks):
                    continue
            except OSError:
                continue

            lower = name.lower()
            dot = lower.rfind(".")
            if dot < 0:
                continue
            ext = lower[dot:]

            if ext in config.MODEL_EXTENSIONS:
                models.append((entry, name[:dot]))
            elif ext in config.IMAGE_EXTENSIONS:
                images.append(name)
            elif ext == ".json" or lower.endswith(".civitai.info"):
                jsons.append(name)

        if not models:
            continue

        single = len(models) == 1
        for entry, stem in models:
            try:
                st = entry.stat(follow_symlinks=follow_symlinks)
            except OSError:
                continue
            found = FoundFile(
                rel_path=_rel(root, Path(entry.path)),
                size=st.st_size,
                mtime=st.st_mtime,
            )
            found.previews = _match_assets(stem, images, single)
            found.sidecars = _match_assets(stem, jsons, single)
            yield found


def _match_assets(stem: str, candidates: list[str], sole_model: bool) -> list[str]:
    """Pick sibling assets belonging to *stem*.

    Matching is stem-prefixed first (``model.safetensors`` -> ``model.png``,
    ``model.preview.png``). When a folder holds exactly one model we also accept
    generic names like ``preview.png`` — that is the folder-per-model layout the
    original manager produces, and it must keep working.
    """
    stem_l = stem.lower()
    exact: list[str] = []
    generic: list[str] = []
    for name in candidates:
        low = name.lower()
        base = low.rsplit(".", 1)[0]
        if base == stem_l or any(base == stem_l + h for h in PREVIEW_HINTS if h):
            exact.append(name)
        elif base.startswith(stem_l + "."):
            exact.append(name)
        elif sole_model and base in ("preview", "metadata", "cover", "thumbnail",
                                     "sample", "user_data"):
            generic.append(name)
    # user_data.json is the previous manager's per-model user file. It has to
    # come through even when a stem-named sidecar already matched, otherwise
    # the ratings and collections in it are silently ignored.
    for name in candidates:
        if name.lower() == "user_data.json" and name not in exact:
            generic.append(name)
    return _dedupe_keep_order(exact + generic) if exact else _dedupe_keep_order(generic)


def _dedupe_keep_order(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            out.append(name)
    return out


def sync(conn, root: Path, progress: Progress | None = None) -> ScanResult:
    """Reconcile the database against the filesystem.

    Returns counts rather than the rows themselves: at 1500+ models the caller
    almost never wants the full list, and materialising it wastes memory.
    """
    import time

    started = time.time()
    result = ScanResult()
    follow = bool(config.get("scan_follow_symlinks"))

    existing = {
        r["path"]: (r["id"], r["size"], r["mtime"])
        for r in conn.execute("SELECT id, path, size, mtime FROM model_file")
    }
    seen: set[str] = set()
    ts = db.now()

    batch: list[FoundFile] = []
    added_ids: list[int] = []
    total_hint = len(existing) or 1

    def flush(items: list[FoundFile]) -> None:
        if not items:
            return
        with db.transaction(conn):
            for f in items:
                prev = existing.get(f.rel_path)
                folder = f.rel_path.rsplit("/", 1)[0] if "/" in f.rel_path else ""
                sidecar_rel = db.jdump(
                    [(f"{folder}/{n}" if folder else n) for n in f.sidecars]
                )
                if prev is None:
                    cur = conn.execute(
                        "INSERT INTO model_file(path, filename, stem, ext, size, mtime, "
                        "hash_state, sidecars, present, first_seen, last_seen) "
                        "VALUES (?,?,?,?,?,?,'pending',?,1,?,?)",
                        (f.rel_path, f.filename, f.stem, f.ext, f.size, f.mtime,
                         sidecar_rel, ts, ts),
                    )
                    fid = int(cur.lastrowid)
                    result.added += 1
                    added_ids.append(fid)
                else:
                    fid, old_size, old_mtime = prev
                    if old_size != f.size or abs(old_mtime - f.mtime) > 1e-6:
                        # Content changed: the old hash and metadata are suspect.
                        conn.execute(
                            "UPDATE model_file SET size=?, mtime=?, sha256=NULL, "
                            "hash_state='pending', sidecars=?, present=1, last_seen=? "
                            "WHERE id=?",
                            (f.size, f.mtime, sidecar_rel, ts, fid),
                        )
                        conn.execute(
                            "UPDATE model SET meta_state='partial' WHERE file_id=?", (fid,)
                        )
                        result.modified += 1
                    else:
                        conn.execute(
                            "UPDATE model_file SET present=1, sidecars=?, last_seen=? "
                            "WHERE id=?",
                            (sidecar_rel, ts, fid),
                        )
                        result.unchanged += 1
                result.previews_linked += _sync_previews(conn, fid, f)
                db.fts_replace(conn, fid)
        items.clear()

    for found in walk(root, follow):
        seen.add(found.rel_path)
        batch.append(found)
        if len(batch) >= 250:
            flush(batch)
            if progress:
                progress(len(seen), max(total_hint, len(seen)), found.rel_path)
    flush(batch)

    gone = [path for path in existing if path not in seen]
    if gone:
        with db.transaction(conn):
            for chunk_start in range(0, len(gone), 500):
                chunk = gone[chunk_start:chunk_start + 500]
                marks = ",".join("?" * len(chunk))
                conn.execute(
                    f"UPDATE model_file SET present=0, last_seen=? WHERE path IN ({marks})",
                    (ts, *chunk),
                )
        result.missing = len(gone)

    # A file that vanished and a file that appeared in the same scan is usually
    # one file renamed or moved outside Atlas. Carry its user data across now,
    # while (size, mtime) still identify it — a rename keeps both, and waiting
    # for a hash would mean waiting for a pass that may never hash this file.
    if result.added and gone:
        with db.transaction(conn):
            for fid in added_ids:
                reattach_moved(conn, fid)

    with db.transaction(conn):
        db.meta_set(conn, "last_scan", str(ts))
    result.duration = time.time() - started
    if progress:
        progress(len(seen), len(seen), "complete")
    return result


def _sync_previews(conn, file_id: int, found: FoundFile) -> int:
    """Replace the preview rows for a file. Cheap: a handful of rows each."""
    folder = found.rel_path.rsplit("/", 1)[0] if "/" in found.rel_path else ""
    wanted = [(f"{folder}/{n}" if folder else n) for n in found.previews]
    have = {r["path"] for r in conn.execute(
        "SELECT path FROM preview WHERE file_id=?", (file_id,))}
    if set(wanted) == have:
        return 0
    conn.execute("DELETE FROM preview WHERE file_id=?", (file_id,))
    for i, path in enumerate(sorted(wanted)):
        conn.execute(
            "INSERT OR IGNORE INTO preview(file_id, path, is_primary) VALUES (?,?,?)",
            (file_id, path, 1 if i == 0 else 0),
        )
    return len(wanted)


_USER_COLUMNS = ("rating, favorite, notes, personal_triggers, tags, weight, "
                 "negative_prompt, use_count, last_used")


def reattach_moved(conn, file_id: int) -> bool:
    """Give a newly seen file the user data of the missing file it replaced.

    Run inside a transaction. A missing row is the same file when its content
    hash matches, or — before either side is hashed — when size and mtime
    both match exactly, which is what a rename or same-drive move preserves.
    An ambiguous (size, mtime) match is skipped rather than guessed at.
    Returns whether anything was carried across.
    """
    new = conn.execute(
        "SELECT size, mtime, sha256 FROM model_file WHERE id=?", (file_id,)).fetchone()
    if new is None:
        return False
    if conn.execute("SELECT 1 FROM user_data WHERE file_id=?", (file_id,)).fetchone():
        return False                  # never overwrite data the new row already has

    candidates = []
    if new["sha256"]:
        candidates = conn.execute(
            "SELECT f.id FROM model_file f JOIN user_data u ON u.file_id=f.id "
            "WHERE f.present=0 AND f.sha256=? AND f.id!=?",
            (new["sha256"], file_id)).fetchall()
    if not candidates:
        rows = conn.execute(
            "SELECT f.id, f.mtime, f.sha256 FROM model_file f "
            "JOIN user_data u ON u.file_id=f.id "
            "WHERE f.present=0 AND f.size=? AND f.id!=?",
            (new["size"], file_id)).fetchall()
        candidates = [
            r for r in rows
            if abs(r["mtime"] - new["mtime"]) < 1e-6
            and not (r["sha256"] and new["sha256"] and r["sha256"] != new["sha256"])
        ]
    if len(candidates) != 1:
        if not new["sha256"]:
            return False
        _reattach_user_data(conn, file_id)          # archived by a backup restore
        return False

    old_id = candidates[0]["id"]
    conn.execute(
        f"INSERT INTO user_data(file_id, {_USER_COLUMNS}, updated_at) "
        f"SELECT ?, {_USER_COLUMNS}, ? FROM user_data WHERE file_id=?",
        (file_id, db.now(), old_id))
    conn.execute(
        "INSERT OR IGNORE INTO collection_member(collection_id, file_id) "
        "SELECT collection_id, ? FROM collection_member WHERE file_id=?",
        (file_id, old_id))
    # Detach from the old row so the data cannot be claimed a second time.
    conn.execute("DELETE FROM user_data WHERE file_id=?", (old_id,))
    conn.execute("DELETE FROM collection_member WHERE file_id=?", (old_id,))
    db.fts_replace(conn, file_id)
    return True


def _reattach_user_data(conn, file_id: int) -> None:
    """Restore user data archived by content hash (a backup restore parks
    entries here when their file is not present yet).

    Only possible once the file has been hashed; the hashing pass calls
    :func:`reattach_moved`, which falls through to this.
    """
    row = conn.execute("SELECT sha256 FROM model_file WHERE id=?", (file_id,)).fetchone()
    if not row or not row["sha256"]:
        return
    orphan = conn.execute(
        "SELECT payload FROM user_data_orphan WHERE sha256=?", (row["sha256"],)
    ).fetchone()
    if not orphan:
        return
    data = db.jload(orphan["payload"], {})
    if not isinstance(data, dict):
        return
    conn.execute(
        "INSERT INTO user_data(file_id, rating, favorite, notes, personal_triggers, "
        "weight, use_count, last_used, updated_at) VALUES (?,?,?,?,?,?,?,?,?) "
        "ON CONFLICT(file_id) DO NOTHING",
        (
            file_id, data.get("rating"), int(data.get("favorite", 0)), data.get("notes"),
            db.jdump(data.get("personal_triggers") or []), data.get("weight"),
            int(data.get("use_count", 0)), data.get("last_used"), db.now(),
        ),
    )
    conn.execute("DELETE FROM user_data_orphan WHERE sha256=?", (row["sha256"],))
