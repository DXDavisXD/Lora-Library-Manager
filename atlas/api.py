"""JSON API handlers.

Handlers receive a :class:`Request` and return a JSON-serialisable object.
Raising :class:`ApiError` produces a structured error response. No handler
touches the socket — that belongs to `server.py`.

Note what is absent: there is no endpoint that returns an API key. Secrets are
reported as ``configured`` / ``key_source`` only, and the check lives in
``config.public_settings()`` so a newly added setting cannot leak by being put
in the wrong dictionary.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

from . import backup, config, db, events, hashing, providers, query, tasks, thumbnails
from .jobs import QUEUE


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class Request:
    def __init__(self, method: str, path: str, query_params: dict[str, Any],
                 body: dict[str, Any], segments: list[str]):
        self.method = method
        self.path = path
        self.query = query_params
        self.body = body
        self.segments = segments

    @property
    def conn(self):
        return db.connect()

    @property
    def root(self) -> Path:
        return config.library_root()

    def param(self, name: str, default: Any = None) -> Any:
        if name in self.body:
            return self.body[name]
        return self.query.get(name, default)

    def int_param(self, name: str, default: int | None = None) -> int | None:
        value = self.param(name, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def bool_param(self, name: str, default: bool = False) -> bool:
        value = self.param(name, None)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def bootstrap(req: Request) -> dict[str, Any]:
    """Everything the UI needs on first paint, in one round trip."""
    conn = req.conn
    return {
        "app": {"name": config.APP_NAME, "version": config.VERSION},
        "library_root": str(req.root),
        "stats": query.stats(conn),
        "facets": query.facets(conn),
        "settings": config.public_settings(),
        "providers": providers.presentations(),
        "jobs": QUEUE.state(),
        "history": QUEUE.history(20),
        "actions": [
            {"id": key, "label": spec["label"], "priority": spec["priority"],
             "group": spec.get("group", "maintenance"), "desc": spec.get("desc", ""),
             "help": spec.get("help", spec.get("desc", "")),
             "primary": spec.get("primary", False)}
            for key, spec in tasks.ACTIONS.items()
        ],
        "action_groups": [{"id": gid, "label": label} for gid, label in tasks.ACTION_GROUPS],
        "thumbnails": thumbnails.available(),
        "sorts": sorted(query.SORTS.keys()),
    }


def list_models(req: Request) -> dict[str, Any]:
    # Numeric filters (min_rating, collection) are parsed inside the query
    # builder; a non-number there is the caller's mistake, not a server fault.
    for name in ("min_rating", "collection"):
        value = req.query.get(name)
        if value not in (None, "") and not str(value).strip().lstrip("-").isdigit():
            raise ApiError(400, f"{name} must be an integer")
    return query.list_models(req.conn, req.query)


def get_model(req: Request) -> dict[str, Any]:
    file_id = _segment_id(req, 2)
    data = query.get_model(req.conn, file_id)
    if data is None:
        raise ApiError(404, f"No model with id {file_id}")
    return data


def update_user_data(req: Request) -> dict[str, Any]:
    file_id = _segment_id(req, 2)
    conn = req.conn
    if not conn.execute("SELECT 1 FROM model_file WHERE id=?", (file_id,)).fetchone():
        raise ApiError(404, f"No model with id {file_id}")

    allowed = {"rating", "favorite", "notes", "personal_triggers", "tags",
               "weight", "negative_prompt"}
    updates = {k: v for k, v in req.body.items() if k in allowed}
    if not updates:
        raise ApiError(400, f"No writable fields supplied. Allowed: {sorted(allowed)}")

    if "rating" in updates and updates["rating"] is not None:
        try:
            rating = int(updates["rating"])
        except (TypeError, ValueError):
            raise ApiError(400, "rating must be an integer 0-5 or null") from None
        if not 0 <= rating <= 5:
            raise ApiError(400, "rating must be between 0 and 5")
        updates["rating"] = rating or None

    with db.transaction(conn):
        conn.execute(
            "INSERT INTO user_data(file_id, updated_at) VALUES (?,?) "
            "ON CONFLICT(file_id) DO NOTHING",
            (file_id, db.now()),
        )
        sets, args = [], []
        for key, value in updates.items():
            sets.append(f"{key}=?")
            if key in ("personal_triggers", "tags"):
                args.append(db.jdump(value if isinstance(value, list) else []))
            elif key == "favorite":
                args.append(int(bool(value)))
            else:
                args.append(value)
        sets.append("updated_at=?")
        args.extend([db.now(), file_id])
        conn.execute(f"UPDATE user_data SET {', '.join(sets)} WHERE file_id=?", args)
        db.fts_replace(conn, file_id)

    # Mirror the edit back to the on-disk user_data.json so the file stays the
    # source of truth (survives a re-index, and the old manager can read it).
    _write_user_data_file(conn, file_id)

    events.publish("model", {"id": file_id})
    return query.get_model(conn, file_id) or {}


def _write_user_data_file(conn, file_id: int) -> None:
    """Persist a model's user data to its folder's user_data.json."""
    if config.get("dry_run"):
        return
    from . import sidecar

    row = conn.execute(
        "SELECT f.path, u.rating, u.notes, u.personal_triggers, u.weight, "
        "u.negative_prompt, u.favorite FROM model_file f "
        "JOIN user_data u ON u.file_id = f.id WHERE f.id = ?", (file_id,)).fetchone()
    if row is None:
        return
    collections = [r["name"] for r in conn.execute(
        "SELECT c.name FROM collection c JOIN collection_member cm ON cm.collection_id=c.id "
        "WHERE cm.file_id=? ORDER BY c.name", (file_id,))]
    try:
        sidecar.write_legacy_user_data(config.library_root() / row["path"], {
            "rating": row["rating"],
            "notes": row["notes"] or "",
            "personal_triggers": db.jload(row["personal_triggers"]),
            "weight": row["weight"],
            "negative_prompt": row["negative_prompt"] or "",
            "favorite": bool(row["favorite"]),
            "collections": collections,
        })
    except OSError:
        pass


def mark_used(req: Request) -> dict[str, Any]:
    file_id = _segment_id(req, 2)
    conn = req.conn
    with db.transaction(conn):
        conn.execute(
            "INSERT INTO user_data(file_id, use_count, last_used, updated_at) VALUES (?,1,?,?) "
            "ON CONFLICT(file_id) DO UPDATE SET use_count=user_data.use_count+1, "
            "last_used=excluded.last_used, updated_at=excluded.updated_at",
            (file_id, db.now(), db.now()),
        )
    return {"ok": True}


def stats(req: Request) -> dict[str, Any]:
    return query.stats(req.conn)


def facets(req: Request) -> dict[str, Any]:
    return query.facets(req.conn)


def analyse_folders(req: Request) -> dict[str, Any]:
    """Report on the existing folder tree. Read-only."""
    from . import folders
    return folders.analyse(req.conn)


def preview_folder_import(req: Request) -> dict[str, Any]:
    """Show exactly which collections an adopt would create, before it runs."""
    from . import folders
    report = folders.analyse(req.conn)
    depth = req.int_param("max_depth", 0) or report["recommended_depth"]
    return folders.preview_import(
        req.conn, max_depth=int(depth),
        min_models=req.int_param("min_models", 1) or 1,
        skip_noise=req.bool_param("skip_noise", True),
    )


def adopt_folders(req: Request) -> dict[str, Any]:
    from . import tasks as task_mod
    job = task_mod.QUEUE.submit(
        "adopt-folders", "Adopt Folder Structure",
        lambda handle: task_mod.adopt_folders(
            handle,
            max_depth=req.int_param("max_depth", 0) or 0,
            min_models=req.int_param("min_models", 1) or 1,
            skip_noise=req.bool_param("skip_noise", True),
            prune=req.bool_param("prune", False),
        ),
        priority="normal", coalesce=False,
    )
    return {"job": job.snapshot()}


def duplicates(req: Request) -> dict[str, Any]:
    groups = hashing.find_duplicates(req.conn)
    return {
        "groups": groups[:200],
        "total_groups": len(groups),
        "reclaimable": sum(g["reclaimable"] for g in groups),
        "unhashed": hashing.pending_count(req.conn),
    }


# -- collections ------------------------------------------------------------

def collections(req: Request) -> dict[str, Any]:
    return {"collections": query.facets(req.conn)["collections"]}


def create_collection(req: Request) -> dict[str, Any]:
    name = str(req.param("name", "")).strip()
    if not name:
        raise ApiError(400, "A collection needs a name")
    conn = req.conn
    # Default sort_order to the nesting depth so "Anime" sorts above
    # "Anime / Fate", matching how folder adoption records hierarchy.
    sort_order = req.int_param("sort_order", None)
    if sort_order is None:
        sort_order = name.count(" / ")
    with db.transaction(conn):
        conn.execute(
            "INSERT OR IGNORE INTO collection(name, color, sort_order, created_at) "
            "VALUES (?,?,?,?)",
            (name, req.param("color"), sort_order, db.now()),
        )
    row = conn.execute("SELECT id, name, color FROM collection WHERE name=?", (name,)).fetchone()
    events.publish("collections", {})
    return dict(row) if row else {}


def delete_collection(req: Request) -> dict[str, Any]:
    cid = _segment_id(req, 2)
    conn = req.conn
    with db.transaction(conn):
        conn.execute("DELETE FROM collection WHERE id=?", (cid,))
    events.publish("collections", {})
    return {"ok": True, "deleted": cid}


def update_membership(req: Request) -> dict[str, Any]:
    """Add or remove collection members, optionally moving the files too."""
    cid = _segment_id(req, 2)
    file_ids = req.param("file_ids") or []
    if not isinstance(file_ids, list):
        raise ApiError(400, "file_ids must be a list")
    try:
        file_ids = [int(fid) for fid in file_ids]
    except (TypeError, ValueError):
        raise ApiError(400, "file_ids must be integers") from None
    add = req.bool_param("add", True)
    conn = req.conn
    with db.transaction(conn):
        for fid in file_ids:
            if add:
                conn.execute(
                    "INSERT OR IGNORE INTO collection_member(collection_id, file_id) VALUES (?,?)",
                    (cid, int(fid)),
                )
            else:
                conn.execute(
                    "DELETE FROM collection_member WHERE collection_id=? AND file_id=?",
                    (cid, int(fid)),
                )
    events.publish("collections", {})

    result: dict[str, Any] = {"ok": True, "count": len(file_ids)}
    if add and req.bool_param("move_files", False):
        from . import tasks as task_mod
        job = task_mod.QUEUE.submit(
            "organise", "Move Into Collection Folder",
            lambda handle: task_mod.organise_collections(handle, collection_id=cid),
            priority="interactive", coalesce=False)
        result["job"] = job.snapshot()
    return result


def organise_preview(req: Request) -> dict[str, Any]:
    """Show the moves a collection layout would make. Touches nothing."""
    from . import organizer
    plan = organizer.plan_collection_layout(
        req.conn, req.root,
        collection_id=req.int_param("collection_id", None),
        group_versions=req.bool_param("group_versions", True))
    return {
        **plan.summary(),
        "sample": [
            {"from": m.from_dir, "to": m.to_dir, "assets": len(m.assets), "why": m.reason}
            for m in plan.real_moves[:60]
        ],
        "orphans": plan.orphans[:60],
    }


# -- jobs -------------------------------------------------------------------

def job_state(req: Request) -> dict[str, Any]:
    return {**QUEUE.state(), "history": QUEUE.history(req.int_param("limit", 25) or 25)}


def run_action(req: Request) -> dict[str, Any]:
    action = req.segments[2] if len(req.segments) > 2 else ""
    if action not in tasks.ACTIONS:
        raise ApiError(404, f"Unknown action '{action}'. Known: {sorted(tasks.ACTIONS)}")
    job = tasks.submit(action)
    return {"job": job.snapshot()}


def cancel_job(req: Request) -> dict[str, Any]:
    job_id = req.segments[2] if len(req.segments) > 2 else ""
    if not QUEUE.cancel(job_id):
        raise ApiError(404, f"Job {job_id} is not cancellable")
    return {"ok": True}


def cancel_all(req: Request) -> dict[str, Any]:
    return {"cancelled": QUEUE.cancel_all()}


# -- settings ---------------------------------------------------------------

def get_settings(req: Request) -> dict[str, Any]:
    return {
        "settings": config.public_settings(),
        "providers": providers.presentations(),
        "thumbnail_cache": dict(zip(("files", "bytes"), thumbnails.cache_size())),
    }


def update_settings(req: Request) -> dict[str, Any]:
    writable = {
        "port", "open_browser", "hash_workers", "request_timeout", "provider_rate_limit",
        "download_concurrency", "thumbnail_size", "thumbnail_quality", "page_size",
        "default_sort", "default_weight", "dry_run", "auto_scan_on_start",
        "scan_follow_symlinks", "library_root", "backup_folder",
    }
    updates = {k: v for k, v in req.body.items() if k in writable}

    # Secrets take a separate, explicit path so they can never arrive as a
    # side effect of a general settings PATCH.
    for name, value in (req.body.get("api_keys") or {}).items():
        if name not in providers.names():
            raise ApiError(400, f"Unknown provider '{name}'")
        if config.secret_source(name) == "env":
            raise ApiError(
                409,
                f"{name} key comes from ATLAS_{name.upper()}_API_KEY; unset it to edit here.",
            )
        config.set_secret(name, str(value))

    if updates:
        config.set_many(updates)
    events.publish("settings", config.public_settings())
    return get_settings(req)


def browse_library_root(req: Request) -> dict[str, Any]:
    """Open a native folder picker for the library root.

    Returns the chosen path *without* applying it — the UI shows it and the
    user confirms, because changing the root repoints the whole database.
    """
    path = _native_pick_dir("Choose your LoRA folder")
    result: dict[str, Any] = {"path": path or ""}
    if path:
        from . import onboarding
        count, size = onboarding._count_models(Path(path))
        result.update({"models": count, "bytes": size})
    return result


def _native_pick_dir(title: str) -> str:
    try:
        import tkinter
        from tkinter import filedialog
    except Exception:                                       # noqa: BLE001
        return ""
    try:
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title=title, mustexist=True)
        root.destroy()
        return path or ""
    except Exception:                                       # noqa: BLE001 - headless
        return ""


def set_provider_key(req: Request) -> dict[str, Any]:
    """Add or replace one provider's API key.

    Kept separate from the general settings PATCH so a secret can never be
    written as a side effect of saving an unrelated preference.
    """
    name = req.segments[3] if len(req.segments) > 3 else ""
    if name not in providers.names():
        raise ApiError(404, f"Unknown provider '{name}'")
    if config.secret_source(name) == "env":
        raise ApiError(
            409,
            f"{name} key comes from ATLAS_{name.upper()}_API_KEY; "
            f"unset that environment variable to manage it here.")
    key = str(req.param("key", "") or "").strip()
    if key:
        config.set_secret(name, key)
    else:
        config.set_secret(name, "")          # blank clears it
    events.publish("settings", config.public_settings())
    return {"ok": True, "provider": name,
            "configured": config.secret_source(name) != "unset"}


# -- backups ----------------------------------------------------------------

def browse_backup(req: Request) -> dict[str, Any]:
    """Open a native file picker for a .llmb file.

    Runs on the server (which is local), so the dialog appears on the user's
    desktop. Falls back to an empty path if no display / tkinter is available,
    and the UI then asks the user to paste a path instead.
    """
    path = _native_pick_file(
        title="Choose a backup to restore",
        patterns=[("Library backup", f"*{backup.EXTENSION}"), ("All files", "*.*")],
    )
    return {"path": path or ""}


def _native_pick_file(title: str, patterns) -> str:
    try:
        import tkinter
        from tkinter import filedialog
    except Exception:                                       # noqa: BLE001
        return ""
    try:
        root = tkinter.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(title=title, filetypes=patterns)
        root.destroy()
        return path or ""
    except Exception:                                       # noqa: BLE001 - headless
        return ""


def list_backups(req: Request) -> dict[str, Any]:
    folder = backup.backup_folder()
    items = []
    for path in sorted(folder.glob(f"*{backup.EXTENSION}"), reverse=True):
        try:
            st = path.stat()
            items.append({"name": path.name, "path": str(path),
                          "bytes": st.st_size, "modified": st.st_mtime})
        except OSError:
            continue
    return {"backups": items, "folder": str(folder)}


def delete_category(req: Request) -> dict[str, Any]:
    """Delete a category (or sub-category) without losing its models.

    Deleting ``Anime`` removes that collection and every collection nested
    under it, and lifts the models' files out of the folder to the nearest
    surviving level: sub-category folders move up to the library root when the
    parent goes, and a sub-category's models move up into the parent's folder
    (or the root if there is no parent folder). Files are never deleted here.
    """
    from . import organizer, scanner

    name = str(req.param("name", "") or "").strip()
    if not name:
        raise ApiError(400, "name is required")
    conn = req.conn
    root = req.root

    parts = [p.strip() for p in name.split(" / ") if p.strip()]
    parent = " / ".join(parts[:-1])

    # Every collection at or beneath this name.
    like = name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    doomed = conn.execute(
        "SELECT id, name FROM collection WHERE name = ? OR name LIKE ? ESCAPE '\\'",
        (name, f"{like} / %"),
    ).fetchall()
    if not doomed:
        raise ApiError(404, f"No collection named '{name}'")

    ids = [r["id"] for r in doomed]
    marks = ",".join("?" * len(ids))
    members = conn.execute(
        f"SELECT DISTINCT f.id, f.path, f.filename FROM collection_member cm "
        f"JOIN model_file f ON f.id = cm.file_id "
        f"WHERE cm.collection_id IN ({marks}) AND f.present = 1", ids).fetchall()

    # Where the freed models should land: the parent's folder, or the root.
    destination = "/".join(organizer.safe_component(p) for p in parts[:-1]) if parent else ""
    moved = 0
    if req.bool_param("move_files", True) and not config.get("dry_run"):
        moves = []
        for row in members:
            current_dir = row["path"].rsplit("/", 1)[0] if "/" in row["path"] else ""
            # Keep each model's own folder, just re-parent it.
            leaf = current_dir.rsplit("/", 1)[-1] if current_dir else ""
            target = f"{destination}/{leaf}" if destination and leaf else (leaf or destination)
            if target == current_dir:
                continue
            moves.append(organizer.Move(
                file_id=row["id"], from_dir=current_dir, to_dir=target,
                assets=organizer._assets_for(conn, root, row["id"], row["path"]),
                reason="collection deleted"))
        if moves:
            result = organizer.apply_plan(conn, root, organizer.Plan(moves=moves))
            moved = result.get("moved", 0)

    with db.transaction(conn):
        conn.execute(f"DELETE FROM collection WHERE id IN ({marks})", ids)
    if moved:
        organizer.prune_empty_dirs(root)
        scanner.sync(conn, root)

    events.publish("collections", {})
    events.publish("stats", query.stats(conn))
    return {"ok": True, "deleted_collections": [r["name"] for r in doomed],
            "models_affected": len(members), "models_moved": moved,
            "moved_to": destination or "(library root)"}


def rename_category(req: Request) -> dict[str, Any]:
    """Rename a category (or sub-category), folder on disk included.

    Only the last part of the name changes — ``Anime / Kiss X Sis`` renamed to
    ``Kissxsis`` becomes ``Anime / Kissxsis``. Anything nested underneath is
    carried along, and every affected model folder is re-parented so the disk
    matches the new name. Nothing is deleted and no user data is touched.
    """
    from . import organizer, scanner

    name = str(req.param("name", "") or "").strip()
    new_leaf = str(req.param("new_name", "") or "").strip()
    if not name:
        raise ApiError(400, "name is required")
    if not new_leaf:
        raise ApiError(400, "new_name is required")
    if "/" in new_leaf or "\\" in new_leaf:
        raise ApiError(400, "A name cannot contain '/' or '\\' — "
                            "rename one level at a time")
    if organizer.safe_component(new_leaf) != new_leaf.strip():
        # Reserved and path-unsafe characters would silently change the folder
        # name, so say so rather than quietly writing something different.
        raise ApiError(400, f"'{new_leaf}' cannot be used as a folder name")

    conn = req.conn
    root = req.root

    parts = [p.strip() for p in name.split(" / ") if p.strip()]
    if not parts:
        raise ApiError(400, "name is required")
    new_parts = parts[:-1] + [new_leaf]
    new_name = " / ".join(new_parts)
    if new_name == name:
        return {"ok": True, "new_name": new_name, "renamed_collections": 0,
                "models_moved": 0}

    like = name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    affected = conn.execute(
        "SELECT id, name FROM collection WHERE name = ? OR name LIKE ? ESCAPE '\\'",
        (name, f"{like} / %"),
    ).fetchall()
    if not affected:
        raise ApiError(404, f"No category named '{name}'")
    if conn.execute("SELECT 1 FROM collection WHERE name = ?", (new_name,)).fetchone():
        raise ApiError(409, f"A category named '{new_name}' already exists")

    # One prefix swap covers every level: the category folder itself, any
    # sub-category folders below it, and each model's own leaf folder.
    old_dir = "/".join(organizer.safe_component(p) for p in parts)
    new_dir = "/".join(organizer.safe_component(p) for p in new_parts)

    # Everything living in the folder moves, not just the collection's members.
    # Renaming a category is a rename of the folder as far as the user is
    # concerned, and leaving half its contents behind under the old name is the
    # last thing they expect. Models are moved row by row rather than with one
    # directory rename so file ids — and the user data hanging off them —
    # survive intact.
    like = old_dir.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    residents = conn.execute(
        "SELECT id, path FROM model_file WHERE present = 1 AND path LIKE ? ESCAPE '\\'",
        (f"{like}/%",)).fetchall()

    moved = 0
    if req.bool_param("move_files", True) and not config.get("dry_run"):
        moves = []
        for row in residents:
            current = row["path"].rsplit("/", 1)[0] if "/" in row["path"] else ""
            if current != old_dir and not current.startswith(old_dir + "/"):
                continue
            target = new_dir + current[len(old_dir):]
            if target == current:
                continue
            moves.append(organizer.Move(
                file_id=row["id"], from_dir=current, to_dir=target,
                assets=organizer._assets_for(conn, root, row["id"], row["path"]),
                reason="category renamed"))
        if moves:
            moved = organizer.apply_plan(
                conn, root, organizer.Plan(moves=moves)).get("moved", 0)

    with db.transaction(conn):
        for row in affected:
            conn.execute("UPDATE collection SET name = ? WHERE id = ?",
                         (new_name + row["name"][len(name):], row["id"]))
    if moved:
        organizer.prune_empty_dirs(root)
        scanner.sync(conn, root)

    events.publish("collections", {})
    events.publish("stats", query.stats(conn))
    return {"ok": True, "new_name": new_name,
            "renamed_collections": len(affected),
            "models_affected": len(residents), "models_moved": moved}


_GENERIC_ASSET_NAMES = {"preview", "metadata", "cover", "thumbnail", "sample", "user_data"}
_ASSET_SUFFIXES = {".json", ".info", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".txt"}


def delete_model(req: Request) -> dict[str, Any]:
    """Delete one model version and everything belonging to it.

    Removes the weights plus that version's own sidecars and previews. Other
    versions sharing the folder are untouched — only files whose stem matches
    this one are removed, and the folder itself only goes if nothing is left.
    """
    from . import scanner

    file_id = _segment_id(req, 2)
    conn = req.conn
    root = req.root
    row = conn.execute(
        "SELECT path, filename, stem, sidecars FROM model_file WHERE id=?",
        (file_id,)).fetchone()
    if row is None:
        raise ApiError(404, f"No model with id {file_id}")
    if config.get("dry_run"):
        raise ApiError(409, "dry run is on — turn it off to delete files")

    model_path = root / row["path"]
    folder = model_path.parent
    stem = row["stem"]

    # Are there other model versions sharing this folder? That decides whether
    # per-folder files (user_data.json, metadata.json) may be touched.
    siblings = []
    if folder.is_dir():
        siblings = [p for p in folder.iterdir()
                    if p.suffix.lower() in config.MODEL_EXTENSIONS
                    and p.resolve() != model_path.resolve()]

    # Files belonging to *this* version: the weights, plus the sidecars and
    # previews recorded against it. Per-folder files are shared, so they are
    # only removed when this was the last model in the folder.
    shared_names = {"user_data.json", "metadata.json"}
    targets: list[Path] = [model_path]
    for rel in db.jload(row["sidecars"]):
        targets.append(root / rel)
    for prow in conn.execute("SELECT path FROM preview WHERE file_id=?", (file_id,)):
        targets.append(root / prow["path"])

    removed, failed, kept = [], [], []
    seen: set[Path] = set()
    for path in targets:
        if path in seen:
            continue
        seen.add(path)
        if path != model_path and siblings and path.name.lower() in shared_names:
            kept.append(path.name)          # belongs to the other versions too
            continue
        try:
            if path.is_file():
                path.unlink()
                removed.append(path.name)
        except OSError as exc:
            failed.append(f"{path.name}: {exc}")
    folder_removed = False
    if folder.is_dir() and not siblings and folder != root:
        # Sweep up this model's own leftovers the index did not know about —
        # stem-named files and the generic per-model names a folder-per-model
        # layout uses. Anything else is the user's (notes, prompt files, their
        # own images) and stays; the folder only goes if that leaves it empty.
        stem_l = stem.lower()
        for leftover in list(folder.iterdir()):
            if not leftover.is_file():
                continue
            low = leftover.name.lower()
            ours = (low.startswith(stem_l + ".")
                    or low.split(".", 1)[0] in _GENERIC_ASSET_NAMES)
            if not ours or leftover.suffix.lower() not in _ASSET_SUFFIXES:
                kept.append(leftover.name)
                continue
            try:
                leftover.unlink()
                removed.append(leftover.name)
            except OSError as exc:
                failed.append(f"{leftover.name}: {exc}")
        try:
            folder.rmdir()
            folder_removed = True
        except OSError:
            pass

    with db.transaction(conn):
        conn.execute("DELETE FROM model_file WHERE id=?", (file_id,))
    scanner.sync(conn, root)
    events.publish("stats", query.stats(conn))
    events.publish("collections", {})
    return {"ok": True, "removed": removed, "failed": failed, "kept": kept,
            "folder_removed": folder_removed,
            "other_versions_kept": len(siblings)}


def browse_backup_folder(req: Request) -> dict[str, Any]:
    """Native folder picker for where backups are written."""
    return {"path": _native_pick_dir("Choose where to save backups") or ""}


def open_folder(req: Request) -> dict[str, Any]:
    """Reveal a path in the OS file manager.

    Only paths inside the library or the backup folder are allowed — this
    endpoint launches a process, so it must never take an arbitrary target
    from a request.
    """
    import subprocess

    raw = str(req.param("path", "") or "").strip()
    if not raw:
        raise ApiError(400, "path is required")

    allowed_roots = [backup.backup_folder().resolve()]
    library: Path | None = None
    try:
        library = config.library_root().resolve()
        allowed_roots.append(library)
    except config.ConfigError:
        pass

    # Model paths are stored relative to the library root, and that is what the
    # gallery sends. Resolving them as-is would measure them against the
    # server's working directory instead, landing outside the library and
    # tripping the guard below — so anchor relative paths to the library first.
    target = Path(raw).expanduser()
    if not target.is_absolute() and library is not None:
        target = library / raw

    try:
        resolved = target.resolve()
    except OSError:
        raise ApiError(400, "unusable path") from None
    if not any(resolved == r or r in resolved.parents for r in allowed_roots):
        raise ApiError(403, "path is outside the library and backup folders")

    # Reveal the file if it is one; otherwise open the directory.
    folder = resolved if resolved.is_dir() else resolved.parent
    if not folder.exists():
        raise ApiError(404, f"no such folder: {folder}")
    try:
        if os.name == "nt":
            if resolved.is_file():
                subprocess.Popen(["explorer", "/select,", str(resolved)])
            else:
                os.startfile(str(folder))               # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])
    except OSError as exc:
        raise ApiError(500, f"could not open the folder: {exc}") from exc
    return {"ok": True, "opened": str(folder)}


def inspect_backup(req: Request) -> dict[str, Any]:
    path = Path(str(req.param("path", "")))
    if not path.is_file():
        raise ApiError(404, f"No backup at {path}")
    try:
        return backup.inspect(path)
    except (KeyError, OSError, ValueError) as exc:
        raise ApiError(400, f"Unreadable backup: {exc}") from exc


def export_backup(req: Request) -> dict[str, Any]:
    job = tasks.submit_export(
        target=str(req.param("path", "") or ""),
        favourites_only=req.bool_param("favourites_only", False),
        include_previews=req.bool_param("include_previews", True),
    )
    return {"job": job.snapshot()}


def restore_backup(req: Request) -> dict[str, Any]:
    archive = str(req.param("path", "") or "")
    if not archive:
        raise ApiError(400, "path is required")
    job = tasks.submit_restore(
        archive=archive,
        restore_metadata=req.bool_param("restore_metadata", True),
        restore_previews=req.bool_param("restore_previews", True),
        download_missing=req.bool_param("download_missing", False),
    )
    return {"job": job.snapshot()}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _segment_id(req: Request, index: int) -> int:
    try:
        return int(req.segments[index])
    except (IndexError, ValueError):
        raise ApiError(400, "Expected a numeric id in the path") from None


Handler = Callable[[Request], Any]

# (method, segment count, matcher) -> handler. Matching is explicit rather than
# regex-based so the table reads as documentation.
ROUTES: list[tuple[str, tuple[str, ...], Handler]] = [
    ("GET",    ("api", "bootstrap"),                 bootstrap),
    ("GET",    ("api", "models"),                    list_models),
    ("GET",    ("api", "models", "*"),               get_model),
    ("POST",   ("api", "models", "*", "user"),       update_user_data),
    ("POST",   ("api", "models", "*", "used"),       mark_used),
    ("GET",    ("api", "stats"),                     stats),
    ("GET",    ("api", "facets"),                    facets),
    ("GET",    ("api", "duplicates"),                duplicates),
    ("GET",    ("api", "folders"),                   analyse_folders),
    ("POST",   ("api", "folders", "preview"),        preview_folder_import),
    ("POST",   ("api", "folders", "adopt"),          adopt_folders),
    ("POST",   ("api", "organise", "preview"),       organise_preview),
    ("GET",    ("api", "collections"),               collections),
    ("POST",   ("api", "collections"),               create_collection),
    ("DELETE", ("api", "collections", "*"),          delete_collection),
    ("POST",   ("api", "collections", "*", "members"), update_membership),
    ("GET",    ("api", "jobs"),                      job_state),
    ("POST",   ("api", "jobs", "*", "cancel"),       cancel_job),
    ("POST",   ("api", "jobs", "cancel-all"),        cancel_all),
    ("POST",   ("api", "run", "*"),                  run_action),
    ("GET",    ("api", "settings"),                  get_settings),
    ("POST",   ("api", "settings"),                  update_settings),
    ("POST",   ("api", "settings", "key", "*"),      set_provider_key),
    ("POST",   ("api", "settings", "browse-root"),   browse_library_root),
    ("GET",    ("api", "backups"),                   list_backups),
    ("POST",   ("api", "backups", "inspect"),        inspect_backup),
    ("POST",   ("api", "backups", "browse"),         browse_backup),
    ("POST",   ("api", "backups", "browse-folder"),  browse_backup_folder),
    ("POST",   ("api", "open-folder"),               open_folder),
    ("POST",   ("api", "collections", "delete-by-name"), delete_category),
    ("POST",   ("api", "collections", "rename-by-name"), rename_category),
    ("DELETE", ("api", "models", "*"),               delete_model),
    ("POST",   ("api", "backups", "export"),         export_backup),
    ("POST",   ("api", "backups", "restore"),        restore_backup),
]


def resolve(method: str, segments: list[str]) -> Handler | None:
    for route_method, pattern, handler in ROUTES:
        if route_method != method or len(pattern) != len(segments):
            continue
        if all(p == "*" or p == s for p, s in zip(pattern, segments)):
            return handler
    return None


def allowed_methods(segments: list[str]) -> list[str]:
    out = []
    for route_method, pattern, _ in ROUTES:
        if len(pattern) == len(segments) and all(
            p == "*" or p == s for p, s in zip(pattern, segments)
        ):
            out.append(route_method)
    return out
