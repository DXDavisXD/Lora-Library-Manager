"""Adopting an existing folder hierarchy.

If you have spent years arranging LoRAs into ``Anime/Fate/Rin/`` then that tree
*is* your metadata, and a manager that insists on its own layout is asking you
to throw it away.

Atlas never moves a file. Instead it reads the hierarchy you already have and
mirrors it as nested collections, so the structure becomes queryable while the
folders stay exactly where they are. Re-running is safe: membership is
reconciled, not duplicated.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from . import db
from .jobs import JobHandle

# Folder names that describe a file rather than a category, and would only add
# noise as collections.
NOISE = {
    "lora", "loras", "models", "new", "misc", "other", "others", "unsorted",
    "temp", "tmp", "downloads", "download", "_", "-", "untitled", "default",
}


def _segments(path: str) -> list[str]:
    parts = path.split("/")[:-1]           # drop the filename
    return [p for p in parts if p.strip()]


def analyse(conn) -> dict[str, Any]:
    """Describe the current folder tree without changing anything.

    This is the report you read *before* importing, so you can see what the
    import would create and pick a sensible depth.
    """
    rows = conn.execute(
        "SELECT path FROM model_file WHERE present = 1").fetchall()
    total = len(rows)

    depths: Counter = Counter()
    folder_counts: Counter = Counter()
    per_level: dict[int, Counter] = {}
    at_root = 0

    for row in rows:
        segs = _segments(row["path"])
        if not segs:
            at_root += 1
        depths[len(segs)] += 1
        if segs:
            folder_counts["/".join(segs)] += 1
        for level, name in enumerate(segs):
            per_level.setdefault(level, Counter())["/".join(segs[: level + 1])] += 1

    suggestions = []
    for level in sorted(per_level):
        counter = per_level[level]
        usable = {k: v for k, v in counter.items() if _leaf(k).lower() not in NOISE}
        suggestions.append({
            "depth": level + 1,
            "distinct_folders": len(counter),
            "usable_folders": len(usable),
            "models_covered": sum(usable.values()),
            "largest": counter.most_common(5),
        })

    # A depth is a good default when it splits the library into a browsable
    # number of groups. Two failure modes to avoid: so few that the filter is
    # pointless, and so many that it approaches one collection per model — at
    # 1500 models a naive "more groups is better" rule picks 400+, which is
    # just the folder tree again with extra steps.
    recommended = 1
    best_score = -1.0
    for entry in suggestions:
        n = entry["usable_folders"]
        if n < 2 or total == 0:
            continue
        average = entry["models_covered"] / n
        if average < 2:
            continue                      # essentially one collection per model
        if n < 5:
            shape = n / 5.0               # too coarse to be a useful filter
        elif n <= 40:
            shape = 1.0                   # the browsable band
        else:
            shape = max(0.15, 40.0 / n)   # decays as it approaches the tree
        if shape > best_score:
            best_score = shape
            recommended = entry["depth"]

    return {
        "total_models": total,
        "models_at_root": at_root,
        "max_depth": max(depths) if depths else 0,
        "depth_histogram": {str(k): v for k, v in sorted(depths.items())},
        "distinct_leaf_folders": len(folder_counts),
        "largest_folders": folder_counts.most_common(15),
        "single_model_folders": sum(1 for v in folder_counts.values() if v == 1),
        "levels": suggestions,
        "recommended_depth": recommended,
    }


def _leaf(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def preview_import(conn, max_depth: int = 1, min_models: int = 1,
                   skip_noise: bool = True) -> dict[str, Any]:
    """What ``import_as_collections`` would create, without creating it."""
    planned = _plan(conn, max_depth, min_models, skip_noise)
    existing = {r["name"] for r in conn.execute("SELECT name FROM collection")}
    return {
        "collections": [
            {"name": name, "models": len(ids), "exists": name in existing}
            for name, ids in sorted(planned.items(), key=lambda kv: -len(kv[1]))
        ],
        "total_collections": len(planned),
        "total_assignments": sum(len(v) for v in planned.values()),
        "max_depth": max_depth,
        "min_models": min_models,
    }


def _plan(conn, max_depth: int, min_models: int,
          skip_noise: bool) -> dict[str, list[int]]:
    rows = conn.execute(
        "SELECT id, path FROM model_file WHERE present = 1").fetchall()
    buckets: dict[str, list[int]] = {}

    for row in rows:
        segs = _segments(row["path"])[:max_depth]
        if not segs:
            continue
        if skip_noise:
            segs = [s for s in segs if s.lower() not in NOISE]
            if not segs:
                continue
        # A model belongs to every ancestor, so filtering by "Anime" finds
        # everything beneath it, not only its immediate folder.
        for depth in range(1, len(segs) + 1):
            buckets.setdefault(" / ".join(segs[:depth]), []).append(row["id"])

    return {name: ids for name, ids in buckets.items() if len(ids) >= min_models}


def import_as_collections(conn, handle: JobHandle | None = None, max_depth: int = 1,
                          min_models: int = 1, skip_noise: bool = True,
                          prune: bool = False) -> dict[str, Any]:
    """Mirror the folder tree as collections. No file is moved or renamed."""
    planned = _plan(conn, max_depth, min_models, skip_noise)
    stats = {"collections_created": 0, "collections_reused": 0,
             "assignments_added": 0, "assignments_removed": 0,
             "models_covered": 0}

    if handle:
        handle.set_total(len(planned))
        handle.progress(0, f"Importing {len(planned)} folder group(s)")

    covered: set[int] = set()
    for name, ids in sorted(planned.items()):
        if handle:
            handle.check_cancelled()

        row = conn.execute("SELECT id FROM collection WHERE name=?", (name,)).fetchone()
        with db.transaction(conn):
            if row is None:
                cur = conn.execute(
                    "INSERT INTO collection(name, sort_order, created_at) VALUES (?,?,?)",
                    (name, name.count(" / "), db.now()),
                )
                collection_id = int(cur.lastrowid)
                stats["collections_created"] += 1
            else:
                collection_id = row["id"]
                stats["collections_reused"] += 1

            current = {
                r["file_id"] for r in conn.execute(
                    "SELECT file_id FROM collection_member WHERE collection_id=?",
                    (collection_id,))
            }
            wanted = set(ids)
            for file_id in wanted - current:
                conn.execute(
                    "INSERT OR IGNORE INTO collection_member(collection_id, file_id) "
                    "VALUES (?,?)", (collection_id, file_id))
                stats["assignments_added"] += 1

            # Only prune when asked: a user may have hand-added members that do
            # not come from the folder tree, and silently dropping them would
            # be exactly the destructive behaviour this module exists to avoid.
            if prune:
                for file_id in current - wanted:
                    conn.execute(
                        "DELETE FROM collection_member WHERE collection_id=? AND file_id=?",
                        (collection_id, file_id))
                    stats["assignments_removed"] += 1

        covered |= set(ids)
        if handle:
            handle.advance(detail=name)

    stats["models_covered"] = len(covered)
    if handle:
        handle.progress(message=(
            f"{stats['collections_created']} created, "
            f"{stats['collections_reused']} updated, "
            f"{stats['assignments_added']} assignment(s) added"
        ))
    return stats
