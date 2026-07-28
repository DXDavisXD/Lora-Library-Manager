"""Content hashing.

SHA-256 over a 1500-model library is hundreds of gigabytes of I/O, so the goal
is to do as little of it as possible:

* Hashing is *lazy*. Change detection uses (size, mtime); a hash is only
  computed when something actually needs one — a provider lookup by hash, a
  duplicate sweep, or a backup export.
* Results are cached in the database and invalidated by the scanner when
  (size, mtime) move.
* Work is spread over a small thread pool. Hashing is I/O bound and releases
  the GIL inside ``hashlib``, so threads are the right primitive; the default
  of 4 is deliberately modest because oversubscribing a spinning disk with
  random reads is slower than doing it serially.
* Reads use a 4 MiB buffer into a preallocated memoryview, which avoids
  allocating a fresh bytes object per chunk.
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

from . import config, db

CHUNK = 4 * 1024 * 1024

Progress = Callable[[int, int, str], None]


def hash_file(path: Path, chunk_size: int = CHUNK, cancel: Callable[[], bool] | None = None) -> str | None:
    """Return the lowercase SHA-256 of *path*, or None if cancelled/unreadable."""
    digest = hashlib.sha256()
    buf = bytearray(chunk_size)
    view = memoryview(buf)
    try:
        with open(path, "rb", buffering=0) as fh:
            while True:
                if cancel and cancel():
                    return None
                read = fh.readinto(buf)
                if not read:
                    break
                digest.update(view[:read])
    except (OSError, ValueError):
        return None
    return digest.hexdigest()


def pending_count(conn) -> int:
    row = conn.execute(
        "SELECT COUNT(*) c FROM model_file WHERE present=1 AND hash_state='pending'"
    ).fetchone()
    return int(row["c"])


def pending_bytes(conn) -> int:
    row = conn.execute(
        "SELECT COALESCE(SUM(size),0) s FROM model_file "
        "WHERE present=1 AND hash_state='pending'"
    ).fetchone()
    return int(row["s"])


def hash_pending(
    conn,
    root: Path,
    limit: int | None = None,
    progress: Progress | None = None,
    cancel: Callable[[], bool] | None = None,
    workers: int | None = None,
) -> int:
    """Hash files whose hash is unknown. Returns the number hashed.

    Smallest files first: it front-loads visible progress and gets the cheap
    wins banked before a cancel, rather than stalling on a 6 GB checkpoint.
    """
    sql = (
        "SELECT id, path, size FROM model_file "
        "WHERE present=1 AND hash_state='pending' ORDER BY size ASC"
    )
    if limit:
        sql += f" LIMIT {int(limit)}"
    targets = [(r["id"], r["path"], r["size"]) for r in conn.execute(sql)]
    if not targets:
        return 0

    total = len(targets)
    done = 0
    n_workers = max(1, int(workers or config.get("hash_workers") or 4))

    def work(item: tuple[int, str, int]) -> tuple[int, str, str | None]:
        fid, rel, _size = item
        return fid, rel, hash_file(root / rel, cancel=cancel)

    results: list[tuple[int, str | None]] = []

    with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="hash") as pool:
        futures = {pool.submit(work, t): t for t in targets}
        for future in as_completed(futures):
            fid, rel, digest = future.result()
            results.append((fid, digest))
            done += 1
            if progress:
                progress(done, total, rel)
            if len(results) >= 100:
                _commit(conn, results)
                results = []
            if cancel and cancel():
                for f in futures:
                    f.cancel()
                break

    _commit(conn, results)
    return done


def _commit(conn, results: list[tuple[int, str | None]]) -> None:
    if not results:
        return
    with db.transaction(conn):
        for fid, digest in results:
            if digest:
                conn.execute(
                    "UPDATE model_file SET sha256=?, hash_state='done' WHERE id=?",
                    (digest, fid),
                )
            else:
                conn.execute(
                    "UPDATE model_file SET hash_state='error' WHERE id=?", (fid,)
                )
    results.clear()


def ensure_hash(conn, root: Path, file_id: int) -> str | None:
    """Hash a single file on demand, reusing a cached value when valid."""
    row = conn.execute(
        "SELECT path, sha256, hash_state FROM model_file WHERE id=?", (file_id,)
    ).fetchone()
    if row is None:
        return None
    if row["sha256"] and row["hash_state"] == "done":
        return row["sha256"]
    digest = hash_file(root / row["path"])
    with db.transaction(conn):
        if digest:
            conn.execute(
                "UPDATE model_file SET sha256=?, hash_state='done' WHERE id=?", (digest, file_id)
            )
        else:
            conn.execute("UPDATE model_file SET hash_state='error' WHERE id=?", (file_id,))
    return digest


def find_duplicates(conn) -> list[dict]:
    """Group present files that share a SHA-256. Requires hashes to exist."""
    rows = conn.execute(
        """
        SELECT f.sha256, f.id, f.path, f.size, m.name
        FROM model_file f
        LEFT JOIN model m ON m.file_id = f.id
        WHERE f.present = 1 AND f.sha256 IS NOT NULL
          AND f.sha256 IN (
              SELECT sha256 FROM model_file
              WHERE present = 1 AND sha256 IS NOT NULL
              GROUP BY sha256 HAVING COUNT(*) > 1
          )
        ORDER BY f.sha256, f.path
        """
    ).fetchall()
    groups: dict[str, dict] = {}
    for r in rows:
        g = groups.setdefault(r["sha256"], {"sha256": r["sha256"], "size": r["size"], "files": []})
        g["files"].append({"id": r["id"], "path": r["path"], "name": r["name"]})
    out = [g for g in groups.values() if len(g["files"]) > 1]
    out.sort(key=lambda g: g["size"] * (len(g["files"]) - 1), reverse=True)
    for g in out:
        g["reclaimable"] = g["size"] * (len(g["files"]) - 1)
    return out
