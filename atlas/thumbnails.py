"""Thumbnail cache.

Preview PNGs from CivitAI routinely run 1–3 MB. A 1500-model gallery that
serves originals asks the browser to decode multiple gigabytes of image, which
is the single biggest reason a large library feels broken in the browser even
when the backend is fast.

Thumbnails are ~15–30 KB WebP, generated once and cached under
``.atlas/thumbs``. The cache key includes the source mtime and size, so an
updated preview supersedes its thumbnail without any invalidation bookkeeping.

Pillow is optional. Without it the originals are served and everything still
works, just heavier.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

from . import config

try:
    from PIL import Image, ImageOps
    HAVE_PILLOW = True
except ImportError:                                        # pragma: no cover
    HAVE_PILLOW = False

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def available() -> bool:
    return HAVE_PILLOW


def cache_key(source: Path, size: int) -> str:
    try:
        st = source.stat()
        stamp = f"{source.as_posix()}|{st.st_size}|{int(st.st_mtime)}|{size}"
    except OSError:
        stamp = f"{source.as_posix()}|missing|{size}"
    return hashlib.sha1(stamp.encode("utf-8")).hexdigest()


def thumb_path(source: Path, size: int) -> Path:
    key = cache_key(source, size)
    # Two-level fan-out: 1500 models with several previews each would otherwise
    # put tens of thousands of files in one directory.
    return config.thumb_dir() / key[:2] / f"{key}.webp"


def get_or_create(source: Path, size: int | None = None) -> Path | None:
    """Return a cached thumbnail path, generating it if needed.

    Returns None when Pillow is unavailable or the source cannot be decoded;
    the caller then serves the original.
    """
    if not source.is_file():
        return None
    size = int(size or config.get("thumbnail_size") or 320)
    target = thumb_path(source, size)
    # A cached thumbnail needs no Pillow to serve. Checking the cache first
    # means a Python without it (a fresh venv, a reinstall) still gets the
    # small files already on disk instead of falling back to multi-MB originals.
    if target.exists():
        return target
    if not HAVE_PILLOW:
        return None

    key = target.stem
    with _locks_guard:
        lock = _locks.setdefault(key, threading.Lock())
    with lock:
        if target.exists():                                # won by another thread
            return target
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode not in ("RGB", "RGBA"):
                    img = img.convert("RGBA" if "A" in img.mode else "RGB")
                img.thumbnail((size, size * 2), Image.LANCZOS)
                tmp = target.with_suffix(".tmp.webp")
                img.save(
                    tmp, "WEBP",
                    quality=int(config.get("thumbnail_quality") or 82),
                    method=4,
                )
                tmp.replace(target)
        except Exception:                                  # noqa: BLE001 - fall back to original
            target.unlink(missing_ok=True)
            return None
        finally:
            with _locks_guard:
                _locks.pop(key, None)
    return target


def warm(conn, root: Path, handle=None, limit: int | None = None) -> dict[str, int]:
    """Pre-generate thumbnails so first scroll is smooth."""
    if not HAVE_PILLOW:
        if handle:
            handle.log("Pillow not installed — thumbnails disabled.", "warn")
        return {"generated": 0, "skipped": 0, "failed": 0}

    sql = "SELECT p.path FROM preview p JOIN model_file f ON f.id=p.file_id WHERE f.present=1"
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql).fetchall()
    stats = {"generated": 0, "skipped": 0, "failed": 0}
    if handle:
        handle.set_total(len(rows))

    size = int(config.get("thumbnail_size") or 320)
    for row in rows:
        if handle:
            handle.check_cancelled()
        source = root / row["path"]
        target = thumb_path(source, size)
        if target.exists():
            stats["skipped"] += 1
        elif get_or_create(source, size):
            stats["generated"] += 1
        else:
            stats["failed"] += 1
        if handle:
            handle.advance(detail=row["path"].rsplit("/", 1)[-1])
    return stats


def cache_size() -> tuple[int, int]:
    """(file count, total bytes) of the thumbnail cache."""
    count = 0
    total = 0
    try:
        for path in config.thumb_dir().rglob("*.webp"):
            count += 1
            try:
                total += path.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return count, total


def purge() -> int:
    removed = 0
    try:
        for path in config.thumb_dir().rglob("*.webp"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    except OSError:
        pass
    return removed
