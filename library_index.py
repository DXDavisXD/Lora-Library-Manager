"""Incremental library index and change detection for LoRA Library Manager.

Maintains a lightweight JSON cache (library_index.json) inside LORA_ROOT that
stores per-file fingerprints (mtime, size, metadata_mtime) and the full
build_gallery_entry() result for each model.  Subsequent runs compare live
fingerprints against the cache to classify files as Added / Modified /
Removed / Unchanged, allowing gallery rebuilds and organize passes to skip
work for files that have not changed.

Cache writes always happen regardless of dry_run mode, because the index file
is not a LoRA model asset and its loss is recoverable (full scan rebuilds it).

Future extension points:
  - Filesystem watching: detect_library_changes() is pure; a watchdog thread
    can call it on FileSystemEvent and trigger targeted rebuilds.
  - Background refreshes: run_quick_refresh() in main.py can move to a thread.
  - Library integrity: add verify_hashes() comparing cached sha256 vs computed.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from config import (
    LORA_EXTENSIONS,
    is_lora_file,
    log_operation,
    PREVIEW_EXTENSIONS,
    is_complete_metadata,
)

# Bump this whenever the cache schema changes in a breaking way.
LIBRARY_INDEX_VERSION = 2
LIBRARY_INDEX_FILENAME = "library_index.json"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_library_index_path(lora_root: Path) -> Path:
    """Return the absolute path to the cache file inside lora_root."""
    return lora_root / LIBRARY_INDEX_FILENAME


def make_index_key(lora_file: Path, lora_root: Path) -> str:
    """Return the posix-style relative path used as the cache dict key.

    Example: ``"FolderName/model.safetensors"``
    Falls back to the bare filename if the file is outside lora_root.
    """
    try:
        return lora_file.resolve().relative_to(lora_root.resolve()).as_posix()
    except ValueError:
        return lora_file.name


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

def load_library_index(lora_root: Path) -> dict:
    """Load the cache from disk and return it.

    Returns an empty-but-valid index when the file is missing, corrupt, or
    written by an incompatible schema version.  Logs the outcome using the
    ``[Index]`` prefix.
    """
    path = get_library_index_path(lora_root)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("version") == LIBRARY_INDEX_VERSION:
                resolved_root = lora_root.resolve()
                models_cache = data.get("models", {})
                valid_models = {}
                discarded_count = 0
                for key, record in models_cache.items():
                    entry = record.get("entry", {})
                    entry_file = entry.get("file")
                    is_valid = False
                    if entry_file:
                        try:
                            if Path(entry_file).resolve().is_relative_to(resolved_root):
                                is_valid = True
                        except Exception:
                            pass
                    if is_valid:
                        valid_models[key] = record
                    else:
                        discarded_count += 1
                
                data["models"] = valid_models
                model_count = len(valid_models)
                if discarded_count > 0:
                    log_operation(f"[Index] Loaded {model_count} cached models (discarded {discarded_count} stale root entries)")
                else:
                    log_operation(f"[Index] Loaded {model_count} cached models")
                return data
            # Wrong version — treat as invalid
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    log_operation("[Index] Cache missing or invalid — full scan required")
    return {"version": LIBRARY_INDEX_VERSION, "last_scan": "", "models": {}}


def save_library_index(lora_root: Path, index: dict) -> None:
    """Persist the cache to disk atomically (write to .tmp then rename).

    Always writes regardless of dry_run mode because the index is a
    recoverable operational artifact, not a LoRA model asset.
    Logs failures but never raises.
    """
    path = get_library_index_path(lora_root)
    tmp_path = path.with_suffix(".tmp")
    index["last_scan"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        tmp_path.replace(path)
    except OSError as e:
        log_operation(f"[Index] Failed to save cache: {e}")


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------

def get_metadata_mtime(lora_file: Path) -> float:
    """Return the mtime of the highest-priority metadata file for *lora_file*.

    Priority order mirrors ``load_metadata()`` in gallery.py:
      1. ``<stem>.json``
      2. ``<stem>.civitai.info``
      3. ``metadata.json``

    Returns ``0.0`` when no metadata file is found.
    """
    candidates = [
        lora_file.parent / f"{lora_file.stem}.json",
        lora_file.parent / f"{lora_file.stem}.civitai.info",
        lora_file.parent / "metadata.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return candidate.stat().st_mtime
            except OSError:
                pass
    return 0.0


def get_file_fingerprint(lora_file: Path) -> dict | None:
    """Return ``{mtime, size, metadata_mtime}`` for *lora_file*.

    Returns ``None`` when the file cannot be stat-ed (e.g. deleted between
    the directory scan and this call).
    """
    try:
        st = lora_file.stat()
        return {
            "mtime": st.st_mtime,
            "size": st.st_size,
            "metadata_mtime": get_metadata_mtime(lora_file),
        }
    except OSError:
        return None


def _fingerprint_matches(cached: dict, live: dict) -> bool:
    """Return True when all three fingerprint fields agree."""
    return (
        cached.get("mtime") == live["mtime"]
        and cached.get("size") == live["size"]
        and cached.get("metadata_mtime") == live["metadata_mtime"]
    )


def is_file_changed(lora_file: Path, lora_root: Path, index: dict) -> bool:
    """Return True when *lora_file* is absent from the cache or any fingerprint
    field has changed.  Convenience wrapper used for single-file checks.
    """
    key = make_index_key(lora_file, lora_root)
    cached_record = index.get("models", {}).get(key)
    if cached_record is None:
        return True
    live_fp = get_file_fingerprint(lora_file)
    if live_fp is None:
        return True
    return not _fingerprint_matches(cached_record, live_fp)


# ---------------------------------------------------------------------------
# Change detection
# ---------------------------------------------------------------------------

def _collect_lora_files(lora_root: Path) -> list[Path]:
    """Return a deduplicated list of every LoRA file under *lora_root*."""
    seen: set[Path] = set()
    result: list[Path] = []
    for ext in LORA_EXTENSIONS:
        for path in lora_root.rglob(f"*{ext}"):
            if is_lora_file(path):
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    result.append(path)
    return result


def detect_library_changes(lora_root: Path, index: dict) -> dict:
    """Scan the filesystem and classify every LoRA file against the cache.

    Returns a dict with four lists::

        {
            "added":     [Path, ...],   # on disk but absent from cache
            "modified":  [Path, ...],   # fingerprint changed or assets incomplete
            "removed":   [str, ...],    # cache key with no matching file
            "unchanged": [Path, ...],   # fingerprint identical and assets complete
        }

    Also logs change counts and cache hit / miss statistics using the
    ``[Index]`` prefix.
    """
    lora_files = _collect_lora_files(lora_root)
    cached_models: dict = index.get("models", {})
    current_keys: set[str] = set()

    added: list[Path] = []
    modified: list[Path] = []
    unchanged: list[Path] = []
    hits = 0
    misses = 0

    for lora_file in lora_files:
        key = make_index_key(lora_file, lora_root)
        current_keys.add(key)
        live_fp = get_file_fingerprint(lora_file)
        if live_fp is None:
            continue  # Can't stat file — skip silently

        # Fast assets-complete validation:
        # 1. Check if complete metadata exists (checking fields, not just presence)
        # 2. Check if a local preview image exists (without parsing JSON)
        has_metadata = False
        if live_fp["metadata_mtime"] > 0.0:
            candidates = [
                lora_file.parent / f"{lora_file.stem}.json",
                lora_file.parent / f"{lora_file.stem}.civitai.info",
                lora_file.parent / "metadata.json",
            ]
            for candidate in candidates:
                if candidate.exists():
                    try:
                        with candidate.open("r", encoding="utf-8") as f:
                            meta = json.load(f)
                        if isinstance(meta, dict) and is_complete_metadata(meta):
                            has_metadata = True
                            break
                    except Exception:
                        pass

        has_preview = False
        for ext in PREVIEW_EXTENSIONS:
            if (lora_file.parent / f"{lora_file.stem}{ext}").exists():
                has_preview = True
                break
        assets_complete = has_metadata and has_preview

        if key not in cached_models:
            added.append(lora_file)
            misses += 1
        elif not _fingerprint_matches(cached_models[key], live_fp) or not assets_complete:
            modified.append(lora_file)
            misses += 1
        else:
            unchanged.append(lora_file)
            hits += 1

    removed: list[str] = [k for k in cached_models if k not in current_keys]

    log_operation(f"[Index] Added: {len(added)}")
    log_operation(f"[Index] Modified: {len(modified)}")
    log_operation(f"[Index] Removed: {len(removed)}")
    log_operation(f"[Index] Unchanged: {len(unchanged)}")
    log_operation(f"[Index] Cache hits: {hits}  Cache misses: {misses}")

    return {
        "added": added,
        "modified": modified,
        "removed": removed,
        "unchanged": unchanged,
    }
