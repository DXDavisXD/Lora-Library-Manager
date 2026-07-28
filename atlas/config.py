"""Configuration and secret handling.

Two rules that differ from a plain settings.json:

1. Secrets are read from the environment first. A key in ``atlas.json`` is a
   fallback, never the preferred home, and is never returned over HTTP.
2. Everything is resolved on read. There is no module-level cache to go stale
   when the dashboard writes a new value while a job is mid-flight.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

APP_NAME = "LoRA Library Manager"
VERSION = "3.0.0"

MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".bin"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

# Files we create inside the library root. Skipped by the scanner.
STATE_DIRNAME = ".atlas"

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "atlas.json"
_lock = threading.RLock()

DEFAULTS: dict[str, Any] = {
    "library_root": "",
    "port": 8077,
    "open_browser": True,
    # Scanning / hashing
    "hash_workers": 4,
    "scan_follow_symlinks": False,
    # Networking
    "request_timeout": 30,
    "provider_rate_limit": 2.0,  # requests/second, per provider
    "download_concurrency": 3,
    # Presentation
    "thumbnail_size": 320,
    "thumbnail_quality": 82,
    "page_size": 120,
    "default_sort": "name",
    "default_weight": 1.0,
    # Behaviour
    "dry_run": False,
    "auto_scan_on_start": True,
    # Empty = <library>/.atlas/backups
    "backup_folder": "",
}

# Keys that must never be serialised to a client, whatever their source.
SECRET_KEYS = {"civitai_api_key", "arcenciel_api_key"}


def _read_raw() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_raw(data: dict[str, Any]) -> None:
    tmp = _CONFIG_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(_CONFIG_PATH)


def get(key: str, default: Any = None) -> Any:
    """Return a non-secret setting, falling back to DEFAULTS."""
    with _lock:
        raw = _read_raw()
    if key in raw:
        return raw[key]
    if default is not None:
        return default
    return DEFAULTS.get(key)


def set_many(values: dict[str, Any]) -> None:
    with _lock:
        raw = _read_raw()
        raw.update(values)
        _write_raw(raw)


def get_secret(name: str) -> str:
    """Resolve a secret: environment first, config file second.

    ``name`` is a bare provider name, e.g. ``civitai`` -> ``ATLAS_CIVITAI_API_KEY``.
    """
    env_key = f"ATLAS_{name.upper()}_API_KEY"
    val = os.environ.get(env_key, "").strip()
    if val:
        return val
    with _lock:
        raw = _read_raw()
    return str(raw.get(f"{name}_api_key", "")).strip()


def set_secret(name: str, value: str) -> None:
    """Persist a secret to the config file.

    Only used when the operator has no environment variable set. If the env var
    *is* set it wins on read, so we warn rather than silently no-op.
    """
    set_many({f"{name}_api_key": value.strip()})


def secret_source(name: str) -> str:
    """Where a secret is coming from: 'env', 'file', or 'unset'."""
    if os.environ.get(f"ATLAS_{name.upper()}_API_KEY", "").strip():
        return "env"
    with _lock:
        raw = _read_raw()
    if str(raw.get(f"{name}_api_key", "")).strip():
        return "file"
    return "unset"


def library_root() -> Path:
    raw = str(get("library_root") or "").strip().strip('"').strip("'")
    if not raw:
        raise ConfigError(
            "No library_root configured. Run `python run.py init <path-to-lora-folder>`."
        )
    return Path(raw).expanduser()


def state_dir() -> Path:
    """Where the database, thumbnails and logs live (inside the library root)."""
    d = library_root() / STATE_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def db_path() -> Path:
    return state_dir() / "atlas.db"


def thumb_dir() -> Path:
    d = state_dir() / "thumbs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def public_settings() -> dict[str, Any]:
    """The settings payload safe to hand to a browser.

    Secrets are represented by presence and origin only — never by value.
    This is enforced here rather than at each call site so a new setting
    cannot accidentally leak by being added to the wrong dict.
    """
    raw = _read_raw()
    merged = {**DEFAULTS, **raw}
    return {k: v for k, v in merged.items() if k not in SECRET_KEYS and not k.endswith("_api_key")}


class ConfigError(RuntimeError):
    pass
