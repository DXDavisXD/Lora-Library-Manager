"""Application configuration for LoRA Library Manager."""

import json
from datetime import datetime
from pathlib import Path


APP_NAME = "LoRA Library Manager"
VERSION = "2.2.0"
LORA_ROOT = Path(r"Path to LoRA folder")  # Default LoRA root folder; can be overridden in settings.json
LORA_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".bin"}
PREVIEW_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SETTINGS_FILE = Path(__file__).parent.parent / "settings.json"
TMP_CLEANUP_AGE_SECONDS = 3600


# Load settings from settings.json.
def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            with SETTINGS_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


# Save settings to settings.json.
def save_settings(settings: dict) -> None:
    try:
        with SETTINGS_FILE.open("w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except OSError:
        pass


# Initialize state flags on startup
_settings = load_settings()
DRY_RUN = _settings.get("dry_run", False)
VERBOSE = _settings.get("verbose", False)
CIVITAI_API_KEY = _settings.get("civitai_api_key", "")
ARCENCIEL_API_KEY = _settings.get("arcenciel_api_key", "")
ENABLE_FILE_LOGGING = True


# Return whether dry-run mode is enabled.
def is_dry_run() -> bool:
    return DRY_RUN


# Enable or disable dry-run mode and persist to settings.json.
def set_dry_run(value: bool) -> None:
    global DRY_RUN
    DRY_RUN = value
    settings = load_settings()
    settings["dry_run"] = value
    save_settings(settings)


# Return whether verbose mode is enabled.
def is_verbose() -> bool:
    return VERBOSE


# Enable or disable verbose mode and persist to settings.json.
def set_verbose(value: bool) -> None:
    global VERBOSE
    VERBOSE = value
    settings = load_settings()
    settings["verbose"] = value
    save_settings(settings)


# Return the current CivitAI API key.
def get_civitai_api_key() -> str:
    return get_provider_api_key("civitai")


# Update the CivitAI API Key and persist to settings.json.
def set_civitai_api_key(key: str) -> None:
    set_provider_api_key("civitai", key)


# Return the current Arcenciel API key.
def get_arcenciel_api_key() -> str:
    return get_provider_api_key("arcenciel")


# Update the Arcenciel API Key and persist to settings.json.
def set_arcenciel_api_key(key: str) -> None:
    set_provider_api_key("arcenciel", key)


def get_provider_api_key(provider: str) -> str:
    """
    Return the stored API key for *provider*.
    Loads dynamically from settings to ensure fresh values are read.
    """
    global CIVITAI_API_KEY, ARCENCIEL_API_KEY
    settings = load_settings()
    val = settings.get(f"{provider}_api_key", "").strip()
    if provider == "civitai":
        CIVITAI_API_KEY = val
    elif provider == "arcenciel":
        ARCENCIEL_API_KEY = val
    return val


def set_provider_api_key(provider: str, key: str) -> None:
    """
    Update the API key for *provider* and persist to settings.json.
    """
    global CIVITAI_API_KEY, ARCENCIEL_API_KEY
    settings = load_settings()
    val = key.strip()
    settings[f"{provider}_api_key"] = val
    if provider == "civitai":
        CIVITAI_API_KEY = val
    elif provider == "arcenciel":
        ARCENCIEL_API_KEY = val
    save_settings(settings)


# Update the LoRA root folder and persist to settings.json.
def set_lora_root(value: str) -> None:
    settings = load_settings()
    settings["lora_root"] = value.strip().strip('"').strip("'").strip()
    save_settings(settings)


# Get preferred download strategy. Defaults to "skip" (skip existing).
def get_download_strategy() -> str:
    _settings = load_settings()
    return _settings.get("download_strategy", "skip")


# Set preferred download strategy.
def set_download_strategy(value: str) -> None:
    settings = load_settings()
    settings["download_strategy"] = value.strip()
    save_settings(settings)


# Get default prompt weight. Defaults to 1.0.
def get_default_weight() -> float:
    _settings = load_settings()
    try:
        return float(_settings.get("default_weight", 1.0))
    except (ValueError, TypeError):
        return 1.0


# Set default prompt weight.
def set_default_weight(value: float) -> None:
    settings = load_settings()
    settings["default_weight"] = float(value)
    save_settings(settings)


# Get default gallery sort. Defaults to "name".
def get_default_sort() -> str:
    _settings = load_settings()
    return _settings.get("default_sort", "name")


# Set default gallery sort.
def set_default_sort(value: str) -> None:
    settings = load_settings()
    settings["default_sort"] = value.strip()
    save_settings(settings)


# Log a timestamped operation message to logs/YYYY-MM-DD.log under LORA_ROOT.
def log_operation(message: str) -> None:
    if ENABLE_FILE_LOGGING:
        lora_root = get_lora_root()
        logs_dir = lora_root / "logs"
        try:
            logs_dir.mkdir(parents=True, exist_ok=True)
            date_str = datetime.now().strftime("%Y-%m-%d")
            log_path = logs_dir / f"{date_str}.log"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] {message}\n")
        except OSError:
            pass


# Return the configured LoRA root folder.
def get_lora_root() -> Path:
    _settings = load_settings()
    custom_root = _settings.get("lora_root")
    if custom_root:
        return Path(str(custom_root).strip().strip('"').strip("'").strip())
    return LORA_ROOT


# Return the current folder layout used by the application.
def get_default_paths() -> dict[str, Path]:
    return {
        "lora_root": get_lora_root(),
    }


# Create the application folders if they do not already exist.
def ensure_directories(paths: dict[str, Path] | None = None) -> dict[str, Path]:
    selected_paths = paths or get_default_paths()

    for folder in selected_paths.values():
        folder.mkdir(parents=True, exist_ok=True)

    return selected_paths


# Check whether a file path looks like a supported LoRA model file.
def is_lora_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in LORA_EXTENSIONS


def normalize_for_comparison(s: str) -> str:
    """Normalize a string by lowercasing and removing all non-alphanumeric characters."""
    if not s:
        return ""
    return "".join(c for c in s.lower() if c.isalnum())


def clean_model_collections(collections: list, folder_path: Path) -> list:
    """Filter out collections that match the model folder name, file stems, or model names from metadata."""
    if not collections:
        return []

    folder_path = Path(folder_path)
    folder_name = folder_path.name
    normalized_targets = {normalize_for_comparison(folder_name)}

    # Add file stems of LoRA files in the folder
    try:
        for f in folder_path.iterdir():
            if f.is_file() and f.suffix.lower() in LORA_EXTENSIONS:
                normalized_targets.add(normalize_for_comparison(f.stem))
    except Exception:
        pass

    # Add model names and version names from metadata files in the same directory
    try:
        for f in folder_path.iterdir():
            if f.is_file() and (f.suffix.lower() == '.json' or f.name.endswith('.civitai.info')):
                try:
                    with f.open("r", encoding="utf-8") as meta_f:
                        meta = json.load(meta_f)
                        if isinstance(meta, dict):
                            # Check civitai metadata format
                            model_obj = meta.get("model")
                            if isinstance(model_obj, dict) and model_obj.get("name"):
                                normalized_targets.add(normalize_for_comparison(model_obj["name"]))
                            if meta.get("name"):
                                normalized_targets.add(normalize_for_comparison(meta["name"]))
                except Exception:
                    pass
    except Exception:
        pass

    cleaned = []
    for c in collections:
        if not c or not isinstance(c, str):
            continue
        c_clean = c.strip()
        if not c_clean:
            continue
        # For path-like tags, check if the last component matches any normalized target.
        # This prevents redundant folder nesting (e.g. Anime / Fate / Rin / Rin).
        parts = [p.strip() for p in c_clean.replace('\\', '/').split('/') if p.strip()]
        if not parts:
            continue
        last_part = parts[-1]
        c_norm = normalize_for_comparison(last_part)
        if c_norm in normalized_targets:
            continue
        cleaned.append(c_clean)

    return cleaned


def is_complete_metadata(metadata: dict) -> bool:
    """
    Backward-compatibility shim.

    The authoritative, provider-aware implementation lives in
    ``downloader.is_complete_metadata()``.  This thin wrapper delegates
    to it so that callers that still import from config continue to work
    without modification.
    """
    # Lazy import avoids a circular dependency at module load time.
    from downloader import is_complete_metadata as _impl
    return _impl(metadata)
