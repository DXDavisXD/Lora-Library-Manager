"""First-run setup.

Running ``python run.py`` with no arguments should get a beginner to a working
gallery without reading anything. That means: find their LoRA folder for them,
explain what is about to happen in plain language, ask before doing anything
that writes, and never require an API key to get started.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable

from . import config

# Where Stable Diffusion front-ends keep LoRAs. Checked in rough order of
# how common they are.
KNOWN_LAYOUTS = [
    "models/Lora", "models/lora", "models/loras",
    "models/Stable-diffusion/Lora",
    "webui/models/Lora", "stable-diffusion-webui/models/Lora",
    "webui-forge/models/Lora", "ComfyUI/models/loras",
    "sd.webui/webui/models/Lora",
]

APP_DIR_HINTS = [
    "stable-diffusion-webui", "webui-forge", "sd-webui", "ComfyUI",
    "SwarmUI", "InvokeAI", "Fooocus", "SD", "AI", "StableDiffusion",
]


def candidate_roots() -> list[Path]:
    """Directories worth probing for a LoRA folder."""
    roots: list[Path] = []
    home = Path.home()
    roots += [home, home / "Documents", home / "Downloads", home / "Desktop"]

    if os.name == "nt":
        import string

        for letter in string.ascii_uppercase:
            drive = Path(f"{letter}:/")
            try:
                if drive.exists():
                    roots.append(drive)
            except OSError:
                continue
    else:
        roots += [Path("/opt"), Path("/srv")]
    return roots


def discover(limit: int = 12) -> list[dict]:
    """Look for plausible LoRA folders and rank them by model count.

    Deliberately shallow: probing known layouts under a handful of roots takes
    well under a second, where walking whole drives could take minutes.
    """
    found: dict[Path, dict] = {}

    def consider(path: Path) -> None:
        try:
            if not path.is_dir() or path in found:
                return
        except OSError:
            return
        count, size = _count_models(path)
        if count:
            found[path] = {"path": str(path), "models": count, "bytes": size}

    for root in candidate_roots():
        for layout in KNOWN_LAYOUTS:
            consider(root / layout)
        # One level of app-directory guessing, e.g. E:/SD/webui-forge/models/Lora
        try:
            entries = list(os.scandir(root))
        except (OSError, PermissionError):
            continue
        for entry in entries[:400]:
            try:
                if not entry.is_dir():
                    continue
            except OSError:
                continue
            name = entry.name
            if not any(hint.lower() in name.lower() for hint in APP_DIR_HINTS):
                continue
            for layout in KNOWN_LAYOUTS:
                consider(Path(entry.path) / layout)

    ranked = sorted(found.values(), key=lambda d: -d["models"])
    return ranked[:limit]


def _count_models(path: Path, cap: int = 4000) -> tuple[int, int]:
    """Count model files under *path*, stopping early on huge trees."""
    count = 0
    size = 0
    try:
        stack = [path]
        while stack and count < cap:
            current = stack.pop()
            try:
                entries = list(os.scandir(current))
            except (OSError, PermissionError):
                continue
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if not entry.name.startswith("."):
                            stack.append(Path(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        if Path(entry.name).suffix.lower() in config.MODEL_EXTENSIONS:
                            count += 1
                            size += entry.stat().st_size
                except OSError:
                    continue
    except OSError:
        pass
    return count, size


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


# ---------------------------------------------------------------------------
# Interactive wizard
# ---------------------------------------------------------------------------

def _ask(prompt: str, default: str = "") -> str:
    try:
        answer = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(130)
    return answer or default


def _yes(prompt: str, default: bool = True) -> bool:
    hint = "[Y/n]" if default else "[y/N]"
    answer = _ask(f"{prompt} {hint} ").lower()
    if not answer:
        return default
    return answer.startswith("y")


def is_configured() -> bool:
    try:
        root = config.library_root()
    except config.ConfigError:
        return False
    return root.is_dir()


def run_wizard() -> int:
    """Walk a first-time user from nothing to a browsable gallery."""
    print()
    print("  " + "─" * 58)
    print(f"   Welcome to {config.APP_NAME}")
    print("  " + "─" * 58)
    print()
    print("   This is a one-time setup. Three questions, then you're done.")
    print("   Nothing on disk is modified without asking.")
    print()

    # 1. Where is the library?
    print("   Looking for your LoRA folder...", end="", flush=True)
    options = discover()
    print(f" found {len(options)}." if options else " none found.")
    print()

    chosen: Path | None = None
    if options:
        for i, entry in enumerate(options, start=1):
            print(f"     {i}. {entry['path']}")
            print(f"        {entry['models']} model(s), {human(entry['bytes'])}")
        print(f"     {len(options) + 1}. Enter a different path")
        print()
        answer = _ask(f"   Which folder? [1-{len(options) + 1}, default 1] ", "1")
        try:
            index = int(answer)
        except ValueError:
            index = 1
        if 1 <= index <= len(options):
            chosen = Path(options[index - 1]["path"])

    while chosen is None:
        raw = _ask("   Full path to your LoRA folder: ").strip('"').strip("'")
        if not raw:
            print("   A path is required. Ctrl+C to quit.")
            continue
        candidate = Path(raw).expanduser()
        if not candidate.is_dir():
            print(f"   Not a folder: {candidate}")
            continue
        count, size = _count_models(candidate)
        print(f"   Found {count} model(s), {human(size)}.")
        if count == 0 and not _yes("   No models found. Use it anyway?", False):
            continue
        chosen = candidate

    config.set_many({"library_root": str(chosen)})
    print(f"\n   Library:  {chosen}")
    print(f"   Database: {config.db_path()}")
    print()

    # 2. Optional API keys — asked for every provider, not just the first.
    from . import providers

    print("   API keys are optional. Without them you still get your whole")
    print("   library, search, ratings and notes — public metadata downloads")
    print("   mostly work unauthenticated; a key helps with rate limits and")
    print("   restricted models.")
    print()
    if _yes("   Add provider API keys now?", False):
        for provider in providers.all_providers():
            env_var = f"ATLAS_{provider.name.upper()}_API_KEY"
            source = config.secret_source(provider.name)
            print()
            if source == "env":
                print(f"   {provider.display_name}: already set via {env_var} "
                      f"(environment wins; skipping).")
                continue
            if source == "file":
                print(f"   {provider.display_name}: a key is already saved.")
                if not _yes("   Replace it?", False):
                    continue
            print(f"   {provider.display_name} — get a key at {provider.website}")
            print(f"   Tip: setting {env_var} in your environment is safer than")
            print("   storing it in a file.")
            key = _ask(f"   Paste {provider.display_name} key (blank to skip): ")
            if key:
                config.set_secret(provider.name, key)
                print("   Saved.")
            else:
                print("   Skipped.")
    print()

    # 3. First scan.
    if _yes("   Scan the library now? (read-only, nothing is moved)", True):
        from . import db, enrich, query, scanner

        conn = db.connect()
        print("\n   Scanning...", end="", flush=True)
        result = scanner.sync(conn, chosen)
        print(f" {result.added} model(s) in {result.duration:.1f}s")
        print("   Reading existing metadata...", end="", flush=True)
        stats = enrich.ingest_sidecars(conn, chosen)
        print(f" described {stats['described']}, "
              f"kept your data on {stats['adopted']}")

        summary = query.stats(conn)
        print()
        print(f"   {summary['total']} model(s) indexed")
        if summary["missing_metadata"]:
            print(f"   {summary['missing_metadata']} still need metadata "
                  f"(run a Full Sync in the app)")
        if summary["collections"]:
            print(f"   {summary['collections']} collection(s) recovered")
    print()
    print("  " + "─" * 58)
    print("   Setup complete. Starting the app...")
    print("  " + "─" * 58)
    print()
    return 0
