"""File organization tools for LoRA model libraries."""

import json
from pathlib import Path
from shutil import move, copy2

from civitai import get_model_for_safetensors
from config import (
    is_dry_run,
    is_verbose,
    log_operation,
    is_lora_file,
    LORA_EXTENSIONS,
    clean_model_collections,
)
SIDECAR_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".json",
    ".txt",
    ".civitai.info",
}


# Build a safe folder name from a model name or filename.
def sanitize_name(name: str) -> str:
    invalid_characters = {'<', '>', ':', '"', '/', '\\', '|', '?', '*'}
    safe_characters = []

    for character in name.strip():
        if character not in invalid_characters and character.isprintable():
            safe_characters.append(character)

    safe_name = "".join(safe_characters).strip(" .")
    return safe_name or "untitled_lora"


# Check whether a file path is a supported LoRA model file.
def is_model_file(path: Path) -> bool:
    return is_lora_file(path)


# Return every supported LoRA file found inside a folder.
def find_lora_files(source_dir: Path) -> list[Path]:
    if not source_dir.exists():
        return []

    lora_files = []
    for ext in LORA_EXTENSIONS:
        for path in source_dir.rglob(f"*{ext}"):
            if is_lora_file(path):
                lora_files.append(path)

    # Deduplicate paths
    seen = set()
    unique_files = []
    for path in lora_files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(path)

    return unique_files


# Check whether a model is already in a folder matching its filename stem.
def is_already_organized(lora_file: Path) -> bool:
    return lora_file.parent.name.casefold() == lora_file.stem.casefold()


# Return the CivitAI official model name from a metadata response.
def extract_official_model_name(civitai_response: dict) -> str:
    if not civitai_response.get("ok"):
        return ""

    data = civitai_response.get("data", {})
    model = data.get("model", {})

    if isinstance(model, dict):
        return model.get("name", "").strip()

    return ""


# Return the best folder name for a model, using CivitAI when possible.
def get_model_folder_name(lora_file: Path) -> str:
    if lora_file.suffix.lower() != ".safetensors":
        return sanitize_name(lora_file.stem)

    official_name = extract_official_model_name(get_model_for_safetensors(lora_file))

    if official_name:
        return sanitize_name(official_name)

    return sanitize_name(lora_file.stem)


# Decide which library folder should contain a LoRA file.
def get_target_folder(lora_file: Path, library_dir: Path) -> Path:
    # 1. Determine model folder name (preserve existing parent folder name if already organized,
    # otherwise compute from metadata/filename)
    if lora_file.parent.resolve() != library_dir.resolve():
        model_folder_name = lora_file.parent.name
    else:
        model_folder_name = get_model_folder_name(lora_file)

    # 2. Check for collection tag in user_data.json
    tag = ""
    user_data_path = lora_file.parent / "user_data.json"
    if user_data_path.exists():
        try:
            with user_data_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # Use clean_model_collections to get real collection names
            colls = clean_model_collections(data.get("collections", []), lora_file.parent)
            if colls:
                # We enforce only 1 tag for now
                tag = colls[0]
        except Exception:
            pass

    # 3. Build target folder path
    if tag:
        parts = [sanitize_name(p.strip()) for p in tag.replace('\\', '/').split('/') if p.strip()]
        if parts:
            return library_dir.joinpath(*parts) / model_folder_name
    return library_dir / model_folder_name


# Check whether moving this file would create a nested duplicate folder.
def would_create_nested_duplicate(lora_file: Path, target_folder: Path) -> bool:
    return lora_file.parent.resolve() == target_folder.resolve()


# Decide whether a file is a sidecar for the given LoRA model file.
def is_matching_sidecar(path: Path, lora_file: Path) -> bool:
    if not path.is_file():
        return False

    if path.name == lora_file.name:
        return False

    if path.name.lower() == f"{lora_file.stem.lower()}.civitai.info":
        return True

    return (
        path.stem.casefold() == lora_file.stem.casefold()
        and path.suffix.lower() in SIDECAR_EXTENSIONS
    )


# Return sidecar files that sit next to a LoRA model file.
def find_sidecar_files(lora_file: Path) -> list[Path]:
    return [
        path
        for path in lora_file.parent.iterdir()
        if is_matching_sidecar(path, lora_file)
    ]


# Move one file only when the destination does not already exist.
def move_without_overwrite(source_path: Path, target_path: Path) -> bool:
    if target_path.exists():
        return False

    if is_dry_run():
        log_operation(f"[DRY RUN] Would move: {source_path} -> {target_path}")
        if is_verbose():
            try:
                print(f"[DRY RUN] Would move: {source_path.name} -> {target_path}")
            except UnicodeEncodeError:
                try:
                    print(f"[DRY RUN] Would move: {source_path.name} -> {target_path.as_posix().encode('ascii', 'replace').decode('ascii')}")
                except Exception:
                    pass
        return True

    target_path.parent.mkdir(parents=True, exist_ok=True)
    move(str(source_path), str(target_path))
    log_operation(f"Moved: {source_path} -> {target_path}")
    if is_verbose():
        try:
            print(f"Moved: {source_path.name} -> {target_path}")
        except UnicodeEncodeError:
            try:
                print(f"Moved: {source_path.name} -> {target_path.as_posix().encode('ascii', 'replace').decode('ascii')}")
            except Exception:
                pass
    return True


# Move one LoRA file into its own library folder.
def organize_lora_file(lora_file: Path, library_dir: Path, target_folder: Path | None = None) -> str:
    folder = target_folder if target_folder is not None else get_target_folder(lora_file, library_dir)
    target_path = folder / lora_file.name

    if would_create_nested_duplicate(lora_file, folder):
        return "skipped"

    # Copy user_data.json if it exists and hasn't been copied to the destination yet.
    # We do this BEFORE moving the model file, so that the metadata is preserved,
    # and sibling model files can still read it in the old folder before they are moved.
    user_data_src = lora_file.parent / "user_data.json"
    if user_data_src.exists():
        user_data_dst = folder / "user_data.json"
        if not user_data_dst.exists():
            if is_dry_run():
                log_operation(f"[DRY RUN] Would copy user data: {user_data_src} -> {user_data_dst}")
            else:
                try:
                    folder.mkdir(parents=True, exist_ok=True)
                    copy2(user_data_src, user_data_dst)
                    log_operation(f"Copied user data sidecar: {user_data_src} -> {user_data_dst}")
                except Exception as e:
                    log_operation(f"Error copying user data sidecar: {e}")

    sidecar_files = find_sidecar_files(lora_file)

    if not move_without_overwrite(lora_file, target_path):
        return "skipped"

    for sidecar_file in sidecar_files:
        sidecar_target = folder / sidecar_file.name
        move_without_overwrite(sidecar_file, sidecar_target)

    # Clean up parent directory and any empty ancestor directories up to library_dir
    current_dir = lora_file.parent
    library_resolved = library_dir.resolve()
    
    while current_dir.resolve() != library_resolved and current_dir.is_dir():
        # Check if current_dir has any model files inside it recursively
        has_remaining_models = False
        for ext in LORA_EXTENSIONS:
            for f in current_dir.rglob(f"*{ext}"):
                if is_lora_file(f):
                    has_remaining_models = True
                    break
            if has_remaining_models:
                break
                
        if not has_remaining_models:
            try:
                # Delete any remaining non-model files or empty subdirectories inside current_dir
                for extra in list(current_dir.glob("*")):
                    if extra.is_file():
                        if is_dry_run():
                            log_operation(f"[DRY RUN] Would remove file: {extra}")
                        else:
                            extra.unlink()
                    elif extra.is_dir():
                        try:
                            if is_dry_run():
                                log_operation(f"[DRY RUN] Would remove directory: {extra}")
                            else:
                                extra.rmdir()
                        except OSError:
                            pass
                if is_dry_run():
                    log_operation(f"[DRY RUN] Would remove empty folder after organizing: {current_dir}")
                else:
                    current_dir.rmdir()
                    log_operation(f"Removed empty folder after organizing: {current_dir}")
            except OSError as e:
                log_operation(f"Error cleaning up parent folder {current_dir}: {e}")
                break
        else:
            break
        # Move up to parent
        current_dir = current_dir.parent

    return "organized"


# Print organizer statistics in a readable format.
def print_statistics(stats: dict[str, int]) -> None:
    print("Organizer summary")
    print(f"organized: {stats['organized']}")
    print(f"skipped: {stats['skipped']}")
    print(f"errors: {stats['errors']}")


# Organize every LoRA file in a source folder into the library.
# When changed_files is provided, only those specific files are processed
# instead of scanning the entire source directory.  This allows the caller
# (run_automatic in main.py) to pass only the added/modified paths from the
# incremental index so unaffected models are never touched.
def organize_library(
    source_dir: Path,
    library_dir: Path | None = None,
    print_summary: bool = True,
    changed_files: list[Path] | None = None,
) -> tuple[dict[str, int], list[Path]]:
    target_library_dir = library_dir or source_dir
    stats = {
        "organized": 0,
        "skipped": 0,
        "errors": 0,
    }
    processed_paths = []

    # Use the caller-supplied list when provided; otherwise full scan.
    files_to_process = changed_files if changed_files is not None else find_lora_files(source_dir)

    for lora_file in files_to_process:
        # Guard: file might have been moved or deleted between detection and here.
        if not lora_file.exists():
            continue
        try:
            target_folder = get_target_folder(lora_file, target_library_dir)
            target_path = target_folder / lora_file.name
            
            result = organize_lora_file(lora_file, target_library_dir, target_folder=target_folder)
            stats[result] += 1
            if result in ("organized", "skipped"):
                processed_paths.append(target_path)
            else:
                processed_paths.append(lora_file)
        except Exception:
            stats["errors"] += 1
            processed_paths.append(lora_file)

    if print_summary:
        print_statistics(stats)

    return stats, processed_paths
