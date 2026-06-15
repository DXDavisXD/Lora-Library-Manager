"""File organization tools for LoRA model libraries."""

from pathlib import Path
from shutil import move

from civitai import get_model_for_safetensors
from config import (
    is_dry_run,
    is_verbose,
    log_operation,
    is_lora_file,
    LORA_EXTENSIONS,
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
    return library_dir / get_model_folder_name(lora_file)


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
def organize_lora_file(lora_file: Path, library_dir: Path) -> str:
    target_folder = get_target_folder(lora_file, library_dir)
    target_path = target_folder / lora_file.name

    if would_create_nested_duplicate(lora_file, target_folder):
        return "skipped"

    sidecar_files = find_sidecar_files(lora_file)

    if not move_without_overwrite(lora_file, target_path):
        return "skipped"

    for sidecar_file in sidecar_files:
        sidecar_target = target_folder / sidecar_file.name
        move_without_overwrite(sidecar_file, sidecar_target)

    # Clean up parent directory if no other model files remain in it
    parent_dir = lora_file.parent
    if parent_dir.resolve() != library_dir.resolve() and parent_dir.is_dir():
        has_remaining_models = False
        for ext in LORA_EXTENSIONS:
            for f in parent_dir.glob(f"*{ext}"):
                if is_lora_file(f):
                    has_remaining_models = True
                    break
            if has_remaining_models:
                break
                
        if not has_remaining_models:
            try:
                # Delete any remaining non-model files (like metadata.json)
                for extra in list(parent_dir.glob("*")):
                    if extra.is_file():
                        extra.unlink()
                parent_dir.rmdir()
                log_operation(f"Removed empty folder after organizing: {parent_dir}")
            except OSError as e:
                log_operation(f"Error cleaning up parent folder {parent_dir}: {e}")

    return "organized"


# Print organizer statistics in a readable format.
def print_statistics(stats: dict[str, int]) -> None:
    print("Organizer summary")
    print(f"organized: {stats['organized']}")
    print(f"skipped: {stats['skipped']}")
    print(f"errors: {stats['errors']}")


# Organize every LoRA file in a source folder into the library.
def organize_library(
    source_dir: Path,
    library_dir: Path | None = None,
    print_summary: bool = True,
) -> dict[str, int]:
    target_library_dir = library_dir or source_dir
    stats = {
        "organized": 0,
        "skipped": 0,
        "errors": 0,
    }

    for lora_file in find_lora_files(source_dir):
        try:
            result = organize_lora_file(lora_file, target_library_dir)
            stats[result] += 1
        except OSError:
            stats["errors"] += 1

    if print_summary:
        print_statistics(stats)

    return stats
