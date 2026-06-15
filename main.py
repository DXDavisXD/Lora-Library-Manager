"""Command-line entry point for LoRA Library Manager.

The GUI is intentionally not implemented yet.
"""

import json
import time
from pathlib import Path

from civitai import get_model_for_safetensors
from config import (
    APP_NAME,
    get_lora_root,
    is_dry_run,
    set_dry_run,
    is_verbose,
    set_verbose,
    log_operation,
    TMP_CLEANUP_AGE_SECONDS,
)
from downloader import (
    download_preview_if_missing,
    metadata_path,
    model_assets_complete,
    preview_image_exists,
    save_metadata_if_missing,
)
from gallery import build_gallery_index, save_gallery_html, metadata_belongs_to_file
from server import start_server
from organizer import find_lora_files, organize_library

try:
    from tqdm import tqdm
except ImportError:
    # Provide a small fallback when tqdm is not installed yet.
    def tqdm(iterable, **kwargs):
        return iterable


# Print a short application header for command-line use.
def print_header() -> None:
    dry_status = "ENABLED" if is_dry_run() else "DISABLED"
    verbose_status = "ENABLED" if is_verbose() else "DISABLED"

    print("==================================================")
    print(f"  {APP_NAME}")
    print(f"  Library Path: {get_lora_root()}")
    print(f"  Dry Run Mode: {dry_status} | Verbose Mode: {verbose_status}")
    print("==================================================")
    print()


# Clean up stale temporary files (.tmp) recursively under lora_root.
def cleanup_stale_tmp_files(lora_root: Path) -> None:
    threshold = TMP_CLEANUP_AGE_SECONDS
    now = time.time()

    try:
        for tmp_file in lora_root.rglob("*.tmp"):
            if tmp_file.is_file():
                try:
                    mtime = tmp_file.stat().st_mtime
                    age = now - mtime
                    if age > threshold:
                        if is_dry_run():
                            log_operation(f"[DRY RUN] Would clean up stale temporary file: {tmp_file} (age: {int(age)}s)")
                            print(f"[DRY RUN] Would clean up stale temporary file: {tmp_file.name}")
                        else:
                            tmp_file.unlink()
                            log_operation(f"Cleaned up stale temporary file: {tmp_file} (age: {int(age)}s)")
                            if is_verbose():
                                print(f"Cleaned up stale temporary file: {tmp_file.name}")
                except OSError as e:
                    log_operation(f"Error checking/deleting temporary file {tmp_file}: {e}")
    except OSError as e:
        log_operation(f"Error scanning for temporary files: {e}")


# Print the main menu options.
def print_menu() -> None:
    print("1. Automatic")
    print("2. Manual")
    print("3. Organize only")
    print("4. Download previews")
    print("5. Download metadata")
    print("6. Generate Gallery")
    print("7. Start Gallery Server")
    print("8. Export Library Manifest")
    print("9. Import Library Manifest")
    print("10. Exit")
    dry_status = "ON" if is_dry_run() else "OFF"
    verbose_status = "ON" if is_verbose() else "OFF"
    print(f"11. Toggle Dry Run Mode (Currently {dry_status})")
    print(f"12. Toggle Verbose Mode (Currently {verbose_status})")
    print("13. Set CivitAI API Key")
    print()


# Prepare the configured LoRA root folder.
def setup_project() -> Path:
    lora_root = get_lora_root()
    lora_root.mkdir(parents=True, exist_ok=True)
    cleanup_stale_tmp_files(lora_root)
    return lora_root


# Return every safetensors file found in a folder.
def find_safetensors_files(folder: Path) -> list[Path]:
    return [
        model_file
        for model_file in find_lora_files(folder)
        if model_file.suffix.lower() == ".safetensors"
    ]


# Load local metadata for a model folder or file when it exists.
def load_existing_metadata(model_path: Path) -> dict | None:
    if model_path.is_dir():
        path = model_path / "metadata.json"
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as metadata_file:
                return json.load(metadata_file)
        except Exception:
            return None
    else:
        # Check <stem>.json first
        path = model_path.parent / f"{model_path.stem}.json"
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as metadata_file:
                    data = json.load(metadata_file)
                    if metadata_belongs_to_file(data, model_path.name):
                        return data
            except Exception:
                pass
                
        # Check metadata.json fallback
        path = model_path.parent / "metadata.json"
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as metadata_file:
                    data = json.load(metadata_file)
                    if metadata_belongs_to_file(data, model_path.name):
                        return data
            except Exception:
                pass

    return None



# Add one result status to a summary counter.
def count_status(summary: dict[str, int], prefix: str, status: str) -> None:
    key = f"{prefix}_{status}"
    summary[key] = summary.get(key, 0) + 1


# Build an empty summary for asset download jobs.
def create_asset_summary() -> dict[str, int]:
    return {
        "models_checked": 0,
        "completed_skipped": 0,
        "api_errors": 0,
        "metadata_saved": 0,
        "metadata_skipped": 0,
        "metadata_error": 0,
        "preview_image_downloaded": 0,
        "preview_image_skipped": 0,
        "preview_image_missing": 0,
        "preview_image_error": 0,
    }


# Fetch CivitAI metadata for a safetensors model file.
def fetch_model_metadata(model_file: Path) -> dict:
    response = get_model_for_safetensors(model_file)

    if response.get("ok"):
        return response

    return {
        "ok": False,
        "error": response.get("error", "unknown_error"),
        "message": response.get("message", "Could not fetch CivitAI metadata."),
    }


# Process one safetensors model for metadata and preview downloads.
def process_model_assets(
    model_file: Path,
    download_metadata: bool,
    download_preview: bool,
    retries: int = 3,
) -> dict[str, int]:
    summary = create_asset_summary()
    summary["models_checked"] = 1

    if model_assets_complete(model_file):
        summary["completed_skipped"] += 1
        return summary

    metadata = load_existing_metadata(model_file)
    needs_remote_metadata = metadata is None and (
        download_metadata or (download_preview and not preview_image_exists(model_file))
    )

    if needs_remote_metadata:
        metadata = fetch_model_metadata(model_file)

        if not metadata.get("ok"):
            summary["api_errors"] += 1
            return summary

    if download_metadata and metadata is not None:
        status = save_metadata_if_missing(model_file, metadata)
        count_status(summary, "metadata", status["status"])

    if download_preview and metadata is not None:
        status = download_preview_if_missing(model_file, metadata, retries)
        count_status(summary, "preview_image", status["status"])

    return summary



# Merge one summary dictionary into another summary dictionary.
def merge_summary(total: dict[str, int], update: dict[str, int]) -> None:
    for key, value in update.items():
        total[key] = total.get(key, 0) + value


# Run an asset job across all safetensors files in a folder.
def run_asset_job(
    library_dir: Path,
    download_metadata: bool,
    download_preview: bool,
    retries: int = 3,
) -> dict[str, int]:
    summary = create_asset_summary()
    model_files = find_safetensors_files(library_dir)

    for model_file in tqdm(model_files, desc="Processing models", unit="model"):
        update = process_model_assets(model_file, download_metadata, download_preview, retries)
        merge_summary(summary, update)

    return summary


# Organize loose models inside the configured LoRA root.
def run_organize(lora_root: Path) -> dict[str, int]:
    return organize_library(lora_root, print_summary=False)


# Refresh the generated HTML gallery inside the configured LoRA root.
def refresh_gallery(lora_root: Path) -> Path:
    entries = build_gallery_index(lora_root)
    return save_gallery_html(entries, lora_root / "gallery.html")


# Run the automatic workflow: organize, download metadata, download previews, and refresh gallery.
def run_automatic(lora_root: Path) -> dict[str, dict[str, int] | str]:
    organize_summary = run_organize(lora_root)
    asset_summary = run_asset_job(lora_root, download_metadata=True, download_preview=True)
    gallery_index = refresh_gallery(lora_root)

    return {
        "organizer": organize_summary,
        "assets": asset_summary,
        "gallery_file": str(gallery_index),
    }


# Run the manual workflow for one user-provided safetensors file.
def run_manual() -> dict[str, int]:
    model_path = Path(input("Enter path to a .safetensors file: ").strip().strip('"'))

    if model_path.suffix.lower() != ".safetensors" or not model_path.exists():
        print("That file does not exist or is not a .safetensors file.")
        return create_asset_summary()

    return process_model_assets(model_path, download_metadata=True, download_preview=True)


# Print a dictionary or other summary in a formatted human-readable format.
def print_summary(title: str, summary: any) -> None:
    print()
    print("=" * 50)
    print(f" {title.upper()}")
    print("=" * 50)

    if isinstance(summary, dict):
        if "organizer" in summary or "assets" in summary or "gallery_file" in summary:
            # Automatic summary structure
            if "organizer" in summary and isinstance(summary["organizer"], dict):
                org = summary["organizer"]
                print("Organization Status:")
                print(f"  - Organized: {org.get('organized', 0)}")
                print(f"  - Skipped:   {org.get('skipped', 0)}")
                print(f"  - Errors:    {org.get('errors', 0)}")
                print()

            if "assets" in summary and isinstance(summary["assets"], dict):
                ast = summary["assets"]
                print("Asset Processing Status:")
                print(f"  - Models Checked:       {ast.get('models_checked', 0)}")
                print(f"  - Already Completed:    {ast.get('completed_skipped', 0)}")
                print(f"  - CivitAI API Errors:   {ast.get('api_errors', 0)}")
                print(f"  - Metadata Saved:       {ast.get('metadata_saved', 0)}")
                print(f"  - Metadata Skipped:     {ast.get('metadata_skipped', 0)}")
                print(f"  - Metadata Errors:      {ast.get('metadata_error', 0)}")
                print(f"  - Previews Downloaded:  {ast.get('preview_image_downloaded', 0)}")
                print(f"  - Previews Skipped:     {ast.get('preview_image_skipped', 0)}")
                print(f"  - Previews Missing:     {ast.get('preview_image_missing', 0)}")
                print(f"  - Previews Errors:      {ast.get('preview_image_error', 0)}")
                print()

            if "gallery_file" in summary:
                print("Gallery Generation:")
                print(f"  - File Path: {summary['gallery_file']}")
        
        elif "models_checked" in summary:
            # Asset job summary structure
            print("Asset Processing Status:")
            print(f"  - Models Checked:       {summary.get('models_checked', 0)}")
            print(f"  - Already Completed:    {summary.get('completed_skipped', 0)}")
            print(f"  - CivitAI API Errors:   {summary.get('api_errors', 0)}")
            print(f"  - Metadata Saved:       {summary.get('metadata_saved', 0)}")
            print(f"  - Metadata Skipped:     {summary.get('metadata_skipped', 0)}")
            print(f"  - Metadata Errors:      {summary.get('metadata_error', 0)}")
            print(f"  - Previews Downloaded:  {summary.get('preview_image_downloaded', 0)}")
            print(f"  - Previews Skipped:     {summary.get('preview_image_skipped', 0)}")
            print(f"  - Previews Missing:     {summary.get('preview_image_missing', 0)}")
            print(f"  - Previews Errors:      {summary.get('preview_image_error', 0)}")

        elif "organized" in summary:
            # Organizer summary structure
            print("Organization Status:")
            print(f"  - Organized: {summary.get('organized', 0)}")
            print(f"  - Skipped:   {summary.get('skipped', 0)}")
            print(f"  - Errors:    {summary.get('errors', 0)}")

        else:
            # Fallback for other dict structures
            for key, value in summary.items():
                print(f"  - {key.replace('_', ' ').title()}: {value}")
    else:
        # For non-dict summaries (e.g. Path or String)
        print(f"  - Result: {summary}")

    print("=" * 50)
    print()


# Run one menu choice and return whether the app should continue.
def handle_choice(choice: str, lora_root: Path) -> bool:
    if choice == "1":
        print_summary("Automatic summary", run_automatic(lora_root))
    elif choice == "2":
        print_summary("Manual summary", run_manual())
    elif choice == "3":
        print_summary("Organize summary", run_organize(lora_root))
    elif choice == "4":
        summary = run_asset_job(lora_root, download_metadata=False, download_preview=True)
        print_summary("Preview download summary", summary)
    elif choice == "5":
        summary = run_asset_job(lora_root, download_metadata=True, download_preview=False)
        print_summary("Metadata download summary", summary)
    elif choice == "6":
        summary = refresh_gallery(lora_root)
        print_summary("Gallery generation summary", summary)
    elif choice == "7":
        start_server(lora_root)
    elif choice == "8":
        manifest_filename = input("Enter output manifest filename (default: library_manifest.json): ").strip()
        if not manifest_filename:
            manifest_filename = "library_manifest.json"
        manifest_path = Path(manifest_filename)
        if not manifest_path.is_absolute():
            manifest_path = lora_root / manifest_path
        from manifest import export_library_manifest
        export_library_manifest(lora_root, manifest_path)
    elif choice == "9":
        manifest_filename = input("Enter input manifest filename (default: library_manifest.json): ").strip()
        if not manifest_filename:
            manifest_filename = "library_manifest.json"
        manifest_path = Path(manifest_filename)
        if not manifest_path.is_absolute():
            manifest_path = lora_root / manifest_path
        from manifest import import_library_manifest
        import_library_manifest(lora_root, manifest_path)
        
        # Automatically organize the imported files into their proper folders
        print("\nRunning library organizer to place imported models in their folders...")
        from organizer import organize_library
        organize_library(lora_root, lora_root)
        
        # Automatically refresh the gallery to reflect changes
        print("\nRefreshing gallery...")
        refresh_gallery(lora_root)
    elif choice == "10":
        print("Goodbye.")
        return False
    elif choice == "11":
        new_val = not is_dry_run()
        set_dry_run(new_val)
        print(f"Dry run mode is now {'ENABLED' if new_val else 'DISABLED'}.")
    elif choice == "12":
        new_val = not is_verbose()
        set_verbose(new_val)
        print(f"Verbose mode is now {'ENABLED' if new_val else 'DISABLED'}.")
    elif choice == "13":
        from config import get_civitai_api_key, set_civitai_api_key
        current_key = get_civitai_api_key()
        masked_key = f"{current_key[:4]}...{current_key[-4:]}" if len(current_key) > 8 else "Not Set"
        print(f"Current API Key: {masked_key}")
        new_key = input("Enter new CivitAI API Key (or type 'clear' to remove, press Enter to cancel): ").strip()
        if new_key:
            if new_key.lower() == "clear":
                set_civitai_api_key("")
                print("API Key cleared.")
            else:
                set_civitai_api_key(new_key)
                print("API Key updated and saved.")
    else:
        print("Please choose a number from 1 to 13.")

    return True


# Run the current non-GUI application workflow.
def main() -> None:
    lora_root = setup_project()

    while True:
        print_header()
        print_menu()
        choice = input("Choose an option: ").strip()

        if not handle_choice(choice, lora_root):
            break


if __name__ == "__main__":
    main()
