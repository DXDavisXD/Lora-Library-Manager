"""Command-line entry point for LoRA Library Manager.

The GUI is intentionally not implemented yet.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import json
import time

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
    is_complete_metadata,
)
from downloader import (
    download_preview_if_missing,
    metadata_path,
    model_assets_complete,
    preview_image_exists,
    save_metadata_if_missing,
    load_json_lenient,
)
from providers.provider_base import ModelInfo
from gallery import build_gallery_index, save_gallery_html, metadata_belongs_to_file
from library_index import load_library_index, detect_library_changes
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
    dry_status = "ON" if is_dry_run() else "OFF"
    verbose_status = "ON" if is_verbose() else "OFF"
    
    print("─── ⚡ CORE WORKFLOW ───")
    print("  1. 🔄 Run Automatic Sync (Organize + Asset Download + Rebuild Gallery)")
    print("  2. 🚀 Start Gallery Web Server")
    print()
    print("─── 🗂️ LIBRARY MAINTENANCE ───")
    print("  3. ⚡ Quick Refresh (Scan files & regenerate gallery)")
    print("  4. 🔧 Open Manual Operations Submenu (Targeted organize, downloads, or rebuilds)")
    print("  5. 🔍 Open Diagnostics & Cleanup Submenu (Duplicates, audits, WebUI migration)")
    print()
    print("─── 🧙‍♂️ IMPORT & BACKUPS ───")
    print("  6. 🗺️ Open Folder Migration Wizard (Convert folder structure to collections)")
    print("  7. 📤 Export Library Backup Manifest")
    print("  8. 📥 Import Library Backup Manifest")
    print()
    print("─── ⚙️ SETTINGS & SYSTEM ───")
    print(f"  9.  🚫 Toggle Dry Run Mode    [Currently: {dry_status}]")
    print(f"  10. 🔊 Toggle Verbose Mode   [Currently: {verbose_status}]")
    print("  11. 🔑 Set CivitAI API Key")
    print()
    print("  0. 🚪 Exit")
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
        # Load and merge both metadata.json and <stem>.json if they exist
        merged_data = {}
        found = False

        # 1. Load from metadata.json (shared/fallback)
        path = model_path.parent / "metadata.json"
        if path.exists():
            data = load_json_lenient(path)
            if isinstance(data, dict) and metadata_belongs_to_file(data, model_path.name):
                from downloader import merge_metadata
                merged_data = merge_metadata(merged_data, data)
                found = True

        # 2. Load from <stem>.json (specific/target)
        path = model_path.parent / f"{model_path.stem}.json"
        if path.exists():
            data = load_json_lenient(path)
            if isinstance(data, dict) and metadata_belongs_to_file(data, model_path.name):
                from downloader import merge_metadata
                merged_data = merge_metadata(merged_data, data)
                found = True

        return merged_data if found else None



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


# Fetch metadata for a safetensors model file from the auto-detected provider.
def fetch_model_info(model_file: Path) -> ModelInfo | None:
    """
    Fetch model version metadata from the auto-detected provider by file hash.
    Returns ModelInfo or None.
    """
    from providers.registry import resolve_provider, get_provider
    from config import get_lora_root, get_provider_api_key
    
    # 1. Load existing metadata to help resolve provider
    metadata = load_existing_metadata(model_file)
    
    # 2. Get file hash if available
    sha256 = ""
    if metadata:
        from gallery import get_sha256_from_metadata
        sha256 = get_sha256_from_metadata(metadata, model_file.name)
    if not sha256:
        # Use cached or compute
        from library_index import load_library_index, make_index_key
        lora_root = get_lora_root()
        index = load_library_index(lora_root)
        key = make_index_key(model_file, lora_root)
        cached = index.get("models", {}).get(key, {})
        sha256 = cached.get("sha256", "")
    if not sha256:
        from civitai import compute_sha256
        try:
            sha256 = compute_sha256(model_file)
        except OSError:
            pass
            
    # 3. Resolve the provider
    provider_name = resolve_provider(meta=metadata, sha256=sha256, filename=model_file.name)
    provider = get_provider(provider_name)
    
    # 4. Fetch from provider if it supports hash lookup
    if provider.capabilities.hash_lookup and sha256:
        api_key = get_provider_api_key(provider_name)
        try:
            return provider.fetch_by_hash(sha256, api_key=api_key)
        except Exception as e:
            log_operation(f"API fetch failed via {provider_name} for {model_file.name}: {e}")
            
    return None


# Process one safetensors model for metadata and preview downloads.
def process_model_assets(
    model_file: Path,
    download_metadata: bool,
    download_preview: bool,
    retries: int = 3,
    force_update: bool = False,
) -> dict[str, int]:
    from jobs import update_job_progress, get_current_job
    summary = create_asset_summary()
    summary["models_checked"] = 1

    if not force_update and model_assets_complete(model_file):
        summary["completed_skipped"] += 1
        return summary

    metadata = load_existing_metadata(model_file)
    needs_remote_metadata = force_update or (
        (metadata is None or not is_complete_metadata(metadata)) and (
            download_metadata or (download_preview and not preview_image_exists(model_file))
        )
    )

    job = get_current_job()
    is_single = job is None or job.progress_total <= 1

    model_info = None
    if needs_remote_metadata:
        update_job_progress(
            percent=30 if is_single else None,
            message=f"Fetching metadata for {model_file.stem}..."
        )
        model_info = fetch_model_info(model_file)

        if model_info is None:
            summary["api_errors"] += 1
            return summary
    elif metadata is not None:
        # Load ModelInfo from existing metadata
        from providers.registry import resolve_provider, get_provider
        p_key = resolve_provider(meta=metadata)
        provider = get_provider(p_key)
        # unwrap and convert
        from civitai import unwrap_civitai_metadata
        unwrapped = unwrap_civitai_metadata(metadata)
        if p_key == "civitai" and hasattr(provider, "_to_model_info"):
            model_info = provider._to_model_info(unwrapped)

    # Use model_info if we have it; otherwise default to metadata (dict) for fallback compatibility
    metadata_arg = model_info if model_info is not None else metadata

    if download_metadata and metadata_arg is not None:
        update_job_progress(
            percent=60 if is_single else None,
            message=f"Saving metadata for {model_file.stem}..."
        )
        status = save_metadata_if_missing(model_file, metadata_arg)
        count_status(summary, "metadata", status["status"])

    if download_preview and metadata_arg is not None:
        if not preview_image_exists(model_file):
            update_job_progress(
                percent=80 if is_single else None,
                message=f"Downloading preview for {model_file.stem}..."
            )
        status = download_preview_if_missing(model_file, metadata_arg, retries)
        count_status(summary, "preview_image", status["status"])

    return summary



# Merge one summary dictionary into another summary dictionary.
def merge_summary(total: dict[str, int], update: dict[str, int]) -> None:
    for key, value in update.items():
        total[key] = total.get(key, 0) + value


# Run an asset job across safetensors files in a folder.
def run_asset_job(
    library_dir: Path,
    download_metadata: bool,
    download_preview: bool,
    retries: int = 3,
    target_files: list[Path] | None = None,
    start_percent: int = 5,
    scale_percent: int = 85,
    force_update: bool = False,
) -> dict[str, int]:
    summary = create_asset_summary()
    model_files = target_files if target_files is not None else find_safetensors_files(library_dir)

    total_files = len(model_files)
    from jobs import update_job_progress

    if force_update:
        msg = "Updating metadata..."
    else:
        msg = "Downloading metadata..." if download_metadata and not download_preview else (
              "Downloading previews..." if download_preview and not download_metadata else 
              "Downloading assets..."
        )

    for idx, model_file in enumerate(model_files):
        current_idx = idx + 1
        percent = int((idx / total_files) * scale_percent) + start_percent
        update_job_progress(
            percent=percent, 
            current=current_idx, 
            total=total_files, 
            message=f"{msg} ({current_idx}/{total_files})", 
            item_name=model_file.name
        )
        update = process_model_assets(model_file, download_metadata, download_preview, retries, force_update=force_update)
        merge_summary(summary, update)

    return summary


# Organize loose models inside the configured LoRA root.
# Pass changed_files to restrict processing to only added/modified paths.
def run_organize(lora_root: Path, changed_files: list[Path] | None = None) -> dict[str, int]:
    stats, _ = organize_library(lora_root, print_summary=False, changed_files=changed_files)
    return stats


# Refresh the generated HTML gallery inside the configured LoRA root.
def refresh_gallery(lora_root: Path) -> Path:
    from jobs import update_job_progress, get_current_job
    job = get_current_job()
    is_dedicated = job is not None and job.name == "Rebuild Gallery"
    
    update_job_progress(
        percent=40 if is_dedicated else None,
        message="Compiling gallery database..."
    )
    entries = build_gallery_index(lora_root, use_cache=not is_dedicated)
    
    update_job_progress(
        percent=80 if is_dedicated else None,
        message="Generating gallery.html..."
    )
    return save_gallery_html(entries, lora_root / "gallery.html")


# Lightweight refresh: detect changes and regenerate gallery only if needed.
# Does NOT trigger organize or asset downloads.
def run_quick_refresh(lora_root: Path) -> dict:
    print("Checking for library changes...")
    log_operation("[Index] Quick Refresh started")
    index = load_library_index(lora_root)
    changes = detect_library_changes(lora_root, index)

    n_added = len(changes["added"])
    n_modified = len(changes["modified"])
    n_removed = len(changes["removed"])

    if n_added == 0 and n_modified == 0 and n_removed == 0:
        msg = "[Index] No library changes detected."
        print(msg)
        log_operation(msg)
        return {"quick_refresh": "no_changes", "gallery_file": str(lora_root / "gallery.html")}

    print(f"Changes detected: {n_added} added, {n_modified} modified, {n_removed} removed")
    gallery_file = refresh_gallery(lora_root)
    log_operation("[Index] Quick Refresh complete")
    return {"quick_refresh": "updated", "gallery_file": str(gallery_file)}


# Run the automatic workflow: organize, download metadata, download previews, and refresh gallery.
# Loads the library index first; exits immediately if no files have changed.
def run_automatic(lora_root: Path) -> dict[str, dict[str, int] | str]:
    print("Checking for library changes...")
    log_operation("[Index] Automatic run started")
    index = load_library_index(lora_root)
    changes = detect_library_changes(lora_root, index)

    added = changes["added"]
    modified = changes["modified"]
    removed = changes["removed"]

    if len(added) == 0 and len(modified) == 0 and len(removed) == 0:
        msg = "[Index] No library changes detected. Skipping organize, asset download, and gallery refresh."
        print(msg)
        log_operation(msg)
        return {
            "organizer": {"organized": 0, "skipped": 0, "errors": 0},
            "assets": create_asset_summary(),
            "gallery_file": str(lora_root / "gallery.html"),
            "no_changes": True,
        }

    # Changes detected — run targeted organize then targeted asset job and gallery refresh.
    from jobs import update_job_progress
    print(f"Changes detected: {len(added)} added, {len(modified)} modified, {len(removed)} removed")
    changed_list = added + modified
    
    update_job_progress(percent=5, message="Organizing library files...")
    organize_summary, post_organize_files = organize_library(lora_root, print_summary=False, changed_files=changed_list)

    update_job_progress(percent=15, message="Syncing assets...")
    asset_summary = run_asset_job(
        lora_root, 
        download_metadata=True, 
        download_preview=True, 
        target_files=post_organize_files,
        start_percent=15,
        scale_percent=70
    )
    
    update_job_progress(percent=85, message="Updating gallery view...")
    gallery_index = refresh_gallery(lora_root)
    
    update_job_progress(percent=100, message="Automatic run complete.")
    log_operation("[Index] Automatic run complete")

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
        if "quick_refresh" in summary:
            # Quick Refresh summary structure
            status = summary.get("quick_refresh", "")
            if status == "no_changes":
                print("  - Status: No library changes detected — gallery unchanged.")
            else:
                print("  - Status: Changes detected — gallery regenerated.")
            if "gallery_file" in summary:
                print(f"  - Gallery File: {summary['gallery_file']}")
        elif "organizer" in summary or "assets" in summary or "gallery_file" in summary:
            # Automatic summary structure
            if summary.get("no_changes"):
                print("  - Status: No library changes detected.")
                print("  - Organize, asset download, and gallery refresh were all skipped.")
                if "gallery_file" in summary:
                    print(f"  - Gallery File: {summary['gallery_file']}")
                return
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
        start_server(lora_root)
    elif choice == "3":
        print_summary("Quick Refresh", run_quick_refresh(lora_root))
    elif choice == "4":
        while True:
            print("\n--- Manual Operations Submenu ---")
            print("1. Process Single File")
            print("2. Organize Only")
            print("3. Download Previews Only")
            print("4. Download Metadata Only")
            print("5. Generate Gallery Only")
            print("6. Generate Missing SHA256 Hashes")
            print("7. Go Back to Main Menu")
            print()
            sub_choice = input("Choose a manual option: ").strip()
            
            if sub_choice == "1":
                print_summary("Manual summary", run_manual())
            elif sub_choice == "2":
                print_summary("Organize summary", run_organize(lora_root))
            elif sub_choice == "3":
                summary = run_asset_job(lora_root, download_metadata=False, download_preview=True)
                print_summary("Preview download summary", summary)
            elif sub_choice == "4":
                summary = run_asset_job(lora_root, download_metadata=True, download_preview=False)
                print_summary("Metadata download summary", summary)
            elif sub_choice == "5":
                summary = refresh_gallery(lora_root)
                print_summary("Gallery generation summary", summary)
            elif sub_choice == "6":
                from migration import generate_missing_sha256_hashes
                generate_missing_sha256_hashes(lora_root)
            elif sub_choice == "7":
                break
            else:
                print("Please choose a number from 1 to 7.")
    elif choice == "5":
        while True:
            print("\n--- Diagnostics & Cleanup Submenu ---")
            print("1. Run CivitAI Update Detector")
            print("2. Run Duplicate LoRA Finder & Cleanup")
            print("3. Run Asset Audit & Bulk Downloader")
            print("4. Migrate External Metadata to User Data")
            print("5. Go Back to Main Menu")
            print()
            diag_choice = input("Choose a diagnostics option: ").strip()
            
            if diag_choice == "1":
                from diagnostics import run_update_detector
                run_update_detector(lora_root)
            elif diag_choice == "2":
                from diagnostics import run_duplicate_finder
                run_duplicate_finder(lora_root)
            elif diag_choice == "3":
                from diagnostics import run_asset_auditer_downloader
                run_asset_auditer_downloader(lora_root)
            elif diag_choice == "4":
                from diagnostics import migrate_metadata_to_user_data
                migrate_metadata_to_user_data(lora_root)
            elif diag_choice == "5":
                break
            else:
                print("Please choose a number from 1 to 5.")
    elif choice == "6":
        while True:
            print("\n--- Migration Wizard ---")
            print("1. Preview Library Organization (Read-Only)")
            print("2. Import Existing Library (Convert Folders to Collections)")
            print("3. Go Back to Main Menu")
            print()
            sub_choice = input("Choose an option: ").strip()
            
            if sub_choice == "1":
                from migration import scan_library_hierarchy, generate_migration_preview
                print("Scanning library for migration preview...")
                data = scan_library_hierarchy(lora_root)
                generate_migration_preview(data)
            elif sub_choice == "2":
                from migration import import_existing_library
                import_existing_library(lora_root)
            elif sub_choice == "3":
                break
            else:
                print("Please choose a number from 1 to 3.")
    elif choice == "7":
        from datetime import datetime
        default_name = f"backup/{datetime.now().strftime('%d.%m.%Y-%H.%M')}.json"
        manifest_filename = input(f"Enter output manifest filename (default: {default_name}): ").strip()
        if not manifest_filename:
            manifest_filename = default_name
        manifest_path = Path(manifest_filename)
        if not manifest_path.is_absolute():
            manifest_path = lora_root / manifest_path
        from manifest import export_library_manifest
        export_library_manifest(lora_root, manifest_path)
    elif choice == "8":
        from datetime import datetime
        backup_dir = lora_root / "backup"
        backups = []
        if backup_dir.exists() and backup_dir.is_dir():
            backups = sorted(
                [p for p in backup_dir.glob("*.json")],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
        manifest_path = None
        if backups:
            print("\nAvailable backups found in backup/ folder:")
            for idx, b in enumerate(backups, 1):
                mtime_str = datetime.fromtimestamp(b.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                print(f"  {idx}. {b.name}  (Modified: {mtime_str})")
            print("  c. Specify custom path/filename")
            print()
            sel = input("Choose a backup to import (default: 1): ").strip().lower()
            if not sel or sel == "1":
                manifest_path = backups[0]
            elif sel == "c":
                manifest_filename = input("Enter manifest path/filename: ").strip()
                if manifest_filename:
                    manifest_path = Path(manifest_filename)
            elif sel.isdigit() and 1 <= int(sel) <= len(backups):
                manifest_path = backups[int(sel) - 1]
            else:
                print("Invalid selection, cancelling.")
                return True
        else:
            manifest_filename = input("Enter input manifest filename (default: library_manifest.json): ").strip()
            if not manifest_filename:
                manifest_filename = "library_manifest.json"
            manifest_path = Path(manifest_filename)
            
        if not manifest_path:
            print("No manifest path selected, cancelling.")
            return True
            
        if not manifest_path.is_absolute():
            manifest_path = lora_root / manifest_path
            
        from manifest import import_library_manifest
        import_library_manifest(lora_root, manifest_path)
        
        # Automatically organize the imported files into their proper folders
        print("\nRunning library organizer to place imported models in their folders...")
        from organizer import organize_library
        organize_library(lora_root, print_summary=False)
        
        # Automatically refresh the gallery to reflect changes
        print("\nRefreshing gallery...")
        refresh_gallery(lora_root)
    elif choice == "9":
        new_val = not is_dry_run()
        set_dry_run(new_val)
        print(f"Dry run mode is now {'ENABLED' if new_val else 'DISABLED'}.")
    elif choice == "10":
        new_val = not is_verbose()
        set_verbose(new_val)
        print(f"Verbose mode is now {'ENABLED' if new_val else 'DISABLED'}.")
    elif choice == "11":
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
    elif choice == "0":
        print("Goodbye.")
        return False
        print("Please choose a number from 0 to 11.")

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
