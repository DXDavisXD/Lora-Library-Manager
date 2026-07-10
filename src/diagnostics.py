"""Diagnostics, duplicate cleanup, and asset audit utilities for LoRA Library Manager."""

import os
import json
import time
from pathlib import Path
from datetime import datetime

from config import (
    is_dry_run,
    is_verbose,
    log_operation,
    LORA_EXTENSIONS,
    is_lora_file,
    PREVIEW_EXTENSIONS,
    is_complete_metadata,
)
from civitai import get_model_by_hash, get_model, compute_sha256
from gallery import load_metadata, unwrap_metadata, get_sha256_from_metadata
from downloader import download_model_assets, preview_image_exists, model_file_assets_complete
from main import refresh_gallery


# ===========================================================================
# 1. CivitAI Update Detector
# ===========================================================================

def run_update_detector(lora_root: Path) -> None:
    print("\n==================================================")
    print(" CIVITAI UPDATE DETECTOR")
    print("==================================================")
    print("Scanning library for models with update metadata...")

    # Collect all safetensors files
    lora_files = []
    for ext in LORA_EXTENSIONS:
        for path in lora_root.rglob(f"*{ext}"):
            if is_lora_file(path):
                lora_files.append(path)

    # Deduplicate
    seen = set()
    unique_files = []
    for path in lora_files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(path)

    total_files = len(unique_files)
    print(f"Found {total_files} unique LoRA files.")

    updates_found = []
    checked_count = 0

    from providers.registry import resolve_provider, get_provider
    from config import get_provider_api_key

    for idx, path in enumerate(unique_files, 1):
        metadata = load_metadata(path)
        if not metadata:
            continue

        provider_name = resolve_provider(meta=metadata, filename=path.name)
        provider = get_provider(provider_name)

        if not provider.capabilities.update_check:
            continue

        # Extract provider-specific model ID and version ID
        if "provider" in metadata:
            provider_model_id = metadata.get("provider_model_id")
            current_version_id = metadata.get("provider_version_id")
        else:
            # Legacy fallback
            from civitai import unwrap_civitai_metadata
            unwrapped = unwrap_civitai_metadata(metadata)
            provider_model_id = str(unwrapped.get("modelId") or (unwrapped.get("model") or {}).get("id") or "")
            current_version_id = str(unwrapped.get("id") or "")

        if not provider_model_id or not current_version_id:
            continue

        checked_count += 1
        model_name = metadata.get("name") or (metadata.get("model") or {}).get("name") or path.stem
        current_version_name = metadata.get("version_name") or metadata.get("name") or "Unknown"

        print(f"[{checked_count}] Checking '{model_name}' (Current: {current_version_name} via {provider_name.title()})...")

        try:
            api_key = get_provider_api_key(provider_name)
            versions = provider.fetch_model_versions(provider_model_id, api_key=api_key)
            if versions:
                # First version in returned list is the latest version
                latest_version = versions[0]
                latest_version_id = latest_version.provider_version_id

                if latest_version_id and latest_version_id != current_version_id:
                    latest_version_name = latest_version.version_name or "Unknown"
                    # Get download URL via capability
                    download_url = ""
                    if provider.capabilities.model_download:
                        download_url = provider.get_download_url(provider_model_id, latest_version_id, api_key=api_key)

                    updates_found.append({
                        "file_name": path.name,
                        "model_name": model_name,
                        "current_version": current_version_name,
                        "latest_version": latest_version_name,
                        "download_url": download_url
                    })
                    print(f"  -> UPDATE AVAILABLE: New version '{latest_version_name}' found!")
            else:
                print(f"  -> No versions returned from {provider_name.title()}.")

        except Exception as e:
            print(f"  -> Error checking updates: {e}")

        # Be polite to APIs
        time.sleep(0.5)

    print("\n==================================================")
    print(" UPDATE CHECK COMPLETE")
    print("==================================================")
    print(f"Total checked: {checked_count} models.")
    
    if updates_found:
        print(f"Updates available: {len(updates_found)} model(s) have new versions.\n")
        for u in updates_found:
            print(f"- {u['model_name']} ({u['file_name']})")
            print(f"  Current: {u['current_version']}")
            print(f"  Latest:  {u['latest_version']}")
            if u["download_url"]:
                print(f"  Download: {u['download_url']}")
            print()
    else:
        print("All checked models are up to date!")


# ===========================================================================
# 2. Duplicate LoRA Finder & Cleanup Tool
# ===========================================================================

def delete_associated_assets(model_path: Path) -> None:
    """Find and delete preview images and metadata json files sharing the same stem."""
    stem = model_path.stem
    folder = model_path.parent
    
    # Previews
    for ext in PREVIEW_EXTENSIONS:
        p_path = folder / f"{stem}{ext}"
        if p_path.exists():
            if is_dry_run():
                print(f"[DRY RUN] Would delete preview image: {p_path.name}")
            else:
                try:
                    p_path.unlink()
                    print(f"Deleted preview image: {p_path.name}")
                except OSError as e:
                    print(f"Error deleting preview image {p_path.name}: {e}")

    # Metadata files
    meta_candidates = [
        folder / f"{stem}.json",
        folder / f"{stem}.civitai.info"
    ]
    for m_path in meta_candidates:
        if m_path.exists():
            if is_dry_run():
                print(f"[DRY RUN] Would delete metadata file: {m_path.name}")
            else:
                try:
                    m_path.unlink()
                    print(f"Deleted metadata file: {m_path.name}")
                except OSError as e:
                    print(f"Error deleting metadata file {m_path.name}: {e}")


def run_duplicate_finder(lora_root: Path, strategy: str = "scan-only") -> None:
    print("\n==================================================")
    print(" DUPLICATE LORA FILE FINDER")
    print("==================================================")
    print("Scanning folders for files of identical sizes...")

    # Group files by size
    files_by_size = {}
    for ext in LORA_EXTENSIONS:
        for path in lora_root.rglob(f"*{ext}"):
            if is_lora_file(path):
                try:
                    size = path.stat().st_size
                    files_by_size.setdefault(size, []).append(path)
                except OSError:
                    pass

    # Keep only groups that have more than 1 file
    candidate_groups = {size: paths for size, paths in files_by_size.items() if len(paths) > 1}

    if not candidate_groups:
        print("No duplicate files found by file size.")
        return

    print(f"Computing SHA256 hashes for {len(candidate_groups)} size-grouped candidates...")
    
    # Hash and group by hash
    files_by_hash = {}
    total_to_hash = sum(len(paths) for paths in candidate_groups.values())
    hashed_count = 0
    from jobs import update_job_progress, get_current_job
    job = get_current_job()

    for size, paths in candidate_groups.items():
        for path in paths:
            hashed_count += 1
            if job:
                percent = int((hashed_count / total_to_hash) * 70) + 15
                update_job_progress(
                    percent=percent,
                    current=hashed_count,
                    total=total_to_hash,
                    message=f"Hashing candidate files... ({hashed_count}/{total_to_hash})",
                    item_name=path.name
                )
            print(f"Hashing '{path.name}' ({size / (1024*1024):.1f} MB)...")
            try:
                sha = compute_sha256(path)
                if sha:
                    files_by_hash.setdefault(sha, []).append(path)
            except Exception as e:
                print(f"Error hashing {path.name}: {e}")

    # Keep groups that have more than 1 file per hash
    duplicate_groups = {sha: paths for sha, paths in files_by_hash.items() if len(paths) > 1}

    if not duplicate_groups:
        print("No duplicate files found by SHA256 hash comparison.")
        return

    print(f"\nFound {len(duplicate_groups)} duplicate group(s):")
    
    groups_list = list(duplicate_groups.items())
    for idx, (sha, paths) in enumerate(groups_list, 1):
        size_bytes = paths[0].stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        print(f"\n[Group {idx}] Hash: {sha[:10]}... (Size: {size_mb:.1f} MB)")
        for file_idx, p in enumerate(paths, 1):
            mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            # Show path relative to lora_root
            rel_path = p.relative_to(lora_root)
            print(f"  {file_idx}. {rel_path} (Modified: {mtime})")

    print("\nDuplicate Cleanup Options:")
    print("  1. Keep OLDEST modified file in each group and delete duplicates")
    print("  2. Keep NEWEST modified file in each group and delete duplicates")
    print("  3. Select files to delete manually")
    print("  4. Cancel (Do nothing)")
    print()
    
    if strategy not in ("keep-oldest", "keep-newest", "scan-only"):
        choice = input("Select a cleanup option (1-4): ").strip()
    else:
        choice = "1" if strategy == "keep-oldest" else "2" if strategy == "keep-newest" else "4"
    
    if choice not in ("1", "2", "3"):
        print("Cleanup cancelled. No files deleted.")
        return

    dry_prefix = "[DRY RUN] " if is_dry_run() else ""
    if is_dry_run():
        print("\n--- DRY RUN ACTIVE: No files will be deleted ---")

    if choice in ("1", "2"):
        keep_oldest = (choice == "1")
        print(f"\nCleaning up using strategy: {'Keep Oldest' if keep_oldest else 'Keep Newest'}...")
        
        for sha, paths in duplicate_groups.items():
            # Sort paths by modified time
            sorted_paths = sorted(paths, key=lambda p: p.stat().st_mtime)
            
            if keep_oldest:
                to_keep = sorted_paths[0]
                to_delete = sorted_paths[1:]
            else:
                to_keep = sorted_paths[-1]
                to_delete = sorted_paths[:-1]
                
            print(f"\nGroup {sha[:8]}: Keeping '{to_keep.relative_to(lora_root)}'")
            for p in to_delete:
                print(f" -> Deleting duplicate file: '{p.relative_to(lora_root)}'")
                if not is_dry_run():
                    try:
                        p.unlink()
                    except OSError as e:
                        print(f"Error deleting file {p.name}: {e}")
                delete_associated_assets(p)

    elif choice == "3":
        for sha, paths in duplicate_groups.items():
            size_mb = paths[0].stat().st_size / (1024 * 1024)
            print(f"\nGroup Hash: {sha[:10]}... ({size_mb:.1f} MB)")
            for file_idx, p in enumerate(paths, 1):
                mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                print(f"  {file_idx}. {p.relative_to(lora_root)} (Modified: {mtime})")
            
            keep_sel = input(f"Which file index (1-{len(paths)}) would you like to KEEP? (Press Enter to skip this group): ").strip()
            if not keep_sel.isdigit() or not (1 <= int(keep_sel) <= len(paths)):
                print("Skipping this group.")
                continue
                
            keep_idx = int(keep_sel) - 1
            to_keep = paths[keep_idx]
            
            for file_idx, p in enumerate(paths):
                if file_idx == keep_idx:
                    continue
                print(f" -> Deleting duplicate file: '{p.relative_to(lora_root)}'")
                if not is_dry_run():
                    try:
                        p.unlink()
                    except OSError as e:
                        print(f"Error deleting file {p.name}: {e}")
                delete_associated_assets(p)

    print("\nCleanup complete.")
    # Rebuild gallery index to clear deleted files
    refresh_gallery(lora_root)


# ===========================================================================
# 3. Asset Audit & Bulk Downloader
# ===========================================================================

def run_asset_auditer_downloader(lora_root: Path) -> None:
    print("\n==================================================")
    print(" ASSET AUDIT & BULK DOWNLOADER")
    print("==================================================")
    print("Auditing library for missing preview images and metadata JSON files...")

    # Collect all safetensors files
    lora_files = []
    for ext in LORA_EXTENSIONS:
        for path in lora_root.rglob(f"*{ext}"):
            if is_lora_file(path):
                lora_files.append(path)

    # Deduplicate
    seen = set()
    unique_files = []
    for path in lora_files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(path)

    total_files = len(unique_files)

    missing_metadata = []
    missing_preview = []
    missing_both = []

    for path in unique_files:
        meta_dict = load_metadata(path)
        has_metadata = bool(meta_dict) and is_complete_metadata(meta_dict)
        has_preview = preview_image_exists(path)

        if not has_metadata and not has_preview:
            missing_both.append(path)
        elif not has_metadata:
            missing_metadata.append(path)
        elif not has_preview:
            missing_preview.append(path)

    print("\n--- Audit Summary ---")
    print(f"Total LoRA Files Audited: {total_files}")
    print(f"Models missing metadata:  {len(missing_metadata)}")
    print(f"Models missing previews:  {len(missing_preview)}")
    print(f"Models missing both:      {len(missing_both)}")
    print("---------------------")

    total_missing = len(missing_metadata) + len(missing_preview) + len(missing_both)
    if total_missing == 0:
        print("All assets (previews and metadata) are complete!")
        return

    download_choice = input(f"\nWould you like to bulk-download missing assets from CivitAI for {total_missing} files? (y/n): ").strip().lower()
    if download_choice != "y":
        print("Bulk download cancelled.")
        return

    # Combine all missing items
    all_missing_files = list(set(missing_metadata + missing_preview + missing_both))
    
    success_count = 0
    fail_count = 0

    print(f"\nStarting bulk download of missing assets...")

    for idx, path in enumerate(all_missing_files, 1):
        print(f"[{idx}/{len(all_missing_files)}] Processing '{path.name}'...")
        
        # 1. Resolve hash
        metadata = load_metadata(path)
        sha256 = get_sha256_from_metadata(metadata, path.name)
        if not sha256:
            print("  -> Hashing local file (can take a moment)...")
            sha256 = compute_sha256(path)
            
        if not sha256:
            print("  -> Could not resolve hash for file. Skipping.")
            fail_count += 1
            continue

        try:
            res = get_model_by_hash(sha256)
            if res.get("ok") and isinstance(res.get("data"), dict):
                civitai_data = res["data"]
                # Save metadata and/or download missing previews
                results = download_model_assets(path, civitai_data)
                
                meta_status = results.get("metadata", {}).get("status")
                prev_status = results.get("preview_image", {}).get("status")
                
                print(f"  -> Metadata: {meta_status.upper() if meta_status else 'SKIP'}")
                print(f"  -> Preview:  {prev_status.upper() if prev_status else 'SKIP'}")
                success_count += 1
            else:
                error_msg = res.get("message") or "Model not found on CivitAI"
                print(f"  -> CivitAI lookup failed: {error_msg}")
                fail_count += 1
        except Exception as e:
            print(f"  -> Error fetching assets: {e}")
            fail_count += 1

        # Be polite to CivitAI API
        time.sleep(0.5)

    print("\nBulk download complete.")
    print(f"Successfully processed: {success_count} models.")
    print(f"Failed / Not Found:     {fail_count} models.")

    if success_count > 0:
        print("\nRegenerating gallery index...")
        refresh_gallery(lora_root)


def migrate_metadata_to_user_data(lora_root: Path) -> None:
    print("\n==================================================")
    print(" MIGRATING EXTERNAL METADATA TO USER DATA")
    print("==================================================")
    print("Scanning library for metadata files...")

    metadata_files = []
    ignored_names = {"collections.json", "library_index.json", "user_data.json", "settings.json"}
    for p in lora_root.rglob("*.json"):
        if p.name in ignored_names:
            continue
        metadata_files.append(p)

    total_files = len(metadata_files)
    print(f"Found {total_files} metadata files.")

    migrated_count = 0
    skipped_count = 0

    from downloader import import_metadata_to_user_data_dict

    for idx, p in enumerate(metadata_files, 1):
        folder = p.parent
        user_data_path = folder / "user_data.json"

        try:
            with p.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            print(f"  [{idx}/{total_files}] Error loading metadata from {p.name}: {e}")
            skipped_count += 1
            continue

        if not isinstance(meta, dict):
            skipped_count += 1
            continue

        user_data = {}
        if user_data_path.exists():
            try:
                with user_data_path.open("r", encoding="utf-8") as f:
                    user_data = json.load(f)
            except Exception:
                pass

        if not isinstance(user_data, dict):
            user_data = {}

        modified = import_metadata_to_user_data_dict(meta, user_data)
        if modified:
            user_data["schema_version"] = 1
            user_data["last_modified"] = datetime.now().isoformat() + "Z"
            if is_dry_run():
                print(f"  [{idx}/{total_files}] [DRY RUN] Would update user_data.json in {folder.name}")
            else:
                try:
                    with user_data_path.open("w", encoding="utf-8") as f:
                        json.dump(user_data, f, indent=2)
                    print(f"  [{idx}/{total_files}] Updated user_data.json in {folder.name}")
                except Exception as e:
                    print(f"  [{idx}/{total_files}] Error saving user_data.json: {e}")
                    skipped_count += 1
                    continue
            migrated_count += 1
        else:
            skipped_count += 1

    print("\nMigration complete.")
    print(f"Updated sidecars: {migrated_count}")
    print(f"Skipped sidecars: {skipped_count}")

    if migrated_count > 0 and not is_dry_run():
        print("\nRegenerating gallery index...")
        refresh_gallery(lora_root)
