"""Library Manifest (Snapshot) export and import helpers for LoRA Library Manager."""

import json
import os
import hashlib
import time
import sys
import re
from pathlib import Path
from datetime import datetime
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from config import (
    VERSION as MANAGER_VERSION,
    get_lora_root,
    is_lora_file,
    LORA_EXTENSIONS,
    get_civitai_api_key,
    log_operation
)
from civitai import unwrap_civitai_metadata, success_response, error_response, fetch_json
from gallery import load_metadata, unwrap_metadata, get_sha256_from_metadata
from downloader import download_to_path, download_preview_if_missing, preview_image_path

MANIFEST_VERSION = 1
DEFAULT_TIMEOUT = 60
CHUNK_SIZE = 1024 * 1024

# Compute a structural checksum of the manifest content (excluding 'checksum' key)
def compute_manifest_checksum(manifest: dict) -> str:
    # Make a copy and omit 'checksum' if present
    manifest_copy = dict(manifest)
    manifest_copy.pop("checksum", None)
    
    # Serialize to deterministic sorted compact JSON
    serialized = json.dumps(manifest_copy, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

# Helper to compute SHA256 from a local file
def compute_file_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    try:
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
                sha256.update(chunk)
        return sha256.hexdigest().lower()
    except Exception as e:
        log_operation(f"Error computing hash for {file_path}: {e}")
        return ""

# Export the library manifest
def export_library_manifest(lora_root: Path, output_file: Path, favorite_hashes: set[str] | None = None) -> Path:
    print("Scanning library files for export...")
    
    # Scan for safetensors files
    lora_files = []
    for ext in LORA_EXTENSIONS:
        for path in lora_root.rglob(f"*{ext}"):
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
            
    models = []
    user_data_dict = {}
    exported_models_count = 0
    exported_user_data_count = 0
    
    for path in unique_files:
        # Load local metadata.json if it exists
        metadata = load_metadata(path)
        unwrapped = unwrap_metadata(metadata)
        
        # 1. Resolve SHA256 (Priority: metadata -> local compute)
        sha256 = get_sha256_from_metadata(metadata, path.name)
        if not sha256:
            # Fallback to computing hash directly from local file
            print(f"Computing hash for {path.name} (fallback)...")
            sha256 = compute_file_sha256(path)
            
        if not sha256:
            continue
            
        # Check favorite filter if active
        if favorite_hashes is not None:
            if sha256 not in favorite_hashes and path.name not in favorite_hashes:
                continue
                
        # Extract metadata parameters
        civitai_version_id = unwrapped.get("id")
        civitai_model_id = unwrapped.get("modelId")
        
        # Try to resolve creator & base model
        creator = ""
        model_obj = unwrapped.get("model") or {}
        creator_obj = model_obj.get("creator") if isinstance(model_obj, dict) else None
        if isinstance(creator_obj, dict):
            creator = creator_obj.get("username", "")
        elif isinstance(unwrapped.get("creator"), dict):
            creator = unwrapped.get("creator", {}).get("username", "")
        elif isinstance(unwrapped.get("creator"), str):
            creator = unwrapped.get("creator", "")
            
        base_model = unwrapped.get("baseModel", "")
        
        model_entry = {
            "filename": path.name,
            "sha256": sha256,
            "civitai_model_id": civitai_model_id,
            "civitai_version_id": civitai_version_id,
            "creator": creator,
            "base_model": base_model
        }
        models.append(model_entry)
        exported_models_count += 1
        
        # Load sidecar user_data.json if present
        user_data_path = path.parent / "user_data.json"
        if user_data_path.exists():
            try:
                with user_data_path.open("r", encoding="utf-8") as f:
                    u_data = json.load(f)
                    
                # Store personal data fields under the stable SHA256 key
                user_data_dict[sha256] = {
                    "rating": u_data.get("rating", 0),
                    "collections": u_data.get("collections", []),
                    "notes": u_data.get("notes", ""),
                    "personal_trigger_words": u_data.get("personal_trigger_words", [])
                }
                exported_user_data_count += 1
            except Exception as e:
                log_operation(f"Error reading user_data sidecar for {path.name}: {e}")
                
    # Build complete manifest payload
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "manager_version": MANAGER_VERSION,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "statistics": {
            "model_count": exported_models_count,
            "user_data_entries": exported_user_data_count,
            "prompt_preset_count": 0
        },
        "models": models,
        "user_data": user_data_dict,
        "prompt_presets": []
    }
    
    # Compute whole-payload checksum
    checksum = compute_manifest_checksum(manifest)
    manifest["checksum"] = checksum
    
    # Save JSON file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    summary_msg = (
        f"Export Complete\n\n"
        f"Models: {exported_models_count}\n"
        f"User Data: {'Included' if exported_user_data_count > 0 else 'None'}\n"
        f"Prompt Presets: None\n"
        f"Checksum:\n{checksum}\n\n"
        f"Saved to:\n{output_file.name}"
    )
    print(f"\n==============================================")
    print(summary_msg)
    print(f"==============================================")
    
    log_operation(
        f"Manifest exported successfully to {output_file.name} | "
        f"Models: {exported_models_count}, User Data Entries: {exported_user_data_count}, "
        f"Checksum: {checksum}"
    )
    
    return output_file

# Detailed live terminal progress printer
def print_detailed_progress(current, total, downloaded, skipped, failed, is_first=False):
    bar_width = 16
    filled = int(bar_width * current // total) if total > 0 else 0
    bar = '#' * filled + '-' * (bar_width - filled)
    
    if not is_first:
        # Move cursor up 11 lines
        sys.stdout.write("\033[11A")
        
    sys.stdout.write(f"Importing Library\n")
    sys.stdout.write(f"[{bar}]\n")
    sys.stdout.write(f"{current} / {total}\n\n")
    sys.stdout.write(f"Downloaded:\n{downloaded}\n\n")
    sys.stdout.write(f"Skipped:\n{skipped}\n\n")
    sys.stdout.write(f"Failed:\n{failed}\n")
    sys.stdout.flush()

# Download model version file with Content-Disposition lookup
def download_model_version(version_id: int, target_dir: Path, fallback_filename: str) -> Path | None:
    api_key = get_civitai_api_key()
    url = f"https://civitai.com/api/download/models/{version_id}"
    if api_key:
        url += f"?token={api_key}"
        
    request = Request(url, headers={"User-Agent": "LoRA-Library-Manager/0.1"})
    tmp_path = None
    
    try:
        with urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
            # Parse Content-Disposition filename
            filename = fallback_filename
            content_disp = response.headers.get("Content-Disposition")
            if content_disp:
                match = re.search(r'filename="([^"]+)"', content_disp)
                if match:
                    filename = match.group(1)
                else:
                    match = re.search(r'filename=([^;]+)', content_disp)
                    if match:
                        filename = match.group(1).strip()
            
            # Clean filename from any path slashes
            filename = Path(filename).name
            target_path = target_dir / filename
            
            # Target path check
            if target_path.exists():
                return target_path
                
            target_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = target_path.with_suffix(".safetensors.tmp")
            
            # Stream download
            with tmp_path.open("wb") as out_file:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    
            os.replace(tmp_path, target_path)
            return target_path
    except Exception as e:
        log_operation(f"Error downloading model version {version_id}: {e}")
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        return None

# Fetch CivitAI metadata for version
def fetch_and_save_metadata(version_id: int, target_dir: Path, filename: str) -> dict | None:
    res = fetch_json(f"model-versions/{version_id}")
    if res.get("ok") is True and isinstance(res.get("data"), dict):
        metadata = res["data"]
        stem = Path(filename).stem
        metadata_file = target_dir / f"{stem}.json"
        shared_file = target_dir / "metadata.json"
        try:
            unwrapped = unwrap_civitai_metadata(metadata)
            with metadata_file.open("w", encoding="utf-8") as f:
                json.dump(unwrapped, f, indent=2)
            if not shared_file.exists():
                try:
                    with shared_file.open("w", encoding="utf-8") as sf:
                        json.dump(unwrapped, sf, indent=2)
                except OSError:
                    pass
            return metadata
        except OSError as e:
            log_operation(f"Error saving metadata for version {version_id}: {e}")
    return None


# Restore user_data sidecar file
def restore_user_data_sidecar(target_folder: Path, u_data: dict) -> bool:
    user_data_path = target_folder / "user_data.json"
    try:
        # Load existing user_data if present to merge details
        existing_data = {}
        if user_data_path.exists():
            try:
                with user_data_path.open("r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except Exception:
                pass
                
        existing_data["schema_version"] = 1
        existing_data["last_modified"] = datetime.now().isoformat() + "Z"
        existing_data["rating"] = u_data.get("rating", 0)
        existing_data["collections"] = u_data.get("collections", [])
        existing_data["notes"] = u_data.get("notes", "")
        existing_data["personal_trigger_words"] = u_data.get("personal_trigger_words", [])
        
        target_folder.mkdir(parents=True, exist_ok=True)
        with user_data_path.open("w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2)
        return True
    except Exception as e:
        log_operation(f"Error restoring user data to {user_data_path}: {e}")
        return False

# Import the library manifest
def import_library_manifest(lora_root: Path, manifest_file: Path) -> None:
    if not manifest_file.exists():
        print(f"Error: Manifest file '{manifest_file}' does not exist.")
        return
        
    try:
        with manifest_file.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"Error reading manifest: {e}")
        return
        
    # Validate version
    m_version = manifest.get("manifest_version")
    if m_version != MANIFEST_VERSION:
        print(f"\nError: Unsupported manifest version '{m_version}'. Supported version: {MANIFEST_VERSION}.")
        return
        
    models_list = manifest.get("models", [])
    user_data_block = manifest.get("user_data", {})
    total_models = len(models_list)
    
    print("Scanning local library for matching models...")
    
    # 1. Build local index of existing models
    local_models = []
    for ext in LORA_EXTENSIONS:
        for path in lora_root.rglob(f"*{ext}"):
            if is_lora_file(path):
                # Load metadata
                metadata = load_metadata(path)

                
                sha256 = get_sha256_from_metadata(metadata, path.name)
                unwrapped = unwrap_metadata(metadata)
                version_id = unwrapped.get("id")
                
                local_models.append({
                    "path": path,
                    "sha256": sha256.lower() if sha256 else "",
                    "version_id": version_id,
                    "model_id": unwrapped.get("modelId"),
                    "filename": path.name.lower(),
                    "parent_path": path.parent
                })
                
    # Helper lookup closure to find a match
    def find_local_match(manifest_model):
        m_sha256 = manifest_model.get("sha256", "").lower()
        m_version_id = manifest_model.get("civitai_version_id")
        m_filename = manifest_model.get("filename", "").lower()
        
        # 1. Match by SHA256
        if m_sha256:
            for lm in local_models:
                if lm["sha256"] == m_sha256:
                    return lm["path"]
                    
        # 2. Match by Version ID
        if m_version_id:
            for lm in local_models:
                if lm["version_id"] == m_version_id:
                    return lm["path"]
                    
        # 3. Match by Filename
        if m_filename:
            for lm in local_models:
                if lm["filename"] == m_filename:
                    return lm["path"]
                    
        return None

    # Analyze manifest matches
    skipped_models = []
    missing_models = []
    
    for m in models_list:
        match_path = find_local_match(m)
        if match_path:
            skipped_models.append((m, match_path))
        else:
            missing_models.append(m)
            
    already_installed = len(skipped_models)
    missing_count = len(missing_models)
    
    # 2. Display Import Confirmation Summary
    print(f"\n==============================================")
    print(f"Library Snapshot Details:")
    print(f"  - Total Models in Snapshot:  {total_models}")
    print(f"  - Already Installed Locally: {already_installed}")
    print(f"  - Missing Locally (to download): {missing_count}")
    print(f"==============================================")
    
    confirm = input("Proceed with import? (Y/N): ").strip().lower()
    if confirm not in ("y", "yes"):
        print("Import cancelled.")
        return
        
    # Initialize import stats
    downloaded = 0
    skipped = already_installed
    failed = 0
    restored_user_data = 0
    failed_details = []
    
    print("\n")
    print_detailed_progress(already_installed, total_models, downloaded, skipped, failed, is_first=True)
    
    # Process already installed models: restore user data if missing locally
    for m, path in skipped_models:
        sha256 = m.get("sha256")
        u_data = user_data_block.get(sha256)
        if u_data:
            user_data_file = path.parent / "user_data.json"
            # Restore user data sidecar if missing or updated
            if restore_user_data_sidecar(path.parent, u_data):
                restored_user_data += 1
                
    # Process missing models sequentially
    current_count = already_installed
    for m in missing_models:
        current_count += 1
        version_id = m.get("civitai_version_id")
        filename = m.get("filename")
        sha256 = m.get("sha256")
        
        if not version_id:
            # Cannot download without version ID
            failed += 1
            failed_details.append({
                "filename": filename,
                "version_id": "Missing",
                "creator": m.get("creator", "Unknown"),
                "base_model": m.get("base_model", "Unknown")
            })
            print_detailed_progress(current_count, total_models, downloaded, skipped, failed)
            continue
            
        # Clear line and print download status
        print(f"\nDownloading {filename} (Version ID: {version_id})...")
        
        # Determine target folder: check if another version of this model is already installed locally
        target_folder = None
        m_model_id = m.get("civitai_model_id")
        if m_model_id:
            for lm in local_models:
                if lm.get("model_id") == m_model_id:
                    target_folder = lm["parent_path"]
                    break
                    
        if not target_folder:
            stem = Path(filename).stem
            target_folder = lora_root / stem
        
        # 1. Download Model File
        saved_model_path = download_model_version(version_id, target_folder, filename)
        
        if saved_model_path:
            downloaded += 1
            
            # 2. Save metadata.json
            metadata = fetch_and_save_metadata(version_id, target_folder, filename)
            
            # 3. Download Preview Image
            if metadata:
                download_preview_if_missing(saved_model_path, metadata)

                
            # 4. Restore user_data.json
            u_data = user_data_block.get(sha256)
            if u_data:
                if restore_user_data_sidecar(target_folder, u_data):
                    restored_user_data += 1
        else:
            failed += 1
            failed_details.append({
                "filename": filename,
                "version_id": version_id,
                "creator": m.get("creator", "Unknown"),
                "base_model": m.get("base_model", "Unknown")
            })
            
        # Overwrite progress output
        print_detailed_progress(current_count, total_models, downloaded, skipped, failed)
        
    # Display final report
    summary_report = (
        f"\n==============================================\n"
        f"Library Import Complete\n\n"
        f"Downloaded: {downloaded}\n"
        f"Skipped:    {skipped}\n"
        f"Failed:     {failed}\n"
        f"Restored User Data: {restored_user_data}\n"
        f"=============================================="
    )
    print(summary_report)
    
    # Print failed details for manual location
    if failed_details:
        print("\nFailed Downloads (Manual Location Details):")
        for f_det in failed_details:
            detail_line = f"  - {f_det['filename']} | Version ID: {f_det['version_id']} | Creator: {f_det['creator']} | Base: {f_det['base_model']}"
            print(detail_line)
            log_operation(f"Import failure manual detail: {detail_line}")
            
    # Write to daily log
    log_operation(
        f"Library Import Complete | "
        f"Downloaded: {downloaded}, Skipped: {skipped}, Failed: {failed}, Restored User Data: {restored_user_data}"
    )
