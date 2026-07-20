"""Library Backup Package (.llmb) export and import helpers for LoRA Library Manager."""

import json
import os
import hashlib
import time
import sys
import re
import zipfile
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
from gallery import load_metadata, unwrap_metadata, get_sha256_from_metadata
from downloader import download_to_path, download_preview_if_missing, preview_image_path
from providers.provider_base import ModelInfo

MANIFEST_VERSION = 1
DEFAULT_TIMEOUT = 60
CHUNK_SIZE = 1024 * 1024


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


# Extract compatibility metadata extras from metadata dict and user_data
def get_metadata_extras(path: Path, u_data: dict) -> dict:
    model_folder = path.parent
    model_stem = path.stem
    meta_path = model_folder / f"{model_stem}.json"
    
    activation_text = ""
    preferred_weight = None
    negative_prompt = ""
    notes = ""
    
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            if isinstance(meta, dict):
                activation_text = meta.get("activation text") or meta.get("activation_text") or ""
                preferred_weight = meta.get("preferred weight") or meta.get("preferred_weight")
                negative_prompt = meta.get("negative text") or meta.get("negative_prompt") or ""
                notes = meta.get("notes") or ""
        except Exception:
            pass
            
    if not activation_text and u_data.get("personal_trigger_words"):
        activation_text = ", ".join(u_data["personal_trigger_words"])
    if preferred_weight is None:
        preferred_weight = u_data.get("preferred_weight")
    if not negative_prompt:
        negative_prompt = u_data.get("negative_prompt") or ""
    if not notes:
        notes = u_data.get("notes") or ""
        
    return {
        "activation text": activation_text,
        "preferred weight": float(preferred_weight) if preferred_weight is not None else None,
        "negative text": negative_prompt,
        "notes": notes
    }


# Export the library backup package (.llmb)
def export_library_backup(
    lora_root: Path,
    output_file: Path,
    favorite_hashes: set[str] | None = None,
    backup_name: str | None = None,
    comment: str | None = None,
    include_previews: bool = True,
    include_user_data: bool = True,
    favorites_only: bool = False,
    favorite_keys: list[str] | None = None,
) -> Path:
    print("Scanning library files for export...")
    
    lora_files = []
    for ext in LORA_EXTENSIONS:
        for path in lora_root.rglob(f"*{ext}"):
            if is_lora_file(path):
                lora_files.append(path)
                
    seen = set()
    unique_files = []
    for path in lora_files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(path)
            
    # If favorites_only, build the allowed hash set matching favorites strictly
    if favorites_only and favorite_hashes is None:
        fav_set: set[str] = set()
        fav_keys_lower = set(str(k).lower() for k in (favorite_keys or []) if k)

        for path in unique_files:
            meta = load_metadata(path)
            sha = get_sha256_from_metadata(meta, path.name)
            sha_lower = sha.lower() if sha else ""
            
            is_fav = False
            
            # 1. Check if hash, filename, stem, or relative path matches localStorage favorites
            try:
                rel_path_str = str(path.relative_to(lora_root)).lower()
            except Exception:
                rel_path_str = ""

            if (sha_lower and sha_lower in fav_keys_lower) or \
               (path.name.lower() in fav_keys_lower) or \
               (path.stem.lower() in fav_keys_lower) or \
               (rel_path_str and rel_path_str in fav_keys_lower):
                is_fav = True

            # 2. Check user_data.json sidecar specifically for is_favorite or "Favorites" collection
            udp = path.parent / "user_data.json"
            if not is_fav and udp.exists():
                try:
                    with udp.open("r", encoding="utf-8") as _f:
                        ud = json.load(_f)
                    
                    is_fav_flag = bool(ud.get("is_favorite") or ud.get("favorite"))
                    colls = [str(c).lower() for c in ud.get("collections", []) if c]
                    
                    if is_fav_flag or "favorites" in colls:
                        is_fav = True
                except Exception:
                    pass
                    
            if is_fav and sha:
                fav_set.add(sha)

        favorite_hashes = fav_set

    models = []
    user_data_dict = {}
    exported_models_count = 0
    exported_user_data_count = 0
    providers_set = set()
    previews_to_export = []
    
    for path in unique_files:
        metadata = load_metadata(path)
        sha256 = get_sha256_from_metadata(metadata, path.name)
        
        # If favorites_only, skip models that are not in favorite_hashes
        if favorite_hashes is not None and sha256 not in favorite_hashes:
            continue
            
        exported_models_count += 1
        
        # Determine provider and IDs
        provider_name = (metadata.get("provider") if metadata else None) or "civitai"
        unwrapped = metadata.get("unwrapped") if (metadata and isinstance(metadata.get("unwrapped"), dict)) else {}
        
        provider_model_id = None
        provider_version_id = None
        
        if metadata:
            provider_model_id = metadata.get("provider_model_id")
            provider_version_id = metadata.get("provider_version_id")
            
        # If IDs are missing, try to infer from legacy metadata
        if not provider_model_id or not provider_version_id:
            if provider_name == "civitai":
                provider_model_id = unwrapped.get("modelId") or (unwrapped.get("model") or {}).get("id")
                provider_version_id = unwrapped.get("id")
            elif provider_name == "arcenciel":
                provider_model_id = unwrapped.get("id") or unwrapped.get("modelId")
                versions = unwrapped.get("versions") or []
                if versions:
                    provider_version_id = versions[0].get("id")
                    for v in versions:
                        v_sha = (v.get("sha256") or "").lower()
                        if v_sha == sha256:
                            provider_version_id = v.get("id")
                            break
                elif "filePath" in unwrapped:
                    provider_version_id = unwrapped.get("id")
                    provider_model_id = unwrapped.get("modelId")

        # If still missing, perform a live hash lookup on the resolved provider
        if not provider_model_id or not provider_version_id:
            try:
                from config import get_provider_api_key
                from providers.registry import get_provider
                provider = get_provider(provider_name)
                if provider.capabilities.hash_lookup:
                    api_key = get_provider_api_key(provider_name)
                    info = provider.fetch_by_hash(sha256, api_key=api_key)
                    if info:
                        provider_model_id = info.provider_model_id
                        provider_version_id = info.provider_version_id
            except Exception as e:
                log_operation(f"Hash lookup failed during export for {path.name} on {provider_name}: {e}")

        # Set provider display name
        try:
            from providers.registry import get_provider
            provider_instance = get_provider(provider_name)
            provider_display_name = provider_instance.presentation.display_name
        except Exception:
            provider_display_name = provider_name.title()

        providers_set.add(provider_name)
        
        creator = ""
        model_obj = unwrapped.get("model") or {}
        creator_obj = model_obj.get("creator") if isinstance(model_obj, dict) else None
        if isinstance(creator_obj, dict):
            creator = creator_obj.get("username", "")
        elif isinstance(unwrapped.get("creator"), dict):
            creator = unwrapped.get("creator", {}).get("username", "")
        elif isinstance(unwrapped.get("creator"), str):
            creator = unwrapped.get("creator", "")
            
        base_model = (metadata.get("base_model") if metadata else None) or unwrapped.get("baseModel", "")
        
        parent_rel = path.parent.relative_to(lora_root)
        relative_path = parent_rel / path.name
        rel_path_str = relative_path.as_posix()
        
        model_entry = {
            "name": path.stem,
            "filename": path.name,
            "relative_path": rel_path_str,
            "sha256": sha256,
            "provider": provider_name,
            "provider_name": provider_display_name,
            "provider_model_id": provider_model_id,
            "provider_version_id": provider_version_id,
            "base_model": base_model,
            "creator": creator
        }
        models.append(model_entry)
        
        # Read user_data.json if enabled
        if include_user_data:
            user_data_path = path.parent / "user_data.json"
            if user_data_path.exists():
                try:
                    with user_data_path.open("r", encoding="utf-8") as f:
                        ud_json = json.load(f)
                    user_data_dict[sha256] = ud_json
                    exported_user_data_count += 1
                except Exception as e:
                    log_operation(f"Error reading {user_data_path} for backup: {e}")
            else:
                user_data_dict[sha256] = {}
        else:
            # Add metadata_extras if present
            meta_extras = metadata.get("metadata_extras") if metadata else None
            user_data_dict[sha256] = {
                "personal_trigger_words": [],
                "preferred_weight": 1.0,
                "negative_prompt": "",
                "notes": "",
                "rating": 0,
                "collections": [],
                "metadata_extras": meta_extras
            }
        
        # Check for local preview
        if include_previews:
            from downloader import preview_image_path
            local_preview = preview_image_path(path)
            if local_preview.exists():
                previews_to_export.append((sha256, local_preview))
                
    backup_info = {
        "schema_version": 2,
        "backup_version": 2,
        "manager_version": MANAGER_VERSION,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "model_count": exported_models_count,
        "providers": sorted(list(providers_set)),
        "backup_name": backup_name or "",
        "comment": comment or ""
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate bytes of each JSON file to compute hashes cleanly
    backup_bytes = json.dumps(backup_info, indent=2).encode("utf-8")
    manifest_bytes = json.dumps(models, indent=2).encode("utf-8")
    user_bytes = json.dumps(user_data_dict, indent=2).encode("utf-8")
    
    # Generate BACKUP_INFO.txt
    created_at_date = datetime.utcnow().strftime("%Y-%m-%d")
    readme_lines = [
        "LoRA Library Backup",
        "",
        "Created:",
        created_at_date,
        "",
        "Models:",
        str(exported_models_count),
        "",
        "Providers:"
    ]
    for p in sorted(list(providers_set)):
        readme_lines.append(f"- {p.title()}")
    readme_lines.extend([
        "",
        "Generated by:",
        f"LoRA Library Manager v{MANAGER_VERSION}"
    ])
    readme_bytes = "\n".join(readme_lines).encode("utf-8")

    checksums = {
        "backup.json": hashlib.sha256(backup_bytes).hexdigest().lower(),
        "library_manifest.json": hashlib.sha256(manifest_bytes).hexdigest().lower(),
        "user_data.json": hashlib.sha256(user_bytes).hexdigest().lower(),
        "BACKUP_INFO.txt": hashlib.sha256(readme_bytes).hexdigest().lower()
    }
    
    # Read preview image bytes and compute checksums
    previews_data = []
    for sha, preview_path in previews_to_export:
        suffix = preview_path.suffix.lower()
        zip_path = f"previews/{sha}{suffix}"
        try:
            img_bytes = preview_path.read_bytes()
            checksums[zip_path] = hashlib.sha256(img_bytes).hexdigest().lower()
            previews_data.append((zip_path, img_bytes))
        except Exception as e:
            log_operation(f"Error reading preview image {preview_path.name} for backup: {e}")
            
    checksums_bytes = json.dumps(checksums, indent=2).encode("utf-8")
    
    # Write backup contents directly to ZIP package
    with zipfile.ZipFile(output_file, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("backup.json", backup_bytes)
        z.writestr("library_manifest.json", manifest_bytes)
        z.writestr("user_data.json", user_bytes)
        z.writestr("BACKUP_INFO.txt", readme_bytes)
        for zip_path, img_bytes in previews_data:
            z.writestr(zip_path, img_bytes)
        z.writestr("checksums.json", checksums_bytes)
        
    abs_save_path = str(output_file.resolve())
    summary_msg = (
        f"Export Complete\n\n"
        f"Models: {exported_models_count}\n"
        f"User Data: {'Included' if include_user_data and exported_user_data_count > 0 else 'Not included'}\n"
        f"Previews: {'Included (' + str(len(previews_to_export)) + ')' if include_previews else 'Not included'}\n"
        f"Saved to: {abs_save_path}"
    )
    print(f"\n==============================================")
    print(f"[Backup System] Backup file saved to:")
    print(f"  --> {abs_save_path}")
    print(f"==============================================")
    print(summary_msg)
    print(f"==============================================")
    
    log_operation(
        f"Backup package exported successfully to {abs_save_path} | "
        f"Models: {exported_models_count}, User Data Entries: {exported_user_data_count}"
    )
    
    return output_file


# Detailed live terminal progress printer
def print_detailed_progress(current, total, downloaded, skipped, failed, is_first=False):
    bar_width = 16
    filled = int(bar_width * current // total) if total > 0 else 0
    bar = '#' * filled + '-' * (bar_width - filled)
    
    if not is_first:
        sys.stdout.write("\033[11A")
        
    sys.stdout.write(f"Importing Library\n")
    sys.stdout.write(f"[{bar}]\n")
    sys.stdout.write(f"{current} / {total}\n\n")
    sys.stdout.write(f"Downloaded:\n{downloaded}\n\n")
    sys.stdout.write(f"Skipped:\n{skipped}\n\n")
    sys.stdout.write(f"Failed:\n{failed}\n")
    sys.stdout.flush()


# Download model version file with Content-Disposition lookup using registered provider
def download_model_version(
    provider_name: str,
    model_id: str | None,
    version_id: str,
    target_dir: Path,
    fallback_filename: str,
) -> Path | None:
    from providers.registry import get_provider
    from config import get_provider_api_key

    provider = get_provider(provider_name)
    api_key = get_provider_api_key(provider_name)

    if not provider.capabilities.model_download:
        log_operation(f"Provider {provider_name} does not support model download.")
        return None

    url = provider.get_download_url(model_id, version_id, api_key=api_key)
    if not url:
        return None

    headers = {"User-Agent": "LoRA-Library-Manager/2.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        
    request = Request(url, headers={k: v for k, v in headers.items() if v})
    tmp_path = None
    
    try:
        with urlopen(request, timeout=120) as response:
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
            
            filename = Path(filename).name
            target_path = target_dir / filename
            
            if target_path.exists():
                return target_path
                
            target_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = target_path.with_suffix(".safetensors.tmp")
            
            with tmp_path.open("wb") as out_file:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    
            os.replace(tmp_path, target_path)
            return target_path
    except Exception as e:
        log_operation(f"Error downloading model version {version_id} via {provider_name}: {e}")
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise e



# Fetch metadata for version using provider and save
def fetch_and_save_metadata(provider_name: str, model_id: str | None, version_id: str, target_dir: Path, filename: str) -> ModelInfo | None:
    from providers.registry import get_provider
    from config import get_provider_api_key
    from downloader import save_metadata_if_missing
    
    provider = get_provider(provider_name)
    api_key = get_provider_api_key(provider_name)
    
    try:
        model_info = provider.fetch_by_id(model_id, version_id, api_key=api_key)
        if model_info:
            save_metadata_if_missing(target_dir / filename, model_info)
            return model_info
    except Exception as e:
        log_operation(f"Error fetching metadata for version {version_id} via {provider_name}: {e}")
    return None


# Restore user_data sidecar file
def restore_user_data_sidecar(target_folder: Path, u_data: dict) -> bool:
    user_data_path = target_folder / "user_data.json"
    try:
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
        
        extras = u_data.get("metadata_extras")
        if isinstance(extras, dict):
            if "preferred weight" in extras:
                existing_data["preferred_weight"] = extras["preferred weight"]
            if "negative text" in extras:
                existing_data["negative_prompt"] = extras["negative text"]
            if "notes" in extras and extras["notes"]:
                if not existing_data.get("notes"):
                    existing_data["notes"] = extras["notes"]
                    
        target_folder.mkdir(parents=True, exist_ok=True)
        with user_data_path.open("w", encoding="utf-8") as f:
            json.dump(existing_data, f, indent=2)
        return True
    except Exception as e:
        log_operation(f"Error restoring user data to {user_data_path}: {e}")
        return False


# Mirror metadata compatibility extras to metadata.json (<stem>.json)
def sync_metadata_extras_to_file(model_path: Path, metadata_extras: dict) -> None:
    model_folder = model_path.parent
    model_stem = model_path.stem
    target_path = model_folder / f"{model_stem}.json"
    
    data = {}
    if target_path.exists():
        try:
            with target_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass
            
    if isinstance(data, dict):
        data["activation text"] = metadata_extras.get("activation text") or ""
        data["preferred weight"] = metadata_extras.get("preferred weight")
        data["negative text"] = metadata_extras.get("negative text") or ""
        data["notes"] = metadata_extras.get("notes") or ""
        
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log_operation(f"Error mirroring metadata_extras to {target_path.name}: {e}")


# Import/Restore the library backup package (.llmb) or legacy manifest (.json)
def import_library_backup(
    lora_root: Path,
    backup_file: Path,
    auto_confirm: bool = False,
    download_models: bool = True,
    restore_previews: bool = True,
    restore_user_data: bool = True,
) -> None:
    if not backup_file.exists():
        print(f"Error: Backup file '{backup_file}' does not exist.")
        return
        
    is_llmb = zipfile.is_zipfile(backup_file)
    
    if is_llmb:
        try:
            with zipfile.ZipFile(backup_file, "r") as z:
                # Validate checksums.json
                namelist = z.namelist()
                if "checksums.json" not in namelist:
                    raise ValueError("Corrupted package: checksums.json missing")
                    
                checksums = json.loads(z.read("checksums.json").decode("utf-8"))
                for file_name, expected_sha in checksums.items():
                    if file_name not in namelist:
                        raise ValueError(f"Corrupted package: missing {file_name}")
                    
                    file_bytes = z.read(file_name)
                    actual_sha = hashlib.sha256(file_bytes).hexdigest().lower()
                    if actual_sha != expected_sha.lower():
                        raise ValueError(f"Corrupted package: checksum mismatch for {file_name}")
                
                backup_data = json.loads(z.read("backup.json").decode("utf-8"))
                models_list = json.loads(z.read("library_manifest.json").decode("utf-8"))
                user_data_block = json.loads(z.read("user_data.json").decode("utf-8"))
                
                # Extract summary info
                backup_name = backup_data.get("backup_name", "N/A")
                comment = backup_data.get("comment", "")
                created_at = backup_data.get("created_at", "N/A")
                if created_at != "N/A":
                    try:
                        created_at = created_at.split("T")[0]
                    except Exception:
                        pass
                providers = backup_data.get("providers", [])
                preview_count = sum(1 for name in namelist if name.lower().startswith("previews/"))
                manager_version = backup_data.get("manager_version", "N/A")
        except Exception as e:
            print(f"Error reading backup package: {e}")
            raise e
    else:
        # Legacy backup JSON
        try:
            with backup_file.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            print(f"Error reading manifest: {e}")
            return
            
        m_version = manifest.get("manifest_version")
        if m_version != MANIFEST_VERSION:
            print(f"\nError: Unsupported manifest version '{m_version}'. Supported version: {MANIFEST_VERSION}.")
            return
            
        models_list = manifest.get("models", [])
        user_data_block = manifest.get("user_data", {})
        
        backup_name = "Legacy Snapshot"
        comment = ""
        created_at = manifest.get("generated_at", "N/A")
        if created_at != "N/A":
            try:
                created_at = created_at.split("T")[0]
            except Exception:
                pass
        providers = list(set(m.get("provider", "civitai") for m in models_list))
        preview_count = 0
        manager_version = manifest.get("manager_version", "N/A")

    total_models = len(models_list)
    print("Scanning local library for matching models...")
    
    local_models = []
    for ext in LORA_EXTENSIONS:
        for path in lora_root.rglob(f"*{ext}"):
            if is_lora_file(path):
                metadata = load_metadata(path)
                sha256 = get_sha256_from_metadata(metadata, path.name)
                
                if metadata and "provider" in metadata:
                    provider_model_id = metadata.get("provider_model_id")
                    provider_version_id = metadata.get("provider_version_id")
                else:
                    unwrapped = unwrap_metadata(metadata)
                    provider_model_id = unwrapped.get("modelId") or (unwrapped.get("model") or {}).get("id")
                    provider_version_id = unwrapped.get("id")
                
                local_models.append({
                    "path": path,
                    "sha256": sha256.lower() if sha256 else "",
                    "version_id": str(provider_version_id) if provider_version_id is not None else "",
                    "model_id": str(provider_model_id) if provider_model_id is not None else "",
                    "filename": path.name.lower(),
                    "parent_path": path.parent
                })
                
    def find_local_match(manifest_model):
        m_sha256 = manifest_model.get("sha256", "").lower()
        m_version_id = manifest_model.get("provider_version_id") or manifest_model.get("civitai_version_id")
        m_filename = manifest_model.get("filename", "").lower()
        
        if m_sha256:
            for lm in local_models:
                if lm["sha256"] == m_sha256:
                    return lm["path"]
                    
        if m_version_id:
            m_version_str = str(m_version_id)
            for lm in local_models:
                if lm["version_id"] == m_version_str:
                    return lm["path"]
                    
        if m_filename:
            for lm in local_models:
                if lm["filename"] == m_filename:
                    return lm["path"]
                    
        return None

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
    
    print(f"\nBackup Summary\n")
    print(f"Backup Name:\n{backup_name or 'N/A'}\n")
    if comment:
        print(f"Comment:\n{comment}\n")
    print(f"Created:\n{created_at}\n")
    print(f"Models:\n{total_models}\n")
    print("Providers:")
    for p in providers:
        print(f"• {p.title()}")
    print()
    print(f"Preview Images:\n{preview_count}\n")
    print(f"Manager Version:\n{manager_version}\n")
    
    print("==============================================")
    print(f"Library Snapshot Details:")
    print(f"  - Total Models in Snapshot:  {total_models}")
    print(f"  - Already Installed Locally: {already_installed}")
    print(f"  - Missing Locally (to download): {missing_count}")
    print("==============================================")
    
    if auto_confirm:
        print("Proceed? (Y/N): Y (Auto-confirmed)")
        confirm = "y"
    else:
        confirm = input("Proceed? (Y/N): ").strip().lower()
        
    if confirm not in ("y", "yes"):
        print("Import cancelled.")
        return
        
    downloaded = 0
    skipped = already_installed
    failed = 0
    restored_user_data = 0
    failed_details = []

    # Lazy import — safe to call outside a job context (no-op when no job active)
    try:
        from jobs import update_job_progress as _upd
    except ImportError:
        def _upd(**_kw): pass

    _upd(
        percent=0,
        current=0,
        total=total_models,
        message=(
            f"Starting restore — {missing_count} to download, "
            f"{already_installed} already installed"
        ),
    )

    print("\n")
    print_detailed_progress(already_installed, total_models, downloaded, skipped, failed, is_first=True)

    for m, path in skipped_models:
        sha256 = m.get("sha256")
        u_data = user_data_block.get(sha256)
        if u_data and restore_user_data:
            if restore_user_data_sidecar(path.parent, u_data):
                restored_user_data += 1

    current_count = already_installed
    for m in missing_models:
        current_count += 1
        filename = m.get("filename") or "Unknown"
        version_id = m.get("provider_version_id") or m.get("civitai_version_id") or "Unknown"
        sha256 = m.get("sha256")

        _provider_label = (m.get("provider") or "civitai").replace("civitai", "CivitAI").replace("arcenciel", "Arc en Ciel")

        try:
            provider_name = m.get("provider")
            model_id = m.get("provider_model_id")

            if not provider_name:
                provider_name = "civitai"
                version_id = m.get("civitai_version_id")
                model_id = m.get("civitai_model_id")

            # Pre-download healing: if IDs are missing or unknown, try to resolve
            # them via a live hash lookup across all capable providers.
            # This auto-heals backups exported with incorrect/missing provider info
            # (e.g. Arcenciel models that were mistakenly tagged as CivitAI).
            if sha256 and (not model_id or not version_id or version_id == "Unknown"):
                log_operation(f"Missing IDs for {filename} — attempting hash-based provider healing...")
                try:
                    from providers.registry import resolve_provider, get_provider
                    from config import get_provider_api_key
                    resolved_p = resolve_provider(sha256=sha256)
                    if resolved_p:
                        heal_provider = get_provider(resolved_p)
                        heal_api_key = get_provider_api_key(resolved_p)
                        if heal_provider.capabilities.hash_lookup:
                            info = heal_provider.fetch_by_hash(sha256, api_key=heal_api_key)
                            if info:
                                provider_name = resolved_p
                                model_id = info.provider_model_id
                                version_id = info.provider_version_id
                                log_operation(
                                    f"Healed provider to '{provider_name}' "
                                    f"(model={model_id}, version={version_id}) for {filename}"
                                )
                except Exception as heal_err:
                    log_operation(f"Hash-based healing failed for {filename}: {heal_err}")

            if not version_id or version_id == "Unknown":
                raise ValueError("Missing provider version ID")

            print(f"\nDownloading {filename} ({provider_name.title()} Version ID: {version_id})...")
            
            if not download_models:
                # Structure-only: restore metadata and previews without downloading the model file
                print(f"  (Skipping download — structure-only mode)")
                # Still restore preview from zip if requested
                if restore_previews and is_llmb:
                    m_relative_path = m.get("relative_path")
                    if m_relative_path:
                        target_folder = (lora_root / m_relative_path).parent
                        target_folder.mkdir(parents=True, exist_ok=True)
                        try:
                            with zipfile.ZipFile(backup_file, "r") as z:
                                preview_prefix = f"previews/{sha256}."
                                for zip_name in z.namelist():
                                    if zip_name.lower().startswith(preview_prefix.lower()):
                                        suffix = Path(zip_name).suffix
                                        stem = Path(m_relative_path).stem
                                        preview_dest = target_folder / f"{stem}{suffix}"
                                        preview_dest.write_bytes(z.read(zip_name))
                                        log_operation(f"Restored preview (structure-only): {preview_dest.name}")
                                        break
                        except Exception as pe:
                            log_operation(f"Preview restore failed for {filename}: {pe}")
                if restore_user_data and sha256:
                    u_data = user_data_block.get(sha256)
                    if u_data and m.get("relative_path"):
                        target_folder = (lora_root / m.get("relative_path")).parent
                        target_folder.mkdir(parents=True, exist_ok=True)
                        if restore_user_data_sidecar(target_folder, u_data):
                            restored_user_data += 1
                downloaded += 1  # count as "processed" for progress
                continue

            # Determine target folder
            m_relative_path = m.get("relative_path")
            if m_relative_path:
                target_path = lora_root / m_relative_path
                target_folder = target_path.parent
                filename = target_path.name
            else:
                # Fallback for legacy backups
                target_folder = None
                if model_id:
                    for lm in local_models:
                        if str(lm.get("model_id")) == str(model_id):
                            target_folder = lm["parent_path"]
                            break
                if not target_folder:
                    stem = Path(filename).stem
                    target_folder = lora_root / stem
                    
            saved_model_path = download_model_version(provider_name, model_id, version_id, target_folder, filename)
            
            if not saved_model_path:
                raise RuntimeError("Download failed (provider returned empty path)")
                
            downloaded += 1
            metadata = fetch_and_save_metadata(provider_name, model_id, version_id, target_folder, saved_model_path.name)
            
            preview_restored = False
            if restore_previews and is_llmb:
                try:
                    with zipfile.ZipFile(backup_file, "r") as z:
                        preview_prefix = f"previews/{sha256}."
                        for zip_name in z.namelist():
                            if zip_name.lower().startswith(preview_prefix.lower()):
                                suffix = Path(zip_name).suffix
                                preview_dest = target_folder / f"{saved_model_path.stem}{suffix}"
                                preview_dest.parent.mkdir(parents=True, exist_ok=True)
                                preview_dest.write_bytes(z.read(zip_name))
                                preview_restored = True
                                log_operation(f"Restored preview image from backup: {preview_dest.name}")
                                break
                except Exception as e:
                    log_operation(f"Error extracting preview image from zip for {filename}: {e}")
                    
            if not preview_restored and metadata:
                download_preview_if_missing(saved_model_path, metadata)
                
            u_data = user_data_block.get(sha256)
            if u_data and restore_user_data:
                if restore_user_data_sidecar(target_folder, u_data):
                    restored_user_data += 1
                
                # Mirror metadata_extras compatibility fields to metadata.json (<stem>.json)
                extras = u_data.get("metadata_extras")
                if isinstance(extras, dict):
                    sync_metadata_extras_to_file(saved_model_path, extras)
        except Exception as e:
            failed += 1
            failed_details.append({
                "filename": filename,
                "version_id": version_id,
                "creator": m.get("creator", "Unknown"),
                "base_model": m.get("base_model", "Unknown"),
                "reason": str(e)
            })
            print(f"  X Download failed: {e}")
            log_operation(f"Error restoring model {filename}: {e}")

        print_detailed_progress(current_count, total_models, downloaded, skipped, failed)

        # Progress: fires AFTER this model fully completes (download + metadata + preview).
        # Maps completed models linearly into 0–85 %: first done ≈ 5 %, all done = 85 %.
        _models_done = downloaded + failed
        _pct = int((_models_done / max(missing_count, 1)) * 85)
        _upd(
            percent=_pct,
            current=current_count,
            total=total_models,
            message=f"[{current_count}/{total_models}] Downloaded {filename}",
            item_name=f"{filename}  •  {_provider_label}",
        )
        
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
    
    if failed_details:
        print("\nFailed Downloads (Manual Location Details):")
        for f_det in failed_details:
            detail_line = f"  - {f_det['filename']} | Version ID: {f_det['version_id']} | Creator: {f_det['creator']} | Base: {f_det['base_model']} | Error: {f_det['reason']}"
            print(detail_line)
            log_operation(f"Import failure manual detail: {detail_line}")
            
    log_operation(
        f"Library Import Complete | "
        f"Downloaded: {downloaded}, Skipped: {skipped}, Failed: {failed}, Restored User Data: {restored_user_data}"
    )

