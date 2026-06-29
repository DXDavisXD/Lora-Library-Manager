"""Migration Wizard core implementation for LoRA Library Manager.

Phase 1: Folder Detection & Preview (Read-only Analysis & Health Report).
Phase 2: Import Existing Library (Convert folders to collections).
"""

import sys
import time
import json
import shutil
from datetime import datetime
from pathlib import Path
from config import LORA_EXTENSIONS, is_lora_file, log_operation, PREVIEW_EXTENSIONS, normalize_for_comparison, clean_model_collections

def safe_print(text: str = "") -> None:
    """Print text safely, replacing characters that cannot be encoded by the terminal."""
    try:
        print(text)
    except UnicodeEncodeError:
        try:
            encoding = sys.stdout.encoding or 'utf-8'
            print(text.encode(encoding, errors='replace').decode(encoding))
        except Exception:
            try:
                print(text.encode('ascii', errors='replace').decode('ascii'))
            except Exception:
                pass


def get_original_filename(metadata: dict) -> str:
    """Extract the original/primary filename from CivitAI metadata."""
    from civitai import unwrap_civitai_metadata as unwrap_metadata
    unwrapped = unwrap_metadata(metadata)
    files = unwrapped.get("files", [])
    if isinstance(files, list):
        # Prefer the one marked primary
        for f in files:
            if isinstance(f, dict) and f.get("primary") is True and f.get("name"):
                return f["name"]
        # Fallback to the first file entry with a name
        for f in files:
            if isinstance(f, dict) and f.get("name"):
                return f["name"]
    return ""


def load_metadata_loose(lora_file: Path) -> tuple[dict, bool]:
    """Load metadata and return (metadata_dict, is_specific_sidecar).

    is_specific_sidecar is True if the metadata file name matches the model stem.
    """
    # 1. Check stem-specific sidecars
    for suffix in (".json", ".civitai.info"):
        candidate = lora_file.parent / f"{lora_file.stem}{suffix}"
        if candidate.exists():
            try:
                with candidate.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data, True
            except Exception:
                pass
                
    # 2. Check folder-level metadata.json
    candidate = lora_file.parent / "metadata.json"
    if candidate.exists():
        try:
            with candidate.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data, False
        except Exception:
            pass
            
    return {}, False


def scan_library_hierarchy(lora_root: Path) -> dict:
    """Scan the configured LoRA root and return a structured data model of the hierarchy.

    This function does NOT modify any files on disk.
    """
    start_time = time.time()
    lora_root = Path(lora_root).resolve()
    
    from library_index import load_library_index, make_index_key, get_file_fingerprint, _fingerprint_matches
    from gallery import get_sha256_from_metadata, get_preview_image
    
    index = load_library_index(lora_root)
    cached_models = index.get("models", {})
    
    models = []
    root_level_models = []
    empty_folders = []
    orphaned_folders = []
    max_depth = 0
    
    # Collect directories and track which ones contain models recursively
    all_dirs = []
    dirs_with_models = set()
    
    for path in lora_root.rglob("*"):
        if path.is_dir():
            try:
                rel_dir = path.relative_to(lora_root)
                parts = rel_dir.parts
                if parts and parts[0] in ("logs", "gallery"):
                    continue
            except ValueError:
                pass
            all_dirs.append(path)
        elif path.is_file() and path.suffix.lower() in LORA_EXTENSIONS and is_lora_file(path):
            try:
                rel_path = path.relative_to(lora_root)
            except ValueError:
                continue
            
            parts = rel_path.parts
            # Skip files inside logs or gallery folder at root level
            if parts and parts[0] in ("logs", "gallery"):
                continue
                
            hierarchy = list(parts[:-1])
            depth = len(hierarchy)
            if depth > max_depth:
                max_depth = depth
                
            key = make_index_key(path, lora_root)
            live_fp = get_file_fingerprint(path)
            
            cached_record = cached_models.get(key)
            use_cached = False
            if cached_record and live_fp and _fingerprint_matches(cached_record, live_fp):
                use_cached = True
                
            sha256 = ""
            version_id = ""
            model_id = ""
            creator = ""
            is_renamed = False
            original_filename = ""
            has_metadata = False
            
            if use_cached:
                entry = cached_record.get("entry", {})
                sha256 = cached_record.get("sha256") or entry.get("sha256", "")
                version_id = entry.get("version_id", "")
                model_id = entry.get("model_id", "")
                creator = entry.get("creator", "")
                is_renamed = cached_record.get("is_renamed", entry.get("is_renamed", False))
                original_filename = cached_record.get("original_filename", entry.get("original_filename", ""))
                has_metadata = cached_record.get("metadata_mtime", 0.0) > 0.0
            
            # Check local metadata sidecar
            metadata_exists = live_fp and live_fp["metadata_mtime"] > 0.0
            if metadata_exists:
                has_metadata = True
                if not use_cached or not sha256 or is_renamed is None or not original_filename:
                    metadata, is_specific = load_metadata_loose(path)
                    if metadata:
                        from civitai import unwrap_civitai_metadata as unwrap_metadata
                        unwrapped = unwrap_metadata(metadata)
                        if not sha256:
                            sha256 = get_sha256_from_metadata(metadata, path.name)
                        if not version_id:
                            version_id = unwrapped.get("id") or ""
                        if not model_id:
                            model_obj = unwrapped.get("model") or {}
                            model_id = unwrapped.get("modelId") or model_obj.get("id") or ""
                        
                        # Check if renamed
                        files = unwrapped.get("files", [])
                        if isinstance(files, list) and len(files) > 0:
                            file_names = [f.get("name") for f in files if isinstance(f, dict) and f.get("name")]
                            if file_names and path.name not in file_names:
                                # Determine if we should count it as renamed based on our heuristics
                                should_mark_renamed = False
                                if is_specific:
                                    should_mark_renamed = True
                                else:
                                    # Count safetensors in parent dir
                                    try:
                                        num_safetensors = sum(1 for p in path.parent.iterdir() if p.is_file() and p.suffix.lower() == ".safetensors")
                                        if num_safetensors == 1:
                                            should_mark_renamed = True
                                    except OSError:
                                        pass
                                        
                                if should_mark_renamed:
                                    is_renamed = True
                                    original_filename = get_original_filename(metadata)
            
            # Fallback if sha256 is empty/unavailable
            if not sha256:
                sha256 = "unavailable"
                
            has_preview = bool(get_preview_image(path))
            
            model_info = {
                "relative_path": rel_path.as_posix(),
                "hierarchy": hierarchy,
                "filename": path.name,
                "model_folder": path.parent.name if depth > 0 else "",
                "depth": depth,
                "sha256": sha256,
                "version_id": version_id,
                "model_id": model_id,
                "has_metadata": has_metadata,
                "has_preview": has_preview,
                "is_renamed": is_renamed,
                "original_filename": original_filename,
            }
            models.append(model_info)
            
            if depth == 0:
                root_level_models.append(path.name)
                
            # Track active folders recursively
            curr = path.parent
            while curr != lora_root and curr.is_relative_to(lora_root):
                dirs_with_models.add(curr.resolve())
                curr = curr.parent

    # Identify empty folders and orphaned folders
    for d in all_dirs:
        # Ignore logs and gallery directories at root level
        if d.parent == lora_root and d.name in ("logs", "gallery"):
            continue
            
        d_res = d.resolve()
        if d_res not in dirs_with_models and d_res != lora_root.resolve():
            # Check for metadata/preview sidecars
            is_orphaned = False
            try:
                for p in d.iterdir():
                    if p.is_file():
                        suffix = p.suffix.lower()
                        if suffix in PREVIEW_EXTENSIONS or suffix in (".json", ".info"):
                            is_orphaned = True
                            break
            except OSError:
                pass
                
            try:
                rel_dir = d.relative_to(lora_root).as_posix()
                if is_orphaned:
                    orphaned_folders.append(rel_dir)
                else:
                    empty_folders.append(rel_dir)
            except ValueError:
                pass
                
    empty_folders.sort()
    orphaned_folders.sort()
    
    # Calculate collections preview mappings & model counts
    # (only folders directly in root, and only if not equal to the model's direct parent folder)
    collection_counts = {}
    for m in models:
        h = m["hierarchy"]
        if len(h) > 1:
            col_path = " / ".join(h[:-1])
            parent_path = lora_root / Path(m["relative_path"]).parent
            cleaned = clean_model_collections([col_path], parent_path)
            if cleaned:
                collection_counts[col_path] = collection_counts.get(col_path, 0) + 1
            
    sorted_collections = sorted(collection_counts.items(), key=lambda x: (-x[1], x[0].lower()))
    
    # Calculate duplicates
    sha256_groups = {}
    version_groups = {}
    
    for m in models:
        sha = m["sha256"]
        if sha and sha != "unavailable":
            sha256_groups.setdefault(sha, []).append(m)
            
        ver = m["version_id"]
        if ver:
            version_groups.setdefault(str(ver), []).append(m)
            
    dup_sha_groups = {k: v for k, v in sha256_groups.items() if len(v) > 1}
    dup_ver_groups = {k: v for k, v in version_groups.items() if len(v) > 1}
    
    # Calculate health rating
    num_models = len(models)
    num_dup_sha_groups = len(dup_sha_groups)
    num_dup_ver_groups = len(dup_ver_groups)
    renamed_models = [m for m in models if m["is_renamed"]]
    missing_metadata_count = sum(1 for m in models if not m["has_metadata"])
    missing_preview_count = sum(1 for m in models if not m["has_preview"])
    missing_sha_count = sum(1 for m in models if m["sha256"] == "unavailable")
    num_root_models = len(root_level_models)
    num_orphaned = len(orphaned_folders)
    num_empty = len(empty_folders)
    
    if num_models == 0:
        health_level = "Excellent"
        explanation = "The library is empty. No issues detected."
    elif num_dup_sha_groups > 0 or num_dup_ver_groups > 0 or len(renamed_models) > 0 or (missing_metadata_count / num_models) > 0.25:
        health_level = "Problems Detected"
        explanation_parts = []
        if num_dup_sha_groups > 0:
            explanation_parts.append(f"{num_dup_sha_groups} duplicate SHA256 group(s)")
        if num_dup_ver_groups > 0:
            explanation_parts.append(f"{num_dup_ver_groups} duplicate Version ID group(s)")
        if len(renamed_models) > 0:
            explanation_parts.append(f"{len(renamed_models)} renamed model file(s)")
        if (missing_metadata_count / num_models) > 0.25:
            explanation_parts.append(f"high metadata gap ({missing_metadata_count}/{num_models} missing)")
        explanation = "Critical issues detected: " + ", ".join(explanation_parts) + "."
    elif missing_metadata_count > 0 or num_root_models > 0 or num_orphaned > 0 or (missing_preview_count / num_models) > 0.25:
        health_level = "Needs Attention"
        explanation_parts = []
        if missing_metadata_count > 0:
            explanation_parts.append(f"{missing_metadata_count} model(s) missing metadata")
        if num_root_models > 0:
            explanation_parts.append(f"{num_root_models} root-level model(s)")
        if num_orphaned > 0:
            explanation_parts.append(f"{num_orphaned} orphaned folder(s)")
        if (missing_preview_count / num_models) > 0.25:
            explanation_parts.append(f"high preview image gap ({missing_preview_count}/{num_models} missing)")
        explanation = "Needs attention: " + ", ".join(explanation_parts) + "."
    elif missing_preview_count > 0 or num_empty > 0 or missing_sha_count > 0:
        health_level = "Good"
        explanation_parts = []
        if missing_preview_count > 0:
            explanation_parts.append(f"{missing_preview_count} model(s) missing preview images")
        if num_empty > 0:
            explanation_parts.append(f"{num_empty} empty folder(s)")
        if missing_sha_count > 0:
            explanation_parts.append(f"{missing_sha_count} missing SHA256 hash(es)")
        explanation = "Generally healthy library with minor gaps: " + ", ".join(explanation_parts) + "."
    else:
        health_level = "Excellent"
        explanation = "No issues detected. Your library is clean and fully organized!"

    scan_duration = time.time() - start_time
    
    # Log summary statistics
    log_operation(
        f"[Migration] Scanned {num_models} models, "
        f"{len(collection_counts)} collections, "
        f"max depth {max_depth}, empty folders {num_empty}, "
        f"orphaned folders {num_orphaned}, renamed {len(renamed_models)}, "
        f"duplicates (sha={num_dup_sha_groups}, ver={num_dup_ver_groups}), "
        f"health={health_level}, duration {scan_duration:.3f}s"
    )
    
    return {
        "models": models,
        "root_level_models": root_level_models,
        "max_depth": max_depth,
        "empty_folders": empty_folders,
        "orphaned_folders": orphaned_folders,
        "collections": sorted_collections,
        "dup_sha_groups": dup_sha_groups,
        "dup_ver_groups": dup_ver_groups,
        "renamed_models": renamed_models,
        "missing_metadata_count": missing_metadata_count,
        "missing_preview_count": missing_preview_count,
        "missing_sha_count": missing_sha_count,
        "health_level": health_level,
        "explanation": explanation,
        "scan_duration": scan_duration,
    }


def generate_migration_preview(data: dict) -> None:
    """Format and display the scanned hierarchy data model to the console."""
    safe_print("=" * 50)
    safe_print("              LIBRARY ANALYSIS REPORT             ")
    safe_print("=" * 50)
    safe_print()
    
    safe_print("Scan Summary:")
    safe_print(f"  - Scan Duration:           {data['scan_duration']:.3f} seconds")
    safe_print(f"  - Total Models Detected:   {len(data['models'])}")
    safe_print(f"  - Root-Level Models:       {len(data['root_level_models'])}")
    safe_print(f"  - Maximum Folder Depth:     {data['max_depth']}")
    safe_print()
    
    safe_print("Folder & Collection Health:")
    safe_print(f"  - Empty Folders:           {len(data['empty_folders'])}")
    safe_print(f"  - Orphaned Asset Folders:   {len(data['orphaned_folders'])}")
    safe_print(f"  - Total Collections:       {len(data['collections'])}")
    safe_print()
    
    safe_print("Metadata & Asset Completeness:")
    safe_print(f"  - Models with Metadata:    {len(data['models']) - data['missing_metadata_count']}")
    safe_print(f"  - Models missing Metadata: {data['missing_metadata_count']}")
    safe_print(f"  - Models with Previews:    {len(data['models']) - data['missing_preview_count']}")
    safe_print(f"  - Models missing Previews: {data['missing_preview_count']}")
    safe_print(f"  - Missing SHA256 Hashes:   {data['missing_sha_count']}")
    safe_print(f"  - Renamed Model Files:     {len(data['renamed_models'])}")
    safe_print()
    
    safe_print("-" * 50)
    safe_print("MIGRATION HEALTH RATING")
    safe_print("-" * 50)
    safe_print(f"Health Rating: {data['health_level']}")
    safe_print(f"Explanation:   {data['explanation']}")
    safe_print()
    
    if data['collections']:
        safe_print("-" * 50)
        safe_print("COLLECTIONS PREVIEW")
        safe_print("-" * 50)
        safe_print(f"The following {len(data['collections'])} collections would be imported:")
        for col_name, count in data['collections'][:10]:
            safe_print(f"  - {col_name} ({count} model{'s' if count != 1 else ''})")
        if len(data['collections']) > 10:
            safe_print(f"  ... and {len(data['collections']) - 10} more.")
        safe_print()
        
    if data['renamed_models']:
        safe_print("-" * 50)
        safe_print("RENAMED FILES DETECTED (showing up to 5)")
        safe_print("-" * 50)
        for rm in data['renamed_models'][:5]:
            orig = rm['original_filename'] or 'unknown'
            safe_print(f"  - {rm['relative_path']}")
            safe_print(f"    (Originally: {orig})")
        if len(data['renamed_models']) > 5:
            safe_print(f"  ... and {len(data['renamed_models']) - 5} more.")
        safe_print()
        
    if data['dup_sha_groups']:
        safe_print("-" * 50)
        safe_print(f"DUPLICATE SHA256 GROUPS (showing up to 3 of {len(data['dup_sha_groups'])})")
        safe_print("-" * 50)
        for idx, (sha, grp) in enumerate(sorted(data['dup_sha_groups'].items())[:3]):
            safe_print(f"  Group {idx + 1} [SHA256: {sha[:10]}...]:")
            for m in grp:
                safe_print(f"    ├── {m['relative_path']}")
        if len(data['dup_sha_groups']) > 3:
            safe_print(f"  ... and {len(data['dup_sha_groups']) - 3} more duplicate SHA256 groups.")
        safe_print()
        
    if data['dup_ver_groups']:
        safe_print("-" * 50)
        safe_print(f"DUPLICATE VERSION ID GROUPS (showing up to 3 of {len(data['dup_ver_groups'])})")
        safe_print("-" * 50)
        for idx, (ver, grp) in enumerate(sorted(data['dup_ver_groups'].items())[:3]):
            safe_print(f"  Group {idx + 1} [Version ID: {ver}]:")
            for m in grp:
                safe_print(f"    ├── {m['relative_path']}")
        if len(data['dup_ver_groups']) > 3:
            safe_print(f"  ... and {len(data['dup_ver_groups']) - 3} more duplicate Version ID groups.")
        safe_print()

    if data['empty_folders']:
        safe_print("-" * 50)
        safe_print("EMPTY FOLDERS DETECTED (showing up to 5)")
        safe_print("-" * 50)
        for ef in data['empty_folders'][:5]:
            safe_print(f"  [Empty] {ef}")
        if len(data['empty_folders']) > 5:
            safe_print(f"  ... and {len(data['empty_folders']) - 5} more empty folders.")
        safe_print()
        
    if data['orphaned_folders']:
        safe_print("-" * 50)
        safe_print("ORPHANED ASSET FOLDERS DETECTED (showing up to 5)")
        safe_print("-" * 50)
        for of in data['orphaned_folders'][:5]:
            safe_print(f"  [Orphaned] {of}")
        if len(data['orphaned_folders']) > 5:
            safe_print(f"  ... and {len(data['orphaned_folders']) - 5} more orphaned folders.")
        safe_print()
        
    safe_print("=" * 50)


def generate_missing_sha256_hashes(lora_root: Path) -> None:
    """Scan the library and generate missing SHA256 hashes, caching them in library_index.json."""
    from library_index import (
        load_library_index,
        save_library_index,
        make_index_key,
        get_file_fingerprint,
    )
    from civitai import compute_sha256
    
    lora_root = Path(lora_root).resolve()
    
    # Collect all model files
    model_files = []
    for ext in LORA_EXTENSIONS:
        for path in lora_root.rglob(f"*{ext}"):
            if is_lora_file(path):
                # Skip logs and gallery folders
                try:
                    rel = path.relative_to(lora_root)
                    parts = rel.parts
                    if parts and parts[0] in ("logs", "gallery"):
                        continue
                except ValueError:
                    continue
                model_files.append(path)
                
    # Deduplicate files
    seen = set()
    unique_files = []
    for path in model_files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(path)
            
    if not unique_files:
        print("No model files found in the library.")
        return
        
    index = load_library_index(lora_root)
    cached_models = index.setdefault("models", {})
    
    # Identify which models are missing SHA256
    missing_hashes = []
    for path in unique_files:
        key = make_index_key(path, lora_root)
        cached_record = cached_models.get(key)
        
        sha256 = ""
        if cached_record:
            entry = cached_record.get("entry", {})
            sha256 = cached_record.get("sha256") or entry.get("sha256", "")
            
        if not sha256:
            missing_hashes.append((path, key))
            
    if not missing_hashes:
        print("All models already have cached or metadata-provided SHA256 hashes. No action needed.")
        return
        
    print(f"Found {len(missing_hashes)} model(s) missing SHA256 hashes.")
    print("Computing hashes... (this may take a while for large files)")
    
    start_time = time.time()
    generated_count = 0
    
    # Use tqdm if available, else standard loop
    try:
        from tqdm import tqdm
        iterator = tqdm(missing_hashes, desc="Generating SHA256", unit="file")
    except ImportError:
        iterator = missing_hashes
        
    for path, key in iterator:
        try:
            if not isinstance(iterator, list):
                # Update progress bar description
                iterator.set_description(f"Hashing {path.name[:30]}")
            else:
                print(f"[{generated_count + 1}/{len(missing_hashes)}] Hashing {path.name}...")
                
            sha256_hash = compute_sha256(path)
            
            # Retrieve live fingerprint
            live_fp = get_file_fingerprint(path) or {}
            
            # Fetch or create cached record
            cached_record = cached_models.setdefault(key, {
                "mtime": live_fp.get("mtime", 0),
                "size": live_fp.get("size", 0),
                "metadata_mtime": live_fp.get("metadata_mtime", 0),
            })
            
            # Store sha256
            cached_record["sha256"] = sha256_hash
            if "entry" in cached_record:
                cached_record["entry"]["sha256"] = sha256_hash
                
            generated_count += 1
        except Exception as e:
            print(f"Error computing hash for {path.name}: {e}")
            
    if generated_count > 0:
        save_library_index(lora_root, index)
        duration = time.time() - start_time
        print(f"Successfully generated and cached {generated_count} SHA256 hash(es) in {duration:.2f} seconds.")
    else:
        print("No hashes were generated due to errors.")


def add_to_global_collections(lora_root: Path, collection_name: str) -> None:
    """Add a collection name to the global collections.json file."""
    collections_file = lora_root / "collections.json"
    collections_list = []
    if collections_file.exists():
        try:
            with collections_file.open("r", encoding="utf-8") as f:
                collections_list = json.load(f)
        except Exception:
            pass
    if not isinstance(collections_list, list):
        collections_list = []
        
    cleaned_list = [c.strip() for c in collections_list if c.strip()]
    exists = any(c.lower() == collection_name.lower() for c in cleaned_list)
    if not exists:
        cleaned_list.append(collection_name)
        cleaned_list.sort(key=lambda x: x.lower())
        try:
            with collections_file.open("w", encoding="utf-8") as f:
                json.dump(cleaned_list, f, indent=2, ensure_ascii=False)
        except OSError:
            pass


def package_loose_models(lora_root: Path) -> int:
    """Find and package all loose model files in the library.

    A model file is considered loose (not in its own folder) if:
    1. It is directly inside lora_root.
    2. Or its parent directory contains more than 1 model file directly.
    """
    from config import is_lora_file
    
    lora_root = Path(lora_root).resolve()
    packaged_count = 0
    
    # 1. Find all directories in the library (including lora_root)
    all_dirs = [lora_root]
    try:
        for p in lora_root.rglob("*"):
            if p.is_dir():
                # Skip logs and gallery
                rel = p.relative_to(lora_root)
                parts = rel.parts
                if parts and parts[0] in ("logs", "gallery"):
                    continue
                all_dirs.append(p)
    except Exception as e:
        log_operation(f"[Import] Error listing directories for packaging: {e}")
        return 0
        
    # 2. For each directory, count model files directly inside it
    for directory in all_dirs:
        try:
            model_files = []
            for p in directory.iterdir():
                if p.is_file() and p.suffix.lower() in LORA_EXTENSIONS and is_lora_file(p):
                    model_files.append(p)
                    
            # Determine if this directory holds loose files
            is_root = (directory == lora_root)
            if (is_root and len(model_files) >= 1) or (not is_root and len(model_files) > 1):
                # We need to package these files!
                for model_file in model_files:
                    stem = model_file.stem
                    target_dir = directory / stem
                    target_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Move the model file and all related files (same stem) in the parent directory
                    for p in directory.iterdir():
                        if p.is_file() and p.stem == stem:
                            dest = target_dir / p.name
                            try:
                                shutil.move(str(p), str(dest))
                            except Exception as e:
                                log_operation(f"[Import] Error packaging related file {p.name} to {dest}: {e}")
                                
                    log_operation(f"[Import] Packaged loose model {model_file.name} into folder {target_dir.name}")
                    packaged_count += 1
        except OSError as e:
            log_operation(f"[Import] Error scanning directory {directory} for loose models: {e}")
            
    return packaged_count


def import_existing_library(lora_root: Path) -> None:
    """Read the existing folder structure and convert it into collections in user_data.json.

    This function is strictly non-destructive for already folder-packaged models.
    For loose models, it packages them first.
    """
    # 1. Package loose models first
    print("Packaging any loose model files at the root level or inside category folders...")
    packaged = package_loose_models(lora_root)
    if packaged > 0:
        print(f"Packaged {packaged} loose model(s) into their own folders.")
        
    print("Scanning library...")
    data = scan_library_hierarchy(lora_root)
    models = data.get("models", [])
    
    if not models:
        print("No model files found to import.")
        return
        
    print(f"Processing collections for {len(models)} model(s)...")
    
    imported = 0
    skipped = 0
    errors = 0
    
    for m in models:
        model_path = lora_root / m["relative_path"]
        user_data_path = model_path.parent / "user_data.json"
        
        # Load user_data
        user_data = {}
        if user_data_path.exists():
            try:
                with user_data_path.open("r", encoding="utf-8") as f:
                    user_data = json.load(f)
            except Exception:
                pass
                
        existing_collections = user_data.get("collections", [])
        if not isinstance(existing_collections, list):
            existing_collections = []
            
        cleaned_existing = clean_model_collections(existing_collections, model_path.parent)
        needs_write = len(cleaned_existing) != len(existing_collections)
        
        # Calculate collection name from folder hierarchy: full subcategory path under lora_root
        hierarchy = m["hierarchy"]
        new_tag_added = False
        
        if len(hierarchy) > 1:
            collection_name = " / ".join(hierarchy[:-1])
            # Don't add a collection tag if it is model-specific
            parent_path = lora_root / Path(m["relative_path"]).parent
            if clean_model_collections([collection_name], parent_path):
                # Check if already imported (case-insensitive check)
                already_imported = any(c.lower() == collection_name.lower() for c in cleaned_existing)
                if not already_imported:
                    cleaned_existing.append(collection_name)
                    new_tag_added = True
                    needs_write = True
                    
        if needs_write:
            user_data["collections"] = cleaned_existing
            user_data["schema_version"] = user_data.get("schema_version", 1)
            user_data["last_modified"] = datetime.now().isoformat() + "Z"
            try:
                model_path.parent.mkdir(parents=True, exist_ok=True)
                with user_data_path.open("w", encoding="utf-8") as f:
                    json.dump(user_data, f, indent=2)
                if new_tag_added:
                    imported += 1
                else:
                    # Cleaned up existing collections
                    pass
            except Exception as e:
                log_operation(f"[Import] Error saving/cleaning user_data.json for {model_path}: {e}")
                errors += 1
        else:
            skipped += 1
    print()
    print("=" * 50)
    print("                 IMPORT COMPLETED")
    print("=" * 50)
    print(f"  - Models Processed:   {len(models)}")
    print(f"  - Collections Added:  {imported}")
    print(f"  - Skipped (Already):  {skipped}")
    print(f"  - Errors:             {errors}")
    print("=" * 50)
    print()
    
    # Rebuild collections.json globally from all cleaned user_data.json files
    print("Cleaning and rebuilding global collections.json...")
    collections_set = set()
    for path in lora_root.rglob("user_data.json"):
        try:
            with path.open("r", encoding="utf-8") as f:
                ud = json.load(f)
            colls = ud.get("collections", [])
            for c in colls:
                if c and c.strip():
                    c_clean = c.strip()
                    if c_clean.lower() == "favorites":
                        c_clean = "Favorites"
                    elif c_clean.lower() == "library":
                        c_clean = "Library"
                    
                    if c_clean not in ("Library", "Favorites"):
                        # Case-insensitive deduplication
                        exists = False
                        for existing in collections_set:
                            if existing.lower() == c_clean.lower():
                                exists = True
                                break
                        if not exists:
                            collections_set.add(c_clean)
        except Exception:
            continue
            
    collections_file = lora_root / "collections.json"
    try:
        with collections_file.open("w", encoding="utf-8") as f:
            json.dump(sorted(list(collections_set), key=lambda x: x.lower()), f, indent=2, ensure_ascii=False)
        print("Global collections.json updated.")
    except Exception as e:
        log_operation(f"[Import] Error saving global collections.json: {e}")
        print(f"Error saving global collections: {e}")

    print("Refreshing gallery...")
    from main import refresh_gallery
    try:
        refresh_gallery(lora_root)
        print("Gallery refreshed successfully.")
    except Exception as e:
        print(f"Error refreshing gallery: {e}")
