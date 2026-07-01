"""Gallery data builders for LoRA library previews."""

import html
import json
import re
from pathlib import Path
from urllib.parse import quote

from civitai import unwrap_civitai_metadata as unwrap_metadata

from config import (
    get_lora_root,
    is_lora_file,
    PREVIEW_EXTENSIONS,
    LORA_EXTENSIONS,
    is_dry_run,
    is_verbose,
    log_operation,
)

METADATA_FILENAME = "metadata.json"


# Extract SHA256 from CivitAI metadata when available.
def get_sha256_from_metadata(metadata: dict, filename: str) -> str:
    unwrapped = unwrap_metadata(metadata)
    for file_info in unwrapped.get("files", []):
        if isinstance(file_info, dict):
            if file_info.get("name") == filename or file_info.get("primary") is True:
                hashes = file_info.get("hashes", {})
                if isinstance(hashes, dict) and hashes.get("SHA256"):
                    return str(hashes["SHA256"]).lower()
    return ""


# Find the standard metadata file inside a LoRA model folder.
def get_metadata_path(lora_file: Path) -> Path:
    return lora_file.parent / METADATA_FILENAME


# Check if metadata belongs to a specific model filename.
def metadata_belongs_to_file(metadata: dict, filename: str) -> bool:
    if not metadata:
        return False
    unwrapped = unwrap_metadata(metadata)
    files = unwrapped.get("files", [])
    if isinstance(files, list) and len(files) > 0:
        # Normalize names: lowercase, strip common prefixes
        def normalize(name_str: str) -> str:
            if not name_str:
                return ""
            # Strip leading digits and underscore (e.g. 1765641621390_)
            s = re.sub(r'^\d+_', '', name_str.lower())
            return s
            
        target = normalize(filename)
        for f in files:
            if isinstance(f, dict):
                f_name = f.get("name")
                if f_name and normalize(f_name) == target:
                    return True
        return False
    return True


def merge_metadata_dicts(dict1: dict, dict2: dict) -> dict:
    merged = dict1.copy()
    for k, v in dict2.items():
        if k not in merged or merged[k] is None:
            merged[k] = v
        elif isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = merge_metadata_dicts(merged[k], v)
        elif isinstance(merged[k], list) and isinstance(v, list):
            for item in v:
                if item not in merged[k]:
                    merged[k].append(item)
    return merged


# Load metadata.json or sidecar metadata files from a LoRA model folder if they exist.
def load_metadata(lora_file: Path) -> dict:
    stem_json = lora_file.parent / f"{lora_file.stem}.json"
    civitai_info = lora_file.parent / f"{lora_file.stem}.civitai.info"
    metadata_json = get_metadata_path(lora_file)

    data_stem = {}
    data_civitai = {}
    data_meta = {}

    # Load stem.json (always trust since it has the exact same name as the safetensors)
    if stem_json.exists():
        try:
            with stem_json.open("r", encoding="utf-8") as f:
                data_stem = json.load(f)
        except Exception:
            pass

    # Load stem.civitai.info (always trust since it has the exact same name as the safetensors)
    if civitai_info.exists():
        try:
            with civitai_info.open("r", encoding="utf-8") as f:
                data_civitai = json.load(f)
        except Exception:
            pass

    # Load metadata.json
    if metadata_json.exists():
        try:
            with metadata_json.open("r", encoding="utf-8") as f:
                d = json.load(f)
                
                # Check if it belongs to this model, or if it is the only model in the folder
                belongs = False
                if metadata_belongs_to_file(d, lora_file.name):
                    belongs = True
                else:
                    try:
                        from config import LORA_EXTENSIONS
                        loras_in_folder = []
                        for ext in LORA_EXTENSIONS:
                            loras_in_folder.extend(list(lora_file.parent.glob(f"*{ext}")))
                        if len(loras_in_folder) == 1:
                            belongs = True
                    except Exception:
                        pass
                
                if belongs:
                    data_meta = d
        except Exception:
            pass

    # Merge all valid metadata dictionaries
    merged_data = {}
    if data_stem:
        merged_data = data_stem
    if data_civitai:
        merged_data = merge_metadata_dicts(merged_data, data_civitai) if merged_data else data_civitai
    if data_meta:
        merged_data = merge_metadata_dicts(merged_data, data_meta) if merged_data else data_meta

    # If we merged multiple files, write the combined metadata back to <stem>.json
    # and clean up standard metadata.json/civitai.info to avoid duplication
    if merged_data:
        try:
            if not is_dry_run():
                with stem_json.open("w", encoding="utf-8") as f:
                    json.dump(merged_data, f, indent=2)
                
                # Delete redundant files
                if metadata_json.exists() and metadata_json.resolve() != stem_json.resolve():
                    try:
                        metadata_json.unlink()
                        log_operation(f"Deleted redundant metadata.json in {lora_file.parent}")
                    except OSError:
                        pass
                if civitai_info.exists() and civitai_info.resolve() != stem_json.resolve():
                    try:
                        civitai_info.unlink()
                        log_operation(f"Deleted redundant civitai.info in {lora_file.parent}")
                    except OSError:
                        pass
        except Exception as e:
            log_operation(f"Error saving merged metadata for {lora_file.name}: {e}")

        return merged_data

    return {}



# Return the official model name from CivitAI metadata when available.
def get_model_name(metadata: dict, lora_file: Path) -> str:
    model = metadata.get("model", {})

    if isinstance(model, dict) and model.get("name"):
        return model["name"]

    return metadata.get("name", lora_file.stem)


# Return trigger words from CivitAI metadata when available.
def get_trigger_words(metadata: dict) -> list[str]:
    trained_words = metadata.get("trainedWords", [])

    if isinstance(trained_words, list):
        flat_words = []
        for group in trained_words:
            if isinstance(group, str):
                for word in group.split(","):
                    cleaned = word.strip()
                    if cleaned:
                        flat_words.append(cleaned)
            elif isinstance(group, list):
                for word in group:
                    cleaned = str(word).strip()
                    if cleaned:
                        flat_words.append(cleaned)
        return flat_words

    return []


# Parse trigger words into style groups from metadata, with custom group names.
def parse_trigger_groups(metadata: dict) -> list[dict]:
    unwrapped = unwrap_metadata(metadata)
    trained_words = unwrapped.get("trainedWords")
    
    if not isinstance(trained_words, list):
        if isinstance(trained_words, str):
            trained_words = [trained_words]
        else:
            return []

    priority_keywords = [
        "outfit", "cosplay", "costume", "dress", "uniform", "swimsuit", "bikini", 
        "lingerie", "kimono", "armor", "hoodie", "suit", "sweater", "vest", 
        "jacket", "coat", "skirt", "shirt", "pant", "shorts", "hat"
    ]
    
    generic_tags = {
        "solo", "1girl", "2girls", "1boy", "2boys", "breasts", "cleavage", "mature female", 
        "sidelocks", "bangs", "anime screencap", "frills", "collarbone", "gloves", 
        "long sleeves", "short sleeves", "sleeveless", "highleg", "blush", "smile", 
        "open mouth", "navel", "butt", "pussy", "ass", "detailed", "cum", "nude", 
        "penis", "urethra", "clitoris", "spread pussy", "sex", "shadow", "flat color", 
        "anime coloring", "monochrome", "greyscale", "background", "legs", "thighhighs"
    }

    groups = []
    used_titles = set()

    for idx, group_str in enumerate(trained_words):
        if not isinstance(group_str, str):
            continue
        
        tags = [t.strip() for t in group_str.split(",") if t.strip()]
        if not tags:
            continue
            
        candidates = []
        for t_idx, tag in enumerate(tags):
            tag_clean = tag.lower()
            if "lora:" in tag_clean or "<" in tag_clean or ">" in tag_clean:
                continue
                
            score = len(tag_clean)
            for keyword in priority_keywords:
                if keyword in tag_clean:
                    score += 100
                    break
                    
            if tag_clean in generic_tags or any(g in tag_clean for g in ["sleeves", "breasts", "cleavage", "screencap"]):
                score -= 100
                
            if t_idx == 0 and tag_clean not in generic_tags:
                score += 30
                
            candidates.append((tag, score))
            
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        title = ""
        for tag, score in candidates:
            tag_title = tag.title()
            if tag_title not in used_titles:
                title = tag_title
                used_titles.add(title)
                break
                
        if not title:
            if candidates:
                title = candidates[0][0].title()
                used_titles.add(title)
            else:
                title = f"Style Group {idx + 1}"
                
        groups.append({
            "title": title,
            "words": tags
        })
        
    return groups


# Return the best description from CivitAI metadata when available.
def get_description(metadata: dict) -> str:
    model = metadata.get("model", {})

    if metadata.get("description"):
        return metadata["description"]

    if isinstance(model, dict) and model.get("description"):
        return model["description"]

    return ""


# Return the local same-basename preview image for a LoRA file if it exists.
def get_preview_image(lora_file: Path) -> str:
    for extension in PREVIEW_EXTENSIONS:
        preview_path = lora_file.parent / f"{lora_file.stem}{extension}"

        if preview_path.exists():
            return str(preview_path)

    return ""


# Known non-creator suffix tokens found in model filenames — used to reject
# false positives when falling back to filename-based creator detection.
_NON_CREATOR_TOKENS = {
    "xl", "v1", "v2", "v3", "v4", "v5", "lora", "loha", "lycoris",
    "pony", "sdxl", "sd15", "sd21", "flux", "illustrious", "animagine",
    "noob", "animefull", "base", "safetensors", "ckpt", "pt", "bin",
    "fp16", "fp32", "bf16", "pruned", "full", "final", "last", "merged",
}


def _extract_creator_from_filename(stem: str) -> str:
    """Try to extract a CivitAI creator/author name from a filename stem.

    Primary strategy — explicit ``-lora-<creator>`` delimiter used by most
    CivitAI auto-downloads:

        ayako-kuroba-s2-illustriousxl-lora-nochekaiser  →  nochekaiser

    Returns empty string when the filename does not follow this naming convention.
    """
    stem_lower = stem.lower()

    # Look for explicit -lora- delimiter
    if "-lora-" in stem_lower:
        after = stem_lower.split("-lora-", 1)[1]
        # Take first segment in case of further hyphens/underscores
        candidate = after.split("-")[0].split("_")[0]
        if candidate and candidate.isalpha() and 3 <= len(candidate) <= 30:
            return candidate

    return ""



# Build one gallery entry for a LoRA file.
def build_gallery_entry(lora_file: Path) -> dict:
    metadata = load_metadata(lora_file)
    unwrapped = unwrap_metadata(metadata)
    
    base_model = unwrapped.get("baseModel", "")
    model_obj = unwrapped.get("model") or {}
    model_name = model_obj.get("name", "") if isinstance(model_obj, dict) else ""
    model_type = model_obj.get("type", "") if isinstance(model_obj, dict) else ""
    
    creator_obj = model_obj.get("creator") if isinstance(model_obj, dict) else None
    creator = ""
    if isinstance(creator_obj, dict):
        creator = creator_obj.get("username", "")
    elif isinstance(unwrapped.get("creator"), dict):
        creator = unwrapped.get("creator", {}).get("username", "")
    elif isinstance(unwrapped.get("creator"), str):
        creator = unwrapped.get("creator", "")

    # Fallback: extract creator from filename when metadata has no creator field.
    # CivitAI downloads follow the pattern: <slug>-lora-<creator>.<ext>
    # e.g. "ayako-kuroba-s2-illustriousxl-lora-nochekaiser.safetensors" → "nochekaiser"
    if not creator:
        creator = _extract_creator_from_filename(lora_file.stem)

    tags = []
    raw_tags = unwrapped.get("tags") or (model_obj.get("tags") if isinstance(model_obj, dict) else None)
    if isinstance(raw_tags, list):
        for tag in raw_tags:
            if isinstance(tag, str):
                tags.append(tag)
            elif isinstance(tag, dict) and tag.get("name"):
                tags.append(tag["name"])

    relative_path = ""
    try:
        relative_path = lora_file.resolve().relative_to(get_lora_root().resolve()).as_posix()
    except ValueError:
        relative_path = lora_file.name

    sha256 = get_sha256_from_metadata(metadata, lora_file.name)
    model_id = unwrapped.get("modelId") or model_obj.get("id") or ""
    version_id = unwrapped.get("id") or ""
    version_name = unwrapped.get("name") or ""

    return {
        "name": get_model_name(metadata, lora_file),
        "folder_name": lora_file.parent.name,
        "file": str(lora_file),
        "file_stem": lora_file.stem,
        "filename": lora_file.name,
        "mtime": int(lora_file.stat().st_mtime),
        "size": lora_file.stat().st_size,
        "description": get_description(metadata),
        "trigger_words": get_trigger_words(metadata),
        "trigger_groups": parse_trigger_groups(metadata),
        "preview_image": get_preview_image(lora_file),
        "base_model": base_model,
        "model_name": model_name,
        "model_type": model_type,
        "creator": creator,
        "tags": tags,
        "relative_path": relative_path,
        "sha256": sha256,
        "model_id": model_id,
        "version_id": version_id,
        "version_name": version_name,
    }




# Build gallery entries for every LoRA file in the library.
# When use_cache is True (default), entries for unchanged files are read from
# library_index.json instead of rebuilding from disk, dramatically reducing
# metadata re-parsing on large unchanged libraries.
def build_gallery_index(library_dir: Path, use_cache: bool = True) -> list[dict]:
    if not library_dir.exists():
        return []

    # --- Filesystem scan (same logic as before) ---
    lora_files = []
    for ext in LORA_EXTENSIONS:
        for path in library_dir.rglob(f"*{ext}"):
            if is_lora_file(path):
                lora_files.append(path)

    # Deduplicate paths
    seen: set = set()
    unique_files: list = []
    for path in lora_files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(path)

    # --- Fast path: cache disabled ---
    if not use_cache:
        return [build_gallery_entry(path) for path in unique_files]

    # --- Cache-assisted path ---
    from library_index import (
        load_library_index,
        save_library_index,
        make_index_key,
        get_file_fingerprint,
        _fingerprint_matches,
    )

    index = load_library_index(library_dir)
    cached_models: dict = index.setdefault("models", {})
    current_keys: set = set()
    entries: list = []
    hits = 0
    misses = 0

    for lora_file in unique_files:
        key = make_index_key(lora_file, library_dir)
        current_keys.add(key)
        live_fp = get_file_fingerprint(lora_file)

        # Cache hit: fingerprint unchanged and entry present
        if live_fp and key in cached_models:
            cached_record = cached_models[key]
            if _fingerprint_matches(cached_record, live_fp) and "entry" in cached_record:
                entries.append(cached_record["entry"])
                hits += 1
                continue

        # Cache miss: rebuild entry from disk
        entry = build_gallery_entry(lora_file)
        entries.append(entry)
        misses += 1

        # Write updated fingerprint + entry back to cache record
        fp = live_fp or {}
        cached_models[key] = {
            "mtime": fp.get("mtime", 0),
            "size": fp.get("size", 0),
            "metadata_mtime": fp.get("metadata_mtime", 0),
            "entry": entry,
        }

    # Prune stale entries for files that no longer exist
    stale_keys = [k for k in cached_models if k not in current_keys]
    for k in stale_keys:
        del cached_models[k]

    log_operation(
        f"[Index] Gallery index: {hits} cache hits, {misses} rebuilds, "
        f"{len(stale_keys)} stale entries removed"
    )
    save_library_index(library_dir, index)

    return entries



# Save the gallery index as a JSON file.
def save_gallery_index(entries: list[dict], output_path: Path) -> Path:
    if is_dry_run():
        log_operation(f"[DRY RUN] Would save gallery index JSON to: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(entries, output_file, indent=2)

    return output_path


# Convert a filesystem path into a browser-friendly relative path.
def relative_asset_path(asset_path: str, output_path: Path) -> str:
    if not asset_path:
        return ""

    try:
        rel_path = Path(asset_path).resolve().relative_to(output_path.parent.resolve()).as_posix()
        return quote(rel_path, safe='/')
    except ValueError:
        return Path(asset_path).resolve().as_uri()


# Escape text so it is safe to place inside HTML.
def escape_text(value: object) -> str:
    return html.escape(str(value), quote=True)


# Strip HTML tags and unescape HTML entities.
def clean_html_text(text: str) -> str:
    if not text:
        return ""
    # Strip HTML tags
    clean = re.sub(r'<[^>]*>', ' ', text)
    # Replace multiple spaces/newlines with a single space
    clean = re.sub(r'\s+', ' ', clean)
    return html.unescape(clean.strip())


def build_trigger_groups_html(trigger_words: list, trigger_groups: list) -> str:
    if not trigger_groups and trigger_words:
        trigger_groups = [{
            "title": "Trigger Words",
            "words": trigger_words
        }]

    groups_html = ""
    for group in trigger_groups:
        title = group.get("title", "Trigger Words")
        words = group.get("words", [])
        words_count = len([w for w in words if w.strip()])
        words_text = ", ".join(words)
        
        list_items = "".join(f"<li>{escape_text(w)}</li>" for w in words if w.strip())
        esc_title = escape_text(title)
        esc_words_text = escape_text(words_text)
        title_display = f"{esc_title} ({words_count})"
        
        groups_html += f"""
        <div class="trigger-group" data-group-words="{esc_words_text}">
            <div class="trigger-group-header" role="button" aria-expanded="false" tabindex="0" aria-label="Toggle {esc_title} triggers">
                <span class="trigger-group-arrow">▶</span>
                <span class="trigger-group-title">{title_display}</span>
                <button class="btn-copy-group" aria-label="Copy {esc_title} triggers" type="button" tabindex="0">
                    <svg class="btn-icon copy-icon-normal" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                    </svg>
                    <svg class="btn-icon copy-icon-success" style="display:none;" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="20 6 9 17 4 12"/>
                    </svg>
                    <span class="copy-group-feedback">Copied!</span>
                </button>
            </div>
            <div class="trigger-group-content-wrapper">
                <div class="trigger-group-content">
                    <ul class="trigger-list">
                        {list_items}
                    </ul>
                </div>
            </div>
        </div>
        """
    return groups_html


# Build one HTML card for a gallery entry group.
def build_gallery_card(group: dict, output_path: Path) -> str:
    versions = group["versions"]
    latest_ver = versions[0]
    
    # Render version selector dropdown
    options_html = ""
    for ver in versions:
        v_name = ver["version_name"] or ver["file_stem"]
        options_html += f'<option value="{escape_text(ver["file_stem"])}">{escape_text(v_name)}</option>\n'
        
    version_selector_html = f"""
    <div class="version-selector-container">
        <span class="version-label">Version:</span>
        <select class="version-select" aria-label="Select model version">
            {options_html}
        </select>
        <button class="btn-delete-version" aria-label="Delete this version" type="button" tabindex="0">
            <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="3 6 5 6 21 6"/>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                <line x1="10" y1="11" x2="10" y2="17"/>
                <line x1="14" y1="11" x2="14" y2="17"/>
            </svg>
        </button>
    </div>
    """
        
    preview_image = relative_asset_path(latest_ver.get("preview_image", ""), output_path)
    trigger_words = latest_ver.get("trigger_words", [])
    trigger_text = ", ".join(trigger_words)
    
    hidden_search_text = " ".join([
        latest_ver.get("base_model", ""),
        group.get("creator", ""),
        " ".join(group.get("tags", []))
    ]).casefold()
    
    desc_raw = latest_ver.get("description", "") or ""
    desc_clean = clean_html_text(desc_raw)
    
    all_triggers = []
    all_stems = []
    for ver in versions:
        all_triggers.extend(ver.get("trigger_words", []))
        all_stems.append(ver.get("file_stem", ""))
        
    search_text = " ".join(
        [
            group.get("name", ""),
            group.get("folder_name", ""),
            ", ".join(all_triggers),
            " ".join(all_stems),
            desc_clean,
            hidden_search_text,
        ]
    ).casefold()

    if preview_image:
        image_html = (
            f'<img class="card-preview-image" src="{escape_text(preview_image)}" '
            f'alt="{escape_text(group.get("name", "LoRA preview"))}" loading="lazy">'
        )
    else:
        image_html = """
            <div class="missing-preview">
                <svg class="placeholder-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                    <circle cx="8.5" cy="8.5" r="1.5"/>
                    <polyline points="21 15 16 10 5 21"/>
                </svg>
                <span>No Preview Available</span>
            </div>
        """

    name = group.get("name", "Untitled LoRA")
    folder_name = group.get("folder_name", "")
    
    # Serialize versions for JS dynamic updates
    versions_serialized = []
    for ver in versions:
        rel_preview = relative_asset_path(ver.get("preview_image", ""), output_path)
        v_trigger_words = ver.get("trigger_words", [])
        v_trigger_groups = ver.get("trigger_groups", [])
        v_groups_html = build_trigger_groups_html(v_trigger_words, v_trigger_groups)
        
        versions_serialized.append({
            "filename": ver["filename"],
            "file_stem": ver["file_stem"],
            "version_name": ver["version_name"],
            "version_id": ver["version_id"],
            "model_id": ver.get("model_id"),
            "preview_image": rel_preview,
            "trigger_words": v_trigger_words,
            "trigger_groups_html": v_groups_html,
            "base_model": ver["base_model"],
            "description": ver["description"],
            "sha256": ver["sha256"],
            "relative_path": ver["relative_path"],
            "mtime": ver["mtime"],
            "size": ver.get("size", 0)
        })
        
    t_groups = latest_ver.get("trigger_groups", [])
    t_groups_count = 1 if (not t_groups and trigger_words) else len(t_groups)

    versions_json = json.dumps(versions_serialized)
    trigger_section_html = f"""
        <div class="trigger-groups-container">
            {build_trigger_groups_html(trigger_words, latest_ver.get("trigger_groups", [])) or '<p class="triggers">None found</p>'}
        </div>
    """

    return f"""
        <article class="card" 
                  data-search="{escape_text(search_text)}" 
                  data-name="{escape_text(name)}" 
                  data-filename="{escape_text(latest_ver['filename'])}" 
                  data-stem="{escape_text(latest_ver['file_stem'])}" 
                  data-mtime="{latest_ver['mtime']}" 
                  data-hash="{escape_text(latest_ver['sha256'])}"
                  data-path="{escape_text(latest_ver['relative_path'])}"
                  data-triggers="{escape_text(trigger_text)}"
                  data-creator="{escape_text(group.get('creator', ''))}"
                  data-base-model="{escape_text(latest_ver.get('base_model', ''))}"
                  data-size="{latest_ver.get('size', 0)}"
                  data-description="{escape_text(desc_clean)}"
                  data-tags="{escape_text(json.dumps(group.get('tags', [])))}"
                  data-official-triggers="{escape_text(json.dumps(trigger_words))}"
                  data-versions="{escape_text(versions_json)}"
                  data-collections="[]"
                  data-rating="0"
                  data-favorite="false">
            <div class="preview">
                {image_html}
                <button class="btn-favorite" aria-label="Toggle favorite" type="button">
                    <svg class="star-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
                    </svg>
                </button>
                {f'''
                <div class="creator-badge" onclick="filterByCreator(event, '{escape_text(group.get("creator", ""))}')">
                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="13.5" cy="6.5" r=".5" fill="currentColor"/>
                        <circle cx="17.5" cy="10.5" r=".5" fill="currentColor"/>
                        <circle cx="8.5" cy="7.5" r=".5" fill="currentColor"/>
                        <circle cx="6.5" cy="12.5" r=".5" fill="currentColor"/>
                        <path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.926 0 1.648-.746 1.648-1.688 0-.437-.18-.835-.437-1.125-.29-.289-.438-.652-.438-1.125a1.64 1.64 0 0 1 1.668-1.668h1.996c3.051 0 5.555-2.503 5.555-5.554C21.965 6.012 17.461 2 12 2z"/>
                    </svg>
                    {escape_text(group.get("creator", ""))}
                </div>
                ''' if group.get("creator") else ''}
                <div class="preview-feedback">Triggers Copied!</div>
            </div>
            <div class="content">
                <h2>{escape_text(name)}</h2>
                <p class="folder">{escape_text(folder_name)}</p>
                
                {version_selector_html}
                
                <div class="card-trigger-section">
                    <p class="label card-trigger-label">Trigger words ({t_groups_count})</p>
                    {trigger_section_html}
                </div>

                <p class="label">Description</p>
                <div class="description-wrapper" title="Click to view full details">
                    <p class="description">{escape_text(desc_clean or "No description available.")}</p>
                </div>
                
                <div class="card-footer">
                    <button class="btn-copy" aria-label="Copy LoRA tag" type="button" tabindex="0">
                        <svg class="btn-icon copy-normal" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                        </svg>
                        <svg class="btn-icon copy-success" style="display:none;" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <polyline points="20 6 9 17 4 12"/>
                        </svg>
                        <span class="btn-text">Copy Tag</span>
                    </button>
                    <button class="btn-add-prompt" aria-label="Add to Prompt" type="button" tabindex="0">
                        <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <line x1="12" y1="5" x2="12" y2="19"/>
                            <line x1="5" y1="12" x2="19" y2="12"/>
                        </svg>
                        <span class="btn-text">Add to Prompt</span>
                    </button>
                </div>
            </div>
        </article>
    """


def group_entries(entries: list[dict]) -> list[dict]:
    # Deduplicate entries by SHA256 to avoid showing duplicate cards for copy/backup files.
    # Prefer the entry that has more complete metadata.
    sha_map = {}
    for entry in entries:
        sha = entry.get("sha256")
        if sha and sha != "unavailable":
            if sha not in sha_map:
                sha_map[sha] = entry
            else:
                # Compare completeness
                existing = sha_map[sha]
                existing_score = 0
                if existing.get("model_id"): existing_score += 10
                if existing.get("trigger_words"): existing_score += len(existing.get("trigger_words"))
                if existing.get("description"): existing_score += 5
                
                entry_score = 0
                if entry.get("model_id"): entry_score += 10
                if entry.get("trigger_words"): entry_score += len(entry.get("trigger_words"))
                if entry.get("description"): entry_score += 5
                
                if entry_score > existing_score:
                    sha_map[sha] = entry
        else:
            # Keep entries without SHA256 as unique entries
            dummy_key = f"no_sha_{len(sha_map)}"
            sha_map[dummy_key] = entry
            
    deduplicated = list(sha_map.values())

    grouped = {}
    for entry in deduplicated:
        model_id = entry.get("model_id") or ""
        folder_name = entry.get("folder_name", "")
        
        if model_id:
            group_key = f"model_{model_id}"
        elif folder_name and folder_name.lower() != "lora":
            group_key = f"folder_{folder_name.lower()}"
        else:
            group_key = f"file_{entry['file_stem'].lower()}"
            
        if group_key not in grouped:
            grouped[group_key] = {
                "name": entry.get("model_name") or entry.get("name") or folder_name or entry.get("file_stem"),
                "folder_name": folder_name,
                "creator": entry.get("creator", ""),
                "tags": set(entry.get("tags", [])),
                "versions": []
            }
            
        version_name = entry.get("version_name")
        if not version_name:
            version_name = entry.get("file_stem")
            
        grouped[group_key]["versions"].append({
            "filename": entry.get("filename"),
            "file_stem": entry.get("file_stem"),
            "version_name": version_name,
            "version_id": entry.get("version_id"),
            "model_id": entry.get("model_id"),
            "preview_image": entry.get("preview_image"),
            "trigger_words": entry.get("trigger_words", []),
            "trigger_groups": entry.get("trigger_groups", []),
            "base_model": entry.get("base_model", ""),
            "description": entry.get("description", ""),
            "sha256": entry.get("sha256", ""),
            "relative_path": entry.get("relative_path", ""),
            "mtime": entry.get("mtime", 0)
        })
        
        if entry.get("creator") and not grouped[group_key]["creator"]:
            grouped[group_key]["creator"] = entry["creator"]
        grouped[group_key]["tags"].update(entry.get("tags", []))
        
    grouped_list = []
    for g in grouped.values():
        g["tags"] = sorted(list(g["tags"]))
        def get_version_sort_key(x):
            v_id = x.get("version_id")
            try:
                v_num = int(v_id) if v_id is not None else 0
            except (ValueError, TypeError):
                v_num = 0
            try:
                mtime_val = float(x.get("mtime", 0))
            except (ValueError, TypeError):
                mtime_val = 0.0
            return (v_num, mtime_val)
        g["versions"].sort(key=get_version_sort_key, reverse=True)
        grouped_list.append(g)
        
    grouped_list.sort(key=lambda x: x["name"].lower())
    return grouped_list


# Build a complete standalone HTML gallery document.
def build_gallery_html(entries: list[dict], output_path: Path) -> str:
    grouped = group_entries(entries)
    cards = "\n".join(build_gallery_card(group, output_path) for group in grouped)

    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>LoRA Library Manager Gallery</title>
    <style>
        :root {{
            color-scheme: dark;
            --bg: #090a0f;
            --panel: #11131c;
            --panel-hover: #161a26;
            --border: rgba(255, 255, 255, 0.05);
            --border-hover: rgba(99, 102, 241, 0.4);
            --text: #f3f4f6;
            --text-muted: #9ca3af;
            --accent: #6366f1;
            --accent-hover: #4f46e5;
            --accent-alpha: rgba(99, 102, 241, 0.1);
            --success: #10b981;
            --shadow: 0 12px 24px -10px rgba(0, 0, 0, 0.5);
            --transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        * {{
            box-sizing: border-box;
        }}

        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: var(--bg);
        }}
        ::-webkit-scrollbar-thumb {{
            background: #272a37;
            border-radius: 4px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: #3c4055;
        }}

        body {{
            margin: 0;
            background-color: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
        }}

        header {{
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgba(9, 10, 15, 0.85);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            padding: 16px 32px;
        }}

        .header-container {{
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            background: rgba(17, 19, 28, 0.45);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 8px 16px;
            gap: 16px;
            width: 100%;
        }}

        .header-left, .header-right {{
            display: flex;
            align-items: center;
            gap: 16px;
        }}

        .header-right {{
            flex-grow: 1;
        }}

        .divider {{
            width: 1px;
            height: 24px;
            background: rgba(255, 255, 255, 0.08);
            flex-shrink: 0;
        }}

        h1 {{
            margin: 0;
            font-size: 20px;
            font-weight: 800;
            background: linear-gradient(135deg, #a5b4fc, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.025em;
            white-space: nowrap;
        }}

        .btn-manage-link {{
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.25);
            color: #a5b4fc;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
            gap: 6px;
        }}

        .btn-manage-link:hover {{
            background: rgba(99, 102, 241, 0.25);
            border-color: var(--accent);
            color: #fff;
        }}

        .search-wrapper {{
            position: relative;
            display: flex;
            align-items: center;
            flex: 1;
            max-width: 100%;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: 8px;
            height: 38px;
            transition: var(--transition);
        }}

        .search-wrapper:focus-within {{
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
            background: #1c202b;
        }}

        .search-icon {{
            width: 16px;
            height: 16px;
            color: var(--text-muted);
            pointer-events: none;
            margin-left: 10px;
            flex-shrink: 0;
        }}

        #search {{
            flex: 1;
            background: transparent;
            border: none;
            color: var(--text);
            padding: 8px 14px 8px 8px;
            height: 100%;
            font-size: 14px;
            outline: none;
        }}

        .collection-chip {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(255, 255, 255, 0.04);
            border-right: 1px solid var(--border);
            color: var(--text);
            padding: 0 12px;
            height: 100%;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
            white-space: nowrap;
            user-select: none;
            border-top-left-radius: 7px;
            border-bottom-left-radius: 7px;
            outline: none;
        }}

        .collection-chip:hover, .collection-chip:focus {{
            background: rgba(255, 255, 255, 0.08);
            color: var(--text);
        }}

        .chip-arrow {{
            font-size: 10px;
            color: var(--text-muted);
        }}

        .chip-clear {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.1);
            color: var(--text-muted);
            font-size: 10px;
            font-weight: bold;
            cursor: pointer;
            transition: var(--transition);
        }}

        .chip-clear:hover {{
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
        }}

        .collections-popup {{
            position: absolute;
            top: calc(100% + 4px);
            left: 0;
            width: 100%;
            max-width: 400px;
            background: rgba(20, 24, 33, 0.98);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            border-radius: 8px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
            z-index: 1000;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            animation: popupFadeIn 0.15s ease-out;
        }}

        .collections-popup[hidden] {{
            display: none !important;
        }}

        @keyframes popupFadeIn {{
            from {{ opacity: 0; transform: translateY(-4px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .popup-search-wrapper {{
            display: flex;
            align-items: center;
            padding: 8px 12px;
            border-bottom: 1px solid var(--border);
            background: rgba(0, 0, 0, 0.2);
        }}

        .popup-search-icon {{
            width: 14px;
            height: 14px;
            color: var(--text-muted);
            margin-right: 8px;
            flex-shrink: 0;
        }}

        #popup-search-input {{
            flex: 1;
            background: transparent;
            border: none;
            color: var(--text);
            font-size: 13px;
            outline: none;
            padding: 2px 0;
        }}

        .popup-collections-list {{
            max-height: 250px;
            overflow-y: auto;
            padding: 4px 0;
        }}

        .popup-item {{
            display: flex;
            align-items: center;
            padding: 8px 12px;
            color: var(--text);
            font-size: 13px;
            cursor: pointer;
            transition: background var(--transition);
            user-select: none;
        }}

        .popup-item:hover, .popup-item.focused {{
            background: var(--accent-alpha);
            color: var(--text);
        }}

        .popup-item.active {{
            background: rgba(99, 102, 241, 0.2);
            color: var(--accent);
            font-weight: 600;
        }}

        .weight-control {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .weight-label {{
            font-size: 11px;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            white-space: nowrap;
        }}

        .weight-selector {{
            display: flex;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 2px;
            height: 38px;
            align-items: center;
        }}

        .btn-weight {{
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 0 12px;
            height: 32px;
            line-height: 32px;
            font-size: 13px;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            transition: var(--transition);
            outline: none;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .btn-weight:hover, .btn-weight:focus {{
            color: var(--text);
        }}

        .btn-weight.active {{
            background: var(--accent);
            color: #ffffff;
            box-shadow: 0 2px 4px rgba(99, 102, 241, 0.2);
        }}

        .select-wrapper {{
            position: relative;
            width: 110px;
        }}

        #sort, #filter-base-model {{
            width: 100%;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text);
            padding: 8px 28px 8px 14px;
            height: 38px;
            font-size: 14px;
            outline: none;
            appearance: none;
            -webkit-appearance: none;
            cursor: pointer;
            transition: var(--transition);
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        #sort:focus, #filter-base-model:focus {{
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
            background: #1c202b;
        }}

        .select-arrow {{
            position: absolute;
            right: 12px;
            top: 50%;
            transform: translateY(-50%);
            width: 16px;
            height: 16px;
            color: var(--text-muted);
            pointer-events: none;
        }}

        .count {{
            color: var(--text-muted);
            font-size: 14px;
            font-weight: 500;
            white-space: nowrap;
        }}

        main {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 32px;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 24px;
        }}

        .card {{
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: var(--shadow);
            transition: var(--transition);
            height: 690px; /* Fixed height so all cards are uniform */
        }}

        .card[hidden] {{
            display: none !important;
        }}

        .card:hover {{
            transform: translateY(-4px);
            border-color: var(--border-hover);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.6);
        }}

        .preview {{
            position: relative;
            aspect-ratio: 4 / 3;
            background: #171923;
            overflow: hidden;
            border-bottom: 1px solid var(--border);
            cursor: pointer;
        }}

        .preview img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            object-position: center top;
            display: block;
            transition: var(--transition);
        }}

        .card:hover .preview img {{
            transform: scale(1.04);
        }}

        /* Subtle artist/creator badge — top-left of the preview image */
        .creator-badge {{
            position: absolute;
            top: 12px;
            left: 12px;
            display: flex;
            align-items: center;
            gap: 4px;
            padding: 4px 8px;
            background: rgba(15, 16, 22, 0.75);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 20px;
            font-size: 10px;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.75);
            letter-spacing: 0.02em;
            pointer-events: auto;
            cursor: pointer;
            max-width: 140px;
            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
            opacity: 0.85;
            z-index: 5;
            transition: all 0.2s ease;
        }}

        .creator-badge:hover {{
            background: var(--accent);
            color: #ffffff;
            border-color: var(--accent);
            opacity: 1;
        }}

        .creator-badge svg {{
            flex-shrink: 0;
            opacity: 0.70;
        }}

        .card:hover .creator-badge {{
            opacity: 1;
        }}

        .btn-favorite {{
            position: absolute;
            top: 12px;
            right: 12px;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: rgba(15, 16, 22, 0.6);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text-muted);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: var(--transition);
            z-index: 10;
            padding: 0;
            outline: none;
        }}

        .btn-favorite:hover, .btn-favorite:focus {{
            background: rgba(15, 16, 22, 0.9);
            color: #fbbf24;
            transform: scale(1.1);
            border-color: rgba(251, 191, 36, 0.3);
        }}

        .btn-favorite.active {{
            color: #fbbf24;
            border-color: rgba(251, 191, 36, 0.4);
            background: rgba(251, 191, 36, 0.15);
        }}

        .btn-favorite svg {{
            width: 18px;
            height: 18px;
            fill: none;
            stroke: currentColor;
            transition: var(--transition);
        }}

        .btn-favorite.active svg {{
            fill: currentColor;
        }}

        .preview-feedback {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(99, 102, 241, 0.9);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 14px;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.2s ease;
            z-index: 5;
        }}

        .preview-feedback.show {{
            opacity: 1;
        }}

        .missing-preview {{
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            gap: 12px;
            color: var(--text-muted);
            background: radial-gradient(circle at center, #1b1e2a 0%, #11131c 100%);
        }}

        .placeholder-icon {{
            width: 48px;
            height: 48px;
            opacity: 0.4;
            transition: var(--transition);
        }}

        .card:hover .placeholder-icon {{
            opacity: 0.7;
            transform: scale(1.1);
        }}

        .missing-preview span {{
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            opacity: 0.6;
        }}

        .content {{
            padding: 16px;
            display: flex;
            flex-direction: column;
            flex: 1;
            min-height: 0;
            overflow: hidden;
        }}

        /* Wrapper that sits between version selector and description — natural height */
        .card-trigger-section {{
            min-height: 0;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            margin-bottom: 12px;
            height: 160px;
            position: relative;
        }}

        /* Fade-out gradient at the bottom of overflow trigger words */
        .card-trigger-section::after {{
            content: "";
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 40px;
            background: linear-gradient(to bottom, transparent, var(--panel));
            pointer-events: none;
        }}

        /* Inside a card, trigger container scrolls — fade at bottom masks the scrollbar edge */
        .card .card-trigger-section .trigger-groups-container {{
            min-height: 0;
            max-height: 160px;
            overflow-y: auto;
            scrollbar-width: none; /* Firefox */
            margin-bottom: 0;
        }}

        .card .card-trigger-section .trigger-groups-container::-webkit-scrollbar {{
            display: none; /* Chrome/Safari — fade is the visual cue */
        }}

        h2 {{
            margin: 0 0 6px 0;
            font-size: 18px;
            font-weight: 700;
            color: var(--text);
            line-height: 1.3;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .folder {{
            margin: 0 0 16px 0;
            font-size: 13px;
            color: var(--text-muted);
            background: rgba(255, 255, 255, 0.03);
            padding: 4px 8px;
            border-radius: 4px;
            align-self: flex-start;
            max-width: 100%;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .version-selector-container {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 0 0 12px 0;
            padding: 6px 10px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }}

        .version-label {{
            font-size: 10px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            white-space: nowrap;
            flex-shrink: 0;
        }}

        .version-select {{
            flex: 1;
            min-width: 0;
            width: 100%;
            background: transparent;
            border: none;
            color: var(--text);
            font-size: 13px;
            font-weight: 500;
            outline: none;
            cursor: pointer;
            padding-right: 20px;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 24 24' stroke='%239ca3af'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 9l-7 7-7-7'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right center;
            background-size: 14px;
            -webkit-appearance: none;
            -moz-appearance: none;
            appearance: none;
            text-overflow: ellipsis;
            white-space: nowrap;
            overflow: hidden;
        }}
        
        .version-select option {{
            background: var(--panel);
            color: var(--text);
        }}

        .btn-delete-version {{
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.2);
            border-radius: 6px;
            width: 28px;
            height: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: var(--transition);
            color: #ef4444;
            padding: 0;
            flex-shrink: 0;
        }}

        .btn-delete-version:hover, .btn-delete-version:focus-visible {{
            background: #ef4444;
            color: #ffffff;
            border-color: #ef4444;
            box-shadow: 0 0 10px rgba(239, 68, 68, 0.4);
        }}

        .btn-delete-version svg {{
            width: 15px;
            height: 15px;
        }}


        .label {{
            margin: 12px 0 4px 0;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #818cf8;
        }}

        .triggers {{
            margin: 0 0 12px 0;
            font-size: 13px;
            background: rgba(99, 102, 241, 0.05);
            border: 1px solid rgba(99, 102, 241, 0.1);
            border-radius: 6px;
            padding: 6px 10px;
            color: var(--text);
            word-break: break-word;
            font-family: SFMono-Regular, Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 12px;
        }}

        .description-wrapper {{
            position: relative;
            flex: 1;          /* fills remaining space between trigger section and footer */
            min-height: 0;
            overflow: hidden;
            cursor: pointer;
            margin-bottom: 2px;
        }}

        /* Fade only shown when text actually overflows */
        .description-wrapper.overflowing::after {{
            content: "";
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 32px;
            background: linear-gradient(to bottom, transparent, var(--panel));
            pointer-events: none;
        }}

        .description-wrapper.overflowing:hover::after {{
            background: linear-gradient(to bottom, transparent, color-mix(in srgb, var(--panel) 85%, var(--accent) 15%));
        }}

        .description {{
            margin: 0;
            font-size: 13px;
            color: var(--text-muted);
            line-height: 1.5;
            overflow: hidden;
        }}

        .card-footer {{
            margin-top: auto;
            padding-top: 10px;
            display: flex;
            gap: 8px;
        }}

        .btn-copy, .btn-add-prompt {{
            flex: 1;
            background: var(--accent-alpha);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 8px;
            color: #a5b4fc;
            padding: 6px 10px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            transition: var(--transition);
            outline: none;
        }}

        .btn-copy:hover, .btn-copy:focus-visible,
        .btn-add-prompt:hover, .btn-add-prompt:focus-visible {{
            background: var(--accent);
            color: #ffffff;
            border-color: var(--accent);
        }}

        .btn-copy:active, .btn-add-prompt:active {{
            transform: scale(0.98);
        }}

        .btn-copy.copied {{
            background: rgba(16, 185, 129, 0.1);
            border-color: rgba(16, 185, 129, 0.3);
            color: #34d399;
        }}

        .btn-copy.copied:hover {{
            background: var(--success);
            color: #ffffff;
            border-color: var(--success);
        }}

        .btn-copy.copied .btn-text {{
            font-family: SFMono-Regular, Consolas, monospace;
            font-size: 11px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 210px;
        }}

        .btn-icon {{
            width: 16px;
            height: 16px;
        }}

        .empty {{
            display: none;
            color: var(--text-muted);
            padding: 60px 0;
            text-align: center;
            font-size: 16px;
        }}

        /* Collapsible Trigger Groups */
        .trigger-groups-container {{
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-bottom: 12px;
            max-height: 280px;
            overflow-y: auto;
            padding-right: 4px;
        }}

        .trigger-group {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            transition: var(--transition);
            flex-shrink: 0;
        }}

        .trigger-group:hover {{
            border-color: rgba(99, 102, 241, 0.2);
            background: rgba(255, 255, 255, 0.04);
        }}

        .trigger-group-header {{
            display: flex;
            align-items: center;
            padding: 8px 12px;
            cursor: pointer;
            user-select: none;
            outline: none;
            gap: 8px;
            position: relative;
        }}

        .trigger-group-header:focus-visible {{
            background: rgba(99, 102, 241, 0.1);
        }}

        .trigger-group-arrow {{
            font-size: 10px;
            color: var(--text-muted);
            transition: transform 0.2s ease;
            width: 12px;
            display: inline-block;
            text-align: center;
        }}

        .trigger-group.expanded .trigger-group-arrow {{
            transform: rotate(90deg);
        }}

        .trigger-group-title {{
            font-size: 13px;
            font-weight: 600;
            color: var(--text);
            flex-grow: 1;
        }}

        .btn-copy-group {{
            background: transparent;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            padding: 4px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: var(--transition);
            outline: none;
            position: relative;
        }}

        .btn-copy-group:hover, .btn-copy-group:focus-visible {{
            color: var(--accent);
            background: rgba(99, 102, 241, 0.1);
        }}

        .btn-copy-group .btn-icon {{
            width: 14px;
            height: 14px;
        }}

        .copy-group-feedback {{
            position: absolute;
            right: 28px;
            background: var(--success);
            color: white;
            font-size: 10px;
            font-weight: 700;
            padding: 2px 6px;
            border-radius: 4px;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.2s ease;
            white-space: nowrap;
        }}

        .copy-group-feedback.show {{
            opacity: 1;
        }}

        .trigger-group-content-wrapper {{
            display: grid;
            grid-template-rows: 0fr;
            transition: grid-template-rows 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        .trigger-group.expanded .trigger-group-content-wrapper {{
            grid-template-rows: 1fr;
        }}

        .trigger-group-content {{
            min-height: 0;
            overflow: hidden;
        }}

        .trigger-list {{
            margin: 0;
            padding: 8px 12px 12px 12px;
            list-style-type: disc;
            list-style-position: inside;
            max-height: 120px;
            overflow-y: auto;
            border-top: 1px solid var(--border);
            background: rgba(0, 0, 0, 0.15);
        }}

        .trigger-list li {{
            font-size: 12px;
            color: var(--text-muted);
            margin-bottom: 4px;
            font-family: SFMono-Regular, Consolas, monospace;
        }}

        .trigger-list li:last-child {{
            margin-bottom: 0;
        }}

        /* Prompt Builder Panel */
        .prompt-builder {{
            position: fixed;
            bottom: 0;
            left: 50%;
            transform: translateX(-50%);
            width: calc(100% - 64px);
            max-width: 800px;
            background: rgba(17, 19, 28, 0.9);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            border-bottom: none;
            border-radius: 12px 12px 0 0;
            box-shadow: 0 -10px 30px rgba(0, 0, 0, 0.5);
            z-index: 1000;
            transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        .prompt-builder.collapsed {{
            transform: translate(-50%, calc(100% - 48px));
        }}

        .prompt-builder-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 24px;
            cursor: pointer;
            user-select: none;
            border-bottom: 1px solid var(--border);
            outline: none;
            height: 48px;
        }}

        .prompt-builder-header:focus-visible {{
            background: rgba(99, 102, 241, 0.1);
        }}

        .prompt-builder-title-container {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .builder-icon {{
            width: 18px;
            height: 18px;
            color: var(--accent);
        }}

        .prompt-builder-title {{
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            color: #a5b4fc;
        }}

        .prompt-badge {{
            background: var(--accent);
            color: white;
            font-size: 11px;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: 10px;
            min-width: 20px;
            text-align: center;
        }}

        .prompt-builder-toggle-icon {{
            font-size: 12px;
            color: var(--text-muted);
            transition: transform 0.3s ease;
        }}

        .prompt-builder:not(.collapsed) .prompt-builder-toggle-icon {{
            transform: rotate(180deg);
        }}

        .prompt-builder-content {{
            padding: 16px 24px 24px 24px;
            max-height: 420px;
            overflow-y: auto;
        }}

        .prompt-builder-inner {{
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}

        .prompt-tokens {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            min-height: 48px;
            padding: 12px;
            background: rgba(0, 0, 0, 0.2);
            border: 1px dashed var(--border);
            border-radius: 8px;
            align-items: center;
        }}

        .prompt-empty-text {{
            margin: 0;
            font-size: 13px;
            color: var(--text-muted);
            width: 100%;
            text-align: center;
        }}

        .prompt-token {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.3);
            border-radius: 6px;
            padding: 4px 8px;
            font-family: SFMono-Regular, Consolas, monospace;
            font-size: 12px;
            color: #a5b4fc;
        }}

        .btn-remove-token {{
            background: transparent;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            padding: 0;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            line-height: 1;
            font-weight: 700;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            transition: var(--transition);
            outline: none;
        }}

        .btn-remove-token:hover, .btn-remove-token:focus-visible {{
            color: #ef4444;
            background: rgba(239, 68, 68, 0.1);
        }}

        .prompt-builder-actions {{
            display: flex;
            justify-content: flex-end;
            gap: 12px;
        }}

        .btn-action-primary, .btn-action-secondary {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 8px 16px;
            font-size: 13px;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            transition: var(--transition);
            outline: none;
        }}

        .btn-action-primary {{
            background: var(--accent);
            border: 1px solid var(--accent);
            color: white;
        }}

        .btn-action-primary:hover, .btn-action-primary:focus-visible {{
            background: var(--accent-hover);
            border-color: var(--accent-hover);
        }}

        .btn-action-secondary {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            color: var(--text);
        }}

        .btn-action-secondary:hover, .btn-action-secondary:focus-visible {{
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
        }}

        .btn-action-civitai {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 8px 16px;
            font-size: 13px;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            transition: var(--transition);
            outline: none;
            text-decoration: none;
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: #94a3b8;
        }}

        .btn-action-civitai:hover, .btn-action-civitai:focus-visible {{
            background: rgba(30, 41, 59, 0.8);
            border-color: rgba(255, 255, 255, 0.2);
            color: white;
        }}

        /* Hover Preview Tooltip */
        .hover-preview-card {{
            position: absolute;
            background: rgba(17, 19, 28, 0.95);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 16px;
            width: 280px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.6);
            z-index: 9999;
            pointer-events: none;
            transition: opacity 0.15s ease, transform 0.15s ease;
            opacity: 0;
            transform: scale(0.96);
            display: none;
        }}
        
        .hover-preview-card.visible {{
            opacity: 1;
            transform: scale(1);
        }}
        
        .hover-preview-title {{
            margin: 0 0 10px 0;
            font-size: 14px;
            font-weight: 700;
            color: var(--text);
            line-height: 1.4;
            word-break: break-word;
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            padding-bottom: 8px;
        }}
        
        .hover-preview-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 6px;
            font-size: 12px;
            line-height: 1.4;
        }}
        
        .hover-preview-row:last-child {{
            margin-bottom: 0;
        }}
        
        .hover-preview-label {{
            color: var(--text-muted);
            font-weight: 500;
        }}
        
        .hover-preview-value {{
            color: var(--text);
            font-weight: 600;
            text-align: right;
            max-width: 160px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        /* Prompt Builder History & Presets list styling */
        .prompt-extras-list::-webkit-scrollbar {{
            width: 4px;
        }}
        .prompt-extras-list::-webkit-scrollbar-track {{
            background: transparent;
        }}
        .prompt-extras-list::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
        }}
        .prompt-extras-list::-webkit-scrollbar-thumb:hover {{
            background: rgba(255, 255, 255, 0.2);
        }}
        
        .preset-item, .history-item {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.04);
            border-radius: 4px;
            padding: 6px 8px;
            font-size: 11px;
            cursor: pointer;
            transition: var(--transition);
        }}
        
        .preset-item:hover, .history-item:hover {{
            background: rgba(99, 102, 241, 0.08) !important;
            border-color: rgba(99, 102, 241, 0.2) !important;
        }}

        @media (max-width: 1024px) {{
            .header-container {{
                flex-direction: column;
                align-items: stretch;
                gap: 16px;
                padding: 16px;
                width: 100%;
            }}

            .header-left, .header-right {{
                width: 100%;
                justify-content: center;
                flex-wrap: wrap;
                gap: 12px;
            }}

            .header-left .divider, .header-right .divider {{
                display: none;
            }}

            .search-wrapper {{
                max-width: 100%;
                width: 100%;
                order: 1;
            }}

            .select-wrapper {{
                flex: 1;
                width: auto;
                order: 2;
            }}

            #sort, #filter-base-model {{
                width: 100%;
            }}

            .count {{
                order: 3;
                text-align: center;
            }}
        }}

        @media (max-width: 768px) {{
            header {{
                padding: 16px;
            }}

            main {{
                padding: 16px;
            }}

            .header-left, .header-right {{
                flex-direction: column;
                align-items: stretch;
                gap: 12px;
            }}

            .weight-control {{
                justify-content: space-between;
                width: 100%;
            }}

            .weight-selector {{
                flex-grow: 1;
                justify-content: space-around;
            }}

            .prompt-builder {{
                width: 100%;
                border-radius: 0;
            }}
        }}



        .card-indicators {{
            position: absolute;
            bottom: 8px;
            left: 8px;
            display: flex;
            gap: 4px;
            pointer-events: none;
            z-index: 5;
        }}

        .indicator-badge {{
            background: rgba(15, 16, 22, 0.75);
            backdrop-filter: blur(4px);
            -webkit-backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text);
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            gap: 3px;
            font-weight: 600;
        }}

        /* Modal Overlay */
        .modal-overlay {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(9, 10, 15, 0.85);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s ease;
        }}

        .modal-overlay.active {{
            opacity: 1;
            pointer-events: auto;
        }}

        /* Modal Container */
        .modal-container {{
            background: var(--panel);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            width: 90%;
            max-width: 960px;
            height: 85vh;
            max-height: 85vh;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.75);
            overflow: hidden;
            position: relative;
            transform: scale(0.9);
            transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
            display: flex;
            flex-direction: column;
        }}

        .modal-overlay.active .modal-container {{
            transform: scale(1);
        }}

        /* Close Button */
        .modal-close {{
            position: absolute;
            top: 12px;
            right: 12px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.05);
            color: var(--text-muted);
            width: 30px;
            height: 30px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10;
            transition: var(--transition);
            outline: none;
        }}

        .modal-close svg {{
            width: 12px;
            height: 12px;
            display: block;
            stroke-width: 2.5;
        }}

        .modal-close:hover, .modal-close:focus {{
            background: rgba(239, 68, 68, 0.1);
            border-color: rgba(239, 68, 68, 0.3);
            color: #ef4444;
        }}

        /* Modal Body Layout */
        .modal-body {{
            display: flex;
            flex-direction: row;
            height: 100%;
            overflow: hidden;
        }}

        /* Left side (Static details, preview) */
        .modal-left {{
            width: 40%;
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            background: rgba(0, 0, 0, 0.15);
            overflow: hidden;
            height: 100%;
        }}

        .modal-left-scrollable {{
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            transition: opacity 0.18s ease;
        }}

        /* Version switch fade transition */
        .modal-version-transitioning {{
            opacity: 0 !important;
            pointer-events: none;
        }}

        .modal-right,
        .modal-middle {{
            transition: opacity 0.18s ease;
        }}

        .modal-left-actions {{
            flex-shrink: 0;
            padding: 20px 24px;
            border-top: 1px solid var(--border);
            background: rgba(17, 19, 28, 0.6);
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
        }}

        .modal-left-actions button, .modal-left-actions a {{
            flex: 1 1 auto;
            min-width: 100px;
            margin: 0;
            text-align: center;
            white-space: nowrap;
        }}

        .modal-preview-container {{
            width: 100%;
            border-radius: 8px;
            overflow: hidden;
            background: #171923;
            border: 1px solid rgba(255, 255, 255, 0.05);
            display: flex;
            align-items: center;
            justify-content: center;
            max-height: 480px;
            margin: auto 0;
        }}

        .modal-preview-container img {{
            max-width: 100%;
            max-height: 480px;
            height: auto;
            width: auto;
            display: block;
            object-fit: contain;
            margin: 0 auto;
        }}        /* Auto-Hiding Scrollbar Overlay styles for specific components */
        #modal-title,
        .modal-left-scrollable,
        .triggers-list-read,
        .triggers-list-edit,
        .modal-popup-collections-list {{
            scrollbar-width: thin;
            scrollbar-color: transparent transparent;
            transition: scrollbar-color 0.3s ease;
        }}

        #modal-title.is-scrolling,
        .modal-left-scrollable.is-scrolling,
        .triggers-list-read.is-scrolling,
        .triggers-list-edit.is-scrolling,
        .modal-popup-collections-list.is-scrolling {{
            scrollbar-color: rgba(255, 255, 255, 0.15) transparent;
        }}

        #modal-title::-webkit-scrollbar,
        .modal-left-scrollable::-webkit-scrollbar,
        .triggers-list-read::-webkit-scrollbar,
        .triggers-list-edit::-webkit-scrollbar,
        .modal-popup-collections-list::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}

        #modal-title::-webkit-scrollbar-track,
        .modal-left-scrollable::-webkit-scrollbar-track,
        .triggers-list-read::-webkit-scrollbar-track,
        .triggers-list-edit::-webkit-scrollbar-track,
        .modal-popup-collections-list::-webkit-scrollbar-track {{
            background: transparent;
        }}
        
        #modal-title::-webkit-scrollbar-thumb,
        .modal-left-scrollable::-webkit-scrollbar-thumb,
        .triggers-list-read::-webkit-scrollbar-thumb,
        .triggers-list-edit::-webkit-scrollbar-thumb,
        .modal-popup-collections-list::-webkit-scrollbar-thumb {{
            background: transparent;
            border-radius: 3px;
            transition: background 0.3s ease;
        }}
        
        #modal-title.is-scrolling::-webkit-scrollbar-thumb,
        .modal-left-scrollable.is-scrolling::-webkit-scrollbar-thumb,
        .triggers-list-read.is-scrolling::-webkit-scrollbar-thumb,
        .triggers-list-edit.is-scrolling::-webkit-scrollbar-thumb,
        .modal-popup-collections-list.is-scrolling::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.15);
        }}
        
        #modal-title.is-scrolling::-webkit-scrollbar-thumb:hover,
        .modal-left-scrollable.is-scrolling::-webkit-scrollbar-thumb:hover,
        .triggers-list-read.is-scrolling::-webkit-scrollbar-thumb:hover,
        .triggers-list-edit.is-scrolling::-webkit-scrollbar-thumb:hover,
        .modal-popup-collections-list.is-scrolling::-webkit-scrollbar-thumb:hover {{
            background: rgba(255, 255, 255, 0.25);
        }}

        /* Persistent Themed Scrollbars for Triggers & Resizable Description */
        .modal-description-text,
        .trigger-groups-container,
        .trigger-list {{
            scrollbar-width: thin;
            scrollbar-color: rgba(255, 255, 255, 0.15) transparent;
        }}

        .modal-description-text::-webkit-scrollbar,
        .trigger-groups-container::-webkit-scrollbar,
        .trigger-list::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}

        .modal-description-text::-webkit-scrollbar-track,
        .trigger-groups-container::-webkit-scrollbar-track,
        .trigger-list::-webkit-scrollbar-track {{
            background: transparent;
        }}

        .modal-description-text::-webkit-scrollbar-thumb,
        .trigger-groups-container::-webkit-scrollbar-thumb,
        .trigger-list::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.15);
            border-radius: 3px;
        }}

        .modal-description-text::-webkit-scrollbar-thumb:hover,
        .trigger-groups-container::-webkit-scrollbar-thumb:hover,
        .trigger-list::-webkit-scrollbar-thumb:hover {{
            background: rgba(255, 255, 255, 0.25);
        }}

        .modal-meta-rows {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 4px;
        }}

        .meta-row {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 10px 12px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-width: 0;
            transition: border-color 0.2s ease, background 0.2s ease;
        }}

        .meta-row:hover {{
            border-color: rgba(255, 255, 255, 0.1);
            background: rgba(255, 255, 255, 0.04);
        }}

        .meta-row strong {{
            color: var(--text-muted);
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }}

        .meta-value-container {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            width: 100%;
            min-width: 0;
        }}

        .meta-value-container span {{
            color: var(--text);
            font-size: 13px;
            word-break: break-all;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            flex: 1;
        }}

        .btn-copy-meta {{
            background: transparent;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            padding: 4px;
            border-radius: 4px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            transition: var(--transition);
            flex-shrink: 0;
            outline: none;
        }}

        .btn-copy-meta:hover {{
            background: rgba(255, 255, 255, 0.08);
            color: var(--text);
        }}

        .btn-open-folder-meta {{
            background: transparent;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            padding: 4px;
            border-radius: 4px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            transition: var(--transition);
            flex-shrink: 0;
            outline: none;
        }}

        .btn-open-folder-meta:hover {{
            background: rgba(255, 255, 255, 0.08);
            color: var(--accent);
        }}

        .btn-open-folder-meta svg {{
            width: 14px;
            height: 14px;
        }}

        .btn-copy-meta svg {{
            width: 14px;
            height: 14px;
        }}

        .btn-copy-meta.copied {{
            color: var(--success);
        }}

        /* Right side (Dynamic/Editable details) */
        .modal-right {{
            width: 60%;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            overflow-y: auto;
            height: 100%;
            box-sizing: border-box;
        }}

        .modal-right h2 {{
            margin: 0;
            font-size: 22px;
            font-weight: 700;
            color: var(--text);
            line-height: 1.3;
        }}

        .modal-section {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        .modal-section h3 {{
            margin: 0;
            font-size: 13px;
            font-weight: 700;
            color: #818cf8;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        /* Description Box */
        .modal-description-text {{
            font-size: 13px;
            color: var(--text-muted);
            line-height: 1.6;
            word-break: break-word;
            overflow-y: auto;
            padding-right: 6px;
        }}

        .modal-description-text a {{
            color: #818cf8;
            text-decoration: underline;
        }}
        .modal-description-text a:hover {{
            color: #a5b4fc;
        }}
        .modal-description-text ul, .modal-description-text ol {{
            padding-left: 20px;
            margin: 8px 0;
        }}
        .modal-description-text li {{
            margin: 4px 0;
        }}
        .modal-description-text p {{
            margin: 8px 0;
        }}
        .modal-description-text h1, .modal-description-text h2, .modal-description-text h3 {{
            color: var(--text);
            margin: 12px 0 6px 0;
            font-size: 14px;
            font-weight: 700;
        }}
        .modal-description-text code {{
            background: rgba(255, 255, 255, 0.08);
            padding: 2px 4px;
            border-radius: 4px;
            font-family: monospace;
        }}

        /* In 3-column layout the description section grows to fill leftover space */
        .modal-body.three-column .modal-middle #modal-section-description {{
            flex: 1;
            min-height: 0;
            display: flex;
            flex-direction: column;
            position: relative;
            overflow: hidden;
        }}

        .modal-body.three-column .modal-middle #modal-section-description .modal-description-text {{
            flex: 1;
            min-height: 0;
            height: auto;
        }}

        /* Fade-out gradient at the bottom of the description when overflowing */
        .modal-body.three-column .modal-middle #modal-section-description::after {{
            content: "";
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 48px;
            background: linear-gradient(to bottom, transparent, rgba(17, 19, 28, 0.95));
            pointer-events: none;
            border-radius: 0 0 8px 8px;
        }}

        /* Tags Container */
        .modal-tags-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }}
        .modal-tag {{
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.2);
            color: #a5b4fc;
            font-size: 11px;
            padding: 2px 8px;
            border-radius: 6px;
            font-weight: 600;
        }}

        .modal-input-field {{
            width: 100%;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text);
            padding: 10px 12px;
            font-size: 13px;
            font-family: inherit;
            outline: none;
            transition: var(--transition);
            box-sizing: border-box;
        }}

        .modal-input-field:focus {{
            border-color: var(--accent);
            background: rgba(99, 102, 241, 0.05);
        }}

        /* Notes Textarea */
        .modal-section textarea {{
            width: 100%;
            min-height: 80px;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text);
            padding: 10px 12px;
            font-size: 14px;
            font-family: inherit;
            resize: vertical;
            outline: none;
            transition: var(--transition);
        }}

        .modal-section textarea:focus {{
            border-color: var(--accent);
            background: rgba(99, 102, 241, 0.05);
        }}

        /* Star Rating */
        .rating-stars {{
            display: flex;
            gap: 8px;
            font-size: 28px;
            cursor: pointer;
            user-select: none;
            color: var(--text-muted);
            width: fit-content;
        }}

        .rating-stars span {{
            transition: transform 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275), color 0.2s, text-shadow 0.2s;
        }}

        .rating-stars span:hover {{
            transform: scale(1.2);
        }}

        .rating-stars span.active {{
            color: #fbbf24;
            text-shadow: 0 0 8px rgba(251, 191, 36, 0.6), 0 0 20px rgba(251, 191, 36, 0.3);
        }}

        .rating-stars span.hover-active {{
            color: #fbbf24;
            transform: scale(1.2);
            text-shadow: 0 0 8px rgba(251, 191, 36, 0.4);
        }}

        /* Collection chips styling */
        .collection-chip-modal {{
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.3);
            color: #a5b4fc;
            font-size: 12px;
            padding: 0;
            border-radius: 6px;
            display: inline-flex;
            align-items: stretch;
            gap: 0;
            font-weight: 600;
            cursor: default;
            overflow: hidden;
            user-select: none;
        }}

        .collection-chip-modal-text {{
            padding: 4px 10px;
            display: inline-flex;
            align-items: center;
        }}

        .collection-chip-modal .btn-remove-chip {{
            background: transparent;
            border: none;
            color: rgba(165, 180, 252, 0.6);
            cursor: pointer;
            font-size: 11px;
            padding: 0 10px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-left: 1px solid rgba(99, 102, 241, 0.3);
            transition: var(--transition);
        }}

        .collection-chip-modal .btn-remove-chip:hover {{
            background: rgba(239, 68, 68, 0.2);
            color: #f87171 !important;
            border-left-color: rgba(239, 68, 68, 0.3);
        }}

        /* Input forms */
        .add-trigger-form {{
            display: flex;
            gap: 8px;
            margin-top: 4px;
        }}

        .add-trigger-form input {{
            flex: 1;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text);
            padding: 6px 10px;
            font-size: 13px;
            outline: none;
        }}

        .add-trigger-form input:focus {{
            border-color: var(--accent);
        }}

        .btn-mini {{
            background: var(--accent);
            border: none;
            color: white;
            padding: 0 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
        }}

        .btn-mini:hover {{
            background: var(--accent-hover);
        }}

        /* Trigger word list layouts */
        .triggers-list-read {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            max-height: 120px;
            overflow-y: auto;
            background: rgba(0, 0, 0, 0.1);
            padding: 8px;
            border-radius: 6px;
            border: 1px solid var(--border);
            width: 100%;
        }}

        .trigger-tag-read {{
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 11px;
            padding: 2px 6px;
            border-radius: 6px;
            font-family: monospace;
        }}

        .triggers-list-edit {{
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            max-height: 120px;
            overflow-y: auto;
            background: rgba(0, 0, 0, 0.1);
            padding: 8px;
            border-radius: 6px;
            border: 1px solid var(--border);
            width: 100%;
        }}

        .trigger-tag-edit {{
            background: rgba(16, 185, 129, 0.08);
            border: 1px solid rgba(16, 185, 129, 0.2);
            color: #34d399;
            font-size: 11px;
            padding: 0;
            border-radius: 6px;
            font-family: monospace;
            display: inline-flex;
            align-items: stretch;
            gap: 0;
            cursor: copy;
            transition: var(--transition);
            user-select: none;
            overflow: hidden;
        }}

        .trigger-tag-edit:hover {{
            background: rgba(16, 185, 129, 0.12);
            border-color: rgba(16, 185, 129, 0.4);
            transform: translateY(-1px);
        }}

        .trigger-tag-edit-text {{
            padding: 4px 10px;
            display: inline-flex;
            align-items: center;
        }}

        .trigger-tag-edit-remove {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0 10px;
            cursor: pointer;
            font-weight: bold;
            font-size: 14px;
            color: rgba(52, 211, 153, 0.6);
            border-left: 1px solid rgba(16, 185, 129, 0.25);
            transition: var(--transition);
        }}

        .trigger-tag-edit-remove:hover {{
            background: rgba(239, 68, 68, 0.2);
            color: #f87171 !important;
            border-left-color: rgba(239, 68, 68, 0.3);
        }}

        .grid-2-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }}

        .modal-save-bar {{
            display: flex;
            gap: 12px;
            border-top: 1px solid var(--border);
            padding-top: 16px;
            margin-top: auto;
        }}

        .modal-save-bar .btn-secondary {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            color: var(--text);
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            flex: 1;
            transition: var(--transition);
        }}

        .modal-save-bar .btn-secondary:hover {{
            background: rgba(255, 255, 255, 0.08);
        }}

        /* 3-Column Portrait Layout Overrides */
        .modal-middle {{
            width: 32%;
            min-width: 280px;
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            gap: 20px;
            overflow-y: auto;
            padding: 24px;
            background: rgba(0, 0, 0, 0.05);
            box-sizing: border-box;
            height: 100%;
        }}

        .modal-body.three-column .modal-left {{
            width: 30%;
            min-width: 280px;
            max-width: 450px;
        }}

        .modal-body.three-column .modal-right {{
            width: 38%;
            min-width: 320px;
        }}

        .modal-section-group {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}

        /* Action button overrides inside modal */
        .btn-action-primary {{
            background: var(--accent);
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
        }}

        .btn-action-primary:hover {{
            background: var(--accent-hover);
        }}

        .btn-action-secondary {{
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text);
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
        }}

        .btn-action-secondary:hover {{
            border-color: var(--accent);
            background: var(--accent-alpha);
        }}

        /* Empty states & Placeholders */
        .empty-placeholder {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 16px;
            border: 1px dashed var(--border);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.01);
            color: var(--text-muted);
            font-size: 13px;
            text-align: center;
            box-sizing: border-box;
        }}

        .popup-section-header {{
            padding: 6px 12px 2px 12px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--accent);
            opacity: 0.85;
        }}

        .popup-divider {{
            height: 1px;
            background: var(--border);
            margin: 4px 0;
        }}

        .popup-footer {{
            padding: 8px 12px;
            border-top: 1px solid var(--border);
            display: flex;
            justify-content: flex-end;
            background: rgba(0, 0, 0, 0.25);
        }}

        .popup-item-check {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 16px;
            height: 16px;
            margin-right: 8px;
            color: var(--accent);
            font-weight: bold;
            font-size: 14px;
        }}

        .btn-delete-collection-popup {{
            background: transparent;
            border: none;
            color: var(--text-muted);
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            padding: 2px 6px;
            border-radius: 4px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            transition: var(--transition);
        }}

        .btn-delete-collection-popup:hover {{
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
        }}

        /* 3-Column Portrait Modal Layout Overrides */
        .modal-body.three-column .modal-left-actions {{
            flex-direction: column;
            align-items: stretch;
            gap: 10px;
            padding: 16px 20px;
        }}

        .modal-body.three-column .modal-left-actions button,
        .modal-body.three-column .modal-left-actions a {{
            width: 100%;
            flex: none;
            box-sizing: border-box;
        }}

        .modal-body.three-column .modal-section h3 {{
            text-align: center;
        }}

        .modal-body.three-column .rating-stars {{
            margin: 0 auto;
        }}

        .modal-body.three-column #modal-collections-chips-list {{
            justify-content: center;
        }}

        .modal-body.three-column .grid-2-col {{
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}

        .modal-body.three-column .modal-left-scrollable {{
            padding: 16px;
            gap: 12px;
        }}

        .modal-body.three-column .collections-popup {{
            max-width: none;
        }}

        /* In three-column mode only, description section should flex-grow (override the wildcard flex-shrink:0) */
        .modal-body.three-column .modal-middle #modal-section-description {{
            flex-shrink: 1;
        }}

        .modal-left-scrollable > *,
        .modal-middle > *,
        .modal-right > * {{
            flex-shrink: 0;
        }}

        @media (max-width: 768px) {{
            .modal-body {{
                flex-direction: column;
                overflow-y: auto;
            }}
            .modal-left {{
                width: 100% !important;
                border-right: none;
                border-bottom: 1px solid var(--border);
                height: auto !important;
            }}
            .modal-middle,
            .modal-right {{
                width: 100% !important;
                height: auto !important;
                overflow: visible !important;
                padding: 16px !important;
            }}
        }}

        @media (max-width: 600px) {{
            .grid-2-col {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <div class="header-container">
            <div class="header-left">
                <h1>LoRA Library</h1>
                <a href="/manage" class="btn-manage-link">
                    <span>🎛️</span>
                    <span>Manager</span>
                </a>
                <div class="divider"></div>
                <div class="weight-control">
                    <span class="weight-label">Weight</span>
                    <div class="weight-selector">
                        <button type="button" class="btn-weight" data-weight="0.6">0.6</button>
                        <button type="button" class="btn-weight" data-weight="0.8">0.8</button>
                        <button type="button" class="btn-weight" data-weight="1.0">1.0</button>
                        <button type="button" class="btn-weight" data-weight="1.2">1.2</button>
                    </div>
                </div>
            </div>

            <div class="header-right">
                <div class="divider"></div>
                <div class="search-wrapper">
                    <div id="collection-chip" class="collection-chip" role="button" tabindex="0">
                        <span class="chip-icon">📚</span>
                        <span class="chip-text">Library</span>
                        <span class="chip-arrow">▼</span>
                    </div>
                    <div id="creator-chip" class="collection-chip active-filter" style="display: none; margin-right: 8px;">
                        <span class="chip-icon">👤</span>
                        <span class="chip-text" id="creator-chip-text"></span>
                        <span class="chip-clear" id="creator-chip-clear" title="Clear creator filter">✕</span>
                    </div>
                    <svg class="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="11" cy="11" r="8"/>
                        <line x1="21" y1="21" x2="16.65" y2="16.65"/>
                    </svg>
                    <input id="search" type="search" placeholder="Search models, triggers, notes..." autocomplete="off">
                    
                    <div id="collections-popup" class="collections-popup" hidden>
                        <div class="popup-search-wrapper">
                            <svg class="popup-search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <circle cx="11" cy="11" r="8"/>
                                <line x1="21" y1="21" x2="16.65" y2="16.65"/>
                            </svg>
                            <input id="popup-search-input" type="text" placeholder="Search collections..." autocomplete="off">
                        </div>
                        <div id="popup-collections-list" class="popup-collections-list"></div>
                        <div class="popup-footer" style="padding: 8px; border-top: 1px solid var(--border); display: flex; justify-content: center; background: rgba(0, 0, 0, 0.2);">
                            <button id="btn-manage-collections" style="background: var(--accent); border: none; color: white; padding: 6px 12px; border-radius: 4px; font-size: 11px; font-weight: 600; cursor: pointer; width: 100%; transition: opacity 0.2s;" onmouseover="this.style.opacity=0.8" onmouseout="this.style.opacity=1.0" type="button">⚙️ Manage Collections</button>
                        </div>
                    </div>
                </div>
                <div class="select-wrapper" style="width: 150px;">
                    <select id="filter-base-model" aria-label="Filter by base model">
                        <option value="">All Base Models</option>
                    </select>
                    <svg class="select-arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="6 9 12 15 18 9"/>
                    </svg>
                </div>
                <div class="select-wrapper">
                    <select id="sort" aria-label="Sort models">
                        <option value="name-asc">A-Z</option>
                        <option value="name-desc">Z-A</option>
                        <option value="favorites-first">Favs First</option>
                        <option value="mtime-desc">Recent</option>
                    </select>
                    <svg class="select-arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="6 9 12 15 18 9"/>
                    </svg>
                </div>
                <div class="divider"></div>
                <span id="count" class="count"></span>
            </div>
        </div>
    </header>
    <main>
        <section id="grid" class="grid">
            {cards}
        </section>
        <p id="empty" class="empty">No matching LoRAs found.</p>
    </main>

    <!-- Manage Collections Modal -->
    <div id="manage-collections-modal" class="modal-overlay" aria-hidden="true" role="dialog" aria-labelledby="manage-title" style="z-index: 10000;">
        <div class="modal-container" style="max-width: 500px; height: 600px; display: flex; flex-direction: column;">
            <button class="modal-close" id="manage-modal-close" aria-label="Close modal">
                <svg class="close-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
            </button>
            <div style="padding: 24px; display: flex; flex-direction: column; height: 100%; min-height: 0;">
                <h2 id="manage-title" style="margin-top: 0; margin-bottom: 16px; font-size: 20px; font-weight: 700; color: var(--text);">Manage Collections</h2>
                <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 16px; line-height: 1.5;">Rename or merge collections. Renaming a parent category will automatically rename all its subcategories on disk.</p>
                <div id="manage-collections-list" style="flex: 1; overflow-y: auto; background: rgba(0, 0, 0, 0.2); border: 1px solid var(--border); border-radius: 8px; padding: 12px; display: flex; flex-direction: column; gap: 8px; margin-bottom: 16px;">
                    <!-- dynamically populated -->
                </div>
                <div style="display: flex; justify-content: flex-end;">
                    <button id="manage-modal-btn-close" style="background: rgba(255, 255, 255, 0.05); border: 1px solid var(--border); color: var(--text); padding: 8px 16px; border-radius: 8px; cursor: pointer; font-size: 13px; font-weight: 600;" type="button">Close</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Details Modal -->
    <div id="details-modal" class="modal-overlay" aria-hidden="true" role="dialog" aria-labelledby="modal-title">
        <div class="modal-container">
            <button class="modal-close" aria-label="Close modal">
                <svg class="close-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18"></line>
                    <line x1="6" y1="6" x2="18" y2="18"></line>
                </svg>
            </button>
            <div class="modal-body">
                <!-- Left Side: Preview & Details -->
                <div class="modal-left">
                    <div class="modal-left-scrollable">
                        <h2 id="modal-title" style="margin-top: 0; margin-bottom: 8px; font-size: 22px; font-weight: 700; color: var(--text); white-space: normal; word-break: break-word; max-height: 115px; overflow-y: auto; scrollbar-width: thin;">Model Name</h2>
                        
                        <div id="modal-version-container" class="version-selector-container" style="margin-bottom: 16px; display: flex; flex-direction: column; gap: 6px; border: none; background: transparent; padding: 0;">
                            <span id="modal-version-count-label" class="version-label" style="font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">Version (0)</span>
                            <div class="modal-version-select-wrapper" style="display: flex; gap: 8px; align-items: center; width: 100%;">
                                <div style="position: relative; flex: 1;">
                                    <select id="modal-version-select" class="version-select" aria-label="Select model version" style="width: 100%; padding: 10px 32px 10px 12px; background-color: rgba(255, 255, 255, 0.03); border: 1px solid var(--border); border-radius: 8px; font-size: 14px; color: var(--text); appearance: none; -webkit-appearance: none; cursor: pointer; background-image: none;">
                                        <!-- dynamically populated -->
                                    </select>
                                    <svg class="select-arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="position: absolute; right: 12px; top: 50%; transform: translateY(-50%); width: 16px; height: 16px; color: var(--text-muted); pointer-events: none;">
                                        <polyline points="6 9 12 15 18 9"/>
                                    </svg>
                                </div>
                                <button id="modal-btn-delete-version" class="btn-delete-version" aria-label="Delete this version" type="button" tabindex="0" style="flex-shrink: 0; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); color: #f87171; padding: 10px; border-radius: 8px; cursor: pointer; display: flex; align-items: center; justify-content: center; height: 38px; width: 38px; box-sizing: border-box; transition: var(--transition);">
                                    <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="width: 16px; height: 16px;">
                                        <polyline points="3 6 5 6 21 6"/>
                                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                                        <line x1="10" y1="11" x2="10" y2="17"/>
                                        <line x1="14" y1="11" x2="14" y2="17"/>
                                    </svg>
                                </button>
                            </div>
                        </div>

                        <div class="modal-preview-container">
                            <img id="modal-image" src="" alt="LoRA Preview">
                        </div>

                        <div id="modal-meta-container" class="modal-section-group">
                            <div class="modal-meta-rows">
                                <div class="meta-row" id="modal-creator-row">
                                    <strong>Creator</strong>
                                    <div class="meta-value-container">
                                        <span id="modal-creator"></span>
                                    </div>
                                </div>
                                <div class="meta-row">
                                    <strong>Base Model</strong>
                                    <div class="meta-value-container">
                                        <span id="modal-base-model"></span>
                                    </div>
                                </div>
                                <div class="meta-row">
                                    <strong>Filename</strong>
                                    <div class="meta-value-container">
                                        <span id="modal-filename"></span>
                                        <button class="btn-copy-meta" data-target="modal-filename" title="Copy Filename" type="button">
                                            <svg class="icon-copy" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                                                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                                            </svg>
                                        </button>
                                    </div>
                                </div>
                                <div class="meta-row">
                                    <strong>Path</strong>
                                    <div class="meta-value-container">
                                        <span id="modal-path"></span>
                                        <button id="modal-btn-open-folder" class="btn-open-folder-meta" title="Open Folder on Disk" type="button">
                                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                                <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
                                            </svg>
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="modal-left-actions">
                        <button id="modal-btn-copy-tag" class="btn-action-primary">Copy Tag</button>
                        <button id="modal-btn-add-prompt" class="btn-action-secondary">Add to Prompt</button>
                        <a id="modal-btn-civitai" class="btn-action-civitai" target="_blank" rel="noopener noreferrer">View on CivitAI</a>
                    </div>
                </div>

                <!-- Middle Column: Metadata (dynamically populated in 3-column mode) -->
                <div class="modal-middle" style="display: none;">
                </div>

                <!-- Right Side: Edit Panel -->
                <div class="modal-right">
                    <!-- Description Section -->
                    <div class="modal-section" id="modal-section-description" style="margin-top: 0;">
                        <h3>Description</h3>
                        <div id="modal-description-placeholder" class="empty-placeholder" style="display: none; padding: 24px 0; color: var(--text-muted); font-size: 13px;">— No description available —</div>
                        <div id="modal-description" class="modal-description-text" style="display: none;"></div>
                    </div>

                    <!-- Tags Section (dynamic display) -->
                    <div class="modal-section" id="modal-section-tags">
                        <h3>Tags</h3>
                        <div id="modal-tags" class="modal-tags-container"></div>
                    </div>
                    
                    <!-- Grid 2-col for Rating/Collections and Notes -->
                    <div class="grid-2-col">
                        <div id="modal-organization-container" class="modal-section-group">
                            <!-- Rating System -->
                            <div class="modal-section">
                                <h3>Personal Rating</h3>
                                <div class="rating-stars" id="modal-rating">
                                    <span data-val="1">&#9734;</span>
                                    <span data-val="2">&#9734;</span>
                                    <span data-val="3">&#9734;</span>
                                    <span data-val="4">&#9734;</span>
                                    <span data-val="5">&#9734;</span>
                                </div>
                            </div>
                            
                            <!-- Collections -->
                            <div class="modal-section">
                                <h3>Collection (Nested Tag)</h3>
                                <div class="collections-container" style="display: flex; flex-direction: column; gap: 10px; width: 100%;">
                                    <!-- Category (Parent) Input -->
                                    <div style="display: flex; flex-direction: column; gap: 4px; width: 100%; position: relative;">
                                        <label style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-muted); font-weight: 600;">Category (e.g. Anime, Concept)</label>
                                        <div class="custom-dropdown-wrapper" style="position: relative; width: 100%;">
                                            <input id="modal-input-parent-collection" class="modal-input-field" type="text" placeholder="Select or type category..." autocomplete="off">
                                            <span class="custom-dropdown-arrow" style="position: absolute; right: 12px; top: 50%; transform: translateY(-50%); cursor: pointer; color: var(--text-muted); pointer-events: none;">▼</span>
                                        </div>
                                        <div id="modal-parent-collections-popup" class="collections-popup" hidden style="width: 100%; box-sizing: border-box; max-height: 320px; display: flex; flex-direction: column;">
                                            <div class="popup-search-wrapper" style="padding: 8px 12px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 8px;">
                                                <svg class="popup-search-icon" style="width: 14px; height: 14px; color: var(--text-muted); flex-shrink: 0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                                    <circle cx="11" cy="11" r="8"/>
                                                    <line x1="21" y1="21" x2="16.65" y2="16.65"/>
                                                </svg>
                                                <input id="modal-parent-collections-search" type="text" placeholder="Search categories..." autocomplete="off" style="background: transparent; border: none; outline: none; color: var(--text); font-size: 13px; flex: 1;">
                                            </div>
                                            <div id="modal-parent-collections-list" class="popup-collections-list" style="flex: 1; max-height: 200px; overflow-y: auto;"></div>
                                            <div class="popup-footer">
                                                <button class="btn-action-primary" style="padding: 4px 12px; font-size: 12px;" type="button">Done</button>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <!-- Sub-Category (Child) Input -->
                                    <div id="modal-sub-collection-wrapper" style="display: none; flex-direction: column; gap: 4px; width: 100%; position: relative;">
                                        <label style="font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-muted); font-weight: 600;">Sub-Category (e.g. Fate, Gridman)</label>
                                        <div class="custom-dropdown-wrapper" style="position: relative; width: 100%;">
                                            <input id="modal-input-sub-collection" class="modal-input-field" type="text" placeholder="Select or type subcategory..." autocomplete="off">
                                            <span class="custom-dropdown-arrow" style="position: absolute; right: 12px; top: 50%; transform: translateY(-50%); cursor: pointer; color: var(--text-muted); pointer-events: none;">▼</span>
                                        </div>
                                        <div id="modal-sub-collections-popup" class="collections-popup" hidden style="width: 100%; box-sizing: border-box; max-height: 320px; display: flex; flex-direction: column;">
                                            <div class="popup-search-wrapper" style="padding: 8px 12px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 8px;">
                                                <svg class="popup-search-icon" style="width: 14px; height: 14px; color: var(--text-muted); flex-shrink: 0;" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                                    <circle cx="11" cy="11" r="8"/>
                                                    <line x1="21" y1="21" x2="16.65" y2="16.65"/>
                                                </svg>
                                                <input id="modal-sub-collections-search" type="text" placeholder="Search subcategories..." autocomplete="off" style="background: transparent; border: none; outline: none; color: var(--text); font-size: 13px; flex: 1;">
                                            </div>
                                            <div id="modal-sub-collections-list" class="popup-collections-list" style="flex: 1; max-height: 200px; overflow-y: auto;"></div>
                                            <div class="popup-footer">
                                                <button class="btn-action-primary" style="padding: 4px 12px; font-size: 12px;" type="button">Done</button>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Notes Column -->
                        <div class="modal-section" style="display: flex; flex-direction: column;">
                            <h3>Notes</h3>
                            <div id="modal-notes-placeholder" class="empty-placeholder" style="display: none; height: 100%;">
                                — No notes yet —
                                <button id="btn-add-note-placeholder" class="btn-mini" type="button" style="margin-top: 8px; width: fit-content;">+ Add Note</button>
                            </div>
                            <textarea id="modal-notes" placeholder="Looks best around 0.8 weight. Add notes..." style="height: 90px; min-height: 60px; max-height: 150px; resize: vertical;"></textarea>
                        </div>
                    </div>
                    
                    <!-- Triggers stacked vertically (No two column layout) -->
                    <div class="modal-section" style="display: flex; flex-direction: column; gap: 16px;">
                        <div>
                            <h3 id="modal-official-triggers-title">Official Triggers</h3>
                            <div id="modal-trigger-search-container" style="display: none; margin-top: 8px; margin-bottom: 4px;">
                                <input type="text" id="modal-trigger-search" placeholder="🔍 Search triggers..." style="width: 100%; padding: 8px 12px; background: rgba(255, 255, 255, 0.03); border: 1px solid var(--border); border-radius: 8px; font-size: 13px; color: var(--text); outline: none;">
                            </div>
                            <div id="modal-official-triggers" class="trigger-groups-container" style="max-height: 280px; border: none; background: transparent; padding: 0; margin-top: 8px;"></div>
                        </div>
                        <div>
                            <div class="trigger-group" id="personal-trigger-group">
                                <div class="trigger-group-header" role="button" aria-expanded="false" tabindex="0" aria-label="Toggle Personal Triggers">
                                    <span class="trigger-group-arrow">▶</span>
                                    <span class="trigger-group-title" id="personal-triggers-header-title">Personal Triggers (0)</span>
                                    <button class="btn-copy-group" aria-label="Copy Personal triggers" type="button" tabindex="0" id="btn-copy-personal-triggers">
                                        <svg class="btn-icon copy-icon-normal" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                                        </svg>
                                        <svg class="btn-icon copy-icon-success" style="display:none;" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                                            <polyline points="20 6 9 17 4 12"/>
                                        </svg>
                                        <span class="copy-group-feedback">Copied!</span>
                                    </button>
                                </div>
                                <div class="trigger-group-content-wrapper">
                                    <div class="trigger-group-content">
                                        <div id="modal-personal-triggers-placeholder" class="empty-placeholder" style="display: none; margin: 12px; padding: 12px; box-sizing: border-box;">
                                            — No personal triggers yet —
                                            <button id="btn-add-trigger-placeholder" class="btn-mini" type="button" style="margin-top: 8px; width: fit-content; display: block;">+ Add Trigger</button>
                                        </div>
                                        <div id="modal-personal-triggers" class="triggers-list-edit" style="border:none; background:transparent; padding:8px 12px 12px 12px; max-height:220px; overflow-y:auto; display:flex; flex-wrap:wrap; gap:6px;">
                                            <!-- pills here -->
                                        </div>
                                        <div class="add-trigger-form" id="personal-trigger-add-form" style="padding: 0 12px 12px 12px; display: none;">
                                            <input type="text" id="input-new-trigger" placeholder="New trigger...">
                                            <button id="btn-add-trigger" class="btn-mini" type="button">Add</button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Action Controls -->
                    <div class="modal-save-bar">
                        <button id="modal-btn-copy-all-triggers" class="btn-secondary">Copy All Triggers</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <div id="prompt-builder" class="prompt-builder collapsed">
        <div class="prompt-builder-header" role="button" aria-expanded="false" tabindex="0" aria-label="Toggle Prompt Builder">
            <div class="prompt-builder-title-container">
                <svg class="builder-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                    <polyline points="14 2 14 8 20 8"/>
                    <line x1="16" y1="13" x2="8" y2="13"/>
                    <line x1="16" y1="17" x2="8" y2="17"/>
                    <polyline points="10 9 9 9 8 9"/>
                </svg>
                <span class="prompt-builder-title">Prompt Builder</span>
                <span id="prompt-badge" class="prompt-badge">0</span>
            </div>
            <div class="prompt-builder-controls-header">
                <span class="prompt-builder-toggle-icon">▲</span>
            </div>
        </div>
        <div class="prompt-builder-content">
            <div class="prompt-builder-inner">
                <div id="prompt-tokens" class="prompt-tokens">
                    <p id="prompt-empty-text" class="prompt-empty-text">No LoRAs added yet. Click "+ Add to Prompt" on any card.</p>
                </div>
                <div class="prompt-builder-actions">
                    <button id="btn-copy-prompt" class="btn-action-primary" type="button" aria-label="Copy prompt to clipboard">
                        <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
                        </svg>
                        <span class="btn-text">Copy Prompt</span>
                    </button>
                    <button id="btn-clear-prompt" class="btn-action-secondary" type="button" aria-label="Clear all prompt tokens">
                        <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <polyline points="3 6 5 6 21 6"/>
                            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                        </svg>
                        <span class="btn-text">Clear All</span>
                    </button>
                </div>
                
                <!-- Divider -->
                <div style="border-top: 1px solid rgba(255, 255, 255, 0.08); margin: 12px 0;"></div>
                
                <!-- History & Presets Panel -->
                <div class="prompt-extras-container" style="display: flex; gap: 16px; min-height: 120px; max-height: 200px;">
                    <!-- Presets Column -->
                    <div class="prompt-extras-column" style="flex: 1; display: flex; flex-direction: column; min-width: 0;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                            <span style="font-size: 11px; font-weight: 700; color: #818cf8; text-transform: uppercase; letter-spacing: 0.05em;">Prompt Presets</span>
                            <button id="btn-save-preset" class="btn-mini" type="button" style="padding: 2px 8px; font-size: 10px; height: auto; line-height: normal;">+ Save Current</button>
                        </div>
                        <div id="prompt-presets-list" class="prompt-extras-list" style="flex: 1; overflow-y: auto; background: rgba(0, 0, 0, 0.2); border: 1px solid var(--border); border-radius: 6px; padding: 6px; display: flex; flex-direction: column; gap: 4px;">
                            <!-- populated dynamically -->
                        </div>
                    </div>
                    
                    <!-- History Column -->
                    <div class="prompt-extras-column" style="flex: 1; display: flex; flex-direction: column; min-width: 0;">
                        <div style="margin-bottom: 6px;">
                            <span style="font-size: 11px; font-weight: 700; color: #818cf8; text-transform: uppercase; letter-spacing: 0.05em;">Prompt History</span>
                        </div>
                        <div id="prompt-history-list" class="prompt-extras-list" style="flex: 1; overflow-y: auto; background: rgba(0, 0, 0, 0.2); border: 1px solid var(--border); border-radius: 6px; padding: 6px; display: flex; flex-direction: column; gap: 4px;">
                            <!-- populated dynamically -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const FAVORITES_KEY = "lora_gallery_favorites";
        const WEIGHT_KEY = "lora_gallery_selected_weight";

        let currentWeight = localStorage.getItem(WEIGHT_KEY) || "1.0";
        let activeCollection = "Library"; // Default state
        let activeCreator = "";

        const SYSTEM_COLLECTIONS = [
            {{
                name: "Library",
                icon: "📚",
                matches: (card, collections, isFav) => true
            }},
            {{
                name: "Favorites",
                icon: "⭐",
                matches: (card, collections, isFav) => isFav
            }},
            {{
                name: "Uncategorized",
                icon: "🗂️",
                matches: (card, collections, isFav) => collections.length === 0
            }}
        ];

        const searchInput = document.getElementById("search");
        const cards = Array.from(document.querySelectorAll(".card"));
        const count = document.getElementById("count");
        const empty = document.getElementById("empty");
        const sortSelect = document.getElementById("sort");
        const grid = document.getElementById("grid");

        function getFavorites() {{
            try {{
                return JSON.parse(localStorage.getItem(FAVORITES_KEY)) || [];
            }} catch(e) {{
                return [];
            }}
        }}

        function saveFavorites(favs) {{
            localStorage.setItem(FAVORITES_KEY, JSON.stringify(favs));
        }}

        function setWeight(weight) {{
            currentWeight = weight;
            localStorage.setItem(WEIGHT_KEY, weight);
            document.querySelectorAll(".btn-weight").forEach(btn => {{
                if (btn.dataset.weight === weight) {{
                    btn.classList.add("active");
                }} else {{
                    btn.classList.remove("active");
                }}
            }});
        }}

        function normalizeBaseModel(baseModelStr) {{
            if (!baseModelStr) return "Other";
            const lower = baseModelStr.toLowerCase();
            if (lower.includes("illustrious")) return "Illustrious";
            if (lower.includes("pony")) return "Pony";
            if (lower.includes("sdxl") || lower.includes("sd xl")) return "SDXL";
            if (lower.includes("noob")) return "NoobAI";
            if (lower.includes("flux")) return "Flux";
            if (lower.includes("sd 1.5") || lower.includes("sd1.5") || lower.includes("sd15") || lower.includes("sd 1.x")) return "SD1.5";
            
            const trimmed = baseModelStr.trim();
            if (!trimmed) return "Other";
            if (trimmed === trimmed.toLowerCase()) {{
                return trimmed.charAt(0).toUpperCase() + trimmed.slice(1);
            }}
            return trimmed;
        }}

        function updateFilter() {{
            const query = searchInput.value.trim().toLowerCase();
            
            const filterBaseModelSelect = document.getElementById("filter-base-model");
            const selectedBaseModel = filterBaseModelSelect ? filterBaseModelSelect.value : "";
            const favsList = getFavorites();
            const sysColl = SYSTEM_COLLECTIONS.find(sc => sc.name === activeCollection);
            
            let visible = 0;

            for (const card of cards) {{
                const matchesSearch = card.dataset.search.includes(query);
                
                const isFav = card.dataset.favorite === "true" || favsList.includes(card.dataset.hash || card.dataset.path);
                let cardColls = [];
                try {{
                    cardColls = JSON.parse(card.dataset.collections || "[]");
                }} catch (e) {{
                    cardColls = [];
                }}
                
                let matchesCollection = true;
                if (sysColl) {{
                    matchesCollection = sysColl.matches(card, cardColls, isFav);
                }} else {{
                    matchesCollection = cardColls.some(c => {{
                        const cParts = c.toLowerCase().split(/\s*[\/\\\\]\s*/);
                        const filterParts = activeCollection.toLowerCase().split(/\s*[\/\\\\]\s*/);
                        if (filterParts.length > cParts.length) return false;
                        for (let i = 0; i < filterParts.length; i++) {{
                            if (cParts[i] !== filterParts[i]) return false;
                        }}
                        return true;
                    }});
                }}
                
                let matchesBaseModel = true;
                if (selectedBaseModel) {{
                    const cardBaseModel = normalizeBaseModel(card.dataset.baseModel);
                    matchesBaseModel = (cardBaseModel === selectedBaseModel);
                }}
                
                let matchesCreator = true;
                if (activeCreator) {{
                    matchesCreator = (card.dataset.creator === activeCreator);
                }}
                
                const matches = matchesSearch && matchesCollection && matchesBaseModel && matchesCreator;
                card.hidden = !matches;

                if (matches) {{
                    visible += 1;
                }}
            }}

            count.textContent = `${{visible}} / ${{cards.length}}`;
            empty.style.display = visible === 0 ? "block" : "none";
        }}

        function performSort() {{
            const option = sortSelect.value;
            
            cards.sort((a, b) => {{
                if (option === "name-asc") {{
                    return a.dataset.name.localeCompare(b.dataset.name);
                }} else if (option === "name-desc") {{
                    return b.dataset.name.localeCompare(a.dataset.name);
                }} else if (option === "mtime-desc") {{
                    return parseInt(b.dataset.mtime) - parseInt(a.dataset.mtime);
                }} else if (option === "favorites-first") {{
                    const aFav = a.dataset.favorite === "true" ? 1 : 0;
                    const bFav = b.dataset.favorite === "true" ? 1 : 0;
                    if (aFav !== bFav) {{
                        return bFav - aFav;
                    }}
                    return a.dataset.name.localeCompare(b.dataset.name);
                }}
                return 0;
            }});

            cards.forEach(card => grid.appendChild(card));
        }}

        // Copy Tag Handler using Event Delegation
        grid.addEventListener("click", (e) => {{
            const btn = e.target.closest(".btn-copy");
            if (!btn) return;
            
            const card = btn.closest(".card");
            if (!card) return;
            
            const stem = card.dataset.stem;
            const tag = `<lora:${{stem}}:${{currentWeight}}>`;
            
            navigator.clipboard.writeText(tag).then(() => {{
                const normalIcon = btn.querySelector(".copy-normal");
                const successIcon = btn.querySelector(".copy-success");
                const textSpan = btn.querySelector(".btn-text");
                
                btn.classList.add("copied");
                if (normalIcon) normalIcon.style.display = "none";
                if (successIcon) successIcon.style.display = "inline";
                if (textSpan) textSpan.textContent = "Copied!";
                
                setTimeout(() => {{
                    btn.classList.remove("copied");
                    if (normalIcon) normalIcon.style.display = "inline";
                    if (successIcon) successIcon.style.display = "none";
                    if (textSpan) textSpan.textContent = "Copy LoRA Tag";
                }}, 1500);
            }}).catch(err => {{
                console.error("Clipboard copy failed:", err);
            }});
        }});

        // Toast notifications utility
        function showToast(message) {{
            let toast = document.getElementById("gallery-toast");
            if (!toast) {{
                toast = document.createElement("div");
                toast.id = "gallery-toast";
                toast.style.cssText = `
                    position: fixed;
                    bottom: 24px;
                    right: 24px;
                    background: #10b981;
                    color: white;
                    padding: 12px 24px;
                    border-radius: 8px;
                    font-size: 14px;
                    font-weight: 600;
                    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
                    z-index: 9999;
                    transition: opacity 0.3s ease, transform 0.3s ease;
                    transform: translateY(100px);
                    opacity: 0;
                `;
                document.body.appendChild(toast);
            }}
            toast.textContent = message;
            toast.style.opacity = "1";
            toast.style.transform = "translateY(0)";
            setTimeout(() => {{
                toast.style.opacity = "0";
                toast.style.transform = "translateY(100px)";
            }}, 3000);
        }}

        // Version Selector Change Handler (Event Delegation)
        grid.addEventListener("change", (e) => {{
            const select = e.target.closest(".version-select");
            if (!select) return;
            
            const card = select.closest(".card");
            if (!card) return;
            
            const versions = JSON.parse(card.dataset.versions || "[]");
            const selectedVer = versions.find(v => v.file_stem === select.value);
            if (!selectedVer) return;
            
            // Update Card dataset attributes
            card.dataset.filename = selectedVer.filename;
            card.dataset.stem = selectedVer.file_stem;
            card.dataset.mtime = selectedVer.mtime;
            card.dataset.hash = selectedVer.sha256;
            card.dataset.path = selectedVer.relative_path;
            card.dataset.triggers = selectedVer.trigger_words.join(", ");
            card.dataset.baseModel = selectedVer.base_model;
            card.dataset.description = selectedVer.description;
            card.dataset.officialTriggers = JSON.stringify(selectedVer.trigger_words);
            
            // Update Image Preview
            const previewContainer = card.querySelector(".preview");
            if (selectedVer.preview_image) {{
                let imgEl = previewContainer.querySelector("img");
                if (!imgEl) {{
                    const missingDiv = previewContainer.querySelector(".missing-preview");
                    if (missingDiv) missingDiv.remove();
                    imgEl = document.createElement("img");
                    imgEl.className = "card-preview-image";
                    imgEl.loading = "lazy";
                    previewContainer.insertBefore(imgEl, previewContainer.firstChild);
                }}
                imgEl.src = selectedVer.preview_image;
                imgEl.alt = card.dataset.name;
            }} else {{
                const imgEl = previewContainer.querySelector("img");
                if (imgEl) imgEl.remove();
                
                let missingDiv = previewContainer.querySelector(".missing-preview");
                if (!missingDiv) {{
                    missingDiv = document.createElement("div");
                    missingDiv.className = "missing-preview";
                    missingDiv.innerHTML = `
                        <svg class="placeholder-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round">
                            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                            <circle cx="8.5" cy="8.5" r="1.5"/>
                            <polyline points="21 15 16 10 5 21"/>
                        </svg>
                        <span>No Preview Available</span>
                    `;
                    previewContainer.insertBefore(missingDiv, previewContainer.firstChild);
                }}
            }}
            
            // Update Trigger Words display
            const triggerContainer = card.querySelector(".trigger-groups-container");
            if (triggerContainer) {{
                triggerContainer.innerHTML = selectedVer.trigger_groups_html || '<p class="triggers">None found</p>';
                const labelEl = card.querySelector(".card-trigger-label");
                if (labelEl) {{
                    const groupCount = triggerContainer.querySelectorAll(".trigger-group").length;
                    labelEl.textContent = `Trigger words (${{groupCount}})`;
                }}
            }}
            
            // Update Description display
            const descEl = card.querySelector(".description");
            if (descEl) {{
                descEl.textContent = selectedVer.description || "No description available.";
            }}
            // Re-evaluate overflow fade for this card's description
            const wrapper = card.querySelector(".description-wrapper");
            if (wrapper) {{
                const d = wrapper.querySelector(".description");
                wrapper.classList.toggle("overflowing", d && d.scrollHeight > wrapper.clientHeight + 2);
            }}
            
            // Dynamic base models dropdown and active filters updates on version changes
            updateBaseModelFilterOptions();
            updateFilter();
        }});

        // Delete Version Click Handler (Event Delegation)
        grid.addEventListener("click", (e) => {{
            const btn = e.target.closest(".btn-delete-version");
            if (!btn) return;
            
            const card = btn.closest(".card");
            if (!card) return;
            
            const select = card.querySelector(".version-select");
            if (!select) return;
            
            const versions = JSON.parse(card.dataset.versions || "[]");
            const selectedVer = versions.find(v => v.file_stem === select.value);
            if (!selectedVer) return;
            
            const confirmMsg = `Are you sure you want to delete the model version '${{selectedVer.version_name || selectedVer.file_stem}}'?\n\nThis will permanently delete the model file:\n${{selectedVer.filename}}\n\nas well as its associated previews and metadata.`;
            if (!confirm(confirmMsg)) return;
            
            fetch('/api/delete_model_version', {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json'
                }},
                body: JSON.stringify({{ relative_path: selectedVer.relative_path }})
            }})
            .then(res => {{
                if (!res.ok) throw new Error("Delete failed on server.");
                return res.json();
            }})
            .then(data => {{
                let remainingVersions = versions.filter(v => v.file_stem !== selectedVer.file_stem);
                if (remainingVersions.length === 0) {{
                    card.remove();
                    const idx = cards.indexOf(card);
                    if (idx > -1) cards.splice(idx, 1);
                    updateFilter();
                    showToast("Model deleted successfully.");
                }} else {{
                    card.dataset.versions = JSON.stringify(remainingVersions);
                    select.innerHTML = remainingVersions.map(v => `<option value="${{v.file_stem}}">${{v.version_name || v.file_stem}}</option>`).join("");
                    select.value = remainingVersions[0].file_stem;
                    select.dispatchEvent(new Event("change"));
                    showToast("Version deleted successfully.");
                }}
            }})
            .catch(err => {{
                console.error("Deletion failed:", err);
                alert("Failed to delete model version: " + err.message);
            }});
        }});

        // Favorite Toggle Handler (Click / Enter / Space on standard button)
        grid.addEventListener("click", (e) => {{
            const btn = e.target.closest(".btn-favorite");
            if (!btn) return;
            
            const card = btn.closest(".card");
            if (!card) return;
            
            const id = card.dataset.hash || card.dataset.path;
            let favs = getFavorites();
            
            if (favs.includes(id)) {{
                favs = favs.filter(x => x !== id);
                btn.classList.remove("active");
                card.dataset.favorite = "false";
            }} else {{
                favs.push(id);
                btn.classList.add("active");
                card.dataset.favorite = "true";
            }}
            
            saveFavorites(favs);
            
            if (activeCollection === "Favorites") {{
                updateFilter();
            }}
            if (sortSelect.value === "favorites-first") {{
                performSort();
            }}
        }});

        // Expand/Collapse Trigger Group Handler
        grid.addEventListener("click", (e) => {{
            const header = e.target.closest(".trigger-group-header");
            if (!header) return;
            if (e.target.closest(".btn-copy-group")) return;
            
            const group = header.closest(".trigger-group");
            if (!group) return;
            
            const isExpanded = group.classList.toggle("expanded");
            header.setAttribute("aria-expanded", isExpanded ? "true" : "false");
        }});

        grid.addEventListener("keydown", (e) => {{
            const header = e.target.closest(".trigger-group-header");
            if (!header) return;
            
            if (e.key === "Enter" || e.key === " ") {{
                e.preventDefault();
                const group = header.closest(".trigger-group");
                if (!group) return;
                
                const isExpanded = group.classList.toggle("expanded");
                header.setAttribute("aria-expanded", isExpanded ? "true" : "false");
            }}
        }});

        // Copy Trigger Group Words Handler
        grid.addEventListener("click", (e) => {{
            const btn = e.target.closest(".btn-copy-group");
            if (!btn) return;
            
            const group = btn.closest(".trigger-group");
            if (!group) return;
            
            const words = group.dataset.groupWords;
            if (!words) return;
            
            navigator.clipboard.writeText(words).then(() => {{
                const normalIcon = btn.querySelector(".copy-icon-normal");
                const successIcon = btn.querySelector(".copy-icon-success");
                const feedback = btn.querySelector(".copy-group-feedback");
                
                if (normalIcon) normalIcon.style.display = "none";
                if (successIcon) successIcon.style.display = "inline";
                if (feedback) feedback.classList.add("show");
                
                setTimeout(() => {{
                    if (normalIcon) normalIcon.style.display = "inline";
                    if (successIcon) successIcon.style.display = "none";
                    if (feedback) feedback.classList.remove("show");
                }}, 1500);
            }}).catch(err => {{
                console.error("Group copy failed:", err);
            }});
        }});

        // Prompt Builder State & Helpers
        const PROMPT_KEY = "lora_gallery_prompt_builder";
        const HISTORY_KEY = "lora_gallery_prompt_history";
        const PRESETS_KEY = "lora_gallery_prompt_presets";
        
        function getPromptHistory() {{
            try {{
                return JSON.parse(localStorage.getItem(HISTORY_KEY)) || [];
            }} catch(e) {{
                return [];
            }}
        }}
        
        function savePromptHistory(history) {{
            localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
        }}
        
        function getPromptPresets() {{
            try {{
                const presets = JSON.parse(localStorage.getItem(PRESETS_KEY)) || [];
                let changed = false;
                const now = Date.now();
                presets.forEach(p => {{
                    if (!p.created_at) {{
                        p.created_at = now;
                        changed = true;
                    }}
                    if (!p.last_used) {{
                        p.last_used = p.created_at;
                        changed = true;
                    }}
                }});
                if (changed) {{
                    localStorage.setItem(PRESETS_KEY, JSON.stringify(presets));
                }}
                return presets;
            }} catch(e) {{
                return [];
            }}
        }}
        
        function savePromptPresets(presets) {{
            localStorage.setItem(PRESETS_KEY, JSON.stringify(presets));
        }}

        function getReadablePromptText(tokens) {{
            return tokens.map(token => {{
                const match = token.match(/<lora:([^:]+):[^>]+>/);
                if (match && match[1]) {{
                    const stem = match[1];
                    return stem.split(/[-_]+/)
                               .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                               .join(" ");
                }}
                return token;
            }}).join(" + ");
        }}
        
        function renderPromptHistoryUI() {{
            const container = document.getElementById("prompt-history-list");
            if (!container) return;
            container.innerHTML = "";
            
            const history = getPromptHistory();
            if (history.length === 0) {{
                container.innerHTML = '<div style="font-size: 11px; color: var(--text-muted); text-align: center; padding: 12px 0;">No history yet.</div>';
                return;
            }}
            
            history.forEach((tokens, idx) => {{
                const div = document.createElement("div");
                div.className = "history-item";
                div.title = tokens.join(" ");
                div.textContent = getReadablePromptText(tokens);
                div.addEventListener("click", () => {{
                    savePromptTokens(tokens);
                    updatePromptBuilderUI();
                }});
                container.appendChild(div);
            }});
        }}
        
        function renderPromptPresetsUI() {{
            const container = document.getElementById("prompt-presets-list");
            if (!container) return;
            container.innerHTML = "";
            
            const presets = getPromptPresets();
            const presetsWithIndex = presets.map((p, idx) => ({{ ...p, originalIndex: idx }}));
            const sortedPresets = presetsWithIndex.sort((a, b) => (b.last_used || 0) - (a.last_used || 0));
            
            if (sortedPresets.length === 0) {{
                container.innerHTML = '<div style="font-size: 11px; color: var(--text-muted); text-align: center; padding: 12px 0;">No presets yet.</div>';
                return;
            }}
            
            sortedPresets.forEach((preset) => {{
                const originalIndex = preset.originalIndex;
                
                const div = document.createElement("div");
                div.className = "preset-item";
                div.style.cssText = "display: flex; align-items: center; justify-content: space-between; gap: 8px;";
                div.addEventListener("click", (e) => {{
                    if (e.target.closest(".btn-preset-rename") || e.target.closest(".btn-preset-delete")) return;
                    applyPreset(originalIndex);
                }});
                
                const nameSpan = document.createElement("span");
                nameSpan.className = "preset-name";
                nameSpan.style.cssText = "flex-grow: 1; font-weight: 600; color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap;";
                nameSpan.textContent = preset.name;
                nameSpan.title = preset.tokens.join(" ");
                
                const actionsDiv = document.createElement("div");
                actionsDiv.style.cssText = "display: flex; gap: 6px; flex-shrink: 0;";
                
                const renameBtn = document.createElement("button");
                renameBtn.type = "button";
                renameBtn.className = "btn-preset-rename";
                renameBtn.innerHTML = "&#9998;";
                renameBtn.title = "Rename Preset";
                renameBtn.style.cssText = "background: transparent; border: none; color: var(--text-muted); cursor: pointer; padding: 2px; font-size: 12px; line-height: 1;";
                renameBtn.addEventListener("click", (e) => {{
                    e.stopPropagation();
                    renamePreset(originalIndex);
                }});
                
                const deleteBtn = document.createElement("button");
                deleteBtn.type = "button";
                deleteBtn.className = "btn-preset-delete";
                deleteBtn.innerHTML = "&#128465;";
                deleteBtn.title = "Delete Preset";
                deleteBtn.style.cssText = "background: transparent; border: none; color: #f87171; cursor: pointer; padding: 2px; font-size: 12px; line-height: 1;";
                deleteBtn.addEventListener("click", (e) => {{
                    e.stopPropagation();
                    deletePreset(originalIndex);
                }});
                
                actionsDiv.appendChild(renameBtn);
                actionsDiv.appendChild(deleteBtn);
                
                div.appendChild(nameSpan);
                div.appendChild(actionsDiv);
                container.appendChild(div);
            }});
        }}
        
        function renamePreset(index) {{
            const presets = getPromptPresets();
            const preset = presets[index];
            if (!preset) return;
            
            const newName = prompt("Rename preset:", preset.name);
            if (!newName || newName.trim() === "") return;
            
            preset.name = newName.trim();
            savePromptPresets(presets);
            renderPromptPresetsUI();
        }}
        
        function deletePreset(index) {{
            const presets = getPromptPresets();
            presets.splice(index, 1);
            savePromptPresets(presets);
            renderPromptPresetsUI();
        }}
        
        function applyPreset(index) {{
            const presets = getPromptPresets();
            const preset = presets[index];
            if (!preset) return;
            
            preset.last_used = Date.now();
            savePromptPresets(presets);
            
            const currentTokens = getPromptTokens();
            const mergedTokens = [...currentTokens];
            
            preset.tokens.forEach(token => {{
                const matchNew = token.match(/<lora:([^:]+):[^>]+>/);
                if (matchNew && matchNew[1]) {{
                    const newStem = matchNew[1];
                    const exists = currentTokens.some(t => {{
                        const m = t.match(/<lora:([^:]+):[^>]+>/);
                        return m && m[1] === newStem;
                    }});
                    if (!exists) {{
                        mergedTokens.push(token);
                    }}
                }} else if (!mergedTokens.includes(token)) {{
                    mergedTokens.push(token);
                }}
            }});
            
            savePromptTokens(mergedTokens);
            updatePromptBuilderUI();
        }}

        function getPromptTokens() {{
            try {{
                return JSON.parse(localStorage.getItem(PROMPT_KEY)) || [];
            }} catch(e) {{
                return [];
            }}
        }}

        function savePromptTokens(tokens) {{
            localStorage.setItem(PROMPT_KEY, JSON.stringify(tokens));
        }}

        function escapeHTML(str) {{
            return str.replace(/&/g, "&amp;")
                      .replace(/</g, "&lt;")
                      .replace(/>/g, "&gt;")
                      .replace(/"/g, "&quot;")
                      .replace(/'/g, "&#039;");
        }}

        function updatePromptBuilderUI() {{
            const tokens = getPromptTokens();
            const badge = document.getElementById("prompt-badge");
            const container = document.getElementById("prompt-tokens");
            const emptyText = document.getElementById("prompt-empty-text");
            
            if (badge) badge.textContent = tokens.length;
            
            // Remove all existing token spans (keep empty text element)
            const tokenElements = container.querySelectorAll(".prompt-token");
            tokenElements.forEach(el => el.remove());
            
            if (tokens.length === 0) {{
                if (emptyText) emptyText.style.display = "block";
            }} else {{
                if (emptyText) emptyText.style.display = "none";
                
                tokens.forEach(token => {{
                    const tokenSpan = document.createElement("span");
                    tokenSpan.className = "prompt-token";
                    tokenSpan.innerHTML = `
                        <span>${{escapeHTML(token)}}</span>
                        <button class="btn-remove-token" type="button" aria-label="Remove ${{escapeHTML(token)}}" data-token="${{escapeHTML(token)}}">×</button>
                    `;
                    container.appendChild(tokenSpan);
                }});
            }}
            
            // Refresh history and presets views
            renderPromptHistoryUI();
            renderPromptPresetsUI();
        }}

        function addToken(stem) {{
            const newToken = `<lora:${{stem}}:${{currentWeight}}>`;
            let tokens = getPromptTokens();
            // Filter out any existing token for this specific model stem to prevent duplicate tags
            tokens = tokens.filter(t => !t.startsWith(`<lora:${{stem}}:`));
            tokens.push(newToken);
            savePromptTokens(tokens);
            updatePromptBuilderUI();
            
            // Auto expand prompt builder when adding a token
            const builder = document.getElementById("prompt-builder");
            const header = builder.querySelector(".prompt-builder-header");
            builder.classList.remove("collapsed");
            header.setAttribute("aria-expanded", "true");
        }}

        // Prompt Builder Expand/Collapse Handler
        const builder = document.getElementById("prompt-builder");
        const builderHeader = builder.querySelector(".prompt-builder-header");
        builderHeader.addEventListener("click", (e) => {{
            if (e.target.closest(".btn-action-primary") || e.target.closest(".btn-action-secondary")) return;
            const isCollapsed = builder.classList.toggle("collapsed");
            builderHeader.setAttribute("aria-expanded", !isCollapsed ? "true" : "false");
        }});

        builderHeader.addEventListener("keydown", (e) => {{
            if (e.key === "Enter" || e.key === " ") {{
                e.preventDefault();
                const isCollapsed = builder.classList.toggle("collapsed");
                builderHeader.setAttribute("aria-expanded", !isCollapsed ? "true" : "false");
            }}
        }});

        // Add to Prompt Click Handler
        grid.addEventListener("click", (e) => {{
            const btn = e.target.closest(".btn-add-prompt");
            if (!btn) return;
            
            const card = btn.closest(".card");
            if (!card) return;
            
            const stem = card.dataset.stem;
            addToken(stem);
            
            // Feedback UI
            const textSpan = btn.querySelector(".btn-text");
            const originalText = textSpan.textContent;
            textSpan.textContent = "Added!";
            btn.style.borderColor = "var(--success)";
            btn.style.color = "var(--success)";
            setTimeout(() => {{
                textSpan.textContent = originalText;
                btn.style.borderColor = "";
                btn.style.color = "";
            }}, 1500);
        }});

        // Remove Token Handler
        document.getElementById("prompt-tokens").addEventListener("click", (e) => {{
            const btn = e.target.closest(".btn-remove-token");
            if (!btn) return;
            
            const tokenToRemove = btn.dataset.token;
            let tokens = getPromptTokens();
            tokens = tokens.filter(t => t !== tokenToRemove);
            savePromptTokens(tokens);
            updatePromptBuilderUI();
        }});

        // Clear All Prompt Handler
        document.getElementById("btn-clear-prompt").addEventListener("click", () => {{
            savePromptTokens([]);
            updatePromptBuilderUI();
        }});

        // Copy Prompt Handler
        document.getElementById("btn-copy-prompt").addEventListener("click", () => {{
            const tokens = getPromptTokens();
            if (tokens.length === 0) return;
            
            const promptText = tokens.join(" ");
            navigator.clipboard.writeText(promptText).then(() => {{
                const btn = document.getElementById("btn-copy-prompt");
                const textSpan = btn.querySelector(".btn-text");
                const originalText = textSpan.textContent;
                textSpan.textContent = "Copied!";
                btn.style.background = "var(--success)";
                btn.style.borderColor = "var(--success)";
                
                setTimeout(() => {{
                    textSpan.textContent = originalText;
                    btn.style.background = "";
                    btn.style.borderColor = "";
                }}, 1500);
                
                // Add to history
                let history = getPromptHistory();
                const promptStr = JSON.stringify(tokens);
                history = history.filter(h => JSON.stringify(h) !== promptStr);
                history.unshift(tokens);
                if (history.length > 20) {{
                    history = history.slice(0, 20);
                }}
                savePromptHistory(history);
                renderPromptHistoryUI();
            }}).catch(err => {{
                console.error("Prompt copy failed:", err);
            }});
        }});
        
        // Save Preset Button Handler
        const savePresetBtn = document.getElementById("btn-save-preset");
        if (savePresetBtn) {{
            savePresetBtn.addEventListener("click", () => {{
                const tokens = getPromptTokens();
                if (tokens.length === 0) {{
                    alert("Prompt builder is empty. Add some LoRAs first!");
                    return;
                }}
                const name = prompt("Enter a name for this preset:");
                if (!name || name.trim() === "") return;
                
                const presets = getPromptPresets();
                const now = Date.now();
                presets.push({{
                    name: name.trim(),
                    tokens: tokens,
                    created_at: now,
                    last_used: now
                }});
                savePromptPresets(presets);
                renderPromptPresetsUI();
            }});
        }}

        // Double Click Preview Handler (Copy trigger words)
        grid.addEventListener("dblclick", (e) => {{
            const preview = e.target.closest(".preview");
            if (!preview) return;
            if (e.target.closest(".btn-favorite")) return;
            
            const card = preview.closest(".card");
            if (!card) return;
            
            const triggers = card.dataset.triggers;
            if (triggers && triggers !== "None found" && triggers.trim() !== "") {{
                navigator.clipboard.writeText(triggers).then(() => {{
                    const feedback = preview.querySelector(".preview-feedback");
                    if (feedback) {{
                        feedback.classList.add("show");
                        setTimeout(() => {{
                            feedback.classList.remove("show");
                        }}, 1500);
                    }}
                }}).catch(err => {{
                    console.error("Failed to copy triggers:", err);
                }});
            }}
        }});

        // Keyboard Shortcuts
        document.addEventListener("keydown", (event) => {{
            if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "f") {{
                event.preventDefault();
                searchInput.focus();
                searchInput.select();
            }}
            if (event.key === "Escape") {{
                const modal = document.getElementById("details-modal");
                if (modal && modal.classList.contains("active")) {{
                    closeDetailsModal();
                    event.stopPropagation();
                    return;
                }}
                searchInput.value = "";
                updateFilter();
                searchInput.blur();
            }}
        }});

        // Details Modal state & handlers
        let currentModalCard = null;
        let currentPersonalTriggers = [];
        let currentCollections = [];
        let currentRating = 0;
        let rawGlobalCollections = [];
        let globalCollections = [];
        const expandedParents = new Set();

        function setGlobalCollections(colls) {{
            rawGlobalCollections = colls || [];
            const set = new Set();
            rawGlobalCollections.forEach(c => {{
                if (c && typeof c === 'string') {{
                    const parts = c.split(/\s*[\/\\\\]\s*/);
                    let path = "";
                    for (let i = 0; i < parts.length; i++) {{
                        path = path ? path + " / " + parts[i] : parts[i];
                        set.add(path);
                    }}
                }}
            }});
            globalCollections = Array.from(set);
        }}
        
        const RECENT_COLLECTIONS_KEY = "lora_gallery_recent_collections";
        let recentCollections = [];
        try {{
            recentCollections = JSON.parse(localStorage.getItem(RECENT_COLLECTIONS_KEY)) || [];
        }} catch(e) {{
            recentCollections = [];
        }}
        
        function markCollectionAsRecent(name) {{
            recentCollections = [name, ...recentCollections.filter(c => c !== name)].slice(0, 5);
            localStorage.setItem(RECENT_COLLECTIONS_KEY, JSON.stringify(recentCollections));
        }}

        // Load global collections from API once
        function loadCollections() {{
            return fetch("/api/collections")
                .then(res => res.json())
                .then(data => {{
                    setGlobalCollections(data.collections);
                    updateCollectionFilterOptions();
                }})
                .catch(err => console.error("Error loading collections:", err));
        }}

        // Open details modal
        function openDetailsModal(card, userData) {{
            currentModalCard = card;
            
            // Extract static details
            const name = card.dataset.name;
            const tags = JSON.parse(card.dataset.tags || "[]");

            // Extract user data
            currentPersonalTriggers = userData.personal_trigger_words || [];
            currentCollections = userData.collections || [];
            currentRating = userData.rating || 0;
            const notes = userData.notes || "";

            // Populate static name
            document.getElementById("modal-title").textContent = name;

            // Hide/Show Tags Section dynamically
            const tagsSection = document.getElementById("modal-section-tags");
            if (tagsSection) {{
                if (tags.length === 0) {{
                    tagsSection.style.display = "none";
                }} else {{
                    tagsSection.style.display = "block";
                    const tagsContainer = document.getElementById("modal-tags");
                    tagsContainer.innerHTML = "";
                    tags.forEach(tag => {{
                        const span = document.createElement("span");
                        span.className = "modal-tag";
                        span.textContent = tag;
                        tagsContainer.appendChild(span);
                    }});
                }}
            }}

            // Populate versions select
            const versions = JSON.parse(card.dataset.versions || "[]");
            const modalVersionSelect = document.getElementById("modal-version-select");
            const modalVersionCountLabel = document.getElementById("modal-version-count-label");
            
            if (modalVersionSelect) {{
                modalVersionSelect.innerHTML = "";
                versions.forEach(v => {{
                    const opt = document.createElement("option");
                    opt.value = v.file_stem;
                    opt.textContent = v.version_name || v.file_stem;
                    modalVersionSelect.appendChild(opt);
                }});
                modalVersionSelect.value = card.dataset.stem;
            }}
            if (modalVersionCountLabel) {{
                modalVersionCountLabel.textContent = `Version (${{versions.length}})`;
            }}

            // Find current selected version details
            const selectedVer = versions.find(v => v.file_stem === card.dataset.stem) || versions[0];
            
            // Populate version-specific static details
            if (selectedVer) {{
                const creator = card.dataset.creator;
                const creatorRow = document.getElementById("modal-creator-row");
                if (creatorRow) {{
                    if (!creator || creator.trim() === "" || creator === "Unknown" || creator === "null") {{
                        creatorRow.style.display = "none";
                    }} else {{
                        creatorRow.style.display = "flex";
                        document.getElementById("modal-creator").textContent = creator;
                    }}
                }}
                document.getElementById("modal-base-model").textContent = selectedVer.base_model || "Unknown";
                document.getElementById("modal-filename").textContent = selectedVer.filename;
                document.getElementById("modal-path").textContent = selectedVer.relative_path;
                
                // Description placeholder check
                const descEl = document.getElementById("modal-description");
                const descPlaceholder = document.getElementById("modal-description-placeholder");
                if (descPlaceholder) {{
                    if (!selectedVer.description || selectedVer.description.trim() === "") {{
                        descPlaceholder.style.display = "block";
                        if (descEl) descEl.style.display = "none";
                    }} else {{
                        descPlaceholder.style.display = "none";
                        if (descEl) {{
                            descEl.style.display = "block";
                            descEl.innerHTML = selectedVer.description;
                        }}
                    }}
                }}

                // Populate CivitAI link
                const civitaiLink = document.getElementById("modal-btn-civitai");
                if (civitaiLink) {{
                    const modelId = selectedVer.model_id;
                    const versionId = selectedVer.version_id;
                    if (modelId) {{
                        civitaiLink.style.display = "inline-flex";
                        let url = `https://civitai.red/models/${{modelId}}`;
                        if (versionId) {{
                            url += `?modelVersionId=${{versionId}}`;
                        }}
                        civitaiLink.href = url;
                    }} else {{
                        civitaiLink.style.display = "none";
                    }}
                }}

                // Image preview
                const imgEl = document.getElementById("modal-image");
                if (selectedVer.preview_image) {{
                    imgEl.style.display = "block";
                    imgEl.src = selectedVer.preview_image;
                }} else {{
                    imgEl.style.display = "none";
                }}

                // Official Triggers
                const offTrigContainer = document.getElementById("modal-official-triggers");
                if (offTrigContainer) {{
                    offTrigContainer.innerHTML = selectedVer.trigger_groups_html || '<span class="text-muted" style="font-size:12px; padding: 4px;">None</span>';
                    
                    const groupCount = offTrigContainer.querySelectorAll(".trigger-group").length;
                    const titleEl = document.getElementById("modal-official-triggers-title");
                    if (titleEl) {{
                        titleEl.textContent = `Official Triggers (${{groupCount}})`;
                    }}
                    
                    const searchContainer = document.getElementById("modal-trigger-search-container");
                    const searchInput = document.getElementById("modal-trigger-search");
                    if (searchContainer) {{
                        if (groupCount > 3) {{
                            searchContainer.style.display = "block";
                            if (searchInput) searchInput.value = "";
                        }} else {{
                            searchContainer.style.display = "none";
                        }}
                    }}
                }}
            }}

            // Render interactive lists
            renderRating(currentRating);
            initializeModalCollectionInputs();
            renderModalNotes(notes);
            renderPersonalTriggers();

            // Set Personal Triggers section collapsed by default
            const personalGroupDiv = document.getElementById("personal-trigger-group");
            if (personalGroupDiv) {{
                personalGroupDiv.classList.remove("expanded");
                const header = personalGroupDiv.querySelector(".trigger-group-header");
                if (header) header.setAttribute("aria-expanded", "false");
            }}

            // Re-bind Action buttons (using current selected value dynamically)
            const copyTagBtn = document.getElementById("modal-btn-copy-tag");
            const addPromptBtn = document.getElementById("modal-btn-add-prompt");

            const newCopyBtn = copyTagBtn.cloneNode(true);
            const newAddBtn = addPromptBtn.cloneNode(true);
            copyTagBtn.parentNode.replaceChild(newCopyBtn, copyTagBtn);
            addPromptBtn.parentNode.replaceChild(newAddBtn, addPromptBtn);

            newCopyBtn.addEventListener("click", () => {{
                const currentStem = document.getElementById("modal-version-select").value;
                const tag = `<lora:${{currentStem}}:${{currentWeight}}>`;
                navigator.clipboard.writeText(tag).then(() => {{
                    newCopyBtn.textContent = "Copied!";
                    setTimeout(() => newCopyBtn.textContent = "Copy Tag", 1500);
                }});
            }});

            newAddBtn.addEventListener("click", () => {{
                const currentStem = document.getElementById("modal-version-select").value;
                addToken(currentStem);
                newAddBtn.textContent = "Added!";
                setTimeout(() => newAddBtn.textContent = "Add to Prompt", 1500);
            }});

            // Adjust layout based on image orientation
            adjustModalLayout();
            setupAutoHideScrollbars();

            // Show Modal
            const modal = document.getElementById("details-modal");
            modal.classList.add("active");
            modal.setAttribute("aria-hidden", "false");
            document.body.style.overflow = "hidden";
        }}

        function closeDetailsModal() {{
            const modal = document.getElementById("details-modal");
            modal.classList.remove("active");
            modal.setAttribute("aria-hidden", "true");
            document.body.style.overflow = "";
            currentModalCard = null;
        }}

        // Render functions
        function renderRating(rating) {{
            const stars = document.querySelectorAll("#modal-rating span");
            stars.forEach(star => {{
                const val = parseInt(star.dataset.val);
                if (val <= rating) {{
                    star.classList.add("active");
                    star.innerHTML = "&#9733;";
                }} else {{
                    star.classList.remove("active");
                    star.innerHTML = "&#9734;";
                }}
            }});
        }}

        function renderPersonalTriggers() {{
            const container = document.getElementById("modal-personal-triggers");
            const placeholder = document.getElementById("modal-personal-triggers-placeholder");
            const form = document.getElementById("personal-trigger-add-form");
            const headerTitle = document.getElementById("personal-triggers-header-title");
            
            if (!container) return;
            container.innerHTML = "";
            
            const count = currentPersonalTriggers.length;
            if (headerTitle) {{
                headerTitle.textContent = `Personal Triggers (${{count}})`;
            }}
            
            if (count === 0) {{
                if (placeholder) {{
                    placeholder.style.display = "flex";
                    placeholder.style.flexDirection = "column";
                    placeholder.style.alignItems = "center";
                }}
                if (form) form.style.display = "none";
                container.style.display = "none";
            }} else {{
                if (placeholder) placeholder.style.display = "none";
                if (form) form.style.display = "flex";
                container.style.display = "flex";
                
                currentPersonalTriggers.forEach((trig, idx) => {{
                    const span = document.createElement("span");
                    span.className = "trigger-tag-edit";
                    span.title = "Click to copy trigger";
                    
                    const textSpan = document.createElement("span");
                    textSpan.className = "trigger-tag-edit-text";
                    textSpan.textContent = trig;
                    
                    const removeSpan = document.createElement("span");
                    removeSpan.className = "trigger-tag-edit-remove";
                    removeSpan.title = "Delete trigger";
                    removeSpan.innerHTML = `
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width: 11px; height: 11px; display: block;">
                            <line x1="18" y1="6" x2="6" y2="18"></line>
                            <line x1="6" y1="6" x2="18" y2="18"></line>
                        </svg>
                    `;
                    
                    span.addEventListener("click", (e) => {{
                        if (e.target.closest(".trigger-tag-edit-remove")) return;
                        navigator.clipboard.writeText(trig).then(() => {{
                            showToast("Trigger copied!");
                        }});
                    }});
                    
                    removeSpan.addEventListener("click", (e) => {{
                        e.stopPropagation();
                        currentPersonalTriggers.splice(idx, 1);
                        renderPersonalTriggers();
                        autoSaveUserData();
                    }});
                    
                    span.appendChild(textSpan);
                    span.appendChild(removeSpan);
                    container.appendChild(span);
                }});
            }}
            
            const groupDiv = document.getElementById("personal-trigger-group");
            if (groupDiv) {{
                groupDiv.dataset.groupWords = currentPersonalTriggers.join(", ");
            }}
        }}

        function normalizeCollectionName(name) {{
            return name.trim().replace(/\s+/g, ' ');
        }}

        function adjustModalLayout() {{
            const imgEl = document.getElementById("modal-image");
            const modalBody = document.querySelector(".modal-body");
            const leftScrollable = document.querySelector(".modal-left-scrollable");
            const middleColumn = document.querySelector(".modal-middle");
            const rightColumn = document.querySelector(".modal-right");
            const metaContainer = document.getElementById("modal-meta-container");
            const orgContainer = document.getElementById("modal-organization-container");
            const tagsSection = document.getElementById("modal-section-tags");
            
            if (!imgEl || !modalBody || !leftScrollable || !middleColumn || !rightColumn || !metaContainer || !orgContainer) return;
            
            // Default to landscape (2-column) if no image or fails to load
            let isPortrait = false;
            if (imgEl.style.display !== "none" && imgEl.naturalWidth && imgEl.naturalHeight) {{
                isPortrait = imgEl.naturalHeight > imgEl.naturalWidth;
            }}
            
            if (isPortrait) {{
                // 3-column portrait layout
                modalBody.classList.add("three-column");
                modalBody.classList.remove("two-column");
                middleColumn.style.display = "flex";
                
                // Move organization container (rating/collections) to middle column (at the top)
                middleColumn.appendChild(orgContainer);
                // Move metadata container to middle column
                middleColumn.appendChild(metaContainer);
                
                // Move description section to middle column (below metadata)
                const descSection = document.getElementById("modal-section-description");
                if (descSection) {{
                    middleColumn.appendChild(descSection);
                }}
                
                // Set modal-left width
                document.querySelector(".modal-left").style.width = "30%";
                document.querySelector(".modal-left").style.borderRight = "1px solid var(--border)";
            }} else {{
                // 2-column landscape layout
                modalBody.classList.add("two-column");
                modalBody.classList.remove("three-column");
                middleColumn.style.display = "none";
                
                // Move description section back to right column (at the top)
                const descSection = document.getElementById("modal-section-description");
                if (descSection) {{
                    rightColumn.insertBefore(descSection, rightColumn.firstChild);
                }}
                
                // Move organization container back to right column grid container before Notes
                const gridContainer = document.querySelector(".modal-right .grid-2-col");
                if (gridContainer) {{
                    gridContainer.insertBefore(orgContainer, gridContainer.firstChild);
                }} else {{
                    if (tagsSection) {{
                        rightColumn.insertBefore(orgContainer, tagsSection);
                    }} else {{
                        rightColumn.insertBefore(orgContainer, rightColumn.firstChild);
                    }}
                }}
                
                // Move metadata container back to left scrollable
                leftScrollable.appendChild(metaContainer);
                
                // Restore widths
                document.querySelector(".modal-left").style.width = "40%";
                document.querySelector(".modal-left").style.borderRight = "1px solid var(--border)";
            }}
        }}

        function setupAutoHideScrollbars() {{
            const scrollableSelectors = [
                '#modal-title',
                '.modal-left-scrollable',
                '.triggers-list-read',
                '.triggers-list-edit',
                '.modal-popup-collections-list'
            ];
            
            document.querySelectorAll(scrollableSelectors.join(', ')).forEach(el => {{
                if (el.dataset.scrollBinded) return;
                el.dataset.scrollBinded = "true";
                
                let timeoutId = null;
                
                const showScrollbar = () => {{
                    el.classList.add("is-scrolling");
                }};
                
                const hideScrollbar = () => {{
                    el.classList.remove("is-scrolling");
                }};
                
                el.addEventListener("scroll", () => {{
                    showScrollbar();
                    clearTimeout(timeoutId);
                    timeoutId = setTimeout(() => {{
                        if (!el.matches(':hover')) {{
                            hideScrollbar();
                        }}
                        timeoutId = null;
                    }}, 1000);
                }});
                
                el.addEventListener("mouseenter", showScrollbar);
                el.addEventListener("mouseleave", () => {{
                    if (!timeoutId) {{
                        hideScrollbar();
                    }}
                }});
            }});
        }}
        
        function findExistingCollectionCaseInsensitive(name) {{
            const normalized = normalizeCollectionName(name).toLowerCase();
            return globalCollections.find(c => c.toLowerCase() === normalized);
        }}

        function createNewCollectionGlobally(name, onCreated) {{
            const normalized = normalizeCollectionName(name);
            const existing = findExistingCollectionCaseInsensitive(normalized);
            if (existing) {{
                onCreated(existing);
                return;
            }}
            
            fetch("/api/collections", {{
                method: "POST",
                headers: {{
                    "Content-Type": "application/json"
                }},
                body: JSON.stringify({{ name: normalized }})
            }})
            .then(res => {{
                if (!res.ok) throw new Error("Failed to save collection globally");
                return res.json();
            }})
            .then(data => {{
                setGlobalCollections(data.collections);
                const serverCased = findExistingCollectionCaseInsensitive(normalized) || normalized;
                onCreated(serverCased);
                updateCollectionFilterOptions();
            }})
            .catch(err => {{
                console.error("Error creating collection:", err);
                if (!globalCollections.includes(normalized)) {{
                    setGlobalCollections([...rawGlobalCollections, normalized]);
                }}
                onCreated(normalized);
            }});
        }}

        function updateModalCollectionInputs() {{
            const parentInput = document.getElementById("modal-input-parent-collection");
            const subWrapper = document.getElementById("modal-sub-collection-wrapper");
            const subInput = document.getElementById("modal-input-sub-collection");
            const parentList = document.getElementById("modal-parent-collections-list");
            const subList = document.getElementById("modal-sub-collections-list");
            const parentPopup = document.getElementById("modal-parent-collections-popup");
            const subPopup = document.getElementById("modal-sub-collections-popup");
            
            if (!parentInput || !parentList || !subList || !subInput) return;
            
            const counts = getCollectionCounts();
            
            // 1. Populate parent categories list
            parentList.innerHTML = "";
            
            // Header
            const parentHeader = document.createElement("div");
            parentHeader.className = "popup-header";
            parentHeader.style.padding = "6px 12px";
            parentHeader.style.fontSize = "11px";
            parentHeader.style.fontWeight = "bold";
            parentHeader.style.color = "var(--accent)";
            parentHeader.style.textTransform = "uppercase";
            parentHeader.style.borderBottom = "1px solid var(--border)";
            parentHeader.style.marginBottom = "4px";
            parentHeader.innerHTML = `
                <span class="popup-item-icon" style="margin-right: 6px;">🥞</span>
                <span>ALL CATEGORIES</span>
            `;
            parentList.appendChild(parentHeader);
            
            // Get unique top-level categories
            const parents = new Set();
            rawGlobalCollections.forEach(c => {{
                if (c && typeof c === 'string') {{
                    const parts = c.split(/\s*[\/\\\\]\s*/);
                    if (parts[0]) parents.add(parts[0]);
                }}
            }});
            
            const sortedParents = Array.from(parents).sort();
            const parentSearchInput = document.getElementById("modal-parent-collections-search");
            const parentTyped = parentSearchInput ? parentSearchInput.value.trim() : "";
            const parentTypedLower = parentTyped.toLowerCase();
            
            // Filter parents
            const filteredParents = sortedParents.filter(p => !parentTyped || p.toLowerCase().includes(parentTypedLower));
            
            // Show "Create Category" option if typing something that is not an exact match
            const exactParentMatch = sortedParents.some(p => p.toLowerCase() === parentTypedLower);
            if (parentTyped && !exactParentMatch) {{
                const createItem = document.createElement("div");
                createItem.className = "popup-item";
                createItem.style.fontWeight = "600";
                createItem.style.color = "var(--accent)";
                createItem.innerHTML = `
                    <span style="margin-right: 8px;">➕</span>
                    <span>Create "${{escapeHTML(parentTyped)}}"</span>
                `;
                createItem.addEventListener("mousedown", (e) => {{
                    e.preventDefault();
                    e.stopPropagation();
                    createNewCollectionGlobally(parentTyped, (name) => {{
                        const parts = name.split(/\s*[\/\\\\]\s*/);
                        parentInput.value = parts[0] || name;
                        subInput.value = parts.slice(1).join(" / ") || "";
                        if (parentPopup) parentPopup.hidden = true;
                        updateModalCollectionInputs();
                        saveModalCollectionsFromInputs();
                    }});
                }});
                parentList.appendChild(createItem);
            }}
            
            // Get currently active category
            const activeParentVal = parentInput.value.trim();
            const activeParentLower = activeParentVal.toLowerCase();
            
            if (filteredParents.length > 0) {{
                filteredParents.forEach(p => {{
                    const item = document.createElement("div");
                    item.className = "popup-item";
                    item.style.display = "flex";
                    item.style.justifyContent = "space-between";
                    item.style.alignItems = "center";
                    item.style.width = "100%";
                    item.style.boxSizing = "border-box";
                    
                    const isActive = activeParentLower === p.toLowerCase();
                    if (isActive) {{
                        item.classList.add("active");
                    }}
                    
                    const count = counts[p] || 0;
                    
                    const nameContainer = document.createElement("div");
                    nameContainer.style.display = "flex";
                    nameContainer.style.alignItems = "center";
                    nameContainer.style.flex = "1";
                    nameContainer.style.overflow = "hidden";
                    nameContainer.style.textOverflow = "ellipsis";
                    nameContainer.style.whiteSpace = "nowrap";
                    
                    const checkSpan = document.createElement("span");
                    checkSpan.className = "popup-item-check";
                    checkSpan.textContent = isActive ? "✓" : "";
                    nameContainer.appendChild(checkSpan);
                    
                    const nameSpan = document.createElement("span");
                    nameSpan.className = "popup-item-name";
                    nameSpan.textContent = p + " (" + count + ")";
                    nameContainer.appendChild(nameSpan);
                    
                    item.appendChild(nameContainer);
                    
                    const delBtn = document.createElement("button");
                    delBtn.className = "btn-delete-collection-popup";
                    delBtn.type = "button";
                    delBtn.textContent = "✕";
                    delBtn.title = "Delete category '" + p + "' globally";
                    delBtn.addEventListener("mousedown", (e) => {{
                        e.stopPropagation();
                        e.preventDefault();
                        deleteCollectionGlobally(p);
                    }});
                    item.appendChild(delBtn);
                    
                    item.addEventListener("mousedown", (e) => {{
                        if (e.target.closest(".btn-delete-collection-popup")) return;
                        e.preventDefault();
                        e.stopPropagation();
                        parentInput.value = p;
                        if (activeParentLower !== p.toLowerCase()) {{
                            subInput.value = "";
                        }}
                        if (parentPopup) parentPopup.hidden = true;
                        updateModalCollectionInputs();
                        saveModalCollectionsFromInputs();
                    }});
                    
                    parentList.appendChild(item);
                }});
            }} else if (!parentTyped) {{
                const item = document.createElement("div");
                item.className = "popup-item text-muted";
                item.style.justifyContent = "center";
                item.style.fontSize = "12px";
                item.textContent = "No categories found";
                parentList.appendChild(item);
            }}
            
            // 2. Populate subcategories list based on currently typed parent
            subList.innerHTML = "";
            const activeParent = parentInput.value.trim();
            if (activeParent) {{
                if (subWrapper) subWrapper.style.display = "flex";
                
                // Header
                const subHeader = document.createElement("div");
                subHeader.className = "popup-header";
                subHeader.style.padding = "6px 12px";
                subHeader.style.fontSize = "11px";
                subHeader.style.fontWeight = "bold";
                subHeader.style.color = "var(--accent)";
                subHeader.style.textTransform = "uppercase";
                subHeader.style.borderBottom = "1px solid var(--border)";
                subHeader.style.marginBottom = "4px";
                subHeader.innerHTML = `
                    <span class="popup-item-icon" style="margin-right: 6px;">📂</span>
                    <span>Sub-category suggestions</span>
                `;
                subList.appendChild(subHeader);
                
                const subs = new Set();
                rawGlobalCollections.forEach(c => {{
                    if (c && typeof c === 'string') {{
                        const parts = c.split(/\s*[\/\\\\]\s*/);
                        if (parts[0] && parts[0].toLowerCase() === activeParent.toLowerCase() && parts[1]) {{
                            subs.add(parts.slice(1).join(" / "));
                        }}
                    }}
                }});
                
                const sortedSubs = Array.from(subs).sort();
                const subSearchInput = document.getElementById("modal-sub-collections-search");
                const subTyped = subSearchInput ? subSearchInput.value.trim() : "";
                const subTypedLower = subTyped.toLowerCase();
                const filteredSubs = sortedSubs.filter(s => !subTyped || s.toLowerCase().includes(subTypedLower));
                
                // Show "Create Sub-Category" option if typing something that is not an exact match
                const exactSubMatch = sortedSubs.some(s => s.toLowerCase() === subTypedLower);
                if (subTyped && !exactSubMatch) {{
                    const createItem = document.createElement("div");
                    createItem.className = "popup-item";
                    createItem.style.fontWeight = "600";
                    createItem.style.color = "var(--accent)";
                    createItem.innerHTML = `
                        <span style="margin-right: 8px;">➕</span>
                        <span>Create "${{escapeHTML(subTyped)}}"</span>
                    `;
                    createItem.addEventListener("mousedown", (e) => {{
                        e.preventDefault();
                        e.stopPropagation();
                        const fullTag = activeParent + " / " + subTyped;
                        createNewCollectionGlobally(fullTag, (name) => {{
                            const parts = name.split(/\s*[\/\\\\]\s*/);
                            subInput.value = parts.slice(1).join(" / ") || subTyped;
                            if (subPopup) subPopup.hidden = true;
                            updateModalCollectionInputs();
                            saveModalCollectionsFromInputs();
                        }});
                    }});
                    subList.appendChild(createItem);
                }}
                
                if (filteredSubs.length > 0) {{
                    filteredSubs.forEach(s => {{
                        const item = document.createElement("div");
                        item.className = "popup-item";
                        item.style.display = "flex";
                        item.style.justifyContent = "space-between";
                        item.style.alignItems = "center";
                        item.style.width = "100%";
                        item.style.boxSizing = "border-box";
                        
                        const isActive = subInput.value.trim().toLowerCase() === s.toLowerCase();
                        if (isActive) {{
                            item.classList.add("active");
                        }}
                        
                        const fullPath = activeParent + " / " + s;
                        const count = counts[fullPath] || 0;
                        
                        const nameContainer = document.createElement("div");
                        nameContainer.style.display = "flex";
                        nameContainer.style.alignItems = "center";
                        nameContainer.style.flex = "1";
                        nameContainer.style.overflow = "hidden";
                        nameContainer.style.textOverflow = "ellipsis";
                        nameContainer.style.whiteSpace = "nowrap";
                        
                        const checkSpan = document.createElement("span");
                        checkSpan.className = "popup-item-check";
                        checkSpan.textContent = isActive ? "✓" : "";
                        nameContainer.appendChild(checkSpan);
                        
                        const nameSpan = document.createElement("span");
                        nameSpan.className = "popup-item-name";
                        nameSpan.textContent = s + " (" + count + ")";
                        nameContainer.appendChild(nameSpan);
                        
                        item.appendChild(nameContainer);
                        
                        const delBtn = document.createElement("button");
                        delBtn.className = "btn-delete-collection-popup";
                        delBtn.type = "button";
                        delBtn.textContent = "✕";
                        delBtn.title = "Delete subcategory '" + s + "' globally";
                        delBtn.addEventListener("mousedown", (e) => {{
                            e.stopPropagation();
                            e.preventDefault();
                            deleteCollectionGlobally(fullPath);
                        }});
                        item.appendChild(delBtn);
                        
                        item.addEventListener("mousedown", (e) => {{
                            if (e.target.closest(".btn-delete-collection-popup")) return;
                            e.preventDefault();
                            e.stopPropagation();
                            subInput.value = s;
                            if (subPopup) subPopup.hidden = true;
                            updateModalCollectionInputs();
                            saveModalCollectionsFromInputs();
                        }});
                        
                        subList.appendChild(item);
                    }});
                }} else if (!subTyped) {{
                    const item = document.createElement("div");
                    item.className = "popup-item text-muted";
                    item.style.justifyContent = "center";
                    item.style.fontSize = "12px";
                    item.textContent = "No subcategories found";
                    subList.appendChild(item);
                }}
            }} else {{
                if (subWrapper) subWrapper.style.display = "none";
                subInput.value = "";
            }}
        }}

        function saveModalCollectionsFromInputs() {{
            const parentInput = document.getElementById("modal-input-parent-collection");
            const subInput = document.getElementById("modal-input-sub-collection");
            if (!parentInput || !subInput) return;
            
            const parentVal = parentInput.value.trim();
            const subVal = subInput.value.trim();
            
            if (!parentVal) {{
                currentCollections = [];
            }} else if (!subVal) {{
                currentCollections = [parentVal];
            }} else {{
                currentCollections = [`${{parentVal}} / ${{subVal}}`];
            }}
            
            autoSaveUserData();
        }}

        function initializeModalCollectionInputs() {{
            let parentVal = "";
            let subVal = "";
            if (currentCollections.length > 0) {{
                const tag = currentCollections[0];
                const parts = tag.split(/\s*[\/\\\\]\s*/);
                parentVal = parts[0] || "";
                subVal = parts.slice(1).join(" / ") || "";
            }}
            const parentInput = document.getElementById("modal-input-parent-collection");
            const subInput = document.getElementById("modal-input-sub-collection");
            if (parentInput) parentInput.value = parentVal;
            if (subInput) subInput.value = subVal;
            
            updateModalCollectionInputs();
        }}

        function renderModalNotes(notesText) {{
            const textarea = document.getElementById("modal-notes");
            const placeholder = document.getElementById("modal-notes-placeholder");
            if (!textarea || !placeholder) return;
            
            textarea.value = notesText || "";
            if (!notesText || notesText.trim() === "") {{
                placeholder.style.display = "flex";
                placeholder.style.flexDirection = "column";
                placeholder.style.alignItems = "center";
                textarea.style.display = "none";
            }} else {{
                placeholder.style.display = "none";
                textarea.style.display = "block";
            }}
        }}

        // Star Rating selector listener and hover dynamics
        const stars = document.querySelectorAll("#modal-rating span");
        stars.forEach(star => {{
            star.addEventListener("click", () => {{
                currentRating = parseInt(star.dataset.val);
                renderRating(currentRating);
                autoSaveUserData();
            }});
            star.addEventListener("mouseenter", () => {{
                const val = parseInt(star.dataset.val);
                stars.forEach(s => {{
                    const sVal = parseInt(s.dataset.val);
                    if (sVal <= val) {{
                        s.classList.add("hover-active");
                    }} else {{
                        s.classList.remove("hover-active");
                    }}
                }});
            }});
        }});

        const ratingContainer = document.getElementById("modal-rating");
        if (ratingContainer) {{
            ratingContainer.addEventListener("mouseleave", () => {{
                stars.forEach(s => s.classList.remove("hover-active"));
            }});
        }}

        // Copy Metadata handler
        document.querySelectorAll(".btn-copy-meta").forEach(btn => {{
            btn.addEventListener("click", (e) => {{
                e.stopPropagation();
                const targetId = btn.dataset.target;
                const text = document.getElementById(targetId).textContent;
                if (!text) return;
                
                navigator.clipboard.writeText(text).then(() => {{
                    btn.classList.add("copied");
                    const origSVG = btn.innerHTML;
                    btn.innerHTML = `
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:14px;height:14px;">
                            <polyline points="20 6 9 17 4 12"/>
                        </svg>
                    `;
                    setTimeout(() => {{
                        btn.classList.remove("copied");
                        btn.innerHTML = origSVG;
                    }}, 1500);
                }});
            }});
        }});

        // Expand/Collapse Trigger Group in Modal Handler
        const modalEl = document.getElementById("details-modal");
        modalEl.addEventListener("click", (e) => {{
            const header = e.target.closest(".trigger-group-header");
            if (!header) return;
            if (e.target.closest(".btn-copy-group")) return;
            
            const group = header.closest(".trigger-group");
            if (!group) return;
            
            const isExpanded = group.classList.toggle("expanded");
            header.setAttribute("aria-expanded", isExpanded ? "true" : "false");
        }});

        modalEl.addEventListener("keydown", (e) => {{
            const header = e.target.closest(".trigger-group-header");
            if (!header) return;
            
            if (e.key === "Enter" || e.key === " ") {{
                e.preventDefault();
                const group = header.closest(".trigger-group");
                if (!group) return;
                
                const isExpanded = group.classList.toggle("expanded");
                header.setAttribute("aria-expanded", isExpanded ? "true" : "false");
            }}
        }});

        // Copy Trigger Group Words in Modal Handler
        modalEl.addEventListener("click", (e) => {{
            const btn = e.target.closest(".btn-copy-group");
            if (!btn) return;
            
            const group = btn.closest(".trigger-group");
            if (!group) return;
            
            const words = group.dataset.groupWords;
            if (!words) return;
            
            navigator.clipboard.writeText(words).then(() => {{
                btn.classList.add("copied");
                const normSVG = btn.querySelector(".copy-icon-normal");
                const succSVG = btn.querySelector(".copy-icon-success");
                const feedbackText = btn.querySelector(".copy-group-feedback");
                
                if (normSVG && succSVG && feedbackText) {{
                    normSVG.style.display = "none";
                    succSVG.style.display = "block";
                    feedbackText.classList.add("show");
                    
                    setTimeout(() => {{
                        normSVG.style.display = "block";
                        succSVG.style.display = "none";
                        feedbackText.classList.remove("show");
                        btn.classList.remove("copied");
                    }}, 1500);
                }} else {{
                    const originalText = btn.textContent;
                    btn.textContent = "Copied!";
                    setTimeout(() => {{
                        btn.textContent = originalText;
                        btn.classList.remove("copied");
                    }}, 1500);
                }}
            }});
        }});

        // Add Personal Trigger handlers
        document.getElementById("btn-add-trigger").addEventListener("click", () => {{
            const input = document.getElementById("input-new-trigger");
            const val = input.value.trim();
            if (val && !currentPersonalTriggers.includes(val)) {{
                currentPersonalTriggers.push(val);
                renderPersonalTriggers();
                autoSaveUserData();
            }}
            input.value = "";
        }});
        document.getElementById("input-new-trigger").addEventListener("keydown", (e) => {{
            if (e.key === "Enter") {{
                e.preventDefault();
                document.getElementById("btn-add-trigger").click();
            }}
        }});

        // Global Collection Deletion handler
        function deleteCollectionGlobally(name) {{
            if (!confirm(`Are you sure you want to delete the collection "${{name}}" globally? This will remove it from all models in the library.`)) {{
                return;
            }}
            
            fetch("/api/delete_collection", {{
                method: "POST",
                headers: {{
                    "Content-Type": "application/json"
                }},
                body: JSON.stringify({{ name: name }})
            }})
            .then(res => {{
                if (!res.ok) throw new Error("Failed to delete collection globally");
                return res.json();
            }})
            .then(data => {{
                setGlobalCollections(data.collections);
                
                // Remove from local active currentCollections if selected
                currentCollections = currentCollections.filter(c => c.toLowerCase() !== name.toLowerCase());
                
                // Remove from localStorage recents
                recentCollections = recentCollections.filter(c => c.toLowerCase() !== name.toLowerCase());
                localStorage.setItem(RECENT_COLLECTIONS_KEY, JSON.stringify(recentCollections));
                
                // Remove from all gallery card element datasets in-memory
                const cardsList = document.querySelectorAll(".card");
                cardsList.forEach(card => {{
                    let cardColls = [];
                    try {{
                        cardColls = JSON.parse(card.dataset.collections || "[]");
                    }} catch(e) {{
                        cardColls = [];
                    }}
                    if (cardColls.some(c => c.toLowerCase() === name.toLowerCase())) {{
                        const newColls = cardColls.filter(c => c.toLowerCase() !== name.toLowerCase());
                        card.dataset.collections = JSON.stringify(newColls);
                        
                        // Update indicators
                        const notes = card.dataset.notes || "";
                        const personalTriggers = JSON.parse(card.dataset.personalTriggers || "[]");
                        const rating = parseInt(card.dataset.rating || "0");
                        updateCardIndicators(card, notes, personalTriggers, rating, newColls);
                    }}
                }});
                
                // Re-render UI
                initializeModalCollectionInputs();
                updateCollectionFilterOptions();
                updateFilter();
            }})
            .catch(err => {{
                console.error("Error deleting collection globally:", err);
                alert("Failed to delete collection: " + err.message);
            }});
        }}

        // Copy Combined Triggers handler
        document.getElementById("modal-btn-copy-all-triggers").addEventListener("click", () => {{
            if (!currentModalCard) return;
            const selectVal = document.getElementById("modal-version-select").value;
            const versions = JSON.parse(currentModalCard.dataset.versions || "[]");
            const selectedVer = versions.find(v => v.file_stem === selectVal);
            const officialTriggers = selectedVer ? selectedVer.trigger_words : [];
            const combined = [...officialTriggers, ...currentPersonalTriggers];
            const text = combined.join(", ");
            
            if (text) {{
                navigator.clipboard.writeText(text).then(() => {{
                    const btn = document.getElementById("modal-btn-copy-all-triggers");
                    const originalText = btn.textContent;
                    btn.textContent = "Copied!";
                    setTimeout(() => btn.textContent = originalText, 1500);
                }});
            }}
        }});

        // Auto Save User Data function
        function autoSaveUserData() {{
            if (!currentModalCard) return;
            
            const selectVal = document.getElementById("modal-version-select").value;
            const versions = JSON.parse(currentModalCard.dataset.versions || "[]");
            const selectedVer = versions.find(v => v.file_stem === selectVal) || versions[0];
            const relPath = selectedVer.relative_path;
            const notesVal = document.getElementById("modal-notes").value;
            
            const payload = {{
                relative_path: relPath,
                user_data: {{
                    personal_trigger_words: currentPersonalTriggers,
                    notes: notesVal,
                    rating: currentRating,
                    collections: currentCollections
                }}
            }};
            
            fetch("/api/save_user_data", {{
                method: "POST",
                headers: {{
                    "Content-Type": "application/json"
                }},
                body: JSON.stringify(payload)
            }})
            .then(res => {{
                if (!res.ok) throw new Error("Save request failed.");
                return res.json();
            }})
            .then(data => {{
                // Update datasets in-memory on the card
                currentModalCard.dataset.collections = JSON.stringify(currentCollections);
                currentModalCard.dataset.rating = currentRating;
                
                if (data.new_relative_path) {{
                    currentModalCard.dataset.path = data.new_relative_path;
                    
                    // Update versions relative_path in-memory too
                    const versions = JSON.parse(currentModalCard.dataset.versions || "[]");
                    versions.forEach(v => {{
                        if (v.file_stem === selectVal) {{
                            v.relative_path = data.new_relative_path;
                        }}
                    }});
                    currentModalCard.dataset.versions = JSON.stringify(versions);
                }}
                
                // Redraw indicators on the card
                updateCardIndicators(currentModalCard, notesVal, currentPersonalTriggers, currentRating, currentCollections);
                
                // Update collection select options in-memory
                updateCollectionFilterOptions();
                
                // Immediately apply current filters
                updateFilter();
            }})
            .catch(err => {{
                console.error("Auto-save user data failed:", err);
            }});
        }}

        // Card Indicator drawing helper
        function updateCardIndicators(card, notes, personalTriggers, rating, collections) {{
            const previewDiv = card.querySelector(".preview");
            let indicatorsContainer = previewDiv.querySelector(".card-indicators");
            
            const hasNotes = notes && notes.trim() !== "";
            const hasTriggers = personalTriggers && personalTriggers.length > 0;
            const hasRating = rating && rating > 0;
            
            if (!hasNotes && !hasTriggers && !hasRating) {{
                if (indicatorsContainer) indicatorsContainer.remove();
                return;
            }}
            
            if (!indicatorsContainer) {{
                indicatorsContainer = document.createElement("div");
                indicatorsContainer.className = "card-indicators";
                previewDiv.appendChild(indicatorsContainer);
            }}
            
            indicatorsContainer.innerHTML = "";
            
            if (hasNotes) {{
                const span = document.createElement("span");
                span.className = "indicator-badge";
                span.title = "Has Personal Notes";
                span.textContent = "📝 Notes";
                indicatorsContainer.appendChild(span);
            }}
            if (hasTriggers) {{
                const span = document.createElement("span");
                span.className = "indicator-badge";
                span.title = "Has Personal Triggers";
                span.textContent = "🏷️ Personal Triggers";
                indicatorsContainer.appendChild(span);
            }}
            if (hasRating) {{
                const span = document.createElement("span");
                span.className = "indicator-badge";
                span.title = `Rating: ${{rating}}/5`;
                span.textContent = `⭐ ${{rating}}`;
                indicatorsContainer.appendChild(span);
            }}
        }}

        // Render collections in the searchable popup list
        // Render collections in the searchable modal popup checklist


        function getCollectionCounts() {{
            const counts = {{}};
            SYSTEM_COLLECTIONS.forEach(sc => {{
                counts[sc.name] = 0;
            }});
            
            const favsList = getFavorites();
            
            cards.forEach(card => {{
                let isFav = false;
                if (currentModalCard && card === currentModalCard) {{
                    isFav = currentRating > 0 || favsList.includes(card.dataset.hash || card.dataset.path);
                }} else {{
                    isFav = card.dataset.favorite === "true" || favsList.includes(card.dataset.hash || card.dataset.path);
                }}
                
                let cardColls = [];
                if (currentModalCard && card === currentModalCard) {{
                    cardColls = currentCollections;
                }} else {{
                    try {{
                        cardColls = JSON.parse(card.dataset.collections || "[]");
                    }} catch(e) {{
                        cardColls = [];
                    }}
                }}
                
                SYSTEM_COLLECTIONS.forEach(sc => {{
                    if (sc.matches(card, cardColls, isFav)) {{
                        counts[sc.name] = (counts[sc.name] || 0) + 1;
                    }}
                }});
                
                cardColls.forEach(c => {{
                    if (c && typeof c === 'string') {{
                        const parts = c.split(/\s*[\/\\\\]\s*/);
                        let path = "";
                        for (let i = 0; i < parts.length; i++) {{
                            path = path ? path + " / " + parts[i] : parts[i];
                            counts[path] = (counts[path] || 0) + 1;
                        }}
                    }}
                }});
            }});
            
            return counts;
        }}

        // Render collections in the searchable toolbar filter popup list
        function renderToolbarCollectionsPopupList(filterText = "") {{
            const listContainer = document.getElementById("popup-collections-list");
            if (!listContainer) return;
            listContainer.innerHTML = "";
            
            const lowerFilter = filterText.trim().toLowerCase();
            const counts = getCollectionCounts();
            
            // System collections
            const virtuals = SYSTEM_COLLECTIONS.map(sc => ({{
                name: sc.name,
                icon: sc.icon,
                countKey: sc.name
            }}));
            
            let html = "";
            let itemIndex = 0;
            
            // Render matching virtual collections
            virtuals.forEach(item => {{
                if (item.name.toLowerCase().includes(lowerFilter)) {{
                    const isActive = activeCollection === item.name;
                    const activeClass = isActive ? "active" : "";
                    const count = counts[item.countKey] || 0;
                    html += `
                        <div class="popup-item ${{activeClass}}" data-name="${{item.name}}" data-index="${{itemIndex++}}">
                            <span class="popup-item-icon" style="margin-right: 8px;">${{item.icon}}</span>
                            <span class="popup-item-name">${{item.name}} (${{count}})</span>
                        </div>
                    `;
                }}
            }});
            
            // Group collections by parent
            const groups = {{}};
            const parentNames = new Set();
            
            globalCollections.forEach(c => {{
                if (c && typeof c === 'string') {{
                    const parts = c.split(/\s*[\/\\\\]\s*/);
                    if (parts[0]) parentNames.add(parts[0]);
                }}
            }});
            
            parentNames.forEach(p => {{
                groups[p.toLowerCase()] = {{
                    parentName: p,
                    parentFullName: p,
                    count: counts[p] || 0,
                    children: []
                }};
            }});
            
            globalCollections.forEach(c => {{
                if (c && typeof c === 'string') {{
                    const parts = c.split(/\s*[\/\\\\]\s*/);
                    const parentKey = parts[0].toLowerCase();
                    if (parts.length > 1) {{
                        const subName = parts.slice(1).join(" / ");
                        groups[parentKey].children.push({{
                            fullName: c,
                            subName: subName,
                            count: counts[c] || 0
                        }});
                    }} else {{
                        groups[parentKey].parentFullName = c;
                    }}
                }}
            }});
            
            const sortedKeys = Object.keys(groups).sort((a, b) => groups[a].parentName.localeCompare(groups[b].parentName));
            const matchingGroups = [];
            
            sortedKeys.forEach(key => {{
                const group = groups[key];
                const parentMatches = group.parentName.toLowerCase().includes(lowerFilter);
                const matchingChildren = group.children.filter(ch =>
                    ch.subName.toLowerCase().includes(lowerFilter) || ch.fullName.toLowerCase().includes(lowerFilter)
                );
                
                if (parentMatches || matchingChildren.length > 0) {{
                    matchingGroups.push({{
                        ...group,
                        displayedChildren: lowerFilter ? matchingChildren : group.children
                    }});
                }}
            }});
            
            if (matchingGroups.length > 0) {{
                if (html) {{
                    html += `<div class="popup-divider"></div>`;
                }}
                html += `<div class="popup-section-header">📁 Collections</div>`;
                
                matchingGroups.forEach(group => {{
                    const parentName = group.parentName;
                    const children = group.displayedChildren;
                    children.sort((a, b) => a.subName.localeCompare(b.subName));
                    
                    const parentTag = group.parentFullName || parentName;
                    const isActive = activeCollection === parentTag;
                    const activeClass = isActive ? "active" : "";
                    const hasChildren = children.length > 0;
                    
                    const isChildActive = children.some(ch => activeCollection === ch.fullName);
                    const isExpanded = !!lowerFilter || isChildActive || expandedParents.has(parentName.toLowerCase());
                    
                    if (hasChildren) {{
                        const isAllActive = activeCollection === parentTag;
                        const allActiveClass = isAllActive ? "active" : "";
                        html += `
                            <div class="popup-item parent-item" data-parent-key="${{escapeHTML(parentName.toLowerCase())}}" data-index="${{itemIndex++}}" style="display: flex; justify-content: space-between; align-items: center; cursor: pointer;">
                                <div style="display: flex; align-items: center; flex: 1; overflow: hidden; height: 100%;">
                                    <span class="popup-item-icon" style="margin-right: 8px;">📁</span>
                                    <span class="popup-item-name" style="text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">${{escapeHTML(parentName)}} (${{group.count}})</span>
                                </div>
                                <span class="parent-toggle-arrow" style="padding: 4px 8px; color: var(--text-muted); display: flex; align-items: center; justify-content: center; font-size: 9px; transition: transform 0.2s; transform: ${{isExpanded ? "rotate(90deg)" : "rotate(0deg)"}};">▶</span>
                            </div>
                            <div class="parent-child-container" id="child-container-${{escapeHTML(parentName.toLowerCase())}}" style="display: ${{isExpanded ? "block" : "none"}}; padding-left: 16px; border-left: 1px solid var(--border); margin-left: 18px; margin-top: 2px; margin-bottom: 2px;">
                                <div class="popup-item ${{allActiveClass}}" data-name="${{escapeHTML(parentTag)}}" data-index="${{itemIndex++}}">
                                    <span class="popup-item-icon" style="margin-right: 8px;">📁</span>
                                    <span class="popup-item-name">All ${{escapeHTML(parentName)}} (${{group.count}})</span>
                                </div>
                        `;
                        
                        children.forEach(ch => {{
                            const isChActive = activeCollection === ch.fullName;
                            const chActiveClass = isChActive ? "active" : "";
                            html += `
                                <div class="popup-item ${{chActiveClass}}" data-name="${{escapeHTML(ch.fullName)}}" data-index="${{itemIndex++}}">
                                    <span class="popup-item-icon" style="margin-right: 8px;">📁</span>
                                    <span class="popup-item-name">${{escapeHTML(ch.subName)}} (${{ch.count}})</span>
                                </div>
                            `;
                        }});
                        
                        html += `</div>`;
                    }} else {{
                        html += `
                            <div class="popup-item ${{activeClass}}" data-name="${{escapeHTML(parentTag)}}" data-index="${{itemIndex++}}">
                                <span class="popup-item-icon" style="margin-right: 8px;">📁</span>
                                <span class="popup-item-name">${{escapeHTML(parentName)}} (${{group.count}})</span>
                            </div>
                        `;
                    }}
                }});
            }}
            
            if (!html) {{
                html = `<div class="popup-item text-muted" style="justify-content: center; font-size: 12px;">No collections found</div>`;
            }}
            
            listContainer.innerHTML = html;
            
            // Attach event listeners to popup items
            const items = listContainer.querySelectorAll(".popup-item");
            items.forEach(item => {{
                if (item.classList.contains("parent-item")) {{
                    item.addEventListener("click", () => {{
                        const parentKey = item.dataset.parentKey;
                        const container = document.getElementById("child-container-" + parentKey);
                        const toggleArrow = item.querySelector(".parent-toggle-arrow");
                        if (container && toggleArrow) {{
                            const isCollapsed = container.style.display === "none";
                            container.style.display = isCollapsed ? "block" : "none";
                            toggleArrow.style.transform = isCollapsed ? "rotate(90deg)" : "rotate(0deg)";
                            
                            if (isCollapsed) {{
                                expandedParents.add(parentKey);
                            }} else {{
                                expandedParents.delete(parentKey);
                            }}
                        }}
                    }});
                }} else {{
                    item.addEventListener("click", () => {{
                        const name = item.dataset.name;
                        if (name) selectCollection(name);
                    }});
                }}
            }});
            
            updatePopupFocus();
        }}

        // Update the collection chip appearance dynamically based on the active selection
        function updateCollectionChip() {{
            const chip = document.getElementById("collection-chip");
            if (!chip) return;

            const iconSpan = chip.querySelector(".chip-icon") || document.createElement("span");
            iconSpan.className = "chip-icon";
            const textSpan = chip.querySelector(".chip-text") || document.createElement("span");
            textSpan.className = "chip-text";
            
            // Clean up old classes
            chip.classList.remove("active-filter");
            
            // Remove old action/arrow elements
            const oldArrow = chip.querySelector(".chip-arrow");
            if (oldArrow) oldArrow.remove();
            const oldClear = chip.querySelector(".chip-clear");
            if (oldClear) oldClear.remove();

            const sysColl = SYSTEM_COLLECTIONS.find(sc => sc.name === activeCollection);
            if (activeCollection === "Library") {{
                iconSpan.textContent = sysColl ? sysColl.icon : "📚";
                textSpan.textContent = "Library";
                
                const arrowSpan = document.createElement("span");
                arrowSpan.className = "chip-arrow";
                arrowSpan.textContent = "▼";
                chip.appendChild(arrowSpan);
            }} else {{
                chip.classList.add("active-filter");
                
                if (sysColl) {{
                    iconSpan.textContent = sysColl.icon;
                    textSpan.textContent = sysColl.name;
                }} else {{
                    iconSpan.textContent = "📁";
                    textSpan.textContent = activeCollection;
                }}

                const clearSpan = document.createElement("span");
                clearSpan.className = "chip-clear";
                clearSpan.textContent = "✕";
                clearSpan.title = "Clear filter";
                
                // Clicking clear ✕ returns to Library view
                clearSpan.addEventListener("click", (e) => {{
                    e.stopPropagation(); // Prevent opening the popup
                    selectCollection("Library");
                }});
                
                chip.appendChild(clearSpan);
            }}
        }}

        // Populate and update collection searchable list and chip
        function updateCollectionFilterOptions() {{
            // Render the searchable collections list
            const popupSearchInput = document.getElementById("popup-search-input");
            renderToolbarCollectionsPopupList(popupSearchInput ? popupSearchInput.value : "");
            
            // Check if activeCollection is still valid (e.g. if it was deleted)
            const isSystem = SYSTEM_COLLECTIONS.some(sc => sc.name === activeCollection);
            if (!isSystem && !globalCollections.includes(activeCollection)) {{
                activeCollection = "Library";
            }}
            
            // Update the chip to display the current selection
            updateCollectionChip();
        }}

        function filterByCreator(e, name) {{
            if (e) e.stopPropagation();
            activeCreator = name;
            updateCreatorChip();
            updateFilter();
        }}
        
        function updateCreatorChip() {{
            const chip = document.getElementById("creator-chip");
            const textSpan = document.getElementById("creator-chip-text");
            if (!chip || !textSpan) return;
            
            if (activeCreator) {{
                textSpan.textContent = activeCreator;
                chip.style.display = "inline-flex";
            }} else {{
                chip.style.display = "none";
            }}
        }}
        
        const creatorClear = document.getElementById("creator-chip-clear");
        if (creatorClear) {{
            creatorClear.addEventListener("click", (e) => {{
                e.stopPropagation();
                activeCreator = "";
                updateCreatorChip();
                updateFilter();
            }});
        }}

        // Manage Collections Modal Logic
        const manageModal = document.getElementById("manage-collections-modal");
        const btnManage = document.getElementById("btn-manage-collections");
        const manageClose = document.getElementById("manage-modal-close");
        const manageBtnClose = document.getElementById("manage-modal-btn-close");
        
        if (btnManage && manageModal) {{
            btnManage.addEventListener("click", (e) => {{
                e.stopPropagation();
                // Close collections list popup
                const collectionsPopup = document.getElementById("collections-popup");
                if (collectionsPopup) collectionsPopup.hidden = true;
                
                // Show modal
                manageModal.classList.add("active");
                manageModal.setAttribute("aria-hidden", "false");
                populateManageCollections();
            }});
        }}
        
        function closeManageModal() {{
            if (manageModal) {{
                manageModal.classList.remove("active");
                manageModal.setAttribute("aria-hidden", "true");
            }}
        }}
        
        if (manageClose) manageClose.addEventListener("click", closeManageModal);
        if (manageBtnClose) manageBtnClose.addEventListener("click", closeManageModal);
        
        function populateManageCollections() {{
            const listContainer = document.getElementById("manage-collections-list");
            if (!listContainer) return;
            
            listContainer.innerHTML = "";
            
            // Sort collections alphabetically
            const sortedColls = [...globalCollections].sort((a, b) => a.localeCompare(b));
            
            if (sortedColls.length === 0) {{
                listContainer.innerHTML = '<div style="color: var(--text-muted); font-size: 13px; text-align: center; padding: 24px;">No collections found. Create collections by tagging LoRAs in the details modal.</div>';
                return;
            }}
            
            sortedColls.forEach(col => {{
                const itemDiv = document.createElement("div");
                itemDiv.style.display = "flex";
                itemDiv.style.justifyContent = "space-between";
                itemDiv.style.alignItems = "center";
                itemDiv.style.padding = "8px 12px";
                itemDiv.style.background = "rgba(255, 255, 255, 0.02)";
                itemDiv.style.border = "1px solid var(--border)";
                itemDiv.style.borderRadius = "6px";
                
                const nameSpan = document.createElement("span");
                nameSpan.textContent = col;
                nameSpan.style.fontSize = "13px";
                nameSpan.style.color = "var(--text)";
                nameSpan.style.fontWeight = "500";
                
                const renameBtn = document.createElement("button");
                renameBtn.textContent = "✏️ Rename";
                renameBtn.style.background = "rgba(129, 140, 248, 0.1)";
                renameBtn.style.border = "1px solid rgba(129, 140, 248, 0.2)";
                renameBtn.style.color = "#a5b4fc";
                renameBtn.style.padding = "4px 8px";
                renameBtn.style.borderRadius = "4px";
                renameBtn.style.fontSize = "11px";
                renameBtn.style.fontWeight = "600";
                renameBtn.style.cursor = "pointer";
                renameBtn.style.transition = "var(--transition)";
                
                renameBtn.addEventListener("mouseover", () => {{
                    renameBtn.style.background = "rgba(129, 140, 248, 0.2)";
                }});
                renameBtn.addEventListener("mouseout", () => {{
                    renameBtn.style.background = "rgba(129, 140, 248, 0.1)";
                }});
                
                renameBtn.addEventListener("click", () => {{
                    const newVal = prompt(`Rename collection "${{col}}" to:`, col);
                    if (newVal !== null) {{
                        const trimmed = newVal.trim();
                        if (trimmed && trimmed.toLowerCase() !== col.toLowerCase()) {{
                            // Call rename API
                            renameBtn.disabled = true;
                            renameBtn.textContent = "Saving...";
                            
                            fetch("/api/rename_collection", {{
                                method: "POST",
                                headers: {{
                                    "Content-Type": "application/json"
                                }},
                                body: JSON.stringify({{
                                    old_name: col,
                                    new_name: trimmed
                                }})
                            }})
                            .then(res => res.json())
                            .then(data => {{
                                if (data.ok) {{
                                    alert(`Successfully renamed collection to "${{trimmed}}" across all files.`);
                                    window.location.reload();
                                }} else {{
                                    alert(`Error: ${{data.message || "Failed to rename collection"}}`);
                                    renameBtn.disabled = false;
                                    renameBtn.textContent = "✏️ Rename";
                                }}
                            }})
                            .catch(err => {{
                                alert(`Network Error: ${{err.message}}`);
                                renameBtn.disabled = false;
                                renameBtn.textContent = "✏️ Rename";
                            }});
                        }}
                    }}
                }});
                
                itemDiv.appendChild(nameSpan);
                itemDiv.appendChild(renameBtn);
                listContainer.appendChild(itemDiv);
            }});
        }}



        let focusedItemIndex = -1;

        function getVisiblePopupItems() {{
            const listContainer = document.getElementById("popup-collections-list");
            if (!listContainer) return [];
            return Array.from(listContainer.querySelectorAll(".popup-item:not(.text-muted)")).filter(el => el.offsetWidth > 0 || el.offsetHeight > 0);
        }}

        function updatePopupFocus() {{
            const items = getVisiblePopupItems();
            items.forEach((item, index) => {{
                if (index === focusedItemIndex) {{
                    item.classList.add("focused");
                    item.scrollIntoView({{ block: "nearest" }});
                }} else {{
                    item.classList.remove("focused");
                }}
            }});
        }}

        function selectCollection(name) {{
            activeCollection = name;
            updateCollectionChip();
            closeCollectionsPopup();
            updateFilter();
        }}

        function toggleCollectionsPopup() {{
            const popup = document.getElementById("collections-popup");
            if (!popup) return;

            if (popup.hidden) {{
                openCollectionsPopup();
            }} else {{
                closeCollectionsPopup();
            }}
        }}

        function openCollectionsPopup() {{
            const popup = document.getElementById("collections-popup");
            const popupInput = document.getElementById("popup-search-input");
            if (!popup) return;

            popup.hidden = false;
            focusedItemIndex = -1;
            if (popupInput) {{
                popupInput.value = "";
                renderToolbarCollectionsPopupList("");
                popupInput.focus();
            }}
            document.addEventListener("click", handleOutsideClick);
        }}

        function closeCollectionsPopup() {{
            const popup = document.getElementById("collections-popup");
            if (!popup) return;

            popup.hidden = true;
            document.removeEventListener("click", handleOutsideClick);
        }}

        function handleOutsideClick(event) {{
            const wrapper = document.querySelector(".search-wrapper");
            if (wrapper && !wrapper.contains(event.target)) {{
                closeCollectionsPopup();
            }}
        }}


        // Populate base model dropdown select dynamically from the unique base models of cards in the gallery
        function updateBaseModelFilterOptions() {{
            const filterBaseModelSelect = document.getElementById("filter-base-model");
            if (!filterBaseModelSelect) return;
            
            const currentSelected = filterBaseModelSelect.value;
            
            const baseModelsSet = new Set();
            cards.forEach(card => {{
                const rawModel = card.dataset.baseModel;
                const normalized = normalizeBaseModel(rawModel);
                if (normalized) {{
                    baseModelsSet.add(normalized);
                }}
            }});
            
            const sortedModels = Array.from(baseModelsSet).sort((a, b) => {{
                if (a === "Other") return 1;
                if (b === "Other") return -1;
                return a.localeCompare(b);
            }});
            
            filterBaseModelSelect.innerHTML = '<option value="">All Base Models</option>';
            sortedModels.forEach(m => {{
                const opt = document.createElement("option");
                opt.value = m;
                opt.textContent = m;
                filterBaseModelSelect.appendChild(opt);
            }});
            
            if (sortedModels.includes(currentSelected)) {{
                filterBaseModelSelect.value = currentSelected;
            }} else {{
                filterBaseModelSelect.value = "";
            }}
        }}

        // Load summaries for indicators on load
        function loadUserDataSummary() {{
            loadCollections().then(() => {{
                fetch("/api/user_data_summary")
                    .then(res => {{
                        if (!res.ok) throw new Error("Summary request failed.");
                        return res.json();
                    }})
                    .then(summary => {{
                        Object.entries(summary).forEach(([relPath, data]) => {{
                            const card = document.querySelector(`.card[data-path="${{relPath}}"]`);
                            if (card) {{
                                card.dataset.rating = data.rating || 0;
                                card.dataset.collections = JSON.stringify(data.collections || []);
                                // Use dummy strings to indicate presence for local state
                                updateCardIndicators(
                                    card,
                                    data.has_notes ? "yes" : "",
                                    data.has_personal_triggers ? ["dummy"] : [],
                                    data.rating,
                                    data.collections || []
                                );
                            }}
                        }});
                        updateCollectionFilterOptions();
                    }})
                    .catch(err => {{
                        console.log("Local API server not active or summary fetch failed. Offline fallback mode.");
                    }});
            }});
        }}

        // Modal Close Button click listener
        document.querySelector(".modal-close").addEventListener("click", closeDetailsModal);

        // Image load listener for layout adjustment
        const imgEl = document.getElementById("modal-image");
        if (imgEl) {{
            imgEl.addEventListener("load", adjustModalLayout);
        }}

        // Modal backdrop click listener
        document.getElementById("details-modal").addEventListener("click", (e) => {{
            if (e.target === document.getElementById("details-modal")) {{
                closeDetailsModal();
            }}
        }});

        // Bind Details clicks (event delegation) - opens details only when clicking the preview image area
        grid.addEventListener("click", (e) => {{
            // Only trigger if clicking inside the .preview container
            const preview = e.target.closest(".preview");
            if (!preview) return;
            
            // Ignore click if they click an interactive element inside preview (like the favorite button or creator badge)
            if (e.target.closest("button") || e.target.closest("a") || e.target.closest(".creator-badge")) {{
                return;
            }}
            
            const card = preview.closest(".card");
            if (!card) return;
            
            const path = card.dataset.path;
            
            // Lazy load user data from API server
            fetch(`/api/get_user_data?relative_path=${{encodeURIComponent(path)}}`)
                .then(res => {{
                    if (!res.ok) throw new Error("Failed to load user data sidecar.");
                    return res.json();
                }})
                .then(userData => {{
                    openDetailsModal(card, userData);
                }})
                .catch(err => {{
                    console.warn("API server not active, using empty fallback state:", err);
                    openDetailsModal(card, {{
                        personal_trigger_words: [],
                        notes: "",
                        rating: 0,
                        collections: []
                    }});
                }});
        }});

        // Bind Details clicks on description wrapper (fade-out area → open modal)
        grid.addEventListener("click", (e) => {{
            const wrapper = e.target.closest(".description-wrapper");
            if (!wrapper) return;
            const card = wrapper.closest(".card");
            if (!card) return;
            const path = card.dataset.path;
            fetch(`/api/get_user_data?relative_path=${{encodeURIComponent(path)}}`)
                .then(res => {{
                    if (!res.ok) throw new Error("Failed to load user data sidecar.");
                    return res.json();
                }})
                .then(userData => {{ openDetailsModal(card, userData); }})
                .catch(() => {{
                    openDetailsModal(card, {{ personal_trigger_words: [], notes: "", rating: 0, collections: [] }});
                }});
        }});

        // Event Listeners for controls
        searchInput.addEventListener("input", updateFilter);
        sortSelect.addEventListener("change", performSort);
        
        // Live search inside Details Modal official triggers
        const triggerSearchInput = document.getElementById("modal-trigger-search");
        if (triggerSearchInput) {{
            triggerSearchInput.addEventListener("input", (e) => {{
                const query = e.target.value.toLowerCase().trim();
                const container = document.getElementById("modal-official-triggers");
                if (!container) return;
                
                const groups = container.querySelectorAll(".trigger-group");
                let visibleCount = 0;
                
                groups.forEach(group => {{
                    const titleEl = group.querySelector(".trigger-group-title");
                    if (!titleEl) return;
                    
                    const titleText = titleEl.textContent.toLowerCase();
                    if (titleText.includes(query)) {{
                        group.style.display = "";
                        visibleCount++;
                    }} else {{
                        group.style.display = "none";
                    }}
                }});
                
                // Show/hide "No matching trigger groups found."
                let noMatchMessage = container.querySelector(".no-match-triggers");
                if (visibleCount === 0 && groups.length > 0) {{
                    if (!noMatchMessage) {{
                        noMatchMessage = document.createElement("p");
                        noMatchMessage.className = "no-match-triggers text-muted";
                        noMatchMessage.style.cssText = "font-size: 13px; text-align: center; padding: 16px 0; margin: 0;";
                        noMatchMessage.textContent = "No matching trigger groups found.";
                        container.appendChild(noMatchMessage);
                    }}
                }} else {{
                    if (noMatchMessage) {{
                        noMatchMessage.remove();
                    }}
                }}
            }});
        }}
        
        // Collection chip click toggles the popup
        const chip = document.getElementById("collection-chip");
        if (chip) {{
            chip.addEventListener("click", (e) => {{
                if (e.target.classList.contains("chip-clear")) return;
                toggleCollectionsPopup();
            }});
            chip.addEventListener("keydown", (e) => {{
                if (e.key === " " || e.key === "Enter") {{
                    e.preventDefault();
                    toggleCollectionsPopup();
                }}
            }});
        }}

        // Keyboard navigation and fuzzy searching inside collections popup
        const popupSearchInput = document.getElementById("popup-search-input");
        if (popupSearchInput) {{
            popupSearchInput.addEventListener("keydown", (e) => {{
                const items = getVisiblePopupItems();
                if (items.length === 0) return;

                if (e.key === "ArrowDown") {{
                    e.preventDefault();
                    focusedItemIndex = (focusedItemIndex + 1) % items.length;
                    updatePopupFocus();
                }} else if (e.key === "ArrowUp") {{
                    e.preventDefault();
                    focusedItemIndex = (focusedItemIndex - 1 + items.length) % items.length;
                    updatePopupFocus();
                }} else if (e.key === "Enter") {{
                    e.preventDefault();
                    if (focusedItemIndex >= 0 && focusedItemIndex < items.length) {{
                        const selectedItem = items[focusedItemIndex];
                        if (selectedItem) {{
                            if (selectedItem.classList.contains("parent-item")) {{
                                selectedItem.click();
                            }} else {{
                                selectCollection(selectedItem.dataset.name);
                            }}
                        }}
                    }} else if (items.length > 0) {{
                        const firstItem = items[0];
                        if (firstItem.classList.contains("parent-item")) {{
                            firstItem.click();
                        }} else {{
                            selectCollection(firstItem.dataset.name);
                        }}
                    }}
                }} else if (e.key === "Escape") {{
                    e.preventDefault();
                    closeCollectionsPopup();
                }}
            }});

            popupSearchInput.addEventListener("input", (e) => {{
                focusedItemIndex = -1;
                renderToolbarCollectionsPopupList(e.target.value);
            }});
        }}



        // Version Selector change listener
        const modalVersionSelect = document.getElementById("modal-version-select");
        if (modalVersionSelect) {{
            modalVersionSelect.addEventListener("change", (e) => {{
                if (!currentModalCard) return;
                const versions = JSON.parse(currentModalCard.dataset.versions || "[]");
                const selectedVer = versions.find(v => v.file_stem === e.target.value);
                if (!selectedVer) return;

                // — Fade out all content panels —
                const leftScrollable = document.querySelector(".modal-left-scrollable");
                const rightCol = document.querySelector(".modal-right");
                const middleCol = document.querySelector(".modal-middle");
                [leftScrollable, rightCol, middleCol].forEach(el => el && el.classList.add("modal-version-transitioning"));

                setTimeout(() => {{
                    // Update parent card select
                    const cardSelect = currentModalCard.querySelector(".version-select");
                    if (cardSelect && cardSelect.value !== selectedVer.file_stem) {{
                        cardSelect.value = selectedVer.file_stem;
                        cardSelect.dispatchEvent(new Event("change"));
                    }}

                    // Update modal preview image
                    const imgEl = document.getElementById("modal-image");
                    if (selectedVer.preview_image) {{
                        imgEl.style.display = "block";
                        imgEl.src = selectedVer.preview_image;
                    }} else {{
                        imgEl.style.display = "none";
                    }}

                    // Update modal description
                    const descEl = document.getElementById("modal-description");
                    const descPlaceholder = document.getElementById("modal-description-placeholder");
                    if (!selectedVer.description || selectedVer.description.trim() === "") {{
                        if (descPlaceholder) descPlaceholder.style.display = "block";
                        if (descEl) descEl.style.display = "none";
                    }} else {{
                        if (descPlaceholder) descPlaceholder.style.display = "none";
                        if (descEl) {{
                            descEl.style.display = "block";
                            descEl.innerHTML = selectedVer.description;
                        }}
                    }}

                    // Update official triggers
                    const offTrigContainer = document.getElementById("modal-official-triggers");
                    if (offTrigContainer) {{
                        offTrigContainer.innerHTML = selectedVer.trigger_groups_html || '<span class="text-muted" style="font-size:12px; padding: 4px;">None</span>';
                        
                        const groupCount = offTrigContainer.querySelectorAll(".trigger-group").length;
                        const titleEl = document.getElementById("modal-official-triggers-title");
                        if (titleEl) {{
                            titleEl.textContent = `Official Triggers (${{groupCount}})`;
                        }}
                        
                        const searchContainer = document.getElementById("modal-trigger-search-container");
                        const searchInput = document.getElementById("modal-trigger-search");
                        if (searchContainer) {{
                            if (groupCount > 3) {{
                                searchContainer.style.display = "block";
                                if (searchInput) searchInput.value = "";
                            }} else {{
                                searchContainer.style.display = "none";
                            }}
                        }}
                    }}

                    // Update metadata
                    const creator = currentModalCard.dataset.creator;
                    const creatorRow = document.getElementById("modal-creator-row");
                    if (creatorRow) {{
                        if (!creator || creator.trim() === "" || creator === "Unknown" || creator === "null") {{
                            creatorRow.style.display = "none";
                        }} else {{
                            creatorRow.style.display = "flex";
                            document.getElementById("modal-creator").textContent = creator;
                        }}
                    }}
                    document.getElementById("modal-base-model").textContent = selectedVer.base_model || "Unknown";
                    document.getElementById("modal-filename").textContent = selectedVer.filename;
                    document.getElementById("modal-path").textContent = selectedVer.relative_path;

                    // Update CivitAI link
                    const civitaiLink = document.getElementById("modal-btn-civitai");
                    if (civitaiLink) {{
                        const modelId = selectedVer.model_id;
                        const versionId = selectedVer.version_id;
                        if (modelId) {{
                            civitaiLink.style.display = "inline-flex";
                            let url = `https://civitai.red/models/${{modelId}}`;
                            if (versionId) url += `?modelVersionId=${{versionId}}`;
                            civitaiLink.href = url;
                        }} else {{
                            civitaiLink.style.display = "none";
                        }}
                    }}

                    // Fetch version-specific user data
                    fetch(`/api/get_user_data?relative_path=${{encodeURIComponent(selectedVer.relative_path)}}`)
                        .then(res => {{
                            if (!res.ok) throw new Error("Failed to load version user data");
                            return res.json();
                        }})
                        .then(userData => {{
                            currentPersonalTriggers = userData.personal_trigger_words || [];
                            currentCollections = userData.collections || [];
                            currentRating = userData.rating || 0;
                            const notes = userData.notes || "";
                            renderRating(currentRating);
                            initializeModalCollectionInputs();
                            renderModalNotes(notes);
                            renderPersonalTriggers();
                            adjustModalLayout();
                            setupAutoHideScrollbars();
                        }})
                        .catch(err => {{
                            console.warn("Failed to load version user data:", err);
                            currentPersonalTriggers = [];
                            currentCollections = [];
                            currentRating = 0;
                            renderRating(0);
                            initializeModalCollectionInputs();
                            renderModalNotes("");
                            renderPersonalTriggers();
                            adjustModalLayout();
                            setupAutoHideScrollbars();
                        }})
                        .finally(() => {{
                            // — Fade back in after content + user data settled —
                            [leftScrollable, rightCol, middleCol].forEach(el => el && el.classList.remove("modal-version-transitioning"));
                        }});
                }}, 160);
            }});
        }}

        // Base Model filter listener
        const filterBaseModelSelect = document.getElementById("filter-base-model");
        if (filterBaseModelSelect) {{
            filterBaseModelSelect.addEventListener("change", updateFilter);
        }}

        // Add Notes placeholder trigger button
        const addNotePlaceholderBtn = document.getElementById("btn-add-note-placeholder");
        if (addNotePlaceholderBtn) {{
            addNotePlaceholderBtn.addEventListener("click", () => {{
                const textarea = document.getElementById("modal-notes");
                const placeholder = document.getElementById("modal-notes-placeholder");
                if (placeholder) placeholder.style.display = "none";
                if (textarea) {{
                    textarea.style.display = "block";
                    textarea.focus();
                }}
            }});
        }}

        // Add Personal Trigger placeholder button
        const addTrigPlaceholderBtn = document.getElementById("btn-add-trigger-placeholder");
        if (addTrigPlaceholderBtn) {{
            addTrigPlaceholderBtn.addEventListener("click", () => {{
                const placeholder = document.getElementById("modal-personal-triggers-placeholder");
                const form = document.getElementById("personal-trigger-add-form");
                if (placeholder) placeholder.style.display = "none";
                if (form) {{
                    form.style.display = "flex";
                    const input = form.querySelector("input");
                    if (input) input.focus();
                }}
            }});
        }}

        // Auto-save notes when textarea loses focus or is changed
        const notesTextarea = document.getElementById("modal-notes");
        if (notesTextarea) {{
            notesTextarea.addEventListener("change", autoSaveUserData);
            notesTextarea.addEventListener("blur", () => {{
                const textarea = document.getElementById("modal-notes");
                const placeholder = document.getElementById("modal-notes-placeholder");
                const val = textarea.value.trim();
                if (val === "") {{
                    if (placeholder) {{
                        placeholder.style.display = "flex";
                        placeholder.style.flexDirection = "column";
                        placeholder.style.alignItems = "center";
                    }}
                    textarea.style.display = "none";
                }}
                autoSaveUserData();
            }});
        }}

        // Category and Sub-category inputs listeners
        const parentInput = document.getElementById("modal-input-parent-collection");
        const subInput = document.getElementById("modal-input-sub-collection");
        const parentPopup = document.getElementById("modal-parent-collections-popup");
        const subPopup = document.getElementById("modal-sub-collections-popup");
        
        if (parentInput && parentPopup) {{
            const focusParentSearch = () => {{
                const searchInput = document.getElementById("modal-parent-collections-search");
                if (searchInput) {{
                    searchInput.value = "";
                    updateModalCollectionInputs();
                    setTimeout(() => {{
                        searchInput.focus();
                    }}, 20);
                }}
            }};

            parentInput.addEventListener("input", () => {{
                parentPopup.hidden = false;
                updateModalCollectionInputs();
            }});
            parentInput.addEventListener("change", () => {{
                updateModalCollectionInputs();
                saveModalCollectionsFromInputs();
            }});
            
            // Show popup and focus search
            parentInput.addEventListener("focus", () => {{
                parentPopup.hidden = false;
                if (subPopup) subPopup.hidden = true;
                focusParentSearch();
            }});
            parentInput.addEventListener("click", (e) => {{
                e.stopPropagation();
                parentPopup.hidden = false;
                if (subPopup) subPopup.hidden = true;
                focusParentSearch();
            }});
            
            const parentDoneBtn = parentPopup.querySelector(".popup-footer button");
            if (parentDoneBtn) {{
                parentDoneBtn.addEventListener("mousedown", (e) => {{
                    e.preventDefault();
                    e.stopPropagation();
                    parentPopup.hidden = true;
                }});
            }}
        }}
        
        if (subInput && subPopup) {{
            const focusSubSearch = () => {{
                const searchInput = document.getElementById("modal-sub-collections-search");
                if (searchInput) {{
                    searchInput.value = "";
                    updateModalCollectionInputs();
                    setTimeout(() => {{
                        searchInput.focus();
                    }}, 20);
                }}
            }};

            subInput.addEventListener("input", () => {{
                subPopup.hidden = false;
                updateModalCollectionInputs();
            }});
            subInput.addEventListener("change", () => {{
                updateModalCollectionInputs();
                saveModalCollectionsFromInputs();
            }});
            
            // Show popup and focus search
            subInput.addEventListener("focus", () => {{
                subPopup.hidden = false;
                if (parentPopup) parentPopup.hidden = true;
                focusSubSearch();
            }});
            subInput.addEventListener("click", (e) => {{
                e.stopPropagation();
                subPopup.hidden = false;
                if (parentPopup) parentPopup.hidden = true;
                focusSubSearch();
            }});
            
            const subDoneBtn = subPopup.querySelector(".popup-footer button");
            if (subDoneBtn) {{
                subDoneBtn.addEventListener("mousedown", (e) => {{
                    e.preventDefault();
                    e.stopPropagation();
                    subPopup.hidden = true;
                }});
            }}
        }}

        // Search inputs change & click listeners
        const parentSearchInputEl = document.getElementById("modal-parent-collections-search");
        if (parentSearchInputEl) {{
            parentSearchInputEl.addEventListener("input", () => {{
                updateModalCollectionInputs();
            }});
            parentSearchInputEl.addEventListener("click", (e) => {{
                e.stopPropagation();
            }});
            parentSearchInputEl.addEventListener("mousedown", (e) => {{
                e.stopPropagation();
            }});
        }}

        const subSearchInputEl = document.getElementById("modal-sub-collections-search");
        if (subSearchInputEl) {{
            subSearchInputEl.addEventListener("input", () => {{
                updateModalCollectionInputs();
            }});
            subSearchInputEl.addEventListener("click", (e) => {{
                e.stopPropagation();
            }});
            subSearchInputEl.addEventListener("mousedown", (e) => {{
                e.stopPropagation();
            }});
        }}
        
        // Setup keyboard navigation for custom popups
        function bindPopupKeyboardNav(inputId, popupId, listId) {{
            const input = document.getElementById(inputId);
            const popup = document.getElementById(popupId);
            const list = document.getElementById(listId);
            if (!input || !popup || !list) return;
            
            let focusedIdx = -1;
            
            const getItems = () => Array.from(list.querySelectorAll(".popup-item:not(.text-muted)"));
            
            const updateFocus = (items) => {{
                items.forEach((item, index) => {{
                    if (index === focusedIdx) {{
                        item.classList.add("focused");
                        item.scrollIntoView({{ block: "nearest" }});
                    }} else {{
                        item.classList.remove("focused");
                    }}
                }});
            }};
            
            input.addEventListener("keydown", (e) => {{
                if (popup.hidden) {{
                    if (e.key === "ArrowDown" || e.key === "ArrowUp") {{
                        e.preventDefault();
                        popup.hidden = false;
                        focusedIdx = -1;
                        updateModalCollectionInputs();
                    }}
                    return;
                }}
                
                const items = getItems();
                if (items.length === 0) return;
                
                if (e.key === "ArrowDown") {{
                    e.preventDefault();
                    focusedIdx = (focusedIdx + 1) % items.length;
                    updateFocus(items);
                }} else if (e.key === "ArrowUp") {{
                    e.preventDefault();
                    focusedIdx = (focusedIdx - 1 + items.length) % items.length;
                    updateFocus(items);
                }} else if (e.key === "Enter") {{
                    e.preventDefault();
                    if (focusedIdx >= 0 && focusedIdx < items.length) {{
                        items[focusedIdx].dispatchEvent(new MouseEvent("mousedown"));
                    }} else if (items.length > 0) {{
                        items[0].dispatchEvent(new MouseEvent("mousedown"));
                    }}
                    popup.hidden = true;
                }} else if (e.key === "Escape") {{
                    e.preventDefault();
                    popup.hidden = true;
                }}
            }});
        }}
        
        bindPopupKeyboardNav("modal-input-parent-collection", "modal-parent-collections-popup", "modal-parent-collections-list");
        bindPopupKeyboardNav("modal-parent-collections-search", "modal-parent-collections-popup", "modal-parent-collections-list");
        bindPopupKeyboardNav("modal-input-sub-collection", "modal-sub-collections-popup", "modal-sub-collections-list");
        bindPopupKeyboardNav("modal-sub-collections-search", "modal-sub-collections-popup", "modal-sub-collections-list");
        
        // Document click listener to close popups
        document.addEventListener("click", (e) => {{
            const pPopup = document.getElementById("modal-parent-collections-popup");
            const sPopup = document.getElementById("modal-sub-collections-popup");
            const pWrapper = document.getElementById("modal-input-parent-collection")?.closest(".custom-dropdown-wrapper")?.parentNode;
            const sWrapper = document.getElementById("modal-input-sub-collection")?.closest(".custom-dropdown-wrapper")?.parentNode;
            
            if (pPopup && !pPopup.hidden && pWrapper && !pWrapper.contains(e.target)) {{
                pPopup.hidden = true;
            }}
            if (sPopup && !sPopup.hidden && sWrapper && !sWrapper.contains(e.target)) {{
                sPopup.hidden = true;
            }}
        }});

        // Open parent folder handler
        const openFolderBtn = document.getElementById("modal-btn-open-folder");
        if (openFolderBtn) {{
            openFolderBtn.addEventListener("click", () => {{
                if (!currentModalCard) return;
                
                const selectVal = document.getElementById("modal-version-select").value;
                const versions = JSON.parse(currentModalCard.dataset.versions || "[]");
                const selectedVer = versions.find(v => v.file_stem === selectVal) || versions[0];
                const path = selectedVer.relative_path;
                
                fetch("/api/open_folder", {{
                    method: "POST",
                    headers: {{
                        "Content-Type": "application/json"
                    }},
                    body: JSON.stringify({{ relative_path: path }})
                }})
                .then(res => {{
                    if (!res.ok) throw new Error("Failed to open folder on server.");
                    return res.json();
                }})
                .then(data => {{
                    showToast("Folder opened successfully.");
                }})
                .catch(err => {{
                    console.error("Failed to open folder:", err);
                    alert("Failed to open folder: " + err.message);
                }});
            }});
        }}

        // Modal delete version handler
        const modalDeleteVersionBtn = document.getElementById("modal-btn-delete-version");
        if (modalDeleteVersionBtn) {{
            modalDeleteVersionBtn.addEventListener("click", () => {{
                if (!currentModalCard) return;
                
                const modalVersionSelect = document.getElementById("modal-version-select");
                if (!modalVersionSelect) return;
                
                const versions = JSON.parse(currentModalCard.dataset.versions || "[]");
                const selectedVer = versions.find(v => v.file_stem === modalVersionSelect.value);
                if (!selectedVer) return;
                
                const confirmMsg = `Are you sure you want to delete the model version '${{selectedVer.version_name || selectedVer.file_stem}}'?\n\nThis will permanently delete the model file:\n${{selectedVer.filename}}\n\nas well as its associated previews and metadata.`;
                if (!confirm(confirmMsg)) return;
                
                fetch('/api/delete_model_version', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify({{ relative_path: selectedVer.relative_path }})
                }})
                .then(res => {{
                    if (!res.ok) throw new Error("Delete failed on server.");
                    return res.json();
                }})
                .then(data => {{
                    let remainingVersions = versions.filter(v => v.file_stem !== selectedVer.file_stem);
                    if (remainingVersions.length === 0) {{
                        currentModalCard.remove();
                        const idx = cards.indexOf(currentModalCard);
                        if (idx > -1) cards.splice(idx, 1);
                        updateFilter();
                        closeDetailsModal();
                        showToast("Model deleted successfully.");
                    }} else {{
                        currentModalCard.dataset.versions = JSON.stringify(remainingVersions);
                        const cardSelect = currentModalCard.querySelector(".version-select");
                        if (cardSelect) {{
                            cardSelect.innerHTML = remainingVersions.map(v => `<option value="${{v.file_stem}}">${{v.version_name || v.file_stem}}</option>`).join("");
                            cardSelect.value = remainingVersions[0].file_stem;
                        }}
                        modalVersionSelect.innerHTML = remainingVersions.map(v => `<option value="${{v.file_stem}}">${{v.version_name || v.file_stem}}</option>`).join("");
                        modalVersionSelect.value = remainingVersions[0].file_stem;
                        modalVersionSelect.dispatchEvent(new Event("change"));
                        showToast("Version deleted successfully.");
                    }}
                }})
                .catch(err => {{
                    console.error("Failed to delete version:", err);
                    alert("Failed to delete version: " + err.message);
                }});
            }});
        }}

        // Initial setup
        const initialFavs = getFavorites();
        cards.forEach(card => {{
            const id = card.dataset.hash || card.dataset.path;
            if (initialFavs.includes(id)) {{
                card.dataset.favorite = "true";
                const favBtn = card.querySelector(".btn-favorite");
                if (favBtn) favBtn.classList.add("active");
            }}
        }});

        setWeight(currentWeight);
        performSort();
        updateBaseModelFilterOptions();
        updateFilter();
        updatePromptBuilderUI();
        setupAutoHideScrollbars();
        loadUserDataSummary();

        // Mark description wrappers that actually overflow so the fade gradient shows
        function markOverflowingDescriptions() {{
            document.querySelectorAll(".description-wrapper").forEach(wrapper => {{
                const desc = wrapper.querySelector(".description");
                if (!desc) return;
                // Compare natural text height against the wrapper's current allocated height
                const overflows = desc.scrollHeight > wrapper.clientHeight + 2;
                wrapper.classList.toggle("overflowing", overflows);
            }});
        }}

        // Hover Preview Card logic
        let hoverTimeout = null;
        let activeHoverCard = null;
        let mouseX = 0;
        let mouseY = 0;
        
        function showHoverPreview(card) {{
            const hoverPreview = document.getElementById("hover-preview-card");
            if (!hoverPreview) return;
            
            const name = card.dataset.name || "Unknown Model";
            const versions = JSON.parse(card.dataset.versions || "[]");
            const selectedVer = versions.find(v => v.file_stem === card.dataset.stem) || versions[0];
            
            if (!selectedVer) return;
            
            const verName = selectedVer.version_name || selectedVer.file_stem || "Default";
            const baseModel = selectedVer.base_model || "Unknown";
            const collections = JSON.parse(card.dataset.collections || "[]");
            const collectionsText = collections.join(", ") || "None";
            const rating = parseInt(card.dataset.rating || "0");
            
            let groupCount = 0;
            if (selectedVer.trigger_groups_html) {{
                const tempDiv = document.createElement("div");
                tempDiv.innerHTML = selectedVer.trigger_groups_html;
                groupCount = tempDiv.querySelectorAll(".trigger-group").length;
            }} else if (selectedVer.trigger_words && selectedVer.trigger_words.length > 0) {{
                groupCount = 1;
            }}
            
            let ratingStars = '<span style="color: var(--text-muted);">None</span>';
            if (rating > 0) {{
                ratingStars = 
                    `<span style="color: #fbbf24;">${{ "★".repeat(rating) }}</span>` + 
                    `<span style="color: rgba(255, 255, 255, 0.2);">${{ "☆".repeat(5 - rating) }}</span>`;
            }}
            
            const creator = card.dataset.creator || "";
            
            document.getElementById("hover-preview-title").textContent = name;
            document.getElementById("hover-preview-version").textContent = verName;
            document.getElementById("hover-preview-base-model").textContent = baseModel;
            document.getElementById("hover-preview-collections").textContent = collectionsText;
            document.getElementById("hover-preview-rating").innerHTML = ratingStars;
            document.getElementById("hover-preview-triggers").textContent = `${{groupCount}} Groups`;
            
            const artistRow = document.getElementById("hover-preview-artist-row");
            const artistVal = document.getElementById("hover-preview-artist");
            if (artistRow && artistVal) {{
                if (creator) {{
                    artistRow.style.display = "";
                    artistVal.textContent = creator;
                }} else {{
                    artistRow.style.display = "none";
                }}
            }}
            
            const x = mouseX;
            const y = mouseY;
            
            hoverPreview.style.display = "block";
            
            const previewWidth = hoverPreview.offsetWidth || 280;
            const previewHeight = hoverPreview.offsetHeight || 180;
            
            const viewportWidth = window.innerWidth;
            const viewportHeight = window.innerHeight;
            
            let top = y + window.scrollY + 15;
            let left = x + window.scrollX + 15;
            
            if (x + 15 + previewWidth > viewportWidth) {{
                left = x + window.scrollX - previewWidth - 15;
            }}
            if (y + 15 + previewHeight > viewportHeight) {{
                top = y + window.scrollY - previewHeight - 15;
            }}
            
            if (left < window.scrollX) left = window.scrollX + 10;
            if (top < window.scrollY) top = window.scrollY + 10;
            
            hoverPreview.style.top = `${{top}}px`;
            hoverPreview.style.left = `${{left}}px`;
            
            hoverPreview.classList.add("visible");
        }}
        
        function hideHoverPreview() {{
            const hoverPreview = document.getElementById("hover-preview-card");
            if (!hoverPreview) return;
            hoverPreview.classList.remove("visible");
            setTimeout(() => {{
                if (!hoverPreview.classList.contains("visible")) {{
                    hoverPreview.style.display = "none";
                }}
            }}, 150);
        }}
        
        if (grid) {{
            grid.addEventListener("mousemove", (e) => {{
                mouseX = e.clientX;
                mouseY = e.clientY;
                
                const modal = document.getElementById("details-modal");
                if (modal && modal.classList.contains("active")) return;
                
                const card = e.target.closest(".card");
                if (card) {{
                    const hoverPreview = document.getElementById("hover-preview-card");
                    const isVisible = hoverPreview && hoverPreview.classList.contains("visible");
                    
                    if (!isVisible) {{
                        if (hoverTimeout) {{
                            clearTimeout(hoverTimeout);
                        }}
                        activeHoverCard = card;
                        hoverTimeout = setTimeout(() => {{
                            showHoverPreview(card);
                        }}, 400); // 400ms preview delay (triggered only when mouse stops moving)
                    }}
                }}
            }}, {{ passive: true }});
            
            grid.addEventListener("mouseout", (e) => {{
                const card = e.target.closest(".card");
                if (!card) return;
                
                const related = e.relatedTarget;
                if (!related || !card.contains(related)) {{
                    clearTimeout(hoverTimeout);
                    hoverTimeout = null;
                    activeHoverCard = null;
                    hideHoverPreview();
                }}
            }});
            
            grid.addEventListener("click", () => {{
                clearTimeout(hoverTimeout);
                hoverTimeout = null;
                activeHoverCard = null;
                hideHoverPreview();
            }});
        }}
        
        window.addEventListener("scroll", () => {{
            clearTimeout(hoverTimeout);
            hoverTimeout = null;
            activeHoverCard = null;
            hideHoverPreview();
        }}, {{ passive: true }});

        // Run once on load (after layout settles)
        requestAnimationFrame(() => requestAnimationFrame(markOverflowingDescriptions));
    </script>
    
    <div id="hover-preview-card" class="hover-preview-card" style="display: none; position: absolute; pointer-events: none; z-index: 10000;">
        <div class="hover-preview-content">
            <h4 id="hover-preview-title" class="hover-preview-title"></h4>
            <div class="hover-preview-row"><span class="hover-preview-label">Version:</span><span id="hover-preview-version" class="hover-preview-value"></span></div>
            <div class="hover-preview-row"><span class="hover-preview-label">Base Model:</span><span id="hover-preview-base-model" class="hover-preview-value"></span></div>
            <div id="hover-preview-artist-row" class="hover-preview-row" style="display:none;"><span class="hover-preview-label">Artist:</span><span id="hover-preview-artist" class="hover-preview-value" style="color: #a78bfa;"></span></div>
            <div class="hover-preview-row"><span class="hover-preview-label">Collections:</span><span id="hover-preview-collections" class="hover-preview-value"></span></div>
            <div class="hover-preview-row"><span class="hover-preview-label">Rating:</span><span id="hover-preview-rating" class="hover-preview-value"></span></div>
            <div class="hover-preview-row"><span class="hover-preview-label">Triggers:</span><span id="hover-preview-triggers" class="hover-preview-value"></span></div>
        </div>
    </div>
</body>
</html>
"""

# Save the gallery as a standalone HTML file.
def save_gallery_html(entries: list[dict], output_path: Path) -> Path:
    if is_dry_run():
        log_operation(f"[DRY RUN] Would save gallery HTML to: {output_path}")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        output_file.write(build_gallery_html(entries, output_path))

    return output_path
