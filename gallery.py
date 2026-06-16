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
        for f in files:
            if isinstance(f, dict) and f.get("name") == filename:
                return True
        return False
    return True


# Load metadata.json or sidecar metadata files from a LoRA model folder if they exist.
def load_metadata(lora_file: Path) -> dict:
    # 1. Try <stem>.json (specific to this version)
    path = lora_file.parent / f"{lora_file.stem}.json"
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as metadata_file:
                data = json.load(metadata_file)
                if metadata_belongs_to_file(data, lora_file.name):
                    return data
        except Exception:
            pass

    # 2. Try <stem>.civitai.info (specific to this version)
    path = lora_file.parent / f"{lora_file.stem}.civitai.info"
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as metadata_file:
                data = json.load(metadata_file)
                if metadata_belongs_to_file(data, lora_file.name):
                    return data
        except Exception:
            pass

    # 3. Try standard metadata.json
    path = get_metadata_path(lora_file)
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as metadata_file:
                data = json.load(metadata_file)
                if metadata_belongs_to_file(data, lora_file.name):
                    return data
        except Exception:
            pass

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
def build_gallery_index(library_dir: Path) -> list[dict]:
    if not library_dir.exists():
        return []

    entries = []
    lora_files = []
    for ext in LORA_EXTENSIONS:
        for path in library_dir.rglob(f"*{ext}"):
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

    for path in unique_files:
        entries.append(build_gallery_entry(path))

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
        words_text = ", ".join(words)
        
        list_items = "".join(f"<li>{escape_text(w)}</li>" for w in words if w.strip())
        esc_title = escape_text(title)
        esc_words_text = escape_text(words_text)
        
        groups_html += f"""
        <div class="trigger-group" data-group-words="{esc_words_text}">
            <div class="trigger-group-header" role="button" aria-expanded="false" tabindex="0" aria-label="Toggle {esc_title} triggers">
                <span class="trigger-group-arrow">▶</span>
                <span class="trigger-group-title">{esc_title}</span>
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
            "description": clean_html_text(ver["description"]),
            "sha256": ver["sha256"],
            "relative_path": ver["relative_path"],
            "mtime": ver["mtime"]
        })
        
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
                  data-description="{escape_text(desc_clean)}"
                  data-tags="{escape_text(json.dumps(group.get('tags', [])))}"
                  data-official-triggers="{escape_text(json.dumps(trigger_words))}"
                  data-versions="{escape_text(versions_json)}"
                  data-favorite="false">
            <div class="preview">
                {image_html}
                <button class="btn-favorite" aria-label="Toggle favorite" type="button">
                    <svg class="star-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
                    </svg>
                </button>

                <div class="preview-feedback">Triggers Copied!</div>
            </div>
            <div class="content">
                <h2>{escape_text(name)}</h2>
                <p class="folder">{escape_text(folder_name)}</p>
                
                {version_selector_html}
                
                <p class="label">Trigger words</p>
                {trigger_section_html}
                
                <p class="label">Description</p>
                <p class="description">{escape_text(desc_clean or "No description available.")}</p>
                
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
    grouped = {}
    for entry in entries:
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
            display: block;
            transition: var(--transition);
        }}

        .card:hover .preview img {{
            transform: scale(1.05);
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
            flex-grow: 1;
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

        .description {{
            margin: 0;
            font-size: 13px;
            color: var(--text-muted);
            line-height: 1.5;
            max-height: 90px;
            overflow-y: auto;
            padding-right: 4px;
        }}

        .card-footer {{
            margin-top: auto;
            padding-top: 16px;
            display: flex;
            gap: 8px;
        }}

        .btn-copy, .btn-add-prompt {{
            flex: 1;
            background: var(--accent-alpha);
            border: 1px solid rgba(99, 102, 241, 0.2);
            border-radius: 8px;
            color: #a5b4fc;
            padding: 10px 12px;
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
            max-height: 160px;
            overflow-y: auto;
            padding-right: 4px;
            scrollbar-width: thin;
            scrollbar-color: rgba(255, 255, 255, 0.15) transparent;
        }}

        .trigger-groups-container::-webkit-scrollbar {{
            width: 4px;
        }}

        .trigger-groups-container::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.15);
            border-radius: 2px;
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

        .trigger-list::-webkit-scrollbar {{
            width: 4px;
        }}
        .trigger-list::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
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
            max-height: 250px;
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
            top: 16px;
            right: 16px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.05);
            color: var(--text-muted);
            font-size: 24px;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10;
            transition: var(--transition);
            outline: none;
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
            padding: 24px;
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            gap: 16px;
            background: rgba(0, 0, 0, 0.15);
            overflow-y: auto;
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
        }}

        .modal-preview-container img {{
            max-width: 100%;
            max-height: 480px;
            height: auto;
            width: auto;
            display: block;
            object-fit: contain;
        }}

        /* Webkit scrollbar customizations for modal elements */
        .modal-left::-webkit-scrollbar,
        .modal-right::-webkit-scrollbar,
        .modal-description-text::-webkit-scrollbar,
        .triggers-list-read::-webkit-scrollbar,
        .triggers-list-edit::-webkit-scrollbar,
        .collections-checkbox-list::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}
        .modal-left::-webkit-scrollbar-track,
        .modal-right::-webkit-scrollbar-track,
        .modal-description-text::-webkit-scrollbar-track,
        .triggers-list-read::-webkit-scrollbar-track,
        .triggers-list-edit::-webkit-scrollbar-track,
        .collections-checkbox-list::-webkit-scrollbar-track {{
            background: transparent;
        }}
        .modal-left::-webkit-scrollbar-thumb,
        .modal-right::-webkit-scrollbar-thumb,
        .modal-description-text::-webkit-scrollbar-thumb,
        .triggers-list-read::-webkit-scrollbar-thumb,
        .triggers-list-edit::-webkit-scrollbar-thumb,
        .collections-checkbox-list::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.12);
            border-radius: 3px;
        }}
        .modal-left::-webkit-scrollbar-thumb:hover,
        .modal-right::-webkit-scrollbar-thumb:hover,
        .modal-description-text::-webkit-scrollbar-thumb:hover,
        .triggers-list-read::-webkit-scrollbar-thumb:hover,
        .triggers-list-edit::-webkit-scrollbar-thumb:hover,
        .collections-checkbox-list::-webkit-scrollbar-thumb:hover {{
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
            gap: 20px;
            overflow-y: auto;
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
            font-size: 12px;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        /* Description Box */
        .modal-description-text {{
            font-size: 13px;
            color: var(--text-muted);
            max-height: 120px;
            overflow-y: auto;
            background: rgba(255, 255, 255, 0.01);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 14px;
            line-height: 1.6;
            scrollbar-width: thin;
            scrollbar-color: rgba(255, 255, 255, 0.15) transparent;
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
            border-radius: 4px;
            font-weight: 600;
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

        /* Collections Checkbox List styling */
        .collections-checkbox-list {{
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

        .collection-checkbox-label {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 11px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border);
            padding: 4px 10px;
            border-radius: 20px;
            cursor: pointer;
            transition: var(--transition);
            user-select: none;
            color: var(--text-muted);
            font-weight: 500;
        }}

        .collection-checkbox-label:hover {{
            background: rgba(255, 255, 255, 0.08);
            border-color: rgba(255, 255, 255, 0.2);
            color: var(--text);
        }}

        .collection-checkbox-label input[type="checkbox"] {{
            margin: 0;
            cursor: pointer;
            accent-color: var(--accent);
        }}

        .collection-checkbox-label.active {{
            background: var(--accent-alpha);
            border-color: rgba(99, 102, 241, 0.4);
            color: #a5b4fc;
        }}

        .collection-checkbox-label .btn-delete-collection-global {{
            background: none;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            font-size: 13px;
            padding: 0 2px;
            margin-left: 4px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            line-height: 1;
            transition: color 0.15s ease;
        }}
        
        .collection-checkbox-label .btn-delete-collection-global:hover {{
            color: #ef4444;
        }}

        /* Input forms */
        .add-collection-form, .add-trigger-form {{
            display: flex;
            gap: 8px;
            margin-top: 4px;
        }}

        .add-collection-form input, .add-trigger-form input {{
            flex: 1;
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: 6px;
            color: var(--text);
            padding: 6px 10px;
            font-size: 13px;
            outline: none;
        }}

        .add-collection-form input:focus, .add-trigger-form input:focus {{
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
            border-radius: 4px;
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
            padding: 4px 10px;
            border-radius: 20px;
            font-family: monospace;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            cursor: pointer;
            transition: var(--transition);
        }}

        .trigger-tag-edit:hover {{
            background: rgba(239, 68, 68, 0.12);
            border-color: rgba(239, 68, 68, 0.3);
            color: #ef4444;
            transform: translateY(-1px);
        }}

        .trigger-tag-edit::after {{
            content: "×";
            font-weight: bold;
        }}

        .grid-2-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }}

        /* Save / Copy Action Bars */
        .modal-actions-bar {{
            display: flex;
            gap: 8px;
        }}

        .modal-actions-bar button {{
            flex: 1;
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

        @media (max-width: 768px) {{
            .modal-body {{
                flex-direction: column;
                overflow-y: auto;
            }}
            .modal-left {{
                width: 100%;
                border-right: none;
                border-bottom: 1px solid var(--border);
            }}
            .modal-right {{
                width: 100%;
            }}
            .modal-container {{
                max-height: 90vh;
                height: 90vh;
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

    <!-- Details Modal -->
    <div id="details-modal" class="modal-overlay" aria-hidden="true" role="dialog" aria-labelledby="modal-title">
        <div class="modal-container">
            <button class="modal-close" aria-label="Close modal">&times;</button>
            <div class="modal-body">
                <!-- Left Side: Preview & Details -->
                <div class="modal-left">
                    <div class="modal-preview-container">
                        <img id="modal-image" src="" alt="LoRA Preview">
                    </div>
                    <div class="modal-meta-rows">
                        <div class="meta-row">
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
                    <div class="modal-actions-bar">
                        <button id="modal-btn-copy-tag" class="btn-action-primary">Copy Tag</button>
                        <button id="modal-btn-add-prompt" class="btn-action-secondary">Add to Prompt</button>
                        <a id="modal-btn-civitai" class="btn-action-civitai" target="_blank" rel="noopener noreferrer">View on CivitAI</a>
                    </div>
                </div>
                <!-- Right Side: Edit Panel -->
                <div class="modal-right">
                    <h2 id="modal-title">Model Name</h2>
                    
                    <div class="modal-section">
                        <h3>Description</h3>
                        <div id="modal-description" class="modal-description-text"></div>
                    </div>

                    <!-- Tags Section (dynamic display) -->
                    <div class="modal-section" id="modal-section-tags">
                        <h3>Tags</h3>
                        <div id="modal-tags" class="modal-tags-container"></div>
                    </div>
                    
                    <!-- Grid 2-col for Rating/Collections and Notes -->
                    <div class="grid-2-col">
                        <div style="display: flex; flex-direction: column; gap: 16px;">
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
                                <h3>Collections</h3>
                                <div class="collections-container">
                                    <div id="modal-collections-list" class="collections-checkbox-list"></div>
                                    <div class="add-collection-form">
                                        <input type="text" id="input-new-collection" placeholder="New collection...">
                                        <button id="btn-add-collection" class="btn-mini" type="button">Add</button>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Notes Column -->
                        <div class="modal-section" style="display: flex; flex-direction: column;">
                            <h3>Notes</h3>
                            <textarea id="modal-notes" placeholder="Looks best around 0.8 weight. Add notes..." style="flex: 1; min-height: 120px;"></textarea>
                        </div>
                    </div>
                    
                    <!-- Triggers -->
                    <div class="modal-section grid-2-col">
                        <div>
                            <h3>Official Triggers</h3>
                            <div id="modal-official-triggers" class="trigger-groups-container" style="max-height: 240px; border: none; background: transparent; padding: 0;"></div>
                        </div>
                        <div>
                            <h3>Personal Triggers</h3>
                            <div id="modal-personal-triggers-container" class="trigger-groups-container" style="max-height: 240px; border: none; background: transparent; padding: 0;">
                                <div class="trigger-group expanded" id="personal-trigger-group">
                                    <div class="trigger-group-header" role="button" aria-expanded="true" tabindex="0" aria-label="Toggle Personal Triggers">
                                        <span class="trigger-group-arrow">▶</span>
                                        <span class="trigger-group-title">Personal Triggers</span>
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
                                            <div id="modal-personal-triggers" class="triggers-list-edit" style="border:none; background:transparent; padding:8px 12px 12px 12px; max-height:none; overflow:visible; display:flex; flex-wrap:wrap; gap:6px;">
                                                <!-- pills here -->
                                            </div>
                                            <div class="add-trigger-form" style="padding: 0 12px 12px 12px;">
                                                <input type="text" id="input-new-trigger" placeholder="New trigger...">
                                                <button id="btn-add-trigger" class="btn-mini" type="button">Add</button>
                                            </div>
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
            </div>
        </div>
    </div>

    <script>
        const FAVORITES_KEY = "lora_gallery_favorites";
        const WEIGHT_KEY = "lora_gallery_selected_weight";

        let currentWeight = localStorage.getItem(WEIGHT_KEY) || "1.0";
        let activeCollection = "Library"; // Default state

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
            
            let visible = 0;

            for (const card of cards) {{
                const matchesSearch = card.dataset.search.includes(query);
                
                // Matches favorites if activeCollection is "Favorites"
                const matchesFav = (activeCollection !== "Favorites") || card.dataset.favorite === "true";
                
                let matchesCollection = true;
                if (activeCollection !== "Library" && activeCollection !== "Favorites") {{
                    try {{
                        const collections = JSON.parse(card.dataset.collections || "[]");
                        // Case-insensitive collection check
                        matchesCollection = collections.some(c => c.toLowerCase() === activeCollection.toLowerCase());
                    }} catch (e) {{
                        matchesCollection = false;
                    }}
                }}
                
                let matchesBaseModel = true;
                if (selectedBaseModel) {{
                    const cardBaseModel = normalizeBaseModel(card.dataset.baseModel);
                    matchesBaseModel = (cardBaseModel === selectedBaseModel);
                }}
                
                const matches = matchesSearch && matchesFav && matchesCollection && matchesBaseModel;
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
            }}
            
            // Update Description display
            const descEl = card.querySelector(".description");
            if (descEl) {{
                descEl.textContent = selectedVer.description || "No description available.";
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
            }}).catch(err => {{
                console.error("Prompt copy failed:", err);
            }});
        }});

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
        let globalCollections = [];

        // Load global collections from API
        function loadCollections() {{
            return fetch("/api/collections")
                .then(res => res.json())
                .then(data => {{
                    globalCollections = data.collections || [];
                    updateCollectionFilterOptions();
                }})
                .catch(err => console.error("Error loading collections:", err));
        }}

        // Open details modal
        function openDetailsModal(card, userData) {{
            currentModalCard = card;
            
            // Extract static details
            const name = card.dataset.name;
            const filename = card.dataset.filename;
            const stem = card.dataset.stem;
            const path = card.dataset.path;
            const creator = card.dataset.creator || "Unknown";
            const baseModel = card.dataset.baseModel || "Unknown";
            const description = card.dataset.description || "No description available.";
            const tags = JSON.parse(card.dataset.tags || "[]");

            // Extract user data
            currentPersonalTriggers = userData.personal_trigger_words || [];
            currentCollections = userData.collections || [];
            currentRating = userData.rating || 0;
            const notes = userData.notes || "";

            // Populate fields
            document.getElementById("modal-title").textContent = name;
            document.getElementById("modal-creator").textContent = creator;
            document.getElementById("modal-base-model").textContent = baseModel;
            document.getElementById("modal-filename").textContent = filename;
            document.getElementById("modal-path").textContent = path;
            document.getElementById("modal-description").innerHTML = description;
            document.getElementById("modal-notes").value = notes;

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

            // Populate CivitAI link
            const select = card.querySelector(".version-select");
            const versions = JSON.parse(card.dataset.versions || "[]");
            const selectedVer = select ? versions.find(v => v.file_stem === select.value) : versions[0];
            const civitaiLink = document.getElementById("modal-btn-civitai");
            if (civitaiLink) {{
                const modelId = selectedVer ? selectedVer.model_id : null;
                const versionId = selectedVer ? selectedVer.version_id : null;
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
            const cardImg = card.querySelector(".preview img");
            if (cardImg) {{
                imgEl.style.display = "block";
                imgEl.src = cardImg.src;
            }} else {{
                imgEl.style.display = "none";
            }}

            // Grouped Official Triggers
            const offTrigContainer = document.getElementById("modal-official-triggers");
            offTrigContainer.innerHTML = "";
            if (selectedVer && selectedVer.trigger_groups_html) {{
                offTrigContainer.innerHTML = selectedVer.trigger_groups_html;
            }} else {{
                offTrigContainer.innerHTML = '<span class="text-muted" style="font-size:12px; padding: 4px;">None</span>';
            }}

            // Render interactive lists
            renderPersonalTriggers();
            renderCollections();
            renderRating(currentRating);

            // Re-bind Action buttons
            const copyTagBtn = document.getElementById("modal-btn-copy-tag");
            const addPromptBtn = document.getElementById("modal-btn-add-prompt");

            const newCopyBtn = copyTagBtn.cloneNode(true);
            const newAddBtn = addPromptBtn.cloneNode(true);
            copyTagBtn.parentNode.replaceChild(newCopyBtn, copyTagBtn);
            addPromptBtn.parentNode.replaceChild(newAddBtn, addPromptBtn);

            newCopyBtn.addEventListener("click", () => {{
                const tag = `<lora:${{stem}}:${{currentWeight}}>`;
                navigator.clipboard.writeText(tag).then(() => {{
                    newCopyBtn.textContent = "Copied!";
                    setTimeout(() => newCopyBtn.textContent = "Copy Tag", 1500);
                }});
            }});

            newAddBtn.addEventListener("click", () => {{
                addToken(stem);
                newAddBtn.textContent = "Added!";
                setTimeout(() => newAddBtn.textContent = "Add to Prompt", 1500);
            }});

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
            container.innerHTML = "";
            if (currentPersonalTriggers.length === 0) {{
                container.innerHTML = '<span class="text-muted" style="font-size:12px; padding: 4px;">No personal triggers.</span>';
            }} else {{
                currentPersonalTriggers.forEach((trig, idx) => {{
                    const span = document.createElement("span");
                    span.className = "trigger-tag-edit";
                    span.textContent = trig;
                    span.addEventListener("click", () => {{
                        currentPersonalTriggers.splice(idx, 1);
                        renderPersonalTriggers();
                        autoSaveUserData();
                    }});
                    container.appendChild(span);
                }});
            }}
            
            // Sync words in the personal trigger group attribute for group copy button
            const groupDiv = document.getElementById("personal-trigger-group");
            if (groupDiv) {{
                groupDiv.dataset.groupWords = currentPersonalTriggers.join(", ");
            }}
        }}

        function renderCollections() {{
            const container = document.getElementById("modal-collections-list");
            container.innerHTML = "";
            
            if (globalCollections.length === 0) {{
                container.innerHTML = '<span class="text-muted" style="font-size:12px; padding: 4px;">No collections defined.</span>';
                return;
            }}
            
            globalCollections.forEach(coll => {{
                const label = document.createElement("label");
                label.className = "collection-checkbox-label";
                
                const checkbox = document.createElement("input");
                checkbox.type = "checkbox";
                checkbox.value = coll;
                
                const isMember = currentCollections.includes(coll);
                checkbox.checked = isMember;
                if (isMember) {{
                    label.classList.add("active");
                }}
                
                checkbox.addEventListener("change", () => {{
                    if (checkbox.checked) {{
                        if (!currentCollections.includes(coll)) {{
                            currentCollections.push(coll);
                        }}
                        label.classList.add("active");
                    }} else {{
                        currentCollections = currentCollections.filter(c => c !== coll);
                        label.classList.remove("active");
                    }}
                    autoSaveUserData();
                }});
                
                label.appendChild(checkbox);
                label.appendChild(document.createTextNode(" " + coll));
                
                // Add Delete button
                const delBtn = document.createElement("button");
                delBtn.type = "button";
                delBtn.className = "btn-delete-collection-global";
                delBtn.innerHTML = "&times;";
                delBtn.title = "Delete collection globally";
                delBtn.addEventListener("click", (e) => {{
                    e.preventDefault();
                    e.stopPropagation();
                    
                    if (confirm(`Are you sure you want to delete the collection "${{coll}}" globally?\n\nThis will remove it from all models in the library.`)) {{
                        fetch("/api/delete_collection", {{
                            method: "POST",
                            headers: {{
                                "Content-Type": "application/json"
                            }},
                            body: JSON.stringify({{ name: coll }})
                        }})
                        .then(res => {{
                            if (!res.ok) throw new Error("Failed to delete collection globally");
                            return res.json();
                        }})
                        .then(data => {{
                            globalCollections = data.collections || [];
                            currentCollections = currentCollections.filter(c => c.toLowerCase() !== coll.toLowerCase());
                            
                            // Update all cards' collections datasets in-memory
                            cards.forEach(card => {{
                                try {{
                                    const cardColls = JSON.parse(card.dataset.collections || "[]");
                                    const updatedCardColls = cardColls.filter(c => c.toLowerCase() !== coll.toLowerCase());
                                    card.dataset.collections = JSON.stringify(updatedCardColls);
                                }} catch (err) {{}}
                            }});
                            
                            renderCollections();
                            updateCollectionFilterOptions();
                            updateFilter();
                            showToast(`Collection "${{coll}}" deleted globally.`);
                        }})
                        .catch(err => {{
                            console.error("Error deleting collection globally:", err);
                            alert("Failed to delete collection globally: " + err.message);
                        }});
                    }}
                }});
                label.appendChild(delBtn);
                
                container.appendChild(label);
            }});
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

        // Add Collection handlers
        document.getElementById("btn-add-collection").addEventListener("click", () => {{
            const input = document.getElementById("input-new-collection");
            const val = input.value.trim();
            if (!val) return;
            
            // Case-insensitive check in global collections
            const existsVal = globalCollections.find(g => g.toLowerCase() === val.toLowerCase());
            
            if (existsVal) {{
                // Collection already exists globally, just select it if not already selected
                if (!currentCollections.includes(existsVal)) {{
                    currentCollections.push(existsVal);
                }}
                renderCollections();
                input.value = "";
                autoSaveUserData();
            }} else {{
                // Save globally
                fetch("/api/collections", {{
                    method: "POST",
                    headers: {{
                        "Content-Type": "application/json"
                    }},
                    body: JSON.stringify({{ name: val }})
                }})
                .then(res => {{
                    if (!res.ok) throw new Error("Failed to save collection globally");
                    return res.json();
                }})
                .then(data => {{
                    globalCollections = data.collections || [];
                    // Find actual casing returned by server
                    const addedColl = globalCollections.find(g => g.toLowerCase() === val.toLowerCase()) || val;
                    if (!currentCollections.includes(addedColl)) {{
                        currentCollections.push(addedColl);
                    }}
                    renderCollections();
                    updateCollectionFilterOptions();
                    autoSaveUserData();
                }})
                .catch(err => {{
                    console.error("Error creating collection:", err);
                    // Fallback to local only if server fails
                    if (!currentCollections.includes(val)) {{
                        currentCollections.push(val);
                    }}
                    if (!globalCollections.includes(val)) {{
                        globalCollections.push(val);
                    }}
                    renderCollections();
                    updateCollectionFilterOptions();
                    autoSaveUserData();
                }});
                input.value = "";
            }}
        }});
        document.getElementById("input-new-collection").addEventListener("keydown", (e) => {{
            if (e.key === "Enter") {{
                e.preventDefault();
                document.getElementById("btn-add-collection").click();
            }}
        }});

        // Copy Combined Triggers handler
        document.getElementById("modal-btn-copy-all-triggers").addEventListener("click", () => {{
            if (!currentModalCard) return;
            const officialTriggers = JSON.parse(currentModalCard.dataset.officialTriggers || "[]");
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
            
            const relPath = currentModalCard.dataset.path;
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
                // Update datasets in-memory
                currentModalCard.dataset.collections = JSON.stringify(currentCollections);
                currentModalCard.dataset.rating = currentRating;
                
                // Redraw indicators
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
        function renderCollectionsPopupList(filterText = "") {{
            const listContainer = document.getElementById("popup-collections-list");
            if (!listContainer) return;

            listContainer.innerHTML = "";
            const lowerFilter = filterText.trim().toLowerCase();

            // Virtual collections pinned at the top
            const virtualItems = [
                {{ name: "Library", icon: "📚" }},
                {{ name: "Favorites", icon: "⭐" }}
            ];

            const filteredVirtual = virtualItems.filter(item => 
                item.name.toLowerCase().includes(lowerFilter)
            );

            // Sort user collections alphabetically
            const sortedUserCollections = [...globalCollections].sort((a, b) => a.localeCompare(b));
            const filteredUser = sortedUserCollections.filter(c => 
                c.toLowerCase().includes(lowerFilter)
            );

            const allItems = [
                ...filteredVirtual.map(v => ({{ ...v, isVirtual: true }})),
                ...filteredUser.map(c => ({{ name: c, icon: "📁", isVirtual: false }}))
            ];

            if (allItems.length === 0) {{
                const noResult = document.createElement("div");
                noResult.className = "popup-item text-muted";
                noResult.style.justifyContent = "center";
                noResult.style.fontSize = "12px";
                noResult.textContent = "No collections found";
                listContainer.appendChild(noResult);
                return;
            }}

            allItems.forEach((item, index) => {{
                const div = document.createElement("div");
                div.className = "popup-item";
                if (item.name === activeCollection) {{
                    div.classList.add("active");
                }}
                div.dataset.name = item.name;
                div.dataset.index = index;
                
                div.innerHTML = `<span style="margin-right: 8px; flex-shrink: 0;">${{item.icon}}</span> <span style="text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">${{item.name}}</span>`;
                
                div.addEventListener("click", () => {{
                    selectCollection(item.name);
                }});
                
                listContainer.appendChild(div);
            }});
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

            if (activeCollection === "Library") {{
                iconSpan.textContent = "📚";
                textSpan.textContent = "Library";
                
                const arrowSpan = document.createElement("span");
                arrowSpan.className = "chip-arrow";
                arrowSpan.textContent = "▼";
                chip.appendChild(arrowSpan);
            }} else {{
                chip.classList.add("active-filter");
                
                if (activeCollection === "Favorites") {{
                    iconSpan.textContent = "⭐";
                    textSpan.textContent = "Favorites";
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
            renderCollectionsPopupList(popupSearchInput ? popupSearchInput.value : "");
            
            // Check if activeCollection is still valid (e.g. if it was deleted)
            if (activeCollection !== "Library" && activeCollection !== "Favorites" && !globalCollections.includes(activeCollection)) {{
                activeCollection = "Library";
            }}
            
            // Update the chip to display the current selection
            updateCollectionChip();
        }}

        let focusedItemIndex = -1;

        function getVisiblePopupItems() {{
            const listContainer = document.getElementById("popup-collections-list");
            if (!listContainer) return [];
            return Array.from(listContainer.querySelectorAll(".popup-item:not(.text-muted)"));
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
                renderCollectionsPopupList("");
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
            
            // Ignore click if they click an interactive element inside preview (like the favorite button)
            if (e.target.closest("button") || e.target.closest("a")) {{
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

        // Event Listeners for controls
        searchInput.addEventListener("input", updateFilter);
        sortSelect.addEventListener("change", performSort);
        
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
                            selectCollection(selectedItem.dataset.name);
                        }}
                    }} else if (items.length > 0) {{
                        selectCollection(items[0].dataset.name);
                    }}
                }} else if (e.key === "Escape") {{
                    e.preventDefault();
                    closeCollectionsPopup();
                }}
            }});

            popupSearchInput.addEventListener("input", (e) => {{
                focusedItemIndex = -1;
                renderCollectionsPopupList(e.target.value);
            }});
        }}

        // Base Model filter listener
        const filterBaseModelSelect = document.getElementById("filter-base-model");
        if (filterBaseModelSelect) {{
            filterBaseModelSelect.addEventListener("change", updateFilter);
        }}

        // Auto-save notes when textarea loses focus or is changed
        const notesTextarea = document.getElementById("modal-notes");
        if (notesTextarea) {{
            notesTextarea.addEventListener("change", autoSaveUserData);
            notesTextarea.addEventListener("blur", autoSaveUserData);
        }}

        // Open parent folder handler
        const openFolderBtn = document.getElementById("modal-btn-open-folder");
        if (openFolderBtn) {{
            openFolderBtn.addEventListener("click", () => {{
                if (!currentModalCard) return;
                const path = currentModalCard.dataset.path;
                
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
        loadUserDataSummary();
    </script>
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
