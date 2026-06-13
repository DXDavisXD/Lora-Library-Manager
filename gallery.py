"""Gallery data builders for LoRA library previews."""

import html
import json
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


# Load metadata.json or sidecar metadata files from a LoRA model folder if they exist.
def load_metadata(lora_file: Path) -> dict:
    # 1. Try standard metadata.json
    path = get_metadata_path(lora_file)
    if not path.exists():
        # 2. Try <stem>.json
        path = lora_file.parent / f"{lora_file.stem}.json"
    if not path.exists():
        # 3. Try <stem>.civitai.info
        path = lora_file.parent / f"{lora_file.stem}.civitai.info"

    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as metadata_file:
            return json.load(metadata_file)
    except (json.JSONDecodeError, OSError):
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


# Build one HTML card for a gallery entry.
def build_gallery_card(entry: dict, output_path: Path) -> str:
    preview_image = relative_asset_path(entry.get("preview_image", ""), output_path)
    trigger_words = entry.get("trigger_words", [])
    trigger_text = ", ".join(trigger_words)
    
    hidden_search_text = " ".join([
        entry.get("base_model", ""),
        entry.get("model_name", ""),
        entry.get("model_type", ""),
        entry.get("creator", ""),
        " ".join(entry.get("tags", []))
    ]).casefold()
    
    search_text = " ".join(
        [
            entry.get("name", ""),
            entry.get("folder_name", ""),
            trigger_text,
            entry.get("description", ""),
            hidden_search_text,
        ]
    ).casefold()

    if preview_image:
        image_html = (
            f'<img src="{escape_text(preview_image)}" '
            f'alt="{escape_text(entry.get("name", "LoRA preview"))}" loading="lazy">'
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

    name = entry.get("name", "Untitled LoRA")
    folder_name = entry.get("folder_name", "")
    filename = entry.get("filename", "")
    file_stem = entry.get("file_stem", "")
    mtime = entry.get("mtime", 0)
    sha256 = entry.get("sha256", "")
    relative_path = entry.get("relative_path", "")

    # Build Trigger Groups HTML with Backwards Compatibility
    trigger_groups = entry.get("trigger_groups", [])
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

    trigger_section_html = f"""
        <div class="trigger-groups-container">
            {groups_html or '<p class="triggers">None found</p>'}
        </div>
    """

    return f"""
        <article class="card" 
                 data-search="{escape_text(search_text)}" 
                 data-name="{escape_text(name)}" 
                 data-filename="{escape_text(filename)}" 
                 data-stem="{escape_text(file_stem)}" 
                 data-mtime="{mtime}" 
                 data-hash="{escape_text(sha256)}"
                 data-path="{escape_text(relative_path)}"
                 data-triggers="{escape_text(trigger_text)}"
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
                
                <p class="label">Trigger words</p>
                {trigger_section_html}
                
                <p class="label">Description</p>
                <p class="description">{escape_text(entry.get("description", "") or "No description available.")}</p>
                
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


# Build a complete standalone HTML gallery document.
def build_gallery_html(entries: list[dict], output_path: Path) -> str:
    cards = "\n".join(build_gallery_card(entry, output_path) for entry in entries)

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
            border-bottom: 1px solid var(--border);
            padding: 16px 32px;
        }}

        .header-container {{
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}

        @media (min-width: 768px) {{
            .header-container {{
                flex-direction: row;
                justify-content: space-between;
                align-items: center;
            }}
        }}

        h1 {{
            margin: 0;
            font-size: 24px;
            font-weight: 800;
            background: linear-gradient(135deg, #a5b4fc, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.025em;
        }}

        .toolbar {{
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            align-items: center;
        }}

        .search-wrapper {{
            position: relative;
            flex: 1 1 300px;
            max-width: 450px;
        }}

        .search-icon {{
            position: absolute;
            left: 14px;
            top: 50%;
            transform: translateY(-50%);
            width: 18px;
            height: 18px;
            color: var(--text-muted);
            pointer-events: none;
        }}

        #search {{
            width: 100%;
            background: #181b24;
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text);
            padding: 10px 14px 10px 42px;
            font-size: 14px;
            outline: none;
            transition: var(--transition);
        }}

        #search:focus {{
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15);
            background: #1c202b;
        }}

        .btn-toggle-fav {{
            background: #181b24;
            border: 1px solid var(--border);
            color: var(--text);
            border-radius: 8px;
            padding: 10px 16px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: var(--transition);
            outline: none;
        }}

        .btn-toggle-fav:hover, .btn-toggle-fav:focus {{
            border-color: var(--accent);
            background: var(--accent-alpha);
            color: var(--text);
        }}

        .btn-toggle-fav.active {{
            background: rgba(251, 191, 36, 0.1);
            border-color: #fbbf24;
            color: #fbbf24;
        }}

        .btn-toggle-fav svg {{
            width: 16px;
            height: 16px;
            fill: none;
            stroke: currentColor;
            transition: var(--transition);
        }}

        .btn-toggle-fav.active svg {{
            fill: currentColor;
        }}

        .weight-control {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .weight-label {{
            font-size: 11px;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .weight-selector {{
            display: flex;
            background: #181b24;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 2px;
        }}

        .btn-weight {{
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 6px 12px;
            font-size: 13px;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            transition: var(--transition);
            outline: none;
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
            width: 180px;
        }}

        #sort {{
            width: 100%;
            background: #181b24;
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text);
            padding: 10px 32px 10px 14px;
            font-size: 14px;
            outline: none;
            appearance: none;
            -webkit-appearance: none;
            cursor: pointer;
            transition: var(--transition);
        }}

        #sort:focus {{
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
        }}

        .trigger-group {{
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            transition: var(--transition);
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

        @media (max-width: 640px) {{
            header {{
                padding: 16px;
            }}

            main {{
                padding: 16px;
            }}

            .toolbar {{
                width: 100%;
            }}

            .search-wrapper {{
                flex: 1 1 100%;
                max-width: 100%;
            }}

            .select-wrapper {{
                width: 100%;
            }}

            .prompt-builder {{
                width: 100%;
                border-radius: 0;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <div class="header-container">
            <h1>LoRA Library</h1>
            <div class="toolbar">
                <div class="search-wrapper">
                    <svg class="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="11" cy="11" r="8"/>
                        <line x1="21" y1="21" x2="16.65" y2="16.65"/>
                    </svg>
                    <input id="search" type="search" placeholder="Search models, folder, triggers, metadata..." autocomplete="off">
                </div>

                <button id="toggle-fav" class="btn-toggle-fav" aria-label="Show favorites only" type="button">
                    <svg viewBox="0 0 24 24" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/>
                    </svg>
                    <span>Show Favorites Only</span>
                </button>

                <div class="weight-control">
                    <span class="weight-label">Weight</span>
                    <div class="weight-selector">
                        <button type="button" class="btn-weight" data-weight="0.6">0.6</button>
                        <button type="button" class="btn-weight" data-weight="0.8">0.8</button>
                        <button type="button" class="btn-weight" data-weight="1.0">1.0</button>
                        <button type="button" class="btn-weight" data-weight="1.2">1.2</button>
                    </div>
                </div>

                <div class="select-wrapper">
                    <select id="sort" aria-label="Sort models">
                        <option value="name-asc">Alphabetical A-Z</option>
                        <option value="name-desc">Alphabetical Z-A</option>
                        <option value="favorites-first">Favorites First</option>
                        <option value="mtime-desc">Recently Added</option>
                    </select>
                    <svg class="select-arrow" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <polyline points="6 9 12 15 18 9"/>
                    </svg>
                </div>
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

    <!-- Prompt Builder -->
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

        const searchInput = document.getElementById("search");
        const cards = Array.from(document.querySelectorAll(".card"));
        const count = document.getElementById("count");
        const empty = document.getElementById("empty");
        const sortSelect = document.getElementById("sort");
        const grid = document.getElementById("grid");
        const toggleFavBtn = document.getElementById("toggle-fav");

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

        function updateFilter() {{
            const query = searchInput.value.trim().toLowerCase();
            const showFavsOnly = toggleFavBtn.classList.contains("active");
            let visible = 0;

            for (const card of cards) {{
                const matchesSearch = card.dataset.search.includes(query);
                const matchesFav = !showFavsOnly || card.dataset.favorite === "true";
                const matches = matchesSearch && matchesFav;
                
                card.hidden = !matches;

                if (matches) {{
                    visible += 1;
                }}
            }}

            count.textContent = `\${{visible}} / \${{cards.length}}`;
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
            const tag = `<lora:\${{stem}}:\${{currentWeight}}>`;
            
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
            
            if (toggleFavBtn.classList.contains("active")) {{
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
                        <span>\${{escapeHTML(token)}}</span>
                        <button class="btn-remove-token" type="button" aria-label="Remove \${{escapeHTML(token)}}" data-token="\${{escapeHTML(token)}}">×</button>
                    `;
                    container.appendChild(tokenSpan);
                }});
            }}
        }}

        function addToken(stem) {{
            const newToken = `<lora:\${{stem}}:\${{currentWeight}}>`;
            let tokens = getPromptTokens();
            // Filter out any existing token for this specific model stem to prevent duplicate tags
            tokens = tokens.filter(t => !t.startsWith(`<lora:\${{stem}}:`));
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
                searchInput.value = "";
                updateFilter();
                searchInput.blur();
            }}
        }});

        // Event Listeners for controls
        searchInput.addEventListener("input", updateFilter);
        sortSelect.addEventListener("change", performSort);
        toggleFavBtn.addEventListener("click", () => {{
            toggleFavBtn.classList.toggle("active");
            updateFilter();
        }});

        document.querySelectorAll(".btn-weight").forEach(btn => {{
            btn.addEventListener("click", () => setWeight(btn.dataset.weight));
        }});

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
        updateFilter();
        updatePromptBuilderUI();
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
