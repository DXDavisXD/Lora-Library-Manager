"""Script to enrich existing local metadata JSON files with creator details from CivitAI."""

import json
import time
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent))

from config import get_lora_root, log_operation
from civitai import get_model, unwrap_civitai_metadata
from main import refresh_gallery

def enrich_all_metadata():
    root = get_lora_root()
    print(f"Scanning library at: {root}")
    
    # Collect all candidate metadata files
    metadata_files = []
    for p in root.rglob("*.json"):
        if p.name in ("collections.json", "library_index.json", "user_data.json", "settings.json"):
            continue
        metadata_files.append(p)
        
    total_files = len(metadata_files)
    print(f"Found {total_files} local metadata JSON files.")
    
    enriched_count = 0
    skipped_count = 0
    error_count = 0
    
    for idx, p in enumerate(metadata_files, 1):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            unwrapped = unwrap_civitai_metadata(data)
            
            if not isinstance(unwrapped, dict) or not unwrapped.get("modelId"):
                skipped_count += 1
                continue
                
            # If creator is already present, skip
            if "creator" in unwrapped:
                skipped_count += 1
                continue
                
            model_id = unwrapped["modelId"]
            model_name = unwrapped.get("name", p.stem)
            print(f"[{idx}/{total_files}] Fetching creator for '{model_name}' (ID: {model_id})...")
            
            # Request model details from CivitAI
            res = get_model(model_id)
            if res.get("ok") and isinstance(res.get("data"), dict):
                model_data = res["data"]
                creator = model_data.get("creator")
                if creator:
                    unwrapped["creator"] = creator
                    
                    # Save back to file
                    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                    print(f"  -> Enriched with creator: {creator.get('username')}")
                    enriched_count += 1
                else:
                    print("  -> No creator details in API response.")
                    # Still mark it as processed to avoid re-querying next time
                    unwrapped["creator"] = None
                    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
                    skipped_count += 1
            else:
                error_msg = res.get("message") or "Unknown API error"
                print(f"  -> Error fetching details: {error_msg}")
                error_count += 1
                
            # Be polite to the CivitAI API
            time.sleep(0.25)
            
        except Exception as e:
            print(f"[{idx}/{total_files}] Error processing {p.name}: {e}")
            error_count += 1
            
    print("\n--- Enrichment Complete ---")
    print(f"Enriched: {enriched_count}")
    print(f"Skipped:  {skipped_count}")
    print(f"Errors:   {error_count}")
    
    if enriched_count > 0:
        print("\nRegenerating gallery index...")
        # Since cache version 2 is already set, this will rebuild the cache using updated creators
        gallery_path = refresh_gallery(root)
        print(f"Gallery updated: {gallery_path}")

if __name__ == "__main__":
    enrich_all_metadata()
