"""Lightweight HTTP server and API endpoints for LoRA Library Manager."""

import http.server
import socketserver
import json
import urllib.parse
from datetime import datetime
from pathlib import Path
import os
import sys

from config import get_lora_root, is_lora_file, LORA_EXTENSIONS, PREVIEW_EXTENSIONS, log_operation

# ---------------------------------------------------------------------------
# In-memory TTL cache for the user-data summary endpoint.
# The summary is computed by an rglob scan across all user_data.json files,
# which is expensive on large libraries.  We cache the result for
# _USER_DATA_CACHE_TTL seconds and invalidate it whenever a save occurs.
# ---------------------------------------------------------------------------
import time as _time

_USER_DATA_CACHE_TTL: float = 30.0   # seconds
_user_data_cache: dict = {}           # {"data": {...}, "computed_at": float}

class GalleryHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, lora_root: Path = None, **kwargs):
        self.lora_root = lora_root or get_lora_root()
        super().__init__(*args, **kwargs)

    def translate_path(self, path):
        # Serve files relative to lora_root instead of the current working directory
        parsed_url = urllib.parse.urlparse(path)
        path_str = urllib.parse.unquote(parsed_url.path)
        
        # Prevent directory traversal by sanitizing path components
        cleaned_path = Path(path_str.lstrip("/"))
        resolved_path = (self.lora_root / cleaned_path).resolve()
        
        # If the resolved path goes outside lora_root, fallback or deny access
        if not resolved_path.is_relative_to(self.lora_root.resolve()):
            return str(self.lora_root / "gallery.html")
            
        return str(resolved_path)

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        if path == "/api/user_data_summary":
            self.handle_get_summary()
        elif path == "/api/get_user_data":
            relative_path = query.get("relative_path", [None])[0]
            self.handle_get_user_data(relative_path)
        elif path == "/api/collections":
            self.handle_get_collections()
        else:
            # Fall back to serving static files
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/save_user_data":
            self.handle_save_user_data()
        elif self.path == "/api/delete_model_version":
            self.handle_delete_model_version()
        elif self.path == "/api/collections":
            self.handle_add_collection()
        elif self.path == "/api/delete_collection":
            self.handle_delete_collection()
        elif self.path == "/api/open_folder":
            self.handle_open_folder()
        else:
            self.send_error_response(404, "Not Found")

    def handle_get_summary(self):
        global _user_data_cache
        try:
            now = _time.monotonic()
            # Return cached result when it is still fresh
            if (
                _user_data_cache
                and "computed_at" in _user_data_cache
                and (now - _user_data_cache["computed_at"]) < _USER_DATA_CACHE_TTL
            ):
                self.send_success_response(_user_data_cache["data"])
                return

            # Cache stale or empty — rebuild from disk
            summary = {}
            for path in self.lora_root.rglob("user_data.json"):
                try:
                    with path.open("r", encoding="utf-8") as f:
                        data = json.load(f)

                    personal_triggers = data.get("personal_trigger_words", [])
                    notes = data.get("notes", "")
                    rating = data.get("rating", 0)
                    collections = data.get("collections", [])

                    # Only include if there is actual user content
                    if not (personal_triggers or notes.strip() or rating > 0 or collections):
                        continue

                    entry = {
                        "has_notes": bool(notes.strip()),
                        "has_personal_triggers": bool(personal_triggers),
                        "rating": rating,
                        "collections": collections
                    }

                    # Assign entry to ALL model files in the same folder
                    # (one user_data.json covers every version/file in that folder)
                    found_any = False
                    for ext in LORA_EXTENSIONS:
                        for f_in_dir in path.parent.glob(f"*{ext}"):
                            if is_lora_file(f_in_dir):
                                rel = f_in_dir.resolve().relative_to(self.lora_root.resolve()).as_posix()
                                summary[rel] = entry
                                found_any = True

                except Exception:
                    continue

            # Store in cache
            _user_data_cache = {"data": summary, "computed_at": _time.monotonic()}
            self.send_success_response(summary)
        except Exception as e:
            self.send_error_response(500, f"Error gathering user data summary: {e}")

    def load_collections_list(self) -> list:
        collections_file = self.lora_root / "collections.json"
        if collections_file.exists():
            try:
                with collections_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
            except Exception as e:
                log_operation(f"Error loading collections.json: {e}")
        
        # Fallback: scan existing user_data.json files recursively to build initial list
        log_operation("collections.json not found or invalid. Scanning library user_data.json files...")
        collections_set = set()
        for path in self.lora_root.rglob("user_data.json"):
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                colls = data.get("collections", [])
                for c in colls:
                    if c and c.strip():
                        c_clean = c.strip()
                        if c_clean.lower() == "favorites":
                            c_clean = "Favorites"
                        elif c_clean.lower() == "library":
                            c_clean = "Library"
                        
                        if c_clean not in ("Library", "Favorites"):
                            # Case-insensitive deduplication: keep first casing encountered
                            exists = False
                            for existing in collections_set:
                                if existing.lower() == c_clean.lower():
                                    exists = True
                                    break
                            if not exists:
                                collections_set.add(c_clean)
            except Exception:
                continue
        
        # Save to collections.json
        collections_list = sorted(list(collections_set))
        self.save_collections_list(collections_list)
        return collections_list

    def save_collections_list(self, collections_list: list):
        collections_file = self.lora_root / "collections.json"
        try:
            with collections_file.open("w", encoding="utf-8") as f:
                json.dump(collections_list, f, indent=2, ensure_ascii=False)
            log_operation(f"Saved collections.json with {len(collections_list)} entries.")
        except Exception as e:
            log_operation(f"Error saving collections.json: {e}")

    def handle_get_collections(self):
        try:
            collections = self.load_collections_list()
            self.send_success_response({"collections": collections})
        except Exception as e:
            self.send_error_response(500, f"Error loading collections: {e}")

    def handle_add_collection(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)
            new_name = data.get("name", "").strip()
            
            if not new_name:
                self.send_error_response(400, "Collection name cannot be empty")
                return
                
            # Normalize Library & Favorites virtual collections
            if new_name.lower() == "favorites":
                new_name = "Favorites"
            elif new_name.lower() == "library":
                new_name = "Library"

            collections = self.load_collections_list()
            
            # Case-insensitive uniqueness check
            exists = False
            for c in collections:
                if c.lower() == new_name.lower():
                    exists = True
                    break
                    
            if not exists and new_name not in ("Library", "Favorites"):
                collections.append(new_name)
                # Keep it sorted
                collections.sort(key=lambda x: x.lower())
                self.save_collections_list(collections)
                
            self.send_success_response({"collections": collections})
        except Exception as e:
            self.send_error_response(500, f"Error adding collection: {e}")

    def handle_delete_collection(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)
            target_name = data.get("name", "").strip()
            
            if not target_name:
                self.send_error_response(400, "Collection name cannot be empty")
                return
                
            collections = self.load_collections_list()
            updated_collections = [c for c in collections if c.lower() != target_name.lower()]
            self.save_collections_list(updated_collections)
            
            # Now update all user_data.json files recursively
            for path in self.lora_root.rglob("user_data.json"):
                try:
                    with path.open("r", encoding="utf-8") as f:
                        user_data = json.load(f)
                    
                    colls = user_data.get("collections", [])
                    has_target = False
                    for c in colls:
                        if c.lower() == target_name.lower():
                            has_target = True
                            break
                            
                    if has_target:
                        new_colls = [c for c in colls if c.lower() != target_name.lower()]
                        user_data["collections"] = new_colls
                        user_data["last_modified"] = datetime.now().isoformat() + "Z"
                        
                        # Check if it has any other content
                        personal_triggers = user_data.get("personal_trigger_words", [])
                        notes = user_data.get("notes", "")
                        rating = user_data.get("rating", 0)
                        
                        has_content = bool(personal_triggers or notes.strip() or rating > 0 or new_colls)
                        if not has_content:
                            try:
                                path.unlink()
                                log_operation(f"Deleted empty user data sidecar after deleting collection: {path}")
                            except OSError as e:
                                log_operation(f"Error deleting user data sidecar {path}: {e}")
                        else:
                            with path.open("w", encoding="utf-8") as f:
                                json.dump(user_data, f, indent=2)
                            log_operation(f"Updated user data sidecar (removed collection): {path}")
                except Exception as e:
                    log_operation(f"Error cleaning collection from sidecar {path}: {e}")
                    continue
            
            # Trigger background refresh of gallery HTML to reflect the updated metadata
            import threading
            def bg_refresh():
                try:
                    from main import refresh_gallery
                    refresh_gallery(self.lora_root)
                except Exception as ex:
                    log_operation(f"Error refreshing gallery HTML after deleting collection in thread: {ex}")
            threading.Thread(target=bg_refresh, daemon=True).start()
            
            self.send_success_response({"collections": updated_collections})
        except json.JSONDecodeError:
            self.send_error_response(400, "Invalid JSON payload")
        except Exception as e:
            self.send_error_response(500, f"Server Error: {e}")

    def handle_open_folder(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)
            relative_path = data.get("relative_path")
            
            if not relative_path:
                self.send_error_response(400, "Missing relative_path")
                return
                
            target_model_file = (self.lora_root / relative_path).resolve()
            if not target_model_file.is_relative_to(self.lora_root.resolve()):
                self.send_error_response(403, "Access Denied: Path traversal detected")
                return
                
            parent_folder = target_model_file.parent
            if parent_folder.exists() and parent_folder.is_dir():
                if sys.platform == "win32":
                    os.startfile(parent_folder)
                    log_operation(f"Opened folder in explorer: {parent_folder}")
                    self.send_success_response({"status": "success", "message": "Folder opened"})
                else:
                    import subprocess
                    if sys.platform == "darwin":
                        subprocess.run(["open", str(parent_folder)])
                    else:
                        subprocess.run(["xdg-open", str(parent_folder)])
                    log_operation(f"Opened folder: {parent_folder}")
                    self.send_success_response({"status": "success", "message": "Folder opened"})
            else:
                self.send_error_response(404, f"Directory not found: {parent_folder}")
        except json.JSONDecodeError:
            self.send_error_response(400, "Invalid JSON payload")
        except Exception as e:
            self.send_error_response(500, f"Server Error: {e}")

    def handle_get_user_data(self, relative_path):
        if not relative_path:
            self.send_error_response(400, "Missing relative_path parameter")
            return

        try:
            target_model_file = (self.lora_root / relative_path).resolve()
            if not target_model_file.is_relative_to(self.lora_root.resolve()):
                self.send_error_response(403, "Access Denied: Path traversal detected")
                return

            user_data_path = target_model_file.parent / "user_data.json"
            
            if user_data_path.exists():
                try:
                    with user_data_path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    self.send_success_response(data)
                except Exception as e:
                    self.send_error_response(500, f"Error reading user data file: {e}")
            else:
                # Return empty defaults if file doesn't exist
                self.send_success_response({
                    "personal_trigger_words": [],
                    "notes": "",
                    "rating": 0,
                    "collections": []
                })
        except Exception as e:
            self.send_error_response(500, f"Server Error: {e}")

    def handle_save_user_data(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            data = json.loads(post_data.decode('utf-8'))
            relative_path = data.get("relative_path")
            user_payload = data.get("user_data", {})
            
            if not relative_path:
                self.send_error_response(400, "Missing relative_path")
                return
            
            target_model_file = (self.lora_root / relative_path).resolve()
            if not target_model_file.is_relative_to(self.lora_root.resolve()):
                self.send_error_response(403, "Access Denied: Path traversal detected")
                return

            if not target_model_file.exists():
                self.send_error_response(404, "Model file not found")
                return
                
            target_folder = target_model_file.parent
            user_data_path = target_folder / "user_data.json"
            
            # Determine if we have actual content to save
            personal_triggers = user_payload.get("personal_trigger_words", [])
            notes = user_payload.get("notes", "")
            rating = user_payload.get("rating", 0)
            collections = user_payload.get("collections", [])
            
            has_content = bool(personal_triggers or notes.strip() or rating > 0 or collections)
            
            if not has_content:
                # If there's no content to save, and the file exists, we delete it lazily
                if user_data_path.exists():
                    try:
                        user_data_path.unlink()
                        log_operation(f"Deleted empty user data sidecar: {user_data_path}")
                    except OSError as e:
                        log_operation(f"Error deleting user data sidecar {user_data_path}: {e}")
                
                self.send_success_response({
                    "status": "success", 
                    "message": "User data cleared",
                    "user_data": {
                        "personal_trigger_words": [],
                        "notes": "",
                        "rating": 0,
                        "collections": []
                    }
                })
                return

            # Load existing file to preserve any other keys (future compatibility)
            existing_data = {}
            if user_data_path.exists():
                try:
                    with user_data_path.open("r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                except Exception:
                    pass

            # Merge and set metadata fields
            existing_data["schema_version"] = 1
            existing_data["last_modified"] = datetime.now().isoformat() + "Z"
            existing_data["personal_trigger_words"] = personal_triggers
            existing_data["notes"] = notes
            existing_data["rating"] = rating

            # Normalize collections and auto-update collections.json
            normalized_collections = []
            current_globals = self.load_collections_list()
            updated = False
            for c in collections:
                c_clean = c.strip()
                if not c_clean:
                    continue
                
                # Normalize virtual collections Library & Favorites
                if c_clean.lower() == "favorites":
                    c_clean = "Favorites"
                elif c_clean.lower() == "library":
                    c_clean = "Library"
                    
                exists = False
                matched_name = c_clean
                for g in current_globals:
                    if g.lower() == c_clean.lower():
                        exists = True
                        matched_name = g
                        break
                        
                normalized_collections.append(matched_name)
                
                if not exists and c_clean not in ("Library", "Favorites"):
                    current_globals.append(c_clean)
                    updated = True
                    
            if updated:
                current_globals.sort(key=lambda x: x.lower())
                self.save_collections_list(current_globals)

            existing_data["collections"] = normalized_collections

            # Save the file
            target_folder.mkdir(parents=True, exist_ok=True)
            with user_data_path.open("w", encoding="utf-8") as f:
                json.dump(existing_data, f, indent=2)
                
            log_operation(f"Saved user data sidecar: {user_data_path}")

            # Invalidate the in-memory summary cache so the next /api/user_data_summary
            # request reflects this write without waiting for the TTL to expire.
            global _user_data_cache
            _user_data_cache = {}

            self.send_success_response({
                "status": "success",
                "message": "User data saved successfully",
                "user_data": existing_data
            })
            
        except json.JSONDecodeError:
            self.send_error_response(400, "Invalid JSON payload")
        except Exception as e:
            self.send_error_response(500, f"Server Error: {e}")

    def handle_delete_model_version(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            data = json.loads(post_data.decode('utf-8'))
            relative_path = data.get("relative_path")
            
            if not relative_path:
                self.send_error_response(400, "Missing relative_path")
                return
            
            target_model_file = (self.lora_root / relative_path).resolve()
            if not target_model_file.is_relative_to(self.lora_root.resolve()):
                self.send_error_response(403, "Access Denied: Path traversal detected")
                return

            if not target_model_file.exists():
                self.send_error_response(404, "Model file not found")
                return
            
            model_folder = target_model_file.parent
            model_stem = target_model_file.stem
            
            files_to_delete = [target_model_file]
            
            # Identify sidecars: <stem>.json and <stem>.civitai.info
            json_sidecar = model_folder / f"{model_stem}.json"
            if json_sidecar.exists():
                files_to_delete.append(json_sidecar)
                
            civitai_info_sidecar = model_folder / f"{model_stem}.civitai.info"
            if civitai_info_sidecar.exists():
                files_to_delete.append(civitai_info_sidecar)
                
            # Previews
            for ext in PREVIEW_EXTENSIONS:
                preview_path = model_folder / f"{model_stem}{ext}"
                if preview_path.exists():
                    files_to_delete.append(preview_path)
            
            # Delete identified files
            deleted_paths = []
            for file_path in files_to_delete:
                if file_path.exists():
                    try:
                        file_path.unlink()
                        deleted_paths.append(file_path.name)
                    except OSError as e:
                        log_operation(f"Error unlinking {file_path}: {e}")
            
            log_operation(f"Deleted model version {target_model_file.name} and sidecars: {deleted_paths}")
            
            # Check if any other model files remain in the folder
            remaining_models = []
            for ext in LORA_EXTENSIONS:
                for f in model_folder.glob(f"*{ext}"):
                    if is_lora_file(f):
                        remaining_models.append(f)
                        
            folder_cleaned = False
            if not remaining_models:
                # No models left in this folder! Delete all remaining files and rmdir
                for extra in list(model_folder.glob("*")):
                    if extra.is_file():
                        try:
                            extra.unlink()
                        except OSError:
                            pass
                try:
                    model_folder.rmdir()
                    folder_cleaned = True
                    log_operation(f"Removed empty model folder: {model_folder}")
                except OSError as e:
                    log_operation(f"Error removing model folder {model_folder}: {e}")
            
            # Automatically refresh gallery HTML in a background thread to reflect the deletion without blocking the UI
            import threading
            def bg_refresh():
                try:
                    from main import refresh_gallery
                    refresh_gallery(self.lora_root)
                except Exception as ex:
                    log_operation(f"Error refreshing gallery HTML after deletion in thread: {ex}")
            threading.Thread(target=bg_refresh, daemon=True).start()
            
            self.send_success_response({
                "status": "success",
                "message": f"Deleted model version {target_model_file.name}",
                "deleted_files": deleted_paths,
                "folder_removed": folder_cleaned
            })
            
        except json.JSONDecodeError:
            self.send_error_response(400, "Invalid JSON payload")
        except Exception as e:
            self.send_error_response(500, f"Server Error: {e}")


    def send_success_response(self, response_data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode('utf-8'))

    def send_error_response(self, code, message):
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({"status": "error", "message": message}).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def make_handler(lora_root: Path):
    class CustomHandler(GalleryHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, lora_root=lora_root, directory=str(lora_root), **kwargs)
    return CustomHandler

def start_server(lora_root: Path, port: int = 8000) -> None:
    handler = make_handler(lora_root)
    server = None
    actual_port = port
    
    # Try finding an open port starting from 8000
    for p in range(port, port + 20):
        try:
            # Bind to localhost specifically
            server = socketserver.TCPServer(("127.0.0.1", p), handler)
            actual_port = p
            break
        except OSError:
            continue
            
    if not server:
        print("Error: Could not find any open ports to start the server.")
        return
        
    # Automatically refresh gallery HTML once on startup to reflect latest code/template changes
    try:
        from main import refresh_gallery
        print("Regenerating gallery HTML to apply template updates...")
        refresh_gallery(lora_root)
    except Exception as e:
        print(f"Warning: Could not regenerate gallery HTML on startup: {e}")
        
    url = f"http://127.0.0.1:{actual_port}/gallery.html"
    print(f"\n==============================================")
    print(f"LoRA Library Server started successfully!")
    print(f"Serving LoRA Root: {lora_root}")
    print(f"Gallery URL: {url}")
    print(f"==============================================")
    print("Opening browser...")
    import webbrowser
    webbrowser.open(url)
    print("Press Ctrl+C to shut down the server and exit.")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.server_close()
        print("Server stopped.")
