"""Lightweight HTTP server and API endpoints for LoRA Library Manager."""

import http.server
import socketserver
import json
import urllib.parse
from datetime import datetime
from pathlib import Path
import os
import sys
import threading
import queue

from config import get_lora_root, is_lora_file, LORA_EXTENSIONS, PREVIEW_EXTENSIONS, log_operation, clean_model_collections

# ---------------------------------------------------------------------------
# In-memory TTL cache for the user-data summary endpoint.
# The summary is computed by an rglob scan across all user_data.json files,
# which is expensive on large libraries.  We cache the result for
# _USER_DATA_CACHE_TTL seconds and invalidate it whenever a save occurs.
# ---------------------------------------------------------------------------
import time as _time

# ---------------------------------------------------------------------------
# Background task runner — only one task runs at a time.
# Log lines are pushed into _task_log_queue so the SSE endpoint can stream
# them to the browser in real-time.
# ---------------------------------------------------------------------------
_task_lock   = threading.Lock()
_task_running = False           # True while a managed task is executing
_task_log_queue: queue.Queue = queue.Queue()   # str lines or sentinel None

_job_queue = None
import jobs
jobs.log_callbacks.append(lambda msg: _task_log_queue.put(msg))

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

        if path == "/" or path == "/index.html":
            self.path = "/gallery.html"
            super().do_GET()
        elif path in ("/favicon.png", "/favicon.ico", "/logo.png"):
            # Serve the logo from the project root directory
            logo_path = Path(__file__).parent.parent / "logo.png"
            if logo_path.exists():
                self.send_response(200)
                if path.endswith(".ico"):
                    self.send_header("Content-Type", "image/x-icon")
                else:
                    self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(logo_path.stat().st_size))
                self.end_headers()
                with open(logo_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "File not found")
                return
        elif path == "/api/user_data_summary":
            self.handle_get_summary()
        elif path == "/api/get_user_data":
            relative_path = query.get("relative_path", [None])[0]
            self.handle_get_user_data(relative_path)
        elif path == "/api/collections":
            self.handle_get_collections()
        elif path == "/manage" or path == "/manage/":
            self.handle_manage_page()
        elif path == "/api/log-stream":
            self.handle_log_stream()
        elif path == "/api/stats":
            self.handle_get_stats()
        elif path == "/api/settings":
            self.handle_get_settings()
        elif path == "/api/tasks":
            self.handle_get_tasks()
        elif path == "/api/task-history":
            self.handle_get_task_history()
        elif path == "/api/backups":
            self.handle_get_backups()
        else:
            # Fall back to serving static files
            super().do_GET()

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        if path == "/api/save_user_data":
            self.handle_save_user_data()
        elif path == "/api/delete_model_version":
            self.handle_delete_model_version()
        elif path == "/api/collections":
            self.handle_add_collection()
        elif path == "/api/delete_collection":
            self.handle_delete_collection()
        elif path == "/api/rename_collection":
            self.handle_rename_collection()
        elif path == "/api/open_folder":
            self.handle_open_folder()
        elif path.startswith("/api/tasks/") and path.endswith("/cancel"):
            parts = path.split("/")
            if len(parts) >= 5:
                job_id = parts[3]
                self.handle_cancel_task(job_id)
            else:
                self.send_error_response(400, "Invalid request path")
        elif path.startswith("/api/run/"):
            action = path[len("/api/run/"):]
            self.handle_run_task(action, query)
        elif path == "/api/settings":
            self.handle_post_settings()
        elif path == "/api/select_folder":
            self.handle_select_folder()
        elif path == "/api/select_model_file":
            self.handle_select_model_file()
        elif path == "/api/select_backup_file":
            self.handle_select_backup_file()
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
                    collections = clean_model_collections(data.get("collections", []), path.parent)

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
        # Rebuild/validate collections from user_data.json files recursively to ensure self-healing and no orphaned collections
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
                            exists = False
                            for existing in collections_set:
                                if existing.lower() == c_clean.lower():
                                    exists = True
                                    break
                            if not exists:
                                collections_set.add(c_clean)
            except Exception:
                continue
        
        collections_list = sorted(list(collections_set), key=lambda x: x.lower())
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
            
            # Delete target_name and all sub-collections (prefix matching)
            target_lower = target_name.lower()
            updated_collections = []
            for c in collections:
                c_lower = c.lower()
                if c_lower == target_lower or c_lower.startswith(target_lower + " / ") or c_lower.startswith(target_lower + "/"):
                    continue
                updated_collections.append(c)
            self.save_collections_list(updated_collections)
            
            # Now update all user_data.json files recursively
            for path in self.lora_root.rglob("user_data.json"):
                try:
                    with path.open("r", encoding="utf-8") as f:
                        user_data = json.load(f)
                    
                    colls = user_data.get("collections", [])
                    has_target = False
                    for c in colls:
                        c_lower = c.lower()
                        if c_lower == target_lower or c_lower.startswith(target_lower + " / ") or c_lower.startswith(target_lower + "/"):
                            has_target = True
                            break
                            
                    if has_target:
                        new_colls = []
                        for c in colls:
                            c_lower = c.lower()
                            if c_lower == target_lower or c_lower.startswith(target_lower + " / ") or c_lower.startswith(target_lower + "/"):
                                continue
                            new_colls.append(c)
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

    def handle_rename_collection(self):
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body)
            old_name = data.get("old_name", "").strip()
            new_name = data.get("new_name", "").strip()
            
            if not old_name or not new_name:
                self.send_error_response(400, "Both old_name and new_name must be specified")
                return
                
            collections = self.load_collections_list()
            old_lower = old_name.lower()
            
            # Update collections list in collections.json
            updated_collections = []
            for c in collections:
                c_lower = c.lower()
                # Rename the exact collection or subcollections
                if c_lower == old_lower:
                    if new_name not in updated_collections:
                        updated_collections.append(new_name)
                elif c_lower.startswith(old_lower + " / "):
                    suffix = c[len(old_name):]
                    renamed = new_name + suffix
                    if renamed not in updated_collections:
                        updated_collections.append(renamed)
                else:
                    if c not in updated_collections:
                        updated_collections.append(c)
                        
            self.save_collections_list(updated_collections)
            
            # Now update all user_data.json files recursively
            updated_count = 0
            for path in self.lora_root.rglob("user_data.json"):
                try:
                    with path.open("r", encoding="utf-8") as f:
                        user_data = json.load(f)
                    
                    colls = user_data.get("collections", [])
                    has_change = False
                    new_colls = []
                    
                    for c in colls:
                        c_lower = c.lower()
                        if c_lower == old_lower:
                            has_change = True
                            if new_name not in new_colls:
                                new_colls.append(new_name)
                        elif c_lower.startswith(old_lower + " / "):
                            has_change = True
                            suffix = c[len(old_name):]
                            renamed = new_name + suffix
                            if renamed not in new_colls:
                                new_colls.append(renamed)
                        else:
                            if c not in new_colls:
                                new_colls.append(c)
                                
                    if has_change:
                        user_data["collections"] = new_colls
                        user_data["last_modified"] = datetime.now().isoformat() + "Z"
                        with path.open("w", encoding="utf-8") as f:
                            json.dump(user_data, f, indent=2)
                        log_operation(f"Renamed collection in user data sidecar: {path}")
                        updated_count += 1
                except Exception as e:
                    log_operation(f"Error renaming collection in sidecar {path}: {e}")
                    continue
            
            # Invalidate summary cache
            global _user_data_cache
            _user_data_cache = {}
            
            # Trigger background refresh of gallery HTML to reflect the updated metadata
            import threading
            def bg_refresh():
                try:
                    from main import refresh_gallery
                    refresh_gallery(self.lora_root)
                except Exception as ex:
                    log_operation(f"Error refreshing gallery HTML after renaming collection in thread: {ex}")
            threading.Thread(target=bg_refresh, daemon=True).start()
            
            self.send_success_response({"ok": True, "updated_files": updated_count, "collections": updated_collections})
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
                    if isinstance(data, dict) and "collections" in data:
                        data["collections"] = clean_model_collections(data["collections"], target_model_file.parent)
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
        global _user_data_cache
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
            collections = clean_model_collections(user_payload.get("collections", []), target_folder)
            collections = collections[:1]  # enforce maximum of 1 collection
            
            has_content = bool(personal_triggers or notes.strip() or rating > 0 or collections)
            
            if not has_content:
                # If there's no content to save, and the file exists, we delete it lazily
                if user_data_path.exists():
                    try:
                        user_data_path.unlink()
                        log_operation(f"Deleted empty user data sidecar: {user_data_path}")
                    except OSError as e:
                        log_operation(f"Error deleting user data sidecar {user_data_path}: {e}")
                
                from organizer import organize_lora_file, get_target_folder
                
                # Compute new path before organization
                new_target_folder = get_target_folder(target_model_file, self.lora_root)
                new_relative_path = (new_target_folder / target_model_file.name).resolve().relative_to(self.lora_root.resolve()).as_posix()

                # Automatically organize sibling model files back to the root folder
                sibling_models = []
                for ext in LORA_EXTENSIONS:
                    for f in target_folder.glob(f"*{ext}"):
                        if is_lora_file(f):
                            sibling_models.append(f)
                            
                for model_file in sibling_models:
                    organize_lora_file(model_file, self.lora_root)
                    
                # Invalidate summary cache
                _user_data_cache = {}
                
                # Trigger background refresh of gallery HTML to reflect the updated metadata and paths
                import threading
                def bg_refresh():
                    try:
                        from main import refresh_gallery
                        refresh_gallery(self.lora_root)
                    except Exception as ex:
                        log_operation(f"Error refreshing gallery HTML after clearing user data: {ex}")
                threading.Thread(target=bg_refresh, daemon=True).start()
                
                self.send_success_response({
                    "status": "success", 
                    "message": "User data cleared",
                    "user_data": {
                        "personal_trigger_words": [],
                        "notes": "",
                        "rating": 0,
                        "collections": []
                    },
                    "new_relative_path": new_relative_path
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

            from organizer import organize_lora_file, get_target_folder
            
            # Compute new path before organization
            new_target_folder = get_target_folder(target_model_file, self.lora_root)
            new_relative_path = (new_target_folder / target_model_file.name).resolve().relative_to(self.lora_root.resolve()).as_posix()

            # Automatically organize sibling model files to the target folder
            sibling_models = []
            for ext in LORA_EXTENSIONS:
                for f in target_folder.glob(f"*{ext}"):
                    if is_lora_file(f):
                        sibling_models.append(f)
                        
            for model_file in sibling_models:
                organize_lora_file(model_file, self.lora_root)
                
            # Invalidate the in-memory summary cache so the next /api/user_data_summary
            # request reflects this write without waiting for the TTL to expire.
            _user_data_cache = {}

            # Trigger background refresh of gallery HTML to reflect the updated metadata and paths
            import threading
            def bg_refresh():
                try:
                    from main import refresh_gallery
                    refresh_gallery(self.lora_root)
                except Exception as ex:
                    log_operation(f"Error refreshing gallery HTML after saving user data: {ex}")
            threading.Thread(target=bg_refresh, daemon=True).start()

            self.send_success_response({
                "status": "success",
                "message": "User data saved successfully",
                "user_data": existing_data,
                "new_relative_path": new_relative_path
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


    # ── Management Dashboard ──────────────────────────────────────────────────

    def handle_manage_page(self):
        """Serve the /manage dashboard HTML page."""
        try:
            import importlib
            import manage_page
            importlib.reload(manage_page)
            from manage_page import build_manage_page
            html = build_manage_page(self.lora_root).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        except Exception as e:
            self.send_error_response(500, f"Error building manage page: {e}")

    def handle_get_stats(self):
        """Return quick library stats as JSON."""
        try:
            import importlib
            import manage_page
            importlib.reload(manage_page)
            from manage_page import get_library_stats
            stats = get_library_stats(self.lora_root)
            self.send_success_response(stats)
        except Exception as e:
            self.send_error_response(500, f"Error computing stats: {e}")

    def handle_get_settings(self):
        """Return current settings (api key, dry_run, verbose, lora_root, download_strategy, weight, sort, providers)."""
        try:
            from config import (is_dry_run, is_verbose, get_lora_root,
                                get_download_strategy, get_default_weight, get_default_sort, get_provider_api_key)
            from providers.registry import get_provider_presentations
            
            providers_status = []
            for name, presentation in get_provider_presentations().items():
                api_key = get_provider_api_key(name)
                status = "Configured" if api_key else "Missing"
                providers_status.append({
                    "id": name,
                    "display_name": presentation["display_name"],
                    "accent_color": presentation["accent_color"],
                    "icon": presentation["icon"],
                    "website": presentation["website"],
                    "status": status,
                    "has_api_key": bool(api_key),
                })

            self.send_success_response({
                "api_key": get_provider_api_key("civitai"), # Legacy backward compatibility
                "dry_run": is_dry_run(),
                "verbose": is_verbose(),
                "lora_root": str(get_lora_root()),
                "download_strategy": get_download_strategy(),
                "default_weight": get_default_weight(),
                "default_sort": get_default_sort(),
                "providers": providers_status,
            })
        except Exception as e:
            self.send_error_response(500, f"Error reading settings: {e}")

    def handle_post_settings(self):
        """Update settings from the settings form."""
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            from config import (set_provider_api_key, set_dry_run, set_verbose, set_lora_root,
                                set_download_strategy, set_default_weight, set_default_sort)
            
            if "api_keys" in body and isinstance(body["api_keys"], dict):
                for provider, key in body["api_keys"].items():
                    set_provider_api_key(provider, key)
            if "api_key" in body: # Legacy backward compatibility
                set_provider_api_key("civitai", body["api_key"])
            if "dry_run" in body:
                set_dry_run(bool(body["dry_run"]))
            if "verbose" in body:
                set_verbose(bool(body["verbose"]))
            if "lora_root" in body:
                set_lora_root(body["lora_root"])
            if "download_strategy" in body:
                set_download_strategy(body["download_strategy"])
            if "default_weight" in body:
                set_default_weight(body["default_weight"])
            if "default_sort" in body:
                set_default_sort(body["default_sort"])
                
            self.send_success_response({"ok": True})
        except Exception as e:
            self.send_error_response(500, f"Error saving settings: {e}")

    def handle_get_tasks(self):
        """Return running and queued tasks from the queue manager."""
        global _job_queue
        if not _job_queue:
            self.send_success_response([])
            return
        self.send_success_response(_job_queue.get_active_tasks())

    def handle_get_task_history(self):
        """Return completed, failed, and interrupted tasks from the queue manager."""
        global _job_queue
        if not _job_queue:
            self.send_success_response([])
            return
        self.send_success_response(_job_queue.get_task_history())

    def handle_cancel_task(self, job_id: str):
        """Request cancellation for a running or pending task."""
        global _job_queue
        if not _job_queue:
            self.send_error_response(500, "Job queue not initialized")
            return
        success = _job_queue.cancel_job(job_id)
        self.send_success_response({"ok": success})

    def handle_get_backups(self):
        """List backup manifest files stored in the backup/ directory."""
        try:
            backup_dir = self.lora_root / "backup"
            backups = []
            if backup_dir.exists() and backup_dir.is_dir():
                backups = sorted(
                    [b.name for b in backup_dir.glob("*.json")],
                    key=lambda x: (backup_dir / x).stat().st_mtime,
                    reverse=True
                )
            self.send_success_response(backups)
        except Exception as e:
            self.send_error_response(500, str(e))

    def handle_run_task(self, action: str, query_params: dict):
        """Enqueue a background task based on client run action."""
        global _job_queue, _task_log_queue
        if not _job_queue:
            self.send_error_response(500, "Job queue not initialized")
            return

        import jobs
        job = None

        try:
            # Drain the SSE log stream before enqueuing a fresh task
            while not _task_log_queue.empty():
                try:
                    _task_log_queue.get_nowait()
                except queue.Empty:
                    break

            if action == "sync":
                job = jobs.SyncJob()
            elif action == "quick-refresh":
                job = jobs.QuickRefreshJob()
            elif action == "organize":
                job = jobs.OrganizeJob()
            elif action == "download-previews":
                job = jobs.DownloadPreviewsJob()
            elif action == "download-metadata":
                job = jobs.DownloadMetadataJob()
            elif action == "update-metadata":
                job = jobs.UpdateMetadataJob()
            elif action == "rebuild-gallery":
                job = jobs.RebuildGalleryJob()
            elif action == "migrate-metadata":
                job = jobs.MigrateMetadataJob()
            elif action == "civitai-update-detector":
                job = jobs.CivitaiUpdateDetectorJob()
            elif action == "duplicate-finder":
                strategy = query_params.get("strategy", ["scan-only"])[0]
                job = jobs.DuplicateFinderJob(strategy=strategy)
            elif action == "generate-hashes":
                job = jobs.GenerateHashesJob()
            elif action == "migration-preview":
                job = jobs.MigrationPreviewJob()
            elif action == "import-existing":
                job = jobs.ImportExistingJob()
            elif action == "export-manifest":
                job = jobs.ExportManifestJob()
            elif action == "import-manifest":
                filename = query_params.get("manifest_path", [""])[0]
                if not filename:
                    self.send_error_response(400, "Missing manifest_path parameter")
                    return
                job = jobs.ImportManifestJob(manifest_filename=filename)
            elif action == "process-single-file":
                file_path = query_params.get("file_path", [""])[0]
                if not file_path:
                    self.send_error_response(400, "Missing file_path parameter")
                    return
                job = jobs.ProcessSingleFileJob(file_path=file_path)
            else:
                self.send_error_response(400, f"Unknown action: {action}")
                return

            _job_queue.enqueue(job)
            self.send_success_response({"ok": True, "job_id": job.id})
        except Exception as e:
            self.send_error_response(500, str(e))

    def handle_log_stream(self):
        """SSE endpoint that streams lines from _task_log_queue to the browser."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            while True:
                try:
                    line = _task_log_queue.get(timeout=30)
                except queue.Empty:
                    # Send a keep-alive comment so the connection doesn't time out
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue

                if line is None:
                    # Sentinel — task finished
                    self.wfile.write(b"data: __DONE__\n\n")
                    self.wfile.flush()
                    break

                encoded = line.replace("\n", " ").encode("utf-8")
                self.wfile.write(b"data: " + encoded + b"\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass   # Client disconnected — just stop

    def handle_select_folder(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            
            folder = filedialog.askdirectory(title="Select Folder")
            root.destroy()
            
            self.send_success_response({"path": folder if folder else ""})
        except Exception as e:
            self.send_error_response(500, f"Error opening folder selector: {str(e)}")

    def handle_select_model_file(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            
            file_path = filedialog.askopenfilename(
                title="Select Safetensors Model File",
                filetypes=[("Safetensors files", "*.safetensors"), ("All files", "*.*")]
            )
            root.destroy()
            
            self.send_success_response({"path": file_path if file_path else ""})
        except Exception as e:
            self.send_error_response(500, f"Error opening file selector: {str(e)}")

    def handle_select_backup_file(self):
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            
            file_path = filedialog.askopenfilename(
                title="Select Backup JSON File",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            root.destroy()
            
            self.send_success_response({"path": file_path if file_path else ""})
        except Exception as e:
            self.send_error_response(500, f"Error opening file selector: {str(e)}")

    # ── Shared helpers ────────────────────────────────────────────────────────

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

def make_handler():
    class CustomHandler(GalleryHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            from config import get_lora_root
            current_root = get_lora_root()
            super().__init__(*args, lora_root=current_root, directory=str(current_root), **kwargs)
    return CustomHandler

def start_server(lora_root: Path, port: int = 8000) -> None:
    global _job_queue
    from jobs import JobQueue
    _job_queue = JobQueue(lora_root)
    _job_queue.start_worker()

    handler = make_handler()
    server = None
    actual_port = port
    
    # Try finding an open port starting from 8000
    for p in range(port, port + 50):
        try:
            # Bind to all interfaces (empty string) for maximum compatibility on Windows
            server = socketserver.ThreadingTCPServer(("", p), handler)
            actual_port = p
            break
        except OSError:
            continue
            
    if not server:
        try:
            # Fallback to port 0 (the OS will automatically assign a free dynamic port)
            server = socketserver.ThreadingTCPServer(("", 0), handler)
            actual_port = server.server_address[1]
        except OSError as e:
            print(f"Error: Could not bind to any port, including dynamic port fallback: {e}")
            return
        
    # Automatically refresh gallery HTML once on startup to reflect latest code/template changes
    try:
        from main import refresh_gallery
        print("Regenerating gallery HTML to apply template updates...")
        refresh_gallery(lora_root)
    except Exception as e:
        print(f"Warning: Could not regenerate gallery HTML on startup: {e}")
        
    url         = f"http://127.0.0.1:{actual_port}/gallery.html"
    manage_url  = f"http://127.0.0.1:{actual_port}/manage"
    print(f"\n==============================================")
    print(f"LoRA Library Server started successfully!")
    print(f"Serving LoRA Root: {lora_root}")
    print(f"Gallery URL:    {url}")
    print(f"Dashboard URL:  {manage_url}")
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
