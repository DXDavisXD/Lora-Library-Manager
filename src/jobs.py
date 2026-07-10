"""Background job queue and task definitions for LoRA Library Manager."""

import json
import uuid
import time
import threading
from pathlib import Path
from datetime import datetime
from config import log_operation

# Thread-local storage to track the active job inside a worker thread
_local_state = threading.local()
SETTINGS_DIR = Path(__file__).parent.parent
JOBS_FILE = SETTINGS_DIR / "jobs.json"
log_callbacks = []

def get_current_job():
    """Return the currently executing job on this thread, or None."""
    return getattr(_local_state, "current_job", None)

def update_job_progress(percent=None, current=None, total=None, message=None, item_name=None):
    """Update progress fields of the active job, triggering a save."""
    job = get_current_job()
    if job:
        if percent is not None:
            job.progress_percent = int(percent)
        if current is not None:
            job.progress_current = int(current)
        if total is not None:
            job.progress_total = int(total)
        if message is not None:
            job.progress_message = str(message)
        if item_name is not None:
            job.progress_item = str(item_name)
        if job.on_update:
            job.on_update(job)


class BaseJob:
    """Base class for all background tasks."""
    def __init__(self, name: str, priority: str = "normal"):
        self.id = uuid.uuid4().hex[:8]
        self.name = name
        self.priority = priority             # "interactive" | "normal" | "maintenance"
        self.status = "pending"              # "pending" | "running" | "completed" | "failed" | "interrupted"
        self.progress_percent = 0
        self.progress_current = 0
        self.progress_total = 0
        self.progress_message = ""
        self.progress_item = ""
        self.start_time = None
        self.end_time = None
        self.error = None
        self.cancel_requested = False
        self.on_update = None                # Callback when fields update
        self.created_at = time.time()

    def check_cancelled(self):
        """Check if cancellation has been requested, raising an exception if so."""
        if self.cancel_requested:
            raise Exception("Task cancelled by user")

    def execute(self, lora_root: Path):
        """Implement task-specific execution logic."""
        raise NotImplementedError()

    def to_dict(self) -> dict:
        """Serialize job details to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "priority": self.priority,
            "status": self.status,
            "progress_percent": self.progress_percent,
            "progress_current": self.progress_current,
            "progress_total": self.progress_total,
            "progress_message": self.progress_message,
            "progress_item": self.progress_item,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'BaseJob':
        """Reconstruct a stub job from serialized state."""
        job = cls(data.get("name", "Unknown Job"), data.get("priority", "normal"))
        job.id = data.get("id", job.id)
        job.status = data.get("status", "pending")
        job.progress_percent = data.get("progress_percent", 0)
        job.progress_current = data.get("progress_current", 0)
        job.progress_total = data.get("progress_total", 0)
        job.progress_message = data.get("progress_message", "")
        job.progress_item = data.get("progress_item", "")
        job.start_time = data.get("start_time")
        job.end_time = data.get("end_time")
        job.error = data.get("error")
        job.cancel_requested = data.get("cancel_requested", False)
        job.created_at = data.get("created_at", time.time())
        return job


# ---------------------------------------------------------------------------
# Concrete Job Subclasses
# ---------------------------------------------------------------------------

class SyncJob(BaseJob):
    def __init__(self):
        super().__init__("Automatic Sync", "normal")

    def execute(self, lora_root: Path):
        update_job_progress(percent=5, message="Starting sync...")
        from main import run_automatic
        run_automatic(lora_root)


class QuickRefreshJob(BaseJob):
    def __init__(self):
        super().__init__("Quick Refresh", "interactive")

    def execute(self, lora_root: Path):
        update_job_progress(percent=10, message="Scanning library...")
        from main import run_quick_refresh
        run_quick_refresh(lora_root)


class OrganizeJob(BaseJob):
    def __init__(self):
        super().__init__("Organize Library", "normal")

    def execute(self, lora_root: Path):
        update_job_progress(percent=5, message="Organizing folders...")
        from main import run_organize
        run_organize(lora_root)


class DownloadPreviewsJob(BaseJob):
    def __init__(self):
        super().__init__("Download Previews", "normal")

    def execute(self, lora_root: Path):
        update_job_progress(percent=5, message="Scanning previews...")
        from main import run_asset_job
        run_asset_job(lora_root, download_metadata=False, download_preview=True)


class DownloadMetadataJob(BaseJob):
    def __init__(self):
        super().__init__("Download Metadata", "normal")

    def execute(self, lora_root: Path):
        update_job_progress(percent=5, message="Scanning metadata...")
        from main import run_asset_job
        run_asset_job(lora_root, download_metadata=True, download_preview=False)


class UpdateMetadataJob(BaseJob):
    def __init__(self):
        super().__init__("Update Metadata", "normal")

    def execute(self, lora_root: Path):
        update_job_progress(percent=5, message="Scanning existing metadata for updates...")
        from main import run_asset_job
        run_asset_job(lora_root, download_metadata=True, download_preview=False, force_update=True)


class RebuildGalleryJob(BaseJob):
    def __init__(self):
        super().__init__("Rebuild Gallery", "interactive")

    def execute(self, lora_root: Path):
        update_job_progress(percent=20, message="Rebuilding gallery.html...")
        from main import refresh_gallery
        refresh_gallery(lora_root)


class MigrateMetadataJob(BaseJob):
    def __init__(self):
        super().__init__("Migrate Metadata to User Data", "normal")

    def execute(self, lora_root: Path):
        update_job_progress(percent=10, message="Migrating metadata fields...")
        from diagnostics import migrate_metadata_to_user_data
        migrate_metadata_to_user_data(lora_root)


class DuplicateFinderJob(BaseJob):
    def __init__(self, strategy: str = "scan-only"):
        super().__init__(f"Duplicate Finder ({strategy})", "maintenance")
        self.strategy = strategy

    def execute(self, lora_root: Path):
        update_job_progress(percent=5, message="Scanning duplicate files...")
        from diagnostics import run_duplicate_finder
        run_duplicate_finder(lora_root, strategy=self.strategy)


class CivitaiUpdateDetectorJob(BaseJob):
    def __init__(self):
        super().__init__("CivitAI Update Detector", "maintenance")

    def execute(self, lora_root: Path):
        update_job_progress(percent=5, message="Querying updates on CivitAI...")
        from diagnostics import run_update_detector
        run_update_detector(lora_root)


class GenerateHashesJob(BaseJob):
    def __init__(self):
        super().__init__("Generate Hashes", "maintenance")

    def execute(self, lora_root: Path):
        update_job_progress(percent=5, message="Computing SHA256 hashes...")
        from migration import generate_missing_sha256_hashes
        generate_missing_sha256_hashes(lora_root)


class MigrationPreviewJob(BaseJob):
    def __init__(self):
        super().__init__("Migration Preview (Read-Only)", "maintenance")

    def execute(self, lora_root: Path):
        update_job_progress(percent=10, message="Scanning folders...")
        from migration import scan_library_hierarchy, generate_migration_preview
        data = scan_library_hierarchy(lora_root)
        generate_migration_preview(data)


class ImportExistingJob(BaseJob):
    def __init__(self):
        super().__init__("Import Existing Library", "normal")

    def execute(self, lora_root: Path):
        update_job_progress(percent=5, message="Converting folders to collections...")
        from migration import import_existing_library
        import_existing_library(lora_root)


class ExportManifestJob(BaseJob):
    def __init__(self):
        super().__init__("Export Manifest", "normal")

    def execute(self, lora_root: Path):
        update_job_progress(percent=10, message="Packaging manifest JSON...")
        from datetime import datetime
        backup_dir = lora_root / "backup"
        backup_dir.mkdir(exist_ok=True)
        filename = f"backup/{datetime.now().strftime('%d.%m.%Y-%H.%M')}.json"
        manifest_path = lora_root / filename
        
        from manifest import export_library_manifest
        export_library_manifest(lora_root, manifest_path)


class ImportManifestJob(BaseJob):
    def __init__(self, manifest_filename: str):
        super().__init__(f"Import Manifest ({manifest_filename})", "normal")
        self.manifest_filename = manifest_filename

    def execute(self, lora_root: Path):
        update_job_progress(percent=5, message="Validating manifest file...")
        manifest_path = Path(self.manifest_filename)
        if not manifest_path.is_absolute():
            manifest_path = lora_root / manifest_path
            
        from manifest import import_library_manifest
        import_library_manifest(lora_root, manifest_path)
        
        update_job_progress(percent=75, message="Organizing library folders...")
        from organizer import organize_library
        organize_library(lora_root, print_summary=False)
        
        update_job_progress(percent=90, message="Rebuilding gallery...")
        from main import refresh_gallery
        refresh_gallery(lora_root)


class ProcessSingleFileJob(BaseJob):
    def __init__(self, file_path: str):
        super().__init__(f"Process File ({Path(file_path).name})", "interactive")
        self.file_path = file_path

    def execute(self, lora_root: Path):
        update_job_progress(percent=10, message="Processing model file...")
        model_path = Path(self.file_path)
        if not model_path.exists():
            raise Exception("File does not exist")
        from main import process_model_assets
        process_model_assets(model_path, download_metadata=True, download_preview=True)


# ---------------------------------------------------------------------------
# Priority-Based Job Queue Manager
# ---------------------------------------------------------------------------

class JobQueue:
    """Thread-safe priority queue manager that persists state to jobs.json."""
    def __init__(self, lora_root: Path):
        self.lora_root = lora_root
        self.lock = threading.Lock()
        self.jobs = []
        self.worker_thread = None
        self.running = False
        
        self.load_history()

    def load_history(self):
        """Load job history from jobs.json, marking active ones as interrupted."""
        if JOBS_FILE.exists():
            try:
                with JOBS_FILE.open("r", encoding="utf-8") as f:
                    records = json.load(f)
                if isinstance(records, list):
                    for r in records:
                        # Construct a stub job
                        job = BaseJob(r.get("name", "Unknown"), r.get("priority", "normal"))
                        job.id = r.get("id", job.id)
                        job.status = r.get("status", "pending")
                        # Mark running/pending jobs as interrupted
                        if job.status in ("running", "pending"):
                            job.status = "interrupted"
                            job.progress_message = "Interrupted due to server stop."
                        else:
                            job.progress_message = r.get("progress_message", "")
                        job.progress_percent = r.get("progress_percent", 0)
                        job.progress_current = r.get("progress_current", 0)
                        job.progress_total = r.get("progress_total", 0)
                        job.progress_item = r.get("progress_item", "")
                        job.start_time = r.get("start_time")
                        job.end_time = r.get("end_time")
                        job.error = r.get("error")
                        job.cancel_requested = r.get("cancel_requested", False)
                        job.created_at = r.get("created_at", time.time())
                        job.on_update = self.save_state
                        self.jobs.append(job)
            except Exception as e:
                log_operation(f"Error loading jobs.json: {e}")

    def save_state(self, job=None):
        """Persist current jobs queue & history to jobs.json."""
        with self.lock:
            try:
                # Keep up to 100 historical records to prevent database bloating
                serializable = [j.to_dict() for j in self.jobs]
                # Sort by created time descending
                serializable.sort(key=lambda x: x["created_at"], reverse=True)
                # Keep active ones plus up to 100 finished ones
                active = [s for s in serializable if s["status"] in ("pending", "running")]
                history = [s for s in serializable if s["status"] not in ("pending", "running")]
                limited_history = history[:100]
                
                with JOBS_FILE.open("w", encoding="utf-8") as f:
                    json.dump(active + limited_history, f, indent=2)
            except OSError as e:
                pass

    def enqueue(self, job: BaseJob):
        """Push a job onto the queue and notify worker."""
        job.on_update = self.save_state
        with self.lock:
            self.jobs.append(job)
        self.save_state()
        log_operation(f"Enqueued job '{job.name}' (ID: {job.id}, Priority: {job.priority})")

    def cancel_job(self, job_id: str) -> bool:
        """Request cancellation for a job."""
        with self.lock:
            for job in self.jobs:
                if job.id == job_id:
                    job.cancel_requested = True
                    log_operation(f"Cancellation requested for job {job_id}")
                    if job.status == "pending":
                        job.status = "failed"
                        job.error = "Cancelled by user"
                        self.save_state()
                    return True
        return False

    def get_active_tasks(self) -> list[dict]:
        """Return running and queued tasks."""
        with self.lock:
            return [j.to_dict() for j in self.jobs if j.status in ("pending", "running")]

    def get_task_history(self) -> list[dict]:
        """Return completed, failed, and interrupted tasks."""
        with self.lock:
            history = [j.to_dict() for j in self.jobs if j.status in ("completed", "failed", "interrupted")]
            history.sort(key=lambda x: x["created_at"], reverse=True)
            return history

    def start_worker(self):
        """Boot the worker daemon thread."""
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def _worker_loop(self):
        """Worker thread loop sorting by priority and executing tasks."""
        import io
        import contextlib
        priority_map = {"interactive": 0, "normal": 1, "maintenance": 2}
        
        class _QueueStream(io.TextIOBase):
            def write(self, s):
                for line in s.splitlines():
                    if line.strip():
                        for cb in log_callbacks:
                            try:
                                cb(line)
                            except Exception:
                                pass
                return len(s)
            def flush(self):
                pass
                
        qs = _QueueStream()
        
        while self.running:
            next_job = None
            with self.lock:
                pending = [j for j in self.jobs if j.status == "pending"]
                if pending:
                    # Sort by priority, then by creation time
                    pending.sort(key=lambda j: (priority_map.get(j.priority, 1), j.created_at))
                    next_job = pending[0]
                    next_job.status = "running"
                    next_job.start_time = datetime.now().isoformat()
            
            if next_job:
                # Update state on startup
                self.save_state()
                log_operation(f"Running job '{next_job.name}' (ID: {next_job.id})")
                
                # Bind the active job to thread local state
                _local_state.current_job = next_job
                
                try:
                    from config import get_lora_root
                    active_root = get_lora_root()
                    with contextlib.redirect_stdout(qs):
                        next_job.execute(active_root)
                    next_job.status = "completed"
                    next_job.progress_percent = 100
                    next_job.progress_message = "Completed successfully."
                    log_operation(f"Job '{next_job.name}' (ID: {next_job.id}) completed successfully.")
                except Exception as e:
                    next_job.status = "failed"
                    next_job.error = str(e)
                    log_operation(f"Job '{next_job.name}' (ID: {next_job.id}) failed: {e}")
                finally:
                    next_job.end_time = datetime.now().isoformat()
                    _local_state.current_job = None
                    self.save_state()
                    # Trigger the __DONE__ sentinel to close any active log streams
                    for cb in log_callbacks:
                        try:
                            cb(None)
                        except Exception:
                            pass
            else:
                time.sleep(0.5)
