"""Background job queue.

One worker thread drains a priority queue. Jobs report progress through a
handle, which coalesces updates before publishing: a hash pass over 1500 files
can emit thousands of progress ticks a second, and pushing every one down an
SSE socket would spend more time serialising JSON than hashing.

Cancellation is cooperative — every long loop polls ``handle.cancelled``.
"""

from __future__ import annotations

import heapq
import itertools
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from . import db, events

PRIORITY_ORDER = {"interactive": 0, "normal": 1, "bulk": 2}
PROGRESS_INTERVAL = 0.2      # seconds between published progress frames
HISTORY_LIMIT = 200


class Cancelled(Exception):
    """Raised inside a job body when cancellation is observed."""


@dataclass(order=True)
class _Entry:
    sort_key: tuple = field(compare=True)
    job: "Job" = field(compare=False)


@dataclass
class Job:
    id: str
    kind: str
    label: str
    fn: Callable[["JobHandle"], Any]
    priority: str = "normal"
    status: str = "pending"
    total: int = 0
    current: int = 0
    message: str = ""
    detail: str = ""
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    ended_at: float | None = None
    result: Any = None
    _cancel: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> dict[str, Any]:
        percent = 0
        if self.total > 0:
            percent = max(0, min(100, int(self.current * 100 / self.total)))
        elif self.status == "done":
            percent = 100
        elapsed = None
        if self.started_at:
            elapsed = (self.ended_at or time.time()) - self.started_at
        eta = None
        if elapsed and self.current > 0 and self.total > self.current and self.status == "running":
            eta = elapsed / self.current * (self.total - self.current)
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "priority": self.priority,
            "status": self.status,
            "total": self.total,
            "current": self.current,
            "percent": percent,
            "message": self.message,
            "detail": self.detail,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "elapsed": elapsed,
            "eta": eta,
            "cancelling": self._cancel.is_set() and self.status == "running",
        }


class JobHandle:
    """The surface a job body sees."""

    def __init__(self, job: Job, queue: "JobQueue"):
        self._job = job
        self._queue = queue
        self._last_publish = 0.0

    @property
    def cancelled(self) -> bool:
        return self._job._cancel.is_set()

    def check_cancelled(self) -> None:
        if self.cancelled:
            raise Cancelled()

    def set_total(self, total: int) -> None:
        self._job.total = max(0, int(total))
        self._publish(force=True)

    def progress(
        self, current: int | None = None, message: str | None = None, detail: str | None = None
    ) -> None:
        if current is not None:
            self._job.current = int(current)
        if message is not None:
            self._job.message = message
        if detail is not None:
            self._job.detail = detail
        self._publish()

    def advance(self, step: int = 1, detail: str | None = None) -> None:
        self._job.current += step
        if detail is not None:
            self._job.detail = detail
        self._publish()

    def log(self, line: str, level: str = "info") -> None:
        events.log(line, level)

    def _publish(self, force: bool = False) -> None:
        now = time.monotonic()
        if force or (now - self._last_publish) >= PROGRESS_INTERVAL:
            self._last_publish = now
            self._queue._emit(self._job)


class JobQueue:
    def __init__(self) -> None:
        self._heap: list[_Entry] = []
        self._counter = itertools.count()
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._jobs: dict[str, Job] = {}
        self._history: list[Job] = []
        self._running: Job | None = None
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        # Create and migrate the database on this thread *before* the worker
        # exists. Otherwise a first run against a brand-new library has the
        # submitting thread and the worker both racing to run the schema
        # script, which surfaces as "database is locked".
        try:
            db.connect()
        except Exception:                                   # noqa: BLE001
            pass          # no library configured yet; jobs will report it
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, name="job-worker", daemon=True)
        self._worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._wake:
            self._wake.notify_all()
        if self._running:
            self._running._cancel.set()
        if self._worker:
            self._worker.join(timeout=timeout)

    # -- submission ---------------------------------------------------------

    def submit(
        self,
        kind: str,
        label: str,
        fn: Callable[[JobHandle], Any],
        priority: str = "normal",
        coalesce: bool = True,
    ) -> Job:
        """Queue a job.

        ``coalesce`` returns the existing job when one of the same kind is
        already pending or running. Double-clicking "Scan Library" should not
        enqueue two full scans.
        """
        with self._lock:
            if coalesce:
                for existing in self._jobs.values():
                    if existing.kind == kind and existing.status in ("pending", "running"):
                        return existing
            job = Job(id=uuid.uuid4().hex[:12], kind=kind, label=label, fn=fn, priority=priority)
            self._jobs[job.id] = job
            rank = PRIORITY_ORDER.get(priority, 1)
            heapq.heappush(self._heap, _Entry((rank, next(self._counter)), job))
            self._wake.notify()
        self._persist(job)
        self._emit(job)
        return job

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status not in ("pending", "running"):
                return False
            job._cancel.set()
            if job.status == "pending":
                job.status = "cancelled"
                job.ended_at = time.time()
                job.message = "Cancelled before start"
                self._retire_locked(job)
        self._persist(job)
        self._emit(job)
        return True

    def cancel_all(self) -> int:
        with self._lock:
            ids = [j.id for j in self._jobs.values() if j.status in ("pending", "running")]
        return sum(1 for jid in ids if self.cancel(jid))

    # -- introspection ------------------------------------------------------

    def active(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = [j for j in self._jobs.values() if j.status in ("pending", "running")]
        jobs.sort(key=lambda j: (PRIORITY_ORDER.get(j.priority, 1), j.created_at))
        return [j.snapshot() for j in jobs]

    def history(self, limit: int = 40) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._history[-limit:])
        return [j.snapshot() for j in reversed(items)]

    def state(self) -> dict[str, Any]:
        active = self.active()
        running = next((j for j in active if j["status"] == "running"), None)
        return {
            "running": running,
            "queued": [j for j in active if j["status"] == "pending"],
            "queue_length": sum(1 for j in active if j["status"] == "pending"),
        }

    # -- internals ----------------------------------------------------------

    def _retire_locked(self, job: Job) -> None:
        self._jobs.pop(job.id, None)
        self._history.append(job)
        if len(self._history) > HISTORY_LIMIT:
            del self._history[: len(self._history) - HISTORY_LIMIT]

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._wake:
                while not self._heap and not self._stop.is_set():
                    self._wake.wait(timeout=1.0)
                if self._stop.is_set():
                    return
                entry = heapq.heappop(self._heap)
                job = entry.job
                if job.status != "pending":
                    continue
                job.status = "running"
                job.started_at = time.time()
                self._running = job

            handle = JobHandle(job, self)
            self._emit(job)
            try:
                job.result = job.fn(handle)
                if job._cancel.is_set():
                    job.status = "cancelled"
                    job.message = job.message or "Cancelled"
                else:
                    job.status = "done"
                    job.message = job.message or "Completed"
                    if job.total:
                        job.current = job.total
            except Cancelled:
                job.status = "cancelled"
                job.message = "Cancelled"
            except Exception as exc:                      # noqa: BLE001 - reported to the user
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = "Failed"
                events.log(f"[{job.label}] {job.error}", "error")
                events.log(traceback.format_exc().strip().splitlines()[-1], "error")
            finally:
                job.ended_at = time.time()
                with self._lock:
                    self._running = None
                    self._retire_locked(job)
                self._persist(job)
                self._emit(job)

    def _emit(self, job: Job) -> None:
        events.publish("job", {"job": job.snapshot(), **self.state()})

    def _persist(self, job: Job) -> None:
        """Mirror job state into the database for post-restart history."""
        try:
            conn = db.connect()
            with db.transaction(conn):
                conn.execute(
                    "INSERT INTO job(id, kind, label, priority, status, total, current, "
                    "message, detail, error, created_at, started_at, ended_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET status=excluded.status, "
                    "total=excluded.total, current=excluded.current, message=excluded.message, "
                    "detail=excluded.detail, error=excluded.error, "
                    "started_at=excluded.started_at, ended_at=excluded.ended_at",
                    (job.id, job.kind, job.label, job.priority, job.status, job.total,
                     job.current, job.message, job.detail, job.error, job.created_at,
                     job.started_at, job.ended_at),
                )
                if job.status in ("done", "error", "cancelled"):
                    conn.execute(
                        "DELETE FROM job WHERE id IN ("
                        "  SELECT id FROM job ORDER BY created_at DESC LIMIT -1 OFFSET ?)",
                        (HISTORY_LIMIT,),
                    )
        except Exception:                                  # noqa: BLE001 - never kill a job over bookkeeping
            pass


QUEUE = JobQueue()
