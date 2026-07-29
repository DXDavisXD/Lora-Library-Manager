"""SQLite storage layer.

The original design kept one JSON sidecar per model plus a whole-library cache
file. That is fine at 50 models and quadratic-feeling at 1500: every stat, every
filter and every search had to walk the tree and parse every sidecar.

Here the filesystem stays the source of truth for *files*, but all derived state
lives in one SQLite database with real indices and an FTS5 index. Sidecars are
still read and written (so Forge/A1111 keep working) — they are just no longer
the query path.

Concurrency: one connection per thread via threading.local. WAL mode lets the
HTTP threads read while a background job writes.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import config

SCHEMA_VERSION = 1

_local = threading.local()
_init_lock = threading.Lock()
_initialised: set[str] = set()

# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

SCHEMA = """
-- A physical model file on disk. `path` is relative to the library root and
-- uses forward slashes so the database is portable across platforms.
CREATE TABLE IF NOT EXISTS model_file (
    id           INTEGER PRIMARY KEY,
    path         TEXT    NOT NULL UNIQUE,
    filename     TEXT    NOT NULL,
    stem         TEXT    NOT NULL,
    ext          TEXT    NOT NULL,
    size         INTEGER NOT NULL,
    mtime        REAL    NOT NULL,
    sha256       TEXT,
    hash_state   TEXT    NOT NULL DEFAULT 'pending',  -- pending|done|error
    -- Sidecar paths discovered during the scan, so enrichment never has to
    -- re-probe the filesystem for files the scanner already saw.
    sidecars     TEXT    NOT NULL DEFAULT '[]',
    present      INTEGER NOT NULL DEFAULT 1,
    first_seen   REAL    NOT NULL,
    last_seen    REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_file_sha     ON model_file(sha256);
CREATE INDEX IF NOT EXISTS ix_file_present ON model_file(present);
CREATE INDEX IF NOT EXISTS ix_file_hstate  ON model_file(hash_state) WHERE hash_state = 'pending';
CREATE INDEX IF NOT EXISTS ix_file_stem    ON model_file(stem);

-- Provider-resolved metadata for a file. One row per file, created lazily.
CREATE TABLE IF NOT EXISTS model (
    file_id             INTEGER PRIMARY KEY REFERENCES model_file(id) ON DELETE CASCADE,
    provider            TEXT,
    provider_model_id   TEXT,
    provider_version_id TEXT,
    name                TEXT,
    version_name        TEXT,
    creator             TEXT,
    base_model          TEXT,
    description         TEXT,
    description_html    TEXT,
    trained_words       TEXT,     -- json array
    tags                TEXT,     -- json array
    preferred_weight    REAL,
    negative_prompt     TEXT,
    nsfw                INTEGER NOT NULL DEFAULT 0,
    meta_state          TEXT NOT NULL DEFAULT 'missing',  -- missing|partial|complete
    fetched_at          REAL,
    source              TEXT      -- provider|sidecar|webui|manual
);
CREATE INDEX IF NOT EXISTS ix_model_base   ON model(base_model);
CREATE INDEX IF NOT EXISTS ix_model_prov   ON model(provider);
CREATE INDEX IF NOT EXISTS ix_model_state  ON model(meta_state);
CREATE INDEX IF NOT EXISTS ix_model_name   ON model(name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS ix_model_ident  ON model(provider, provider_model_id);

-- Everything the user owns. Kept separate from `model` so a metadata refetch
-- can never clobber it, and so it survives a full re-index.
CREATE TABLE IF NOT EXISTS user_data (
    file_id           INTEGER PRIMARY KEY REFERENCES model_file(id) ON DELETE CASCADE,
    rating            INTEGER,
    favorite          INTEGER NOT NULL DEFAULT 0,
    notes             TEXT,
    personal_triggers TEXT,   -- json array
    -- Tags the *user* applied, kept apart from provider tags so a metadata
    -- refresh can never touch them.
    tags              TEXT,   -- json array
    weight            REAL,
    negative_prompt   TEXT,
    use_count         INTEGER NOT NULL DEFAULT 0,
    last_used         REAL,
    updated_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_user_fav    ON user_data(favorite) WHERE favorite = 1;
CREATE INDEX IF NOT EXISTS ix_user_rating ON user_data(rating);

-- User data keyed by content hash rather than path, so it can be reattached
-- after a file is renamed or moved. This is what makes reorganising safe.
CREATE TABLE IF NOT EXISTS user_data_orphan (
    sha256     TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,   -- json blob of the user_data row
    orphaned_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS collection (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    parent_id  INTEGER REFERENCES collection(id) ON DELETE CASCADE,
    color      TEXT,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    UNIQUE(name, parent_id)
);

CREATE TABLE IF NOT EXISTS collection_member (
    collection_id INTEGER NOT NULL REFERENCES collection(id) ON DELETE CASCADE,
    file_id       INTEGER NOT NULL REFERENCES model_file(id) ON DELETE CASCADE,
    PRIMARY KEY (collection_id, file_id)
);
CREATE INDEX IF NOT EXISTS ix_member_file ON collection_member(file_id);

CREATE TABLE IF NOT EXISTS preview (
    id         INTEGER PRIMARY KEY,
    file_id    INTEGER NOT NULL REFERENCES model_file(id) ON DELETE CASCADE,
    path       TEXT NOT NULL,
    is_primary INTEGER NOT NULL DEFAULT 0,
    width      INTEGER,
    height     INTEGER,
    UNIQUE(file_id, path)
);
CREATE INDEX IF NOT EXISTS ix_preview_file ON preview(file_id);

-- Append-only job log. Replaces jobs.json; bounded by a retention sweep.
CREATE TABLE IF NOT EXISTS job (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    label       TEXT NOT NULL,
    priority    TEXT NOT NULL DEFAULT 'normal',
    status      TEXT NOT NULL,           -- pending|running|done|error|cancelled
    total       INTEGER NOT NULL DEFAULT 0,
    current     INTEGER NOT NULL DEFAULT 0,
    message     TEXT,
    detail      TEXT,
    error       TEXT,
    created_at  REAL NOT NULL,
    started_at  REAL,
    ended_at    REAL
);
CREATE INDEX IF NOT EXISTS ix_job_status  ON job(status);
CREATE INDEX IF NOT EXISTS ix_job_created ON job(created_at DESC);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- Search index. Regular (not contentless) FTS5 keyed by rowid = file_id, so a
-- row can be replaced with a plain DELETE + INSERT.
CREATE VIRTUAL TABLE IF NOT EXISTS model_fts USING fts5(
    name, creator, base_model, tags, trigger_words, filename, notes,
    tokenize = 'unicode61 remove_diacritics 2'
);
"""


# --------------------------------------------------------------------------
# Connection management
# --------------------------------------------------------------------------

def connect(path: Path | None = None) -> sqlite3.Connection:
    """Return this thread's connection, creating and initialising on demand."""
    target = str(path or config.db_path())
    conns: dict[str, sqlite3.Connection] = getattr(_local, "conns", None)  # type: ignore[assignment]
    if conns is None:
        conns = {}
        _local.conns = conns
    conn = conns.get(target)
    if conn is not None:
        return conn

    Path(target).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-32000")   # ~32 MB page cache
    conn.execute("PRAGMA busy_timeout=30000")
    conns[target] = conn

    with _init_lock:
        if target not in _initialised:
            _migrate(conn)
            _initialised.add(target)
    return conn


def close_thread_connections() -> None:
    conns = getattr(_local, "conns", None)
    if not conns:
        return
    for conn in conns.values():
        try:
            conn.close()
        except sqlite3.Error:
            pass
    conns.clear()


def _migrate(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?)", (str(SCHEMA_VERSION),)
        )

    # CREATE TABLE IF NOT EXISTS cannot add a column to a table that already
    # exists, so new columns are applied here. Checked against the live schema
    # rather than the version counter: an ALTER that has already happened is
    # then simply skipped, which is what makes this safe to run on every open.
    have = {r["name"] for r in conn.execute("PRAGMA table_info(model)")}
    for column, decl in (("description_html", "TEXT"),):
        if column not in have:
            conn.execute(f"ALTER TABLE model ADD COLUMN {column} {decl}")

    # Future migrations dispatch on int(row["value"]) here.


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

class transaction:
    """Context manager for an explicit IMMEDIATE transaction.

    isolation_level=None means we drive BEGIN/COMMIT ourselves, which keeps
    bulk upserts in a single fsync instead of one per row.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def __enter__(self) -> sqlite3.Connection:
        self.conn.execute("BEGIN IMMEDIATE")
        return self.conn

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.conn.execute("COMMIT")
        else:
            try:
                self.conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
        return False


def jdump(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def jload(value: Any, fallback: Any = None) -> Any:
    if not value:
        return [] if fallback is None else fallback
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return [] if fallback is None else fallback


def meta_get(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


# --------------------------------------------------------------------------
# FTS maintenance
# --------------------------------------------------------------------------

def fts_replace(conn: sqlite3.Connection, file_id: int) -> None:
    """Rebuild the FTS row for one file from its current joined state."""
    row = conn.execute(
        """
        SELECT f.filename,
               m.name, m.creator, m.base_model, m.tags, m.trained_words,
               u.notes, u.personal_triggers, u.tags AS user_tags
        FROM model_file f
        LEFT JOIN model     m ON m.file_id = f.id
        LEFT JOIN user_data u ON u.file_id = f.id
        WHERE f.id = ?
        """,
        (file_id,),
    ).fetchone()
    conn.execute("DELETE FROM model_fts WHERE rowid = ?", (file_id,))
    if row is None:
        return
    triggers = jload(row["trained_words"]) + jload(row["personal_triggers"])
    tags = jload(row["tags"]) + jload(row["user_tags"])
    conn.execute(
        "INSERT INTO model_fts(rowid, name, creator, base_model, tags, trigger_words, filename, notes) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (
            file_id,
            row["name"] or Path(row["filename"]).stem,
            row["creator"] or "",
            row["base_model"] or "",
            " ".join(str(t) for t in tags),
            " ".join(str(t) for t in triggers),
            row["filename"],
            row["notes"] or "",
        ),
    )


def fts_rebuild_all(conn: sqlite3.Connection) -> int:
    conn.execute("DELETE FROM model_fts")
    ids = [r["id"] for r in conn.execute("SELECT id FROM model_file WHERE present = 1")]
    for fid in ids:
        fts_replace(conn, fid)
    return len(ids)


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


def now() -> float:
    return time.time()
