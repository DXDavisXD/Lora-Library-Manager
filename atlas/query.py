"""Read queries for the gallery.

Everything the browser asks for is answered by SQL with a LIMIT. Nothing here
loads the whole library into memory, and nothing here reads a file from disk —
that is the difference between a gallery that opens instantly at 1500 models and
one that generates a 40 MB HTML document first.

Search goes through FTS5 when the term looks like text, and falls back to a
LIKE scan only for very short fragments where a prefix match is more useful than
a token match.
"""

from __future__ import annotations

import re
from typing import Any

from . import db

SORTS: dict[str, str] = {
    "name":     "COALESCE(NULLIF(m.name,''), f.stem) COLLATE NOCASE ASC",
    "name_desc": "COALESCE(NULLIF(m.name,''), f.stem) COLLATE NOCASE DESC",
    "added":    "f.first_seen DESC",
    "oldest":   "f.first_seen ASC",
    "size":     "f.size DESC",
    "rating":   "COALESCE(u.rating,0) DESC, COALESCE(NULLIF(m.name,''), f.stem) COLLATE NOCASE ASC",
    "used":     "COALESCE(u.use_count,0) DESC, COALESCE(u.last_used,0) DESC",
    "base":     "m.base_model COLLATE NOCASE ASC, COALESCE(NULLIF(m.name,''), f.stem) COLLATE NOCASE ASC",
}

_FTS_UNSAFE = re.compile(r'["\^\*\(\):]')


SELECT_FIELDS = """
    f.id, f.path, f.filename, f.stem, f.size, f.mtime, f.sha256, f.hash_state,
    m.name, m.version_name, m.creator, m.base_model, m.provider,
    m.provider_model_id, m.provider_version_id, m.trained_words, m.tags,
    m.preferred_weight, m.nsfw, m.meta_state,
    u.rating, u.favorite, u.notes, u.personal_triggers, u.tags AS user_tags,
    u.weight, u.negative_prompt AS user_negative, u.use_count, u.last_used,
    (SELECT p.path FROM preview p WHERE p.file_id = f.id
       ORDER BY p.is_primary DESC, p.path LIMIT 1) AS preview_path,
    (SELECT COUNT(*) FROM preview p WHERE p.file_id = f.id) AS preview_count
"""


def _fts_query(term: str) -> str | None:
    """Build an FTS5 MATCH expression, or None if a LIKE scan suits better."""
    cleaned = _FTS_UNSAFE.sub(" ", term).strip()
    if not cleaned:
        return None
    tokens = [t for t in cleaned.split() if t]
    if not tokens:
        return None
    if len(tokens) == 1 and len(tokens[0]) < 3:
        return None                      # too short to tokenise usefully
    # Prefix-match the final token so results narrow as the user types.
    parts = [f'"{t}"' for t in tokens[:-1]]
    parts.append(f'"{tokens[-1]}"*')
    return " AND ".join(parts)


def build_filters(params: dict[str, Any]) -> tuple[str, list[Any], list[str]]:
    """Translate request parameters into a WHERE clause."""
    where = ["f.present = 1"]
    args: list[Any] = []
    joins: list[str] = []

    search = (params.get("q") or "").strip()
    if search:
        match = _fts_query(search)
        if match:
            joins.append("JOIN model_fts fts ON fts.rowid = f.id")
            where.append("model_fts MATCH ?")
            args.append(match)
        else:
            where.append(
                "(f.filename LIKE ? ESCAPE '\\' OR COALESCE(m.name,'') LIKE ? ESCAPE '\\')"
            )
            like = "%" + search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            args.extend([like, like])

    base = params.get("base_model")
    if base:
        where.append("m.base_model = ?")
        args.append(base)

    creator = (params.get("creator") or "").strip()
    if creator:
        where.append("m.creator = ?")
        args.append(creator)

    # Everything no collection has claimed — the working set while you are
    # partway through organising, and otherwise only reachable by elimination.
    # A separate flag, not a reserved category name: safe_component() allows
    # leading dots, so any sentinel string could be created for real.
    if params.get("uncategorised"):
        where.append(
            "NOT EXISTS (SELECT 1 FROM collection_member cmc WHERE cmc.file_id = f.id)")

    provider = params.get("provider")
    if provider:
        where.append("m.provider = ?")
        args.append(provider)

    if params.get("favorite"):
        where.append("COALESCE(u.favorite,0) = 1")

    rating = params.get("min_rating")
    if rating:
        where.append("COALESCE(u.rating,0) >= ?")
        args.append(int(rating))

    state = params.get("meta_state")
    if state:
        where.append("COALESCE(m.meta_state,'missing') = ?")
        args.append(state)

    if params.get("missing_preview"):
        where.append("NOT EXISTS (SELECT 1 FROM preview p WHERE p.file_id = f.id)")

    if params.get("nsfw") == "hide":
        where.append("COALESCE(m.nsfw,0) = 0")

    if params.get("rated"):
        where.append("u.rating IS NOT NULL AND u.rating > 0")

    if params.get("has_notes"):
        where.append("COALESCE(u.notes,'') != ''")

    # "Has custom triggers" — only models where the user has defined their own
    # trigger words (personal_triggers), not provider-supplied ones.
    if params.get("has_triggers"):
        where.append("COALESCE(u.personal_triggers,'[]') NOT IN ('[]','')")

    collection = params.get("collection")
    category = (params.get("category") or "").strip()
    subcategory = (params.get("subcategory") or "").strip()
    if collection:
        joins.append("JOIN collection_member cm ON cm.file_id = f.id")
        where.append("cm.collection_id = ?")
        args.append(int(collection))
    elif category:
        # Match a category and everything nested under it. "Anime" catches
        # "Anime", "Anime / Fate", "Anime / Fate / Rin", …; adding a
        # sub-category narrows the prefix to "Anime / Fate". EXISTS (not a
        # JOIN) so a model in several matching collections is counted once.
        prefix = f"{category} / {subcategory}" if subcategory else category
        like = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        where.append(
            "EXISTS (SELECT 1 FROM collection_member cmc "
            "JOIN collection cc ON cc.id = cmc.collection_id "
            "WHERE cmc.file_id = f.id AND (cc.name = ? OR cc.name LIKE ? ESCAPE '\\'))"
        )
        args.extend([prefix, f"{like} / %"])

    trigger = (params.get("trigger") or "").strip()
    if trigger:
        where.append(
            "(COALESCE(m.trained_words,'[]') LIKE ? OR COALESCE(u.personal_triggers,'[]') LIKE ?)"
        )
        args.extend([f"%{trigger}%", f"%{trigger}%"])

    return " AND ".join(where), args, joins


def list_models(conn, params: dict[str, Any]) -> dict[str, Any]:
    where, args, joins = build_filters(params)
    join_sql = " ".join(dict.fromkeys(joins))

    sort_key = str(params.get("sort") or "name")
    order = SORTS.get(sort_key, SORTS["name"])

    try:
        page = max(1, int(params.get("page") or 1))
    except (TypeError, ValueError):
        page = 1
    try:
        size = int(params.get("page_size") or 0)
    except (TypeError, ValueError):
        size = 0
    size = max(1, min(size or 120, 500))
    offset = (page - 1) * size

    base_from = f"""
        FROM model_file f
        LEFT JOIN model m     ON m.file_id = f.id
        LEFT JOIN user_data u ON u.file_id = f.id
        {join_sql}
        WHERE {where}
    """

    total = int(conn.execute(f"SELECT COUNT(*) c {base_from}", args).fetchone()["c"])
    rows = conn.execute(
        f"SELECT {SELECT_FIELDS} {base_from} ORDER BY {order} LIMIT ? OFFSET ?",
        (*args, size, offset),
    ).fetchall()

    return {
        "items": [serialise(r) for r in rows],
        "total": total,
        "page": page,
        "page_size": size,
        "pages": (total + size - 1) // size,
    }


def serialise(row) -> dict[str, Any]:
    name = row["name"] or row["stem"]
    return {
        "id": row["id"],
        "path": row["path"],
        "filename": row["filename"],
        "name": name,
        "version_name": row["version_name"] or "",
        "creator": row["creator"] or "",
        "base_model": row["base_model"] or "",
        "provider": row["provider"] or "",
        "provider_model_id": row["provider_model_id"] or "",
        "provider_version_id": row["provider_version_id"] or "",
        "size": row["size"],
        "sha256": row["sha256"] or "",
        "folder": row["path"].rsplit("/", 1)[0] if "/" in row["path"] else "",
        "trained_words": db.jload(row["trained_words"]),
        "personal_triggers": db.jload(row["personal_triggers"]),
        "tags": db.jload(row["tags"]),
        "user_tags": db.jload(row["user_tags"]),
        "user_negative": row["user_negative"] or "",
        "preferred_weight": row["preferred_weight"],
        "weight": row["weight"],
        "nsfw": bool(row["nsfw"]),
        "meta_state": row["meta_state"] or "missing",
        "rating": row["rating"],
        "favorite": bool(row["favorite"]),
        "notes": row["notes"] or "",
        "use_count": row["use_count"] or 0,
        "last_used": row["last_used"],
        "preview": row["preview_path"],
        "preview_count": row["preview_count"] or 0,
    }


def get_model(conn, file_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        f"""SELECT {SELECT_FIELDS}
            FROM model_file f
            LEFT JOIN model m     ON m.file_id = f.id
            LEFT JOIN user_data u ON u.file_id = f.id
            WHERE f.id = ?""",
        (file_id,),
    ).fetchone()
    if row is None:
        return None
    data = serialise(row)
    detail = conn.execute(
        "SELECT description, description_html, negative_prompt, source, fetched_at "
        "FROM model WHERE file_id=?",
        (file_id,),
    ).fetchone()
    if detail:
        data.update({
            "description": detail["description"] or "",
            "description_html": detail["description_html"] or "",
            "negative_prompt": detail["negative_prompt"] or "",
            "source": detail["source"] or "",
            "fetched_at": detail["fetched_at"],
        })
    data["previews"] = [
        r["path"] for r in conn.execute(
            "SELECT path FROM preview WHERE file_id=? ORDER BY is_primary DESC, path", (file_id,)
        )
    ]
    data["collections"] = [
        {"id": r["id"], "name": r["name"]}
        for r in conn.execute(
            "SELECT c.id, c.name FROM collection c "
            "JOIN collection_member cm ON cm.collection_id = c.id WHERE cm.file_id=? "
            "ORDER BY c.name",
            (file_id,),
        )
    ]
    if data["sha256"]:
        data["duplicates"] = [
            {"id": r["id"], "path": r["path"]}
            for r in conn.execute(
                "SELECT id, path FROM model_file WHERE sha256=? AND id!=? AND present=1",
                (data["sha256"], file_id),
            )
        ]
    else:
        data["duplicates"] = []

    # Other versions of the same model, so the drawer can offer a version switch.
    data["versions"] = []
    if data["provider"] and data["provider_model_id"]:
        data["versions"] = [
            {"id": r["id"], "version_name": r["version_name"] or "",
             "version_id": r["provider_version_id"] or "", "path": r["path"],
             "current": r["id"] == file_id}
            for r in conn.execute(
                "SELECT f.id, f.path, m.version_name, m.provider_version_id "
                "FROM model_file f JOIN model m ON m.file_id=f.id "
                "WHERE f.present=1 AND m.provider=? AND m.provider_model_id=? "
                "ORDER BY m.provider_version_id DESC",
                (data["provider"], data["provider_model_id"]))
        ]

    prov = providers_module().get(data["provider"]) if data["provider"] else None
    data["page_url"] = (
        f"{getattr(prov, 'mirror', prov.website)}/models/{data['provider_model_id']}"
        if prov and data["provider_model_id"] else ""
    )
    return data


def providers_module():
    from . import providers
    return providers


def facets(conn) -> dict[str, Any]:
    """Filter options with counts, so the UI never offers an empty filter."""
    def rows(sql: str) -> list[dict[str, Any]]:
        return [{"value": r[0], "count": r[1]} for r in conn.execute(sql) if r[0]]

    return {
        "base_models": rows(
            "SELECT m.base_model, COUNT(*) FROM model_file f JOIN model m ON m.file_id=f.id "
            "WHERE f.present=1 AND m.base_model != '' GROUP BY m.base_model ORDER BY 2 DESC"
        ),
        "providers": rows(
            "SELECT m.provider, COUNT(*) FROM model_file f JOIN model m ON m.file_id=f.id "
            "WHERE f.present=1 AND m.provider != '' GROUP BY m.provider ORDER BY 2 DESC"
        ),
        "collections": [
            {"id": r["id"], "name": r["name"], "count": r["c"], "color": r["color"]}
            for r in conn.execute(
                "SELECT c.id, c.name, c.color, COUNT(cm.file_id) c FROM collection c "
                "LEFT JOIN collection_member cm ON cm.collection_id=c.id "
                "GROUP BY c.id ORDER BY c.sort_order, c.name"
            )
        ],
        "categories": category_tree(conn),
        "uncategorised": int(conn.execute(
            "SELECT COUNT(*) FROM model_file f WHERE f.present=1 AND NOT EXISTS "
            "(SELECT 1 FROM collection_member cm WHERE cm.file_id = f.id)"
        ).fetchone()[0]),
    }


def category_tree(conn) -> list[dict[str, Any]]:
    """Collections shaped as a category → sub-category tree for the cascade.

    A collection named "Anime / Fate / Rin" contributes category "Anime" and
    sub-category "Fate" (only the first two levels drive the picker; deeper
    nesting still filters via prefix match). Counts are distinct model counts,
    so a model in two sub-categories of one category is counted once there.
    """
    rows = conn.execute(
        "SELECT c.name, cm.file_id FROM collection c "
        "JOIN collection_member cm ON cm.collection_id = c.id "
        "JOIN model_file f ON f.id = cm.file_id AND f.present = 1"
    ).fetchall()

    cats: dict[str, dict[str, Any]] = {}
    for row in rows:
        parts = [p.strip() for p in row["name"].split(" / ") if p.strip()]
        if not parts:
            continue
        category = parts[0]
        entry = cats.setdefault(category, {"name": category, "files": set(), "subs": {}})
        entry["files"].add(row["file_id"])
        if len(parts) > 1:
            sub = parts[1]
            sentry = entry["subs"].setdefault(sub, {"name": sub, "files": set()})
            sentry["files"].add(row["file_id"])

    out = []
    for category in sorted(cats):
        entry = cats[category]
        subs = [
            {"name": s, "count": len(entry["subs"][s]["files"])}
            for s in sorted(entry["subs"])
        ]
        out.append({
            "name": category,
            "count": len(entry["files"]),
            "subcategories": subs,
        })
    return out


def stats(conn) -> dict[str, Any]:
    """Library health. One query per figure, all index-backed."""
    def scalar(sql: str, args: tuple = ()) -> int:
        row = conn.execute(sql, args).fetchone()
        return int(row[0] or 0)

    total = scalar("SELECT COUNT(*) FROM model_file WHERE present=1")
    return {
        "total": total,
        "missing_files": scalar("SELECT COUNT(*) FROM model_file WHERE present=0"),
        "total_bytes": scalar("SELECT COALESCE(SUM(size),0) FROM model_file WHERE present=1"),
        "missing_preview": scalar(
            "SELECT COUNT(*) FROM model_file f WHERE f.present=1 AND NOT EXISTS "
            "(SELECT 1 FROM preview p WHERE p.file_id=f.id)"
        ),
        "missing_metadata": scalar(
            "SELECT COUNT(*) FROM model_file f LEFT JOIN model m ON m.file_id=f.id "
            "WHERE f.present=1 AND COALESCE(m.meta_state,'missing')='missing'"
        ),
        "partial_metadata": scalar(
            "SELECT COUNT(*) FROM model_file f JOIN model m ON m.file_id=f.id "
            "WHERE f.present=1 AND m.meta_state='partial'"
        ),
        "unhashed": scalar(
            "SELECT COUNT(*) FROM model_file WHERE present=1 AND sha256 IS NULL"
        ),
        "favorites": scalar(
            "SELECT COUNT(*) FROM user_data u JOIN model_file f ON f.id=u.file_id "
            "WHERE f.present=1 AND u.favorite=1"
        ),
        "rated": scalar(
            "SELECT COUNT(*) FROM user_data u JOIN model_file f ON f.id=u.file_id "
            "WHERE f.present=1 AND u.rating IS NOT NULL"
        ),
        "collections": scalar("SELECT COUNT(*) FROM collection"),
        "duplicate_groups": scalar(
            "SELECT COUNT(*) FROM (SELECT sha256 FROM model_file WHERE present=1 "
            "AND sha256 IS NOT NULL GROUP BY sha256 HAVING COUNT(*)>1)"
        ),
        "last_scan": db.meta_get(conn, "last_scan"),
    }
