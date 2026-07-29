"""Sidecar JSON: reading, repairing and writing.

The library must stay usable from Forge / AUTOMATIC1111 while Atlas is running,
so sidecars remain first-class:

* ``<stem>.civitai.info``  — CivitAI's own dump, written by several tools.
* ``<stem>.json``          — the WebUI's user-facing file (activation text,
                             preferred weight, notes, negative text).
* ``<stem>.atlas.json``    — ours, for anything the others have no field for.

Writes are atomic and *merge-preserving*: unknown keys written by another tool
are read back and re-emitted untouched. A metadata refresh must never be able to
silently drop a field Atlas does not understand.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Lenient loading
# ---------------------------------------------------------------------------

_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_MISSING_COMMA = re.compile(r'("(?:[^"\\]|\\.)*"\s*:\s*(?:"(?:[^"\\]|\\.)*"|[\d.eE+-]+|true|false|null))\s*\n(\s*")')
_COMMENTS = re.compile(r"^\s*//.*$", re.MULTILINE)


def load_lenient(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Parse JSON, repairing the failure modes seen in the wild.

    Returns ``(data, repair_note)``. ``repair_note`` is None when the file
    parsed cleanly, otherwise a short description of what had to be fixed — the
    caller surfaces it so the user can choose to rewrite the file.
    """
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        try:
            raw = path.read_bytes().decode("utf-8", errors="replace")
        except OSError:
            return None, "unreadable"

    if not raw.strip():
        return None, "empty"

    try:
        return json.loads(raw), None
    except json.JSONDecodeError:
        pass

    repairs: list[str] = []
    fixed = raw

    if "//" in fixed:
        candidate = _COMMENTS.sub("", fixed)
        if candidate != fixed:
            fixed, _ = candidate, repairs.append("comments")

    candidate = _TRAILING_COMMA.sub(r"\1", fixed)
    if candidate != fixed:
        fixed = candidate
        repairs.append("trailing comma")

    candidate = _MISSING_COMMA.sub(r"\1,\n\2", fixed)
    if candidate != fixed:
        fixed = candidate
        repairs.append("missing comma")

    try:
        return json.loads(fixed), ", ".join(repairs) or "reformatted"
    except json.JSONDecodeError:
        pass

    # Last resort: the largest balanced {...} span. Recovers files truncated by
    # an interrupted write, which is the common case for a killed download.
    start = fixed.find("{")
    if start >= 0:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(fixed)):
            ch = fixed[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(fixed[start:i + 1]), "truncated tail discarded"
                    except json.JSONDecodeError:
                        break
    return None, "unparseable"


def write_atomic(path: Path, data: dict[str, Any]) -> None:
    """Write JSON such that a crash can never leave a half-file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=".atlas-")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_HTML_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t]+")


def clean_text(value: Any, limit: int = 8000) -> str:
    if not value:
        return ""
    text = _HTML_TAG.sub(" ", str(value))
    text = (text.replace("&nbsp;", " ").replace("&amp;", "&")
                .replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"'))
    text = _WS.sub(" ", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())[:limit]


# Provider descriptions are author-written HTML. Flattening them to one
# paragraph, which is what clean_text() does, throws away the headings, lists
# and line breaks that make a long description readable at all.
#
# This is a strict allowlist, not a blocklist: anything not named here is
# dropped, so a tag nobody thought about cannot arrive later and be trusted by
# default. Attributes go entirely, except href on a link, which is then
# restricted to http/https/mailto — that rules out javascript: and data: URLs.
# The result is sanitised once, on the way in, so the browser is only ever
# handed markup that has already been through this.
_ALLOWED_TAGS = {
    "p", "br", "b", "strong", "i", "em", "u", "s", "code", "pre",
    "ul", "ol", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6",
    "a", "hr", "span", "div",
}
# Rendered as their block equivalent rather than kept: these carry layout or
# fetch behaviour we do not want, but their text is worth keeping.
_UNWRAP_TAGS = {"span", "div"}
_SAFE_SCHEME = re.compile(r"^\s*(https?:|mailto:)", re.I)
_TAG_RE = re.compile(r"<(/?)([a-zA-Z0-9]+)([^>]*)>")
_HREF_RE = re.compile(r"""href\s*=\s*("([^"]*)"|'([^']*)'|([^\s>]+))""", re.I)


def _richest(*candidates: Any) -> str:
    """Pick the candidate that still carries markup, else the first non-empty.

    A sidecar usually holds the description twice: once flattened at the top
    level, where write_metadata_sidecar puts the plain text the WebUI reads,
    and once as the provider wrote it under ``model``. Taking the first
    non-empty candidate therefore reliably picked the flattened copy and threw
    the formatting away — which is exactly what it looked like from the outside:
    a fix that changed nothing.
    """
    for candidate in candidates:
        if candidate and "<" in str(candidate):
            return str(candidate)
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return ""


def clean_html(value: Any, limit: int = 20000) -> str:
    """Reduce provider HTML to a small, safe subset for display."""
    if not value:
        return ""
    raw = str(value)[:limit * 2]

    def replace(match: re.Match) -> str:
        closing, name, attrs = match.group(1), match.group(2).lower(), match.group(3)
        if name not in _ALLOWED_TAGS:
            return ""                       # drop the tag, keep its text
        if name in _UNWRAP_TAGS:
            return ""                       # unwrap: the text stays, the box goes
        if name == "a":
            if closing:
                return "</a>"
            href = ""
            found = _HREF_RE.search(attrs or "")
            if found:
                href = found.group(2) or found.group(3) or found.group(4) or ""
            if not _SAFE_SCHEME.match(href):
                return ""                   # unsafe or relative target: unwrap it
            safe = (href.replace("&", "&amp;").replace('"', "&quot;")
                        .replace("<", "&lt;").replace(">", "&gt;"))
            return f'<a href="{safe}" target="_blank" rel="noopener noreferrer nofollow">'
        return f"<{'/' if closing else ''}{name}>"

    out = _TAG_RE.sub(replace, raw)
    # Any surviving angle bracket is literal text, not markup.
    out = re.sub(r"<(?![/a-zA-Z])", "&lt;", out)
    return out[:limit].strip()


def _as_list(value: Any) -> list[str]:
    """Coerce the several shapes trigger words arrive in into a clean list."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace("\n", ",").split(",")]
        return [p for p in parts if p]
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                for key in ("name", "tagName", "word", "value"):
                    if item.get(key):
                        out.append(str(item[key]).strip())
                        break
        return out
    return []


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def candidate_paths(model_path: Path) -> list[Path]:
    """Sidecar paths for a model file, most authoritative first."""
    base = model_path.with_suffix("")
    return [
        Path(str(base) + ".civitai.info"),
        base.with_suffix(".json"),
        Path(str(base) + ".atlas.json"),
        model_path.parent / "metadata.json",
    ]


def _authority(path: Path) -> int:
    """Rank a known sidecar so merge order matches :func:`candidate_paths`."""
    name = path.name.lower()
    if name.endswith(".civitai.info"):
        return 0
    if name.endswith(".atlas.json"):
        return 2
    if name == "metadata.json":
        return 3
    return 1                       # <stem>.json, the WebUI file


def read_all(
    model_path: Path, known: list[Path] | None = None
) -> tuple[dict[str, Any], list[str], list[tuple[Path, dict[str, Any]]]]:
    """Merge every sidecar next to *model_path*.

    Returns ``(merged, notes, parsed)``. ``parsed`` is the per-file result in
    merge order, so a caller that needs to know *which* file a value came from
    does not have to read and parse them a second time.

    Pass ``known`` when the caller already knows which sidecars exist — the
    scanner collects them during its single directory pass, and reusing that
    avoids four speculative ``stat`` calls per model. Without it we fall back
    to probing, which is what a one-off CLI call wants.
    """
    if known is not None:
        paths = sorted(known, key=_authority)
        probe = False
    else:
        paths = candidate_paths(model_path)
        probe = True

    merged: dict[str, Any] = {}
    notes: list[str] = []
    parsed: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        if probe and not path.exists():
            continue
        data, repair = load_lenient(path)
        if data is None:
            notes.append(f"{path.name}: {repair}")
            continue
        if repair:
            notes.append(f"{path.name}: repaired ({repair})")
        if isinstance(data, dict):
            parsed.append((path, data))
            # User-data files are returned in `parsed` for the adoption step but
            # never merged into the metadata blob — their "notes"/"tags" mean
            # something different from the provider fields of the same name.
            lower = path.name.lower()
            if lower == LEGACY_USER_DATA or lower.endswith(USER_DATA_SUFFIX):
                continue
            # Earlier files win; later ones only fill gaps.
            for key, value in data.items():
                merged.setdefault(key, value)
    return merged, notes, parsed


def normalise(raw: dict[str, Any], filename: str) -> dict[str, Any]:
    """Flatten a sidecar of any known shape into Atlas's field names."""
    model = raw.get("model") if isinstance(raw.get("model"), dict) else {}

    name = (model.get("name") or raw.get("modelName") or raw.get("name") or "").strip()
    version_name = (raw.get("versionName") or raw.get("version_name") or "").strip()
    if not version_name and raw.get("name") and name and raw["name"] != name:
        version_name = str(raw["name"]).strip()

    creator = ""
    for source in (raw.get("creator"), model.get("creator")):
        if isinstance(source, dict):
            creator = str(source.get("username") or "").strip()
        elif isinstance(source, str):
            creator = source.strip()
        if creator:
            break

    # Trigger words. CivitAI ships these as "groups": a list where each element
    # is a comma-joined outfit ("long hair, blue eyes, sailor uniform"). Prefer
    # that structured list and keep each element whole — splitting it would turn
    # one group into a dozen loose words. Only fall back to splitting a flat
    # activation-text string when no list form exists, so a model that has both
    # never shows the groups AND their split-out words together.
    grouped = raw.get("trainedWords")
    if not isinstance(grouped, list):
        grouped = raw.get("trained_words")
    if isinstance(grouped, list) and grouped:
        triggers = _dedupe([str(t).strip() for t in grouped if str(t).strip()])
    else:
        # Only a flat activation-text string. Keep it as a single group chip
        # rather than splitting into loose words — that is how the WebUI treats
        # it, and it matches the grouped display everywhere else.
        activation = str(raw.get("activation text") or raw.get("activation_text") or "").strip()
        if activation:
            triggers = [activation]
        else:
            triggers = _dedupe(_as_list(raw.get("trigger_words")))

    tags = _dedupe(_as_list(raw.get("tags")) or _as_list(model.get("tags")))

    weight = raw.get("preferred weight", raw.get("preferred_weight", raw.get("preferredWeight")))
    try:
        weight = float(weight) if weight not in (None, "") else None
    except (TypeError, ValueError):
        weight = None

    # Provenance. The previous manager exports a flat schema with explicit
    # ``provider`` / ``provider_model_id`` / ``provider_version_id`` /
    # ``file_sha256`` keys — recognising those is what keeps an adopted
    # library's metadata graded "complete" instead of "partial".
    provider = str(raw.get("provider") or raw.get("_atlas_provider") or "").strip()
    model_id = (raw.get("provider_model_id") or raw.get("modelId")
                or model.get("id") or raw.get("_atlas_model_id") or "")
    version_id = (raw.get("provider_version_id") or raw.get("_atlas_version_id") or "")
    if not version_id and isinstance(raw.get("id"), (int, str)):
        version_id = raw.get("id")
    if not provider and (model_id or version_id):
        provider = "civitai"   # historical sidecars predate multi-provider support

    sha = ""
    if isinstance(raw.get("file_sha256"), str):
        sha = raw["file_sha256"].lower()
    if not sha:
        files = raw.get("files")
        if isinstance(files, list):
            for entry in files:
                if not isinstance(entry, dict):
                    continue
                hashes = entry.get("hashes") or {}
                if isinstance(hashes, dict) and hashes.get("SHA256"):
                    sha = str(hashes["SHA256"]).lower()
                    break
    if not sha and isinstance(raw.get("sha256"), str):
        sha = raw["sha256"].lower()

    return {
        "provider": str(provider),
        "provider_model_id": str(model_id),
        "provider_version_id": str(version_id),
        "name": name or Path(filename).stem,
        "version_name": version_name,
        "creator": creator,
        "base_model": str(raw.get("baseModel") or raw.get("base_model") or "").strip(),
        # Plain text stays the canonical field — it is what the WebUI reads and
        # what the search index tokenises. The formatted version rides alongside
        # for our own detail panel.
        "description": clean_text(raw.get("description") or model.get("description")),
        "description_html": clean_html(_richest(
            raw.get("description_html"),
            model.get("description"),
            raw.get("description"))),
        "trained_words": triggers,
        "tags": tags,
        "preferred_weight": weight,
        "negative_prompt": str(raw.get("negative text") or raw.get("negative_prompt") or "").strip(),
        "nsfw": 1 if (raw.get("nsfw") or model.get("nsfw")) else 0,
        "sha256": sha,
        "notes": str(raw.get("notes") or "").strip(),
    }


def completeness(fields: dict[str, Any]) -> str:
    """Grade normalised metadata: complete / partial / missing."""
    have_identity = bool(fields.get("name")) and bool(fields.get("base_model"))
    have_provenance = bool(fields.get("provider_version_id")) or bool(fields.get("sha256"))
    have_usage = bool(fields.get("trained_words")) or bool(fields.get("description"))
    score = sum((have_identity, have_provenance, have_usage))
    if score == 3:
        return "complete"
    if score == 0:
        return "missing"
    return "partial"


# Files a *human* edits, as opposed to files a downloader generates.
# ``user_data.json`` is the previous LoRA Library Manager's own user-data file;
# importing it is the difference between adopting a library and starting over.
USER_AUTHORED_NAMES = ("metadata.json", "user_data.json")

LEGACY_USER_DATA = "user_data.json"

# Per-model user data. The legacy file is per-FOLDER, which was fine when every
# model had its own folder — but versions of one model deliberately share a
# folder now, and two versions writing to one file means the second silently
# overwrites the first's rating and notes. So each model also gets its own
# ``<stem>.user.json``, which is unambiguous. The folder-level file is still
# maintained when a folder holds a single model, so the previous app keeps
# working in the common case.
USER_DATA_SUFFIX = ".user.json"


def user_data_path(model_path: Path) -> Path:
    """The per-model user-data file for *model_path*."""
    return Path(str(model_path.with_suffix("")) + USER_DATA_SUFFIX)


def _folder_model_count(folder: Path) -> int:
    from . import config
    try:
        return sum(1 for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in config.MODEL_EXTENSIONS)
    except OSError:
        return 0


def is_user_authored(path: Path, model_path: Path) -> bool:
    name = path.name.lower()
    if name.endswith(".civitai.info") or name.endswith(".atlas.json"):
        return False        # machine-written; safe to regenerate
    if name == user_data_path(model_path).name.lower():
        return True         # this model's own user file
    stem_json = model_path.with_suffix("").with_suffix(".json").name.lower()
    return name == stem_json or name in USER_AUTHORED_NAMES


def read_legacy_user_data(model_path: Path,
                          parsed: list[tuple[Path, dict[str, Any]]] | None = None
                          ) -> dict[str, Any] | None:
    """Read a ``user_data.json`` written by the previous manager.

    Schema (v1)::

        {"schema_version": 1, "last_modified": "...", "personal_trigger_words": [],
         "notes": "", "rating": 0, "collections": ["Anime / Kiss X Sis"],
         "preferred_weight": 0.8, "negative_prompt": "..."}

    Returns None when there is no such file, so the caller can tell "absent"
    from "present but empty".

    The per-model ``<stem>.user.json`` wins over the folder-level file: when a
    folder holds several versions, only the per-model file can be trusted to
    describe *this* one.
    """
    own_name = user_data_path(model_path).name.lower()
    data: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None

    if parsed is not None:
        for path, payload in parsed:
            lower = path.name.lower()
            if lower == own_name:
                data = payload
                break
            if lower == LEGACY_USER_DATA:
                fallback = payload
    else:
        own = user_data_path(model_path)
        if own.exists():
            loaded, _ = load_lenient(own)
            if isinstance(loaded, dict):
                data = loaded
        if data is None:
            candidate = model_path.parent / LEGACY_USER_DATA
            if candidate.exists():
                loaded, _ = load_lenient(candidate)
                if isinstance(loaded, dict):
                    fallback = loaded

    if data is None:
        data = fallback
    if data is None:
        return None

    rating = data.get("rating")
    try:
        rating = int(rating) if rating not in (None, "", 0) else None
    except (TypeError, ValueError):
        rating = None

    weight = data.get("preferred_weight")
    try:
        weight = float(weight) if weight not in (None, "") else None
    except (TypeError, ValueError):
        weight = None
    if weight == 0.0:
        weight = None                     # the old app wrote 0.0 for "unset"

    return {
        "rating": rating,
        "notes": str(data.get("notes") or "").strip(),
        "personal_triggers": _dedupe(_as_list(
            data.get("personal_trigger_words") or data.get("personal_triggers"))),
        "collections": [str(c).strip() for c in (data.get("collections") or []) if str(c).strip()],
        "weight": weight,
        "negative_prompt": str(data.get("negative_prompt") or "").strip(),
        "tags": _dedupe(_as_list(data.get("tags") or data.get("user_tags"))),
        "favorite": bool(data.get("favorite") or data.get("favourite")),
    }


def write_legacy_user_data(model_path: Path, payload: dict[str, Any]) -> None:
    """Persist a model's user data, in the previous manager's schema.

    Written to ``<stem>.user.json`` — one file per model, so two versions
    sharing a folder each keep their own rating, notes and trigger words.

    The folder-level ``user_data.json`` is also kept in step, but only when the
    folder holds a single model. With several versions present there is no
    honest way to say which one that shared file describes, so it is left
    alone rather than being overwritten by whichever version was saved last.
    """
    import datetime

    def _render(existing: dict[str, Any]) -> dict[str, Any]:
        record = dict(existing)
        record.update({
            "schema_version": 1,
            "last_modified": datetime.datetime.utcnow().isoformat() + "Z",
            "personal_trigger_words": payload.get("personal_triggers") or [],
            "notes": payload.get("notes") or "",
            "rating": payload.get("rating") or 0,
            "collections": payload.get("collections") or [],
        })
        if payload.get("weight") is not None:
            record["preferred_weight"] = payload["weight"]
        if payload.get("negative_prompt"):
            record["negative_prompt"] = payload["negative_prompt"]
        if payload.get("tags"):
            record["tags"] = payload["tags"]
        record["favorite"] = bool(payload.get("favorite"))
        return record

    def _load(path: Path) -> dict[str, Any]:
        if path.exists():
            loaded, _ = load_lenient(path)
            if isinstance(loaded, dict):
                return loaded
        return {}

    own = user_data_path(model_path)
    write_atomic(own, _render(_load(own)))

    shared = model_path.parent / LEGACY_USER_DATA
    if _folder_model_count(model_path.parent) <= 1:
        write_atomic(shared, _render(_load(shared)))


def read_user_authored(
    model_path: Path,
    known: list[Path] | None = None,
    parsed: list[tuple[Path, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Extract the fields a person curated by hand in the WebUI.

    These are the ones worth years of accumulated effort — hand-picked trigger
    words and personal notes — so they are lifted into Atlas's own user data
    where no metadata refresh can reach them.

    Pass ``parsed`` from :func:`read_all` to reuse its work instead of
    re-reading the same files.
    """
    if parsed is not None:
        pairs = [(p, d) for p, d in parsed if is_user_authored(p, model_path)]
    else:
        if known is not None:
            targets = [p for p in known if is_user_authored(p, model_path)]
        else:
            candidates = [model_path.with_suffix("").with_suffix(".json"),
                          model_path.parent / "metadata.json"]
            targets = [p for p in candidates if p.exists()]
        pairs = []
        for path in targets:
            data, _ = load_lenient(path)
            if isinstance(data, dict):
                pairs.append((path, data))

    # NOTE: "triggers" is intentionally NOT populated from activation text.
    # Activation text is the model's *official* trigger list (it becomes
    # trained_words in normalise); copying it here too produced a duplicate,
    # split-up "personal" list sitting under the real one. Personal triggers
    # come only from the user's explicit personal_trigger_words.
    out: dict[str, Any] = {"notes": "", "triggers": [], "weight": None, "negative": ""}
    for path, data in sorted(pairs, key=lambda pair: _authority(pair[0])):
        if not out["notes"]:
            note = data.get("notes") or data.get("description_user") or ""
            out["notes"] = str(note).strip()
        if out["weight"] is None:
            raw = data.get("preferred weight", data.get("preferred_weight"))
            try:
                weight = float(raw) if raw not in (None, "") else None
            except (TypeError, ValueError):
                weight = None
            # Both the WebUI and the previous manager write 0.0 to mean "not
            # set". Taking it literally would pin every such LoRA to zero
            # strength, which silently disables it in a prompt.
            out["weight"] = None if weight == 0.0 else weight
        if not out["negative"]:
            out["negative"] = str(data.get("negative text") or "").strip()
    out["triggers"] = _dedupe(out["triggers"])
    return out


# Keys in the metadata sidecar that hold the *user's* own curation. A metadata
# refresh must update everything else but never touch these.
USER_OWNED_KEYS = ("activation text", "activation_text", "preferred weight",
                   "preferred_weight", "negative text", "notes")

# An earlier version of this tool wrote a second, thin ``<stem>.atlas.json``
# beside the main metadata file. Everything in it is machine-written provenance
# that the rich sidecar now carries under its own key names, so it is dead
# weight — but its fields are folded in before it is removed, just in case the
# main file predates them. It never held user data (that lives in
# user_data.json), so retiring it cannot lose anything the user typed.
LEGACY_ATLAS_SUFFIX = ".atlas.json"

# old key -> key in the current schema
_LEGACY_ATLAS_MAP = {
    "_atlas_provider": "provider",
    "_atlas_model_id": "provider_model_id",
    "_atlas_version_id": "provider_version_id",
    "name": "name",
    "versionName": "version_name",
    "creator": "creator",
    "baseModel": "base_model",
    "tags": "tags",
    "trainedWords": "trained_words",
    "sha256": "file_sha256",
}


def legacy_atlas_path(model_path: Path) -> Path:
    return Path(str(model_path.with_suffix("")) + LEGACY_ATLAS_SUFFIX)


def retire_legacy_atlas(model_path: Path, dry_run: bool = False) -> dict[str, Any] | None:
    """Fold a legacy ``<stem>.atlas.json`` into the main sidecar and delete it.

    Returns a summary, or None when there is nothing to do. Only fills gaps in
    the main file — never overwrites a value that is already there, and never
    touches the user-owned keys.
    """
    legacy = legacy_atlas_path(model_path)
    if not legacy.is_file():
        return None

    data, _ = load_lenient(legacy)
    main_path = model_path.with_suffix("").with_suffix(".json")
    carried: list[str] = []

    if isinstance(data, dict) and data:
        main: dict[str, Any] = {}
        if main_path.is_file():
            loaded, _ = load_lenient(main_path)
            if isinstance(loaded, dict):
                main = loaded
        for old_key, new_key in _LEGACY_ATLAS_MAP.items():
            if old_key not in data:
                continue
            value = data[old_key]
            if value in (None, "", [], {}):
                continue
            if new_key in USER_OWNED_KEYS:
                continue                    # never touch curated fields
            if main.get(new_key) in (None, "", [], {}):
                main[new_key] = value
                carried.append(new_key)
        if carried and not dry_run:
            write_atomic(main_path, main)

    if not dry_run:
        try:
            legacy.unlink()
        except OSError:
            return {"path": str(legacy), "removed": False, "carried": carried}
    return {"path": str(legacy), "removed": not dry_run, "carried": carried}


def write_metadata_sidecar(model_path: Path, fields: dict[str, Any],
                           raw: dict[str, Any] | None = None) -> None:
    """Write ONE rich ``<stem>.json`` — the model's full metadata record.

    Matches the previous manager's format: flat ``provider_*`` / ``file_sha256``
    fields for our own reading, plus the complete provider payload (description,
    images, stats, trainedWords, model, files…) so the model can be described
    and rebuilt later. One file per model, not two.

    Non-destructive:

    * If the file already exists, provider-sourced fields are refreshed but the
      user-owned keys (activation text, preferred weight, negative text, notes)
      are left exactly as they are — that is where hand-curated trigger words
      live.
    * Those keys are never *created* either. A new sidecar gets provider data
      and nothing else, so the WebUI's editable fields stay empty until a
      person puts something in them.
    """
    # Fold away any legacy .atlas.json first, so a metadata refresh also tidies
    # up the second file the old design left behind.
    retire_legacy_atlas(model_path)

    path = model_path.with_suffix("").with_suffix(".json")
    existing: dict[str, Any] = {}
    if path.exists():
        data, _ = load_lenient(path)
        if isinstance(data, dict):
            existing = data

    # Preserve everything the user owns before we overlay provider data.
    preserved = {k: existing[k] for k in USER_OWNED_KEYS if k in existing}

    record: dict[str, Any] = dict(existing)          # keep any unknown keys too
    if raw:
        record.update(raw)                           # full provider payload

    # Flat fields we rely on for fast reads (mirror the previous app's schema).
    record.update({
        "provider": fields.get("provider", ""),
        "provider_display_name": (fields.get("provider") or "").title(),
        "provider_model_id": fields.get("provider_model_id", ""),
        "provider_version_id": fields.get("provider_version_id", ""),
        "name": fields.get("name", "") or record.get("name", ""),
        "version_name": fields.get("version_name", ""),
        "creator": fields.get("creator", ""),
        "description": fields.get("description") or record.get("description", ""),
        # Only ever replaced by something non-empty: a refresh that did not
        # carry markup must not wipe markup already on disk.
        "description_html": (fields.get("description_html")
                             or record.get("description_html") or ""),
        "base_model": fields.get("base_model", ""),
        "trained_words": fields.get("trained_words", []),
        "tags": fields.get("tags", []),
        "file_sha256": fields.get("sha256", "") or record.get("file_sha256", ""),
        "sd version": fields.get("base_model") or record.get("sd version", "Unknown"),
    })

    # Restore user-owned keys untouched.
    record.update(preserved)

    # `activation text` is deliberately never written here, not even when the
    # file is new. It is the WebUI's own field: whatever is in it is what gets
    # pasted into a prompt, so it belongs to the person using the WebUI. Filling
    # it with the official trigger list on first download looks helpful, but it
    # silently turns a provider's guess into something indistinguishable from a
    # hand-written prompt, and there is then no way to tell which entries the
    # user actually chose. The official list is already stored as
    # `trained_words`, and the UI shows it separately.
    write_atomic(path, record)
