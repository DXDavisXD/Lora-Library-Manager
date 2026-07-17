"""
src/providers/arcenciel_provider.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Arcenciel (arcenciel.io) implementation of ProviderBase.

This module is the ONLY place in the application that knows about
Arcenciel-specific JSON field names (title, activationTags, sha256webui,
filePath, versionName, uploader …).  All other modules receive normalised
ModelInfo objects.

Architecture notes
------------------
* HTML stripping uses Python's stdlib ``html.parser.HTMLParser`` — no
  regex, no third-party dependencies.
* A lightweight per-instance model cache avoids duplicate
  ``GET /api/models/{id}`` calls within the same logical operation.
  The cache is bounded in practice by the number of unique model IDs
  fetched during a single run.
* ``get_download_url`` accepts separate ``model_id`` and ``version_id``
  parameters (no composite keys) because Arcenciel has no
  ``/api/versions/{id}`` endpoint — the parent model must be fetched to
  locate the correct version's ``filePath``.
"""

from __future__ import annotations

import html as _html_module
from html.parser import HTMLParser
from typing import TYPE_CHECKING

from providers.provider_base import ModelInfo, ProviderBase, ProviderCapabilities, ProviderPresentation

if TYPE_CHECKING:
    pass


# ── HTML → plain text ─────────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    """Minimal HTML parser that accumulates visible text content."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_tags = {"script", "style", "head"}
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:  # type: ignore[override]
        if tag.lower() in self._skip_tags:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag.lower() in self._skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._parts.append(text)

    def get_text(self) -> str:
        return " ".join(self._parts).strip()


def _strip_html(raw: str) -> str:
    """Return the plain-text content of an HTML string.

    Uses ``html.unescape`` first so HTML entities (``&amp;``, ``&lt;`` …)
    are decoded correctly, then extracts visible text with the stdlib
    ``HTMLParser``.  Never uses regular expressions.
    """
    if not raw:
        return ""
    unescaped = _html_module.unescape(raw)
    extractor = _TextExtractor()
    try:
        extractor.feed(unescaped)
    except Exception:
        # Malformed HTML: fall back to the unescaped string without tags
        return unescaped
    return extractor.get_text()


# ── Trigger-word helpers ──────────────────────────────────────────────────────

def _parse_activation_tags(tags: list) -> list[str]:
    """
    Normalise Arcenciel ``activationTags`` into a flat list of unique strings.

    Each entry may be:
    * A plain comma-delimited string:  ``"tag1, tag2"``
    * A labelled group:                ``"Default | tag1, tag2"``

    The label prefix (everything up to and including ``|``) is stripped.
    Tags are split on commas, stripped of whitespace, and deduplicated
    while preserving order of first appearance.
    """
    seen: set[str] = set()
    result: list[str] = []
    for entry in tags:
        if not isinstance(entry, str):
            continue
        # Strip label prefix if present
        text = entry.split("|", 1)[-1]
        for part in text.split(","):
            word = part.strip()
            if word and word not in seen:
                seen.add(word)
                result.append(word)
    return result


# ── Provider ──────────────────────────────────────────────────────────────────

class ArcencielProvider(ProviderBase):
    """Provider implementation for the Arc en Ciel model repository."""

    def __init__(self) -> None:
        # Lightweight per-instance response cache.
        # Key: str(model_id), value: parsed model dict from the API.
        # Populated by _fetch_model(); cleared between tests via _clear_cache().
        self._model_cache: dict[str, dict] = {}

    # ── Capabilities ──────────────────────────────────────────────────────────

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            hash_lookup=True,       # /api/models/search?search={sha256}
            search=True,
            update_check=True,      # /api/models/{id} → versions[]
            preview_download=True,
            model_download=True,    # /uploads/{filePath} for platform-hosted files
        )

    @property
    def presentation(self) -> ProviderPresentation:
        return ProviderPresentation(
            display_name="Arcenciel",
            accent_color="#7b68ee",
            icon="▲",
            website="https://arcenciel.io",
        )

    def get_provider_name(self) -> str:
        return "arcenciel"

    def get_model_page_url(self, model_id: str, version_id: str | None = None) -> str:
        url = f"https://arcenciel.io/models/{model_id}"
        if version_id:
            url += f"?modelVersionId={version_id}"
        return url

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _clear_cache(self) -> None:
        """Clear the in-memory model cache (useful between test cases)."""
        self._model_cache.clear()

    def _fetch_model(self, model_id: str, api_key: str | None = None) -> dict | None:
        """
        Return the full model dict for *model_id*, using the cache.

        On a cache miss, calls ``GET /api/models/{model_id}``.  Successful
        responses are stored so that subsequent calls within the same
        operation skip the network round-trip.

        Note: the model-by-ID endpoint does NOT embed images inside
        ``versions[].images``.  Use ``_enrich_with_images`` to populate
        images before building a ``ModelInfo`` that requires preview URLs.
        """
        key = str(model_id)
        if key in self._model_cache:
            return self._model_cache[key]

        from arcenciel import get_model
        resp = get_model(model_id, api_key=api_key)
        if not resp.get("ok"):
            return None
        data = resp.get("data") or {}
        self._model_cache[key] = data
        return data

    def _enrich_with_images(
        self,
        model_obj: dict,
        version_obj: dict,
        api_key: str | None = None,
    ) -> dict:
        """
        Return a copy of *model_obj* with image data populated on
        *version_obj*'s ``images`` key.

        The model-by-ID endpoint (``GET /api/models/{id}``) omits the
        ``images`` array from version objects.  Images ARE present in
        search results.  When a version has no images but has an
        ``imageOrder`` list, this method re-fetches via search using the
        model title to obtain the full version objects with images embedded.
        """
        from arcenciel import search_models

        # Already has images — nothing to do
        if version_obj.get("images"):
            return model_obj

        # No imageOrder either — model genuinely has no images
        if not version_obj.get("imageOrder"):
            return model_obj

        # Re-query by model title (most reliable free-text identifier)
        title = model_obj.get("title", "")
        if not title:
            return model_obj

        resp = search_models(title, limit=5, api_key=api_key)
        if not resp.get("ok"):
            return model_obj

        items = (resp.get("data") or {}).get("data") or []
        target_model_id = str(model_obj.get("id"))

        for item in items:
            if str(item.get("id")) != target_model_id:
                continue
            
            # Build map of version_id -> images from search result
            search_ver_images = {}
            for sv in item.get("versions") or []:
                sv_id = str(sv.get("id"))
                if sv.get("images"):
                    search_ver_images[sv_id] = sv["images"]

            # Graft images onto all versions in model_obj where missing
            new_versions = []
            any_enriched = False
            for vv in (model_obj.get("versions") or []):
                vv_id = str(vv.get("id"))
                if vv_id in search_ver_images and not vv.get("images"):
                    vv_copy = dict(vv)
                    vv_copy["images"] = search_ver_images[vv_id]
                    new_versions.append(vv_copy)
                    any_enriched = True
                else:
                    new_versions.append(vv)

            if any_enriched:
                model_obj = dict(model_obj)
                model_obj["versions"] = new_versions
                return model_obj
        return model_obj

    # ── Normalisation ─────────────────────────────────────────────────────────

    def _to_model_info(self, model_obj: dict, version_obj: dict) -> ModelInfo:
        """
        Translate raw Arcenciel model + version dicts into a normalised
        ModelInfo.  This is the ONLY place Arcenciel field names are used.
        """
        # ── IDs ───────────────────────────────────────────────────────────────
        model_id   = str(model_obj.get("id") or "")
        version_id = str(version_obj.get("id") or "")

        # ── Name / version ────────────────────────────────────────────────────
        name         = model_obj.get("title") or ""
        version_name = version_obj.get("versionName") or ""

        # ── Creator ───────────────────────────────────────────────────────────
        uploader = model_obj.get("uploader") or {}
        creator  = (
            uploader.get("username")
            if isinstance(uploader, dict)
            else str(uploader)
        ) or ""

        # ── Description (HTML → plain text) ───────────────────────────────────
        description = _strip_html(model_obj.get("description") or "")

        # ── Base model ────────────────────────────────────────────────────────
        base_model = version_obj.get("baseModel") or ""

        # ── Trigger words ─────────────────────────────────────────────────────
        raw_tags     = version_obj.get("activationTags") or []
        trained_words = _parse_activation_tags(raw_tags)

        # ── Tags ──────────────────────────────────────────────────────────────
        tags: list[str] = []
        for tag_obj in model_obj.get("tags") or []:
            if isinstance(tag_obj, dict):
                name_val = tag_obj.get("name")
                if name_val:
                    tags.append(str(name_val))
            elif isinstance(tag_obj, str) and tag_obj:
                tags.append(tag_obj)

        # ── Preview URLs ──────────────────────────────────────────────────────
        from arcenciel import build_cdn_url
        preview_urls: list[str] = []
        for img in (version_obj.get("images") or []):
            if not isinstance(img, dict):
                continue
            fp = img.get("filePath")
            if fp and isinstance(fp, str):
                url = build_cdn_url(fp)
                if url not in preview_urls:
                    preview_urls.append(url)

        # ── File identity ─────────────────────────────────────────────────────
        file_sha256 = (version_obj.get("sha256") or "").lower()
        original_filename = (
            version_obj.get("originalName")
            or version_obj.get("fileName")
            or ""
        )

        # ── Download URL (transient — not stored in sidecar) ─────────────────
        download_url = _build_download_url(version_obj)

        trained_words_grouped = []
        for entry in raw_tags:
            if isinstance(entry, str):
                text = entry.split("|", 1)[-1].strip()
                if text and text not in trained_words_grouped:
                    trained_words_grouped.append(text)

        return ModelInfo(
            provider="arcenciel",
            provider_display_name="Arc en Ciel",
            provider_model_id=model_id,
            provider_version_id=version_id,
            name=name,
            version_name=version_name,
            creator=creator,
            description=description,
            base_model=base_model,
            trained_words=trained_words,
            trained_words_grouped=trained_words_grouped,
            preview_urls=preview_urls,
            tags=tags,
            preferred_weight=None,
            negative_prompt="",
            file_sha256=file_sha256,
            original_filename=original_filename,
            download_url=download_url,
        )

    # ── ProviderBase implementation ───────────────────────────────────────────

    def fetch_by_hash(self, sha256: str, api_key: str | None = None) -> ModelInfo | None:
        """
        Resolve a model by its SHA-256 hash.

        Arcenciel has no dedicated ``/by-hash/`` endpoint.  Instead the
        general search endpoint accepts a hash as the query string.

        Strategy:
          1. Search the full SHA-256 value and check the results client-side.
          2. If no match, search the AutoV3 (``sha256webui``) value directly.
        """
        from arcenciel import search_models

        clean = sha256.strip().lower()
        if not clean:
            return None

        # Step 1: try SHA-256 directly
        resp = search_models(clean, limit=5, api_key=api_key)
        if resp.get("ok"):
            # The API returns: {"ok": True, "data": {"page":1, "data":[...], ...}}
            items = (resp.get("data") or {}).get("data") or []
            match = _find_version_by_hash(items, clean)
            if match:
                model_obj, version_obj = match
                model_obj = self._enrich_with_images(model_obj, version_obj, api_key)
                self._model_cache[str(model_obj.get("id"))] = model_obj
                return self._to_model_info(model_obj, version_obj)

        # Step 2: SHA-256 wasn't found; it may be an AutoV3 hash
        resp2 = search_models(clean, limit=10, api_key=api_key)
        if resp2.get("ok"):
            items2 = (resp2.get("data") or {}).get("data") or []
            for item in items2:
                for v in item.get("versions") or []:
                    if (v.get("sha256webui") or "").lower() == clean:
                        item = self._enrich_with_images(item, v, api_key)
                        self._model_cache[str(item.get("id"))] = item
                        return self._to_model_info(item, v)

        return None

    def fetch_by_id(
        self,
        model_id: str | None,
        version_id: str,
        api_key: str | None = None,
    ) -> ModelInfo | None:
        """
        Fetch a specific model version by Arcenciel model ID and version ID.

        Because Arcenciel provides no ``/api/versions/{id}`` endpoint, this
        method fetches the parent model and finds the matching version inside
        the embedded ``versions[]`` array.  Images are enriched via a search
        fallback since the model-by-ID endpoint omits them.
        """
        if not model_id:
            return None
        model_obj = self._fetch_model(model_id, api_key=api_key)
        if not model_obj:
            return None

        target_vid = str(version_id)
        for v in model_obj.get("versions") or []:
            if str(v.get("id")) == target_vid:
                model_obj = self._enrich_with_images(model_obj, v, api_key)
                # Store the enriched model back in cache so subsequent lookups skip network enrichment
                self._model_cache[str(model_id)] = model_obj
                # Re-find version after possible dict replacement in enrich
                for vv in model_obj.get("versions") or []:
                    if str(vv.get("id")) == target_vid:
                        return self._to_model_info(model_obj, vv)
        return None

    def fetch_model_versions(
        self,
        model_id: str,
        api_key: str | None = None,
    ) -> list[ModelInfo]:
        """
        Return all known versions of an Arcenciel model, sorted newest-first.
        Used by the update detector.
        """
        model_obj = self._fetch_model(model_id, api_key=api_key)
        if not model_obj:
            return []

        versions_raw = sorted(
            model_obj.get("versions") or [],
            key=lambda v: v.get("id") or 0,
            reverse=True,
        )
        return [self._to_model_info(model_obj, v) for v in versions_raw]

    def search(
        self,
        query: str,
        limit: int = 20,
        api_key: str | None = None,
    ) -> list[ModelInfo]:
        """Search Arc en Ciel for LoRA models matching *query*."""
        from arcenciel import search_models

        resp = search_models(query, limit=limit, api_key=api_key)
        if not resp.get("ok"):
            return []

        # API envelope: {"ok": True, "data": {"page":1, "data":[...], ...}}
        items = (resp.get("data") or {}).get("data") or []
        results: list[ModelInfo] = []
        for item in items:
            # Search results already include images embedded in versions
            self._model_cache[str(item.get("id"))] = item
            versions = sorted(
                item.get("versions") or [],
                key=lambda v: v.get("id") or 0,
                reverse=True,
            )
            if versions:
                results.append(self._to_model_info(item, versions[0]))
        return results

    def get_download_url(
        self,
        model_id: str | None,
        version_id: str,
        api_key: str | None = None,
    ) -> str:
        """
        Return the download URL for a specific version.

        Arcenciel's CDN path (``/uploads/…``) is not publicly accessible and
        returns 404.  The stable API download endpoint is used instead:
        ``GET /api/models/{model_id}/versions/{version_id}/download``

        For externally-hosted files the ``externalDownloadUrl`` is returned
        directly (requires a model fetch to read the version dict).

        Both ``model_id`` and ``version_id`` are required.
        """
        if not model_id or not version_id:
            return ""

        # Check for externalDownloadUrl first (requires model fetch)
        model_obj = self._fetch_model(model_id, api_key=api_key)
        if model_obj:
            target_vid = str(version_id)
            for v in model_obj.get("versions") or []:
                if str(v.get("id")) == target_vid:
                    ext = v.get("externalDownloadUrl")
                    if ext and isinstance(ext, str) and ext.strip():
                        return ext.strip()
                    break

        # Use the stable API download endpoint for platform-hosted files
        from arcenciel import build_api_url
        return build_api_url(f"models/{model_id}/versions/{version_id}/download")


# ── Module-level helpers ──────────────────────────────────────────────────────

def _build_download_url(version_obj: dict) -> str:
    """
    Derive the download URL from a version dict.

    Preference:
    1. ``externalDownloadUrl`` (non-null) — externally hosted file
    2. ``filePath`` via the CDN base — platform-hosted file
    """
    from arcenciel import build_cdn_url

    ext = version_obj.get("externalDownloadUrl")
    if ext and isinstance(ext, str) and ext.strip():
        return ext.strip()

    fp = version_obj.get("filePath")
    if fp and isinstance(fp, str) and fp.strip():
        return build_cdn_url(fp.strip())

    return ""


def _find_version_by_hash(
    items: list[dict],
    sha256: str,
) -> tuple[dict, dict] | None:
    """
    Scan a list of model objects and return the (model, version) pair whose
    ``sha256`` or ``sha256webui`` matches *sha256* (case-insensitive).
    """
    for item in items:
        for v in item.get("versions") or []:
            if (v.get("sha256") or "").lower() == sha256:
                return item, v
            if (v.get("sha256webui") or "").lower() == sha256:
                return item, v
    return None
