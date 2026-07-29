"""CivitAI provider.

Traffic goes to whichever CivitAI host is configured. It defaults to
``civitai.red`` — a full mirror of the CivitAI API and CDN that stays reachable
where ``civitai.com`` is throttled or blocked, which is the usual cause of
metadata resolving only partially. The base is one setting (``civitai_base``)
used uniformly for the API, downloads and page links, so there is never a split
where metadata comes from one host and files from another.

Auth always travels in an ``Authorization`` header, never a ``?token=`` query
parameter (query strings leak into logs and proxy caches).
"""

from __future__ import annotations

from typing import Any

from .base import ModelInfo, ProviderBase

# The default primary host. CivitAI's own domain remains selectable via config.
DEFAULT_BASE = "https://civitai.red"

KNOWN_BASES = {
    "civitai.red": "https://civitai.red",
    "civitai.com": "https://civitai.com",
    "www.civitai.com": "https://civitai.com",
}


class CivitaiProvider(ProviderBase):
    name = "civitai"
    display_name = "CivitAI"
    website = DEFAULT_BASE
    accent = "#f0475a"   # civitai.red — red-themed to match the domain
    domains = ("civitai.com", "www.civitai.com", "civitai.red")
    supports_download = True
    supports_hash_lookup = True

    @property
    def base(self) -> str:
        """Configured host root, e.g. ``https://civitai.red``.

        Accepts a bare domain (``civitai.red``), a known alias, or a full URL.
        """
        from .. import config

        raw = str(config.get("civitai_base") or config.get("civitai_mirror") or "").strip().rstrip("/")
        if not raw:
            return DEFAULT_BASE
        if raw in KNOWN_BASES:
            return KNOWN_BASES[raw]
        if raw.startswith("https://") or raw.startswith("http://"):
            return raw
        return DEFAULT_BASE

    @property
    def api(self) -> str:
        return f"{self.base}/api/v1"

    # Backwards-compatible alias; some call sites and tests refer to `mirror`.
    @property
    def mirror(self) -> str:
        return self.base

    # -- lookups ------------------------------------------------------------

    def fetch_by_hash(self, sha256: str, api_key: str | None = None) -> ModelInfo | None:
        if not sha256:
            return None
        data = self.get_json(f"{self.api}/model-versions/by-hash/{sha256.upper()}", api_key)
        if not data:
            return None
        return self._with_parent(self._version_to_info(data), api_key)

    def fetch_by_id(
        self, model_id: str | None, version_id: str, api_key: str | None = None
    ) -> ModelInfo | None:
        if not version_id:
            return None
        data = self.get_json(f"{self.api}/model-versions/{version_id}", api_key)
        if not data:
            return None
        return self._with_parent(self._version_to_info(data), api_key)

    def _with_parent(self, info: ModelInfo, api_key: str | None) -> ModelInfo:
        """Fill in the fields that only exist on the parent model.

        A version response (by-hash or by-id) nests just
        ``model: {name, type, nsfw, poi}`` — the description, creator and tags
        live on ``/models/{id}``. Those are exactly the fields needed to
        describe a model in the gallery and to rebuild it later, so fetch the
        parent once when they are missing. Results are cached per run so
        several versions of one model cost a single request.
        """
        if not info or not info.provider_model_id:
            return info
        if info.description and info.creator and info.tags:
            return info                      # already complete

        cache = getattr(self, "_parent_cache", None)
        if cache is None:
            cache = self._parent_cache = {}
        # Providers are long-lived singletons, so bound the cache — a 1500-model
        # resolve would otherwise hold every parent response for the session.
        if len(cache) > 400:
            cache.clear()
        key = info.provider_model_id
        if key not in cache:
            try:
                cache[key] = self.get_json(f"{self.api}/models/{key}", api_key) or {}
            except Exception:                # noqa: BLE001 - parent is a bonus
                cache[key] = {}
        parent = cache[key]
        if not parent:
            return info

        from ..sidecar import clean_text

        if not info.description:
            info.description = clean_text(parent.get("description"))
        if not info.description_html:
            info.description_html = str(parent.get("description") or "")
        if not info.creator:
            creator = parent.get("creator")
            info.creator = str((creator or {}).get("username", "") if isinstance(creator, dict)
                               else creator or "")
        if not info.tags:
            info.tags = [str(t) for t in (parent.get("tags") or []) if t]
        info.nsfw = bool(info.nsfw or parent.get("nsfw"))

        # Mirror the parent block into the raw payload so the sidecar carries
        # the same detail the previous manager stored.
        merged = dict(info.raw)
        merged["model"] = {
            "name": parent.get("name", ""),
            "creator": parent.get("creator"),
            "tags": parent.get("tags", []),
            "nsfw": parent.get("nsfw", False),
            "description": parent.get("description", ""),
            "type": parent.get("type", ""),
            "id": parent.get("id", key),
        }
        merged.setdefault("description", parent.get("description", ""))
        if parent.get("stats"):
            merged.setdefault("modelStats", parent["stats"])
        info.raw = merged
        return info

    def fetch_versions(self, model_id: str, api_key: str | None = None) -> list[ModelInfo]:
        data = self.get_json(f"{self.api}/models/{model_id}", api_key)
        if not data:
            return []
        model_meta = {
            "name": data.get("name", ""),
            "creator": (data.get("creator") or {}).get("username", ""),
            "tags": data.get("tags") or [],
            "nsfw": bool(data.get("nsfw")),
            "description": data.get("description", ""),
            "id": data.get("id", model_id),
        }
        out = [
            self._version_to_info(v, model_meta)
            for v in (data.get("modelVersions") or [])
            if isinstance(v, dict)
        ]
        out.sort(key=lambda i: i.published_at, reverse=True)
        return out

    def search(self, query: str, limit: int = 20, api_key: str | None = None) -> list[ModelInfo]:
        from urllib.parse import urlencode

        params = urlencode({"query": query, "limit": max(1, min(limit, 100)), "types": "LORA"})
        data = self.get_json(f"{self.api}/models?{params}", api_key)
        results: list[ModelInfo] = []
        for item in (data or {}).get("items", []):
            versions = item.get("modelVersions") or []
            if not versions:
                continue
            results.append(self._version_to_info(versions[0], {
                "name": item.get("name", ""),
                "creator": (item.get("creator") or {}).get("username", ""),
                "tags": item.get("tags") or [],
                "nsfw": bool(item.get("nsfw")),
                "description": item.get("description", ""),
                "id": item.get("id", ""),
            }))
        return results

    def download_url(self, model_id: str | None, version_id: str) -> str:
        """Download endpoint on the configured base host.

        The key travels in an Authorization header (see
        :meth:`download_headers`), never as a query parameter.
        """
        return f"{self.base}/api/download/models/{version_id}"

    # -- normalisation ------------------------------------------------------

    def _version_to_info(
        self, version: dict[str, Any], model_meta: dict[str, Any] | None = None
    ) -> ModelInfo:
        from ..sidecar import clean_text

        parent = model_meta or version.get("model") or {}
        model_id = str(parent.get("id") or version.get("modelId") or "")

        primary: dict[str, Any] = {}
        for entry in version.get("files") or []:
            if isinstance(entry, dict) and entry.get("primary"):
                primary = entry
                break
        if not primary:
            files = [f for f in (version.get("files") or []) if isinstance(f, dict)]
            primary = files[0] if files else {}

        sha = str(((primary.get("hashes") or {}).get("SHA256") or "")).lower()

        previews: list[str] = []
        preview_meta: list[dict] = []
        for image in version.get("images") or []:
            if isinstance(image, dict) and image.get("url"):
                previews.append(str(image["url"]))
                preview_meta.append({
                    "url": str(image["url"]),
                    "width": image.get("width"), "height": image.get("height"),
                    "nsfwLevel": image.get("nsfwLevel"), "type": image.get("type"),
                })

        creator = parent.get("creator")
        if isinstance(creator, dict):
            creator = creator.get("username", "")

        # Full payload for the rich sidecar: the version object plus the parent
        # model fields the previous manager also kept (model, stats, etc.).
        raw = dict(version)
        if model_meta:
            raw.setdefault("modelId", model_id)
            raw["model"] = {k: model_meta.get(k) for k in
                            ("name", "creator", "tags", "nsfw", "description", "id")
                            if k in model_meta}
        elif isinstance(version.get("model"), dict):
            raw["model"] = version["model"]

        return ModelInfo(
            provider=self.name,
            provider_model_id=model_id,
            provider_version_id=str(version.get("id") or ""),
            name=str(parent.get("name") or version.get("name") or ""),
            version_name=str(version.get("name") or ""),
            creator=str(creator or ""),
            description=clean_text(version.get("description") or parent.get("description")),
            # Kept unflattened; sanitised later, once, in sidecar.clean_html.
            description_html=str(version.get("description")
                                 or parent.get("description") or ""),
            base_model=str(version.get("baseModel") or ""),
            trained_words=[str(w) for w in (version.get("trainedWords") or []) if w],
            tags=[str(t) for t in (parent.get("tags") or []) if t],
            preview_urls=previews,
            preview_meta=preview_meta,
            preferred_weight=_weight(version),
            nsfw=bool(parent.get("nsfw")),
            file_sha256=sha,
            original_filename=str(primary.get("name") or ""),
            download_url=str(version.get("downloadUrl") or ""),
            page_url=f"{self.base}/models/{model_id}" if model_id else "",
            published_at=str(version.get("publishedAt") or version.get("createdAt") or ""),
            raw=raw,
        )


def _weight(version: dict[str, Any]) -> float | None:
    settings = version.get("settings")
    if isinstance(settings, dict):
        for key in ("strength", "preferredWeight"):
            try:
                if settings.get(key) is not None:
                    return float(settings[key])
            except (TypeError, ValueError):
                pass
    return None
