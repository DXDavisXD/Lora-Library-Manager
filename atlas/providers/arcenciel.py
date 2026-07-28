"""Arcenciel (arcenciel.io) provider.

Written against the live API rather than by analogy with CivitAI — the two
differ substantially. Verified endpoints:

* ``GET /api/models/search?search=<q>&limit=N&page=1&status=all``
  Returns ``{page, limit, data: [model], totalCount, totalPages}``. The search
  index covers SHA-256 hashes and file names as well as free text, which is
  what makes hash lookup possible: there is no dedicated by-hash route.
* ``GET /api/models/{modelId}`` — full model, versions embedded.
* ``GET /api/models/{modelId}/versions/{versionId}`` — one version.
* ``GET /api/models/{modelId}/versions/{versionId}/download`` — the file.
* Media lives at ``https://arcenciel.io/uploads/{filePath}``.

Field names differ from CivitAI throughout: ``title`` not ``name``,
``versionName`` not ``name``, ``activationTags`` not ``trainedWords``,
``uploader.username`` not ``creator.username``, and tags are ``{id, name}``
objects. Getting these wrong is why nothing resolved before.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

from .base import ModelInfo, ProviderBase

API = "https://arcenciel.io/api"
SITE = "https://arcenciel.io"
UPLOADS = "https://arcenciel.io/uploads"


class ArcencielProvider(ProviderBase):
    name = "arcenciel"
    display_name = "Arcenciel"
    website = SITE
    accent = "#4f9df7"   # blue-themed
    domains = ("arcenciel.io", "www.arcenciel.io")
    supports_download = True
    supports_hash_lookup = True

    # -- lookups ------------------------------------------------------------

    def fetch_by_hash(self, sha256: str, api_key: str | None = None) -> ModelInfo | None:
        """Resolve by SHA-256 through the search index.

        Arcenciel has no by-hash endpoint, but its search matches on the stored
        hash, so searching for the digest returns the owning model. The result
        already embeds every version, so the matching one is picked locally
        without a second request.
        """
        if not sha256:
            return None
        digest = sha256.strip().lower()
        model = self._search_one(digest, api_key)
        if model is None:
            return None
        version = self._pick_version(model, sha256=digest)
        return self._to_info(model, version) if version else None

    def fetch_by_id(
        self, model_id: str | None, version_id: str, api_key: str | None = None
    ) -> ModelInfo | None:
        """Resolve a specific version.

        Two requests are needed: ``/models/{id}`` carries the model-level
        fields (title, description, tags, uploader) but its embedded versions
        omit ``images``, while ``/models/{id}/versions/{vid}`` has the images
        but only ``{uploaderId}`` for the model. Neither alone is enough to
        build a complete record — missing this is why previews came back empty.
        """
        if not model_id:
            return None
        model = self.get_json(f"{API}/models/{model_id}", api_key)
        if not isinstance(model, dict):
            return None
        version = self._pick_version(model, version_id=str(version_id or ""))
        if version is None:
            return None
        if version_id and not version.get("images"):
            detail = self.get_json(
                f"{API}/models/{model_id}/versions/{version_id}", api_key)
            if isinstance(detail, dict):
                merged = dict(version)
                merged.update({k: v for k, v in detail.items() if k != "model"})
                version = merged
        return self._to_info(model, version)

    def fetch_versions(self, model_id: str, api_key: str | None = None) -> list[ModelInfo]:
        model = self.get_json(f"{API}/models/{model_id}", api_key)
        if not isinstance(model, dict):
            return []
        out = [self._to_info(model, v) for v in (model.get("versions") or [])
               if isinstance(v, dict)]
        out.sort(key=lambda i: i.published_at, reverse=True)
        return out

    def search(self, query: str, limit: int = 20, api_key: str | None = None) -> list[ModelInfo]:
        params = urlencode({"search": query, "limit": max(1, min(limit, 100)),
                            "page": 1, "status": "all"})
        data = self.get_json(f"{API}/models/search?{params}", api_key)
        results: list[ModelInfo] = []
        for model in (data or {}).get("data", []):
            if not isinstance(model, dict):
                continue
            version = self._pick_version(model)
            if version:
                results.append(self._to_info(model, version))
        return results

    def download_url(self, model_id: str | None, version_id: str) -> str:
        return f"{API}/models/{model_id}/versions/{version_id}/download"

    # -- helpers ------------------------------------------------------------

    def _search_one(self, query: str, api_key: str | None) -> dict[str, Any] | None:
        params = urlencode({"search": query, "limit": 5, "page": 1, "status": "all"})
        data = self.get_json(f"{API}/models/search?{params}", api_key)
        items = (data or {}).get("data") or []
        return items[0] if items and isinstance(items[0], dict) else None

    @staticmethod
    def _pick_version(model: dict[str, Any], sha256: str = "",
                      version_id: str = "") -> dict[str, Any] | None:
        """Choose the version matching a hash or id, else the newest."""
        versions = [v for v in (model.get("versions") or []) if isinstance(v, dict)]
        if not versions:
            return None
        if sha256:
            for v in versions:
                # sha256webui is Arcenciel's AutoV3-style hash; match either.
                if str(v.get("sha256") or "").lower() == sha256 \
                        or str(v.get("sha256webui") or "").lower() == sha256:
                    return v
        if version_id:
            for v in versions:
                if str(v.get("id")) == version_id:
                    return v
            return None
        return max(versions, key=lambda v: str(v.get("publishedAt") or ""))

    def _to_info(self, model: dict[str, Any], version: dict[str, Any]) -> ModelInfo:
        from ..sidecar import clean_text

        uploader = model.get("uploader")
        creator = ""
        if isinstance(uploader, dict):
            creator = str(uploader.get("username") or "")

        # tags are {id, name} objects.
        tags = [str(t.get("name")) for t in (model.get("tags") or [])
                if isinstance(t, dict) and t.get("name")]

        # activationTags look like "Default | tag, tag, tag" — the part before
        # the pipe is a group label, not something you would paste in a prompt.
        # Keep each entry as one trigger group with the label stripped.
        triggers: list[str] = []
        for entry in (version.get("activationTags") or []):
            text = str(entry).strip()
            if not text:
                continue
            if "|" in text:
                text = text.split("|", 1)[1].strip()
            if text:
                triggers.append(text)

        previews: list[str] = []
        preview_meta: list[dict] = []
        for image in (version.get("images") or []):
            if not isinstance(image, dict):
                continue
            path = image.get("filePath")
            if not path:
                continue
            url = f"{UPLOADS}/{str(path).lstrip('/')}"
            previews.append(url)
            preview_meta.append({"url": url, "width": image.get("width"),
                                 "height": image.get("height")})

        model_id = str(model.get("id") or version.get("modelId") or "")
        description = clean_text(model.get("description")
                                 or version.get("aboutThisVersion"))

        # Full payload for the rich sidecar, mirroring the CivitAI provider.
        raw = dict(version)
        raw["model"] = {
            "id": model_id,
            "name": model.get("title", ""),
            "description": model.get("description", ""),
            "tags": tags,
            "type": model.get("type", ""),
            "creator": creator,
            "civitaiModelId": model.get("civitaiModelId"),
        }

        return ModelInfo(
            provider=self.name,
            provider_model_id=model_id,
            provider_version_id=str(version.get("id") or ""),
            name=str(model.get("title") or ""),
            version_name=str(version.get("versionName") or ""),
            creator=creator,
            description=description,
            base_model=str(version.get("baseModel") or ""),
            trained_words=triggers,
            tags=tags,
            preview_urls=previews,
            preview_meta=preview_meta,
            preferred_weight=None,
            nsfw=False,                      # not exposed on the model record
            file_sha256=str(version.get("sha256") or "").lower(),
            original_filename=str(version.get("fileName")
                                  or version.get("originalName") or ""),
            download_url=self.download_url(model_id, str(version.get("id") or "")),
            page_url=f"{SITE}/models/{model_id}" if model_id else "",
            published_at=str(version.get("publishedAt")
                             or version.get("createdAt") or ""),
            raw=raw,
        )
