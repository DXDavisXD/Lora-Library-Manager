"""
src/providers/civitai_provider.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CivitAI implementation of ProviderBase.

This module is the ONLY place in the application that knows about
CivitAI-specific JSON field names (modelId, id, trainedWords, images, files…).
All other modules receive normalised ModelInfo objects.
"""

from __future__ import annotations

from pathlib import Path

from providers.provider_base import (
    ModelInfo,
    ProviderBase,
    ProviderCapabilities,
    ProviderPresentation,
)


class CivitaiProvider(ProviderBase):
    """Provider implementation for the CivitAI model repository."""

    # ── Capabilities ─────────────────────────────────────────────────────────

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            hash_lookup=True,
            search=True,
            update_check=True,
            preview_download=True,
            model_download=True,
        )

    @property
    def presentation(self) -> ProviderPresentation:
        return ProviderPresentation(
            display_name="CivitAI",
            accent_color="#ef4444",
            icon="◉",
            website="https://civitai.red",
        )

    def get_provider_name(self) -> str:
        return "civitai"

    def get_model_page_url(self, model_id: str, version_id: str | None = None) -> str:
        url = f"https://civitai.red/models/{model_id}"
        if version_id:
            url += f"?modelVersionId={version_id}"
        return url

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _unwrap(response: dict) -> dict:
        """Strip the {ok, data} wrapper if present."""
        if isinstance(response, dict) and "ok" in response and "data" in response:
            return response.get("data") or {}
        return response

    def _to_model_info(self, version_data: dict, model_data: dict | None = None) -> ModelInfo:
        """
        Translate a raw CivitAI version dict (plus optional full model dict)
        into a normalised ModelInfo.  This is the ONLY place CivitAI field
        names are referenced.
        """
        # Merge model sub-object embedded in version_data with the explicit
        # model_data (from a separate /models/{id} call) when available.
        embedded_model = version_data.get("model") or {}
        model_obj: dict = {}
        if model_data:
            model_obj = model_data
        elif isinstance(embedded_model, dict):
            model_obj = embedded_model

        # ── IDs ───────────────────────────────────────────────────────────────
        model_id   = str(
            version_data.get("modelId")
            or model_obj.get("id")
            or ""
        )
        version_id = str(version_data.get("id") or "")

        # ── Name / version ────────────────────────────────────────────────────
        name         = model_obj.get("name") or version_data.get("name") or ""
        version_name = version_data.get("name") or ""

        # ── Creator ───────────────────────────────────────────────────────────
        creator_obj = (
            model_obj.get("creator")
            or version_data.get("creator")
            or {}
        )
        if isinstance(creator_obj, dict):
            creator = creator_obj.get("username", "")
        elif isinstance(creator_obj, str):
            creator = creator_obj
        else:
            creator = ""

        # ── Description ───────────────────────────────────────────────────────
        description = (
            version_data.get("description")
            or model_obj.get("description")
            or ""
        )

        # ── Base model ────────────────────────────────────────────────────────
        base_model = version_data.get("baseModel", "")

        # ── Trigger words ─────────────────────────────────────────────────────
        raw_words = version_data.get("trainedWords") or []
        trained_words: list[str] = []
        for entry in raw_words:
            if isinstance(entry, str):
                entry_clean = entry.strip()
                if entry_clean:
                    trained_words.append(entry_clean)
            elif isinstance(entry, list):
                entry_str = ", ".join(str(w).strip() for w in entry if str(w).strip())
                if entry_str:
                    trained_words.append(entry_str)

        # ── Tags ──────────────────────────────────────────────────────────────
        tags: list[str] = []
        raw_tags = model_obj.get("tags") or version_data.get("tags") or []
        if isinstance(raw_tags, list):
            tags = [str(t) for t in raw_tags if t]

        # ── Preview URLs ──────────────────────────────────────────────────────
        preview_urls: list[str] = []
        for images_source in [
            version_data.get("images", []),
            model_obj.get("images", []),
        ]:
            if isinstance(images_source, list):
                for img in images_source:
                    if not isinstance(img, dict):
                        continue
                    if img.get("type") in ("video", "gif"):
                        continue
                    url = img.get("url", "")
                    if url and url not in preview_urls:
                        preview_urls.append(url)

        # ── Download URL (transient — not stored in sidecar) ─────────────────
        download_url = ""
        for f_info in version_data.get("files", []):
            if not isinstance(f_info, dict):
                continue
            url = f_info.get("downloadUrl", "")
            if url:
                download_url = url
                break

        # ── File identity ─────────────────────────────────────────────────────
        file_sha256       = ""
        original_filename = ""
        for f_info in version_data.get("files", []):
            if not isinstance(f_info, dict):
                continue
            if f_info.get("primary") or not original_filename:
                original_filename = f_info.get("name", "") or original_filename
                hashes = f_info.get("hashes") or {}
                if isinstance(hashes, dict):
                    file_sha256 = hashes.get("SHA256", "") or file_sha256

        # ── Preferred weight / Negative prompt ───────────────────────────────
        preferred_weight = version_data.get("preferred weight")
        if preferred_weight is not None:
            try:
                preferred_weight = float(preferred_weight)
            except (ValueError, TypeError):
                preferred_weight = None

        negative_prompt = version_data.get("negative text") or ""

        return ModelInfo(
            provider="civitai",
            provider_display_name="CivitAI",
            provider_model_id=model_id,
            provider_version_id=version_id,
            name=name,
            version_name=version_name,
            creator=creator,
            description=description,
            base_model=base_model,
            trained_words=trained_words,
            trained_words_grouped=[entry.strip() for entry in raw_words if isinstance(entry, str) and entry.strip()],
            preview_urls=preview_urls,
            tags=tags,
            preferred_weight=preferred_weight,
            negative_prompt=negative_prompt,
            file_sha256=file_sha256.lower(),
            original_filename=original_filename,
            download_url=download_url,
        )

    # ── ProviderBase implementation ───────────────────────────────────────────

    def fetch_by_hash(self, sha256: str, api_key: str | None = None) -> ModelInfo | None:
        """Resolve a model by its SHA-256 hash via CivitAI /model-versions/by-hash/."""
        from civitai import get_model_by_hash, get_model

        response = get_model_by_hash(sha256, api_key=api_key)
        if not response.get("ok"):
            return None
        version_data = self._unwrap(response)

        # Fetch the full model for complete tags / creator info
        model_id = version_data.get("modelId")
        model_data: dict | None = None
        if model_id:
            model_resp = get_model(model_id, api_key=api_key)
            if model_resp.get("ok"):
                raw = self._unwrap(model_resp)
                model_data = {
                    "id":          raw.get("id"),
                    "name":        raw.get("name"),
                    "type":        raw.get("type"),
                    "creator":     raw.get("creator"),
                    "tags":        raw.get("tags", []),
                    "description": raw.get("description", ""),
                    "images":      raw.get("images", []),
                }

        return self._to_model_info(version_data, model_data)

    def fetch_by_id(
        self,
        model_id: str | None,
        version_id: str,
        api_key: str | None = None,
    ) -> ModelInfo | None:
        """Fetch a specific model version by CivitAI model ID and version ID."""
        from civitai import fetch_json, get_model

        version_resp = fetch_json(f"model-versions/{version_id}", api_key=api_key)
        if not version_resp.get("ok"):
            return None
        version_data = self._unwrap(version_resp)

        model_data: dict | None = None
        actual_model_id = model_id or version_data.get("modelId")
        if actual_model_id:
            model_resp = get_model(int(actual_model_id), api_key=api_key)
            if model_resp.get("ok"):
                raw = self._unwrap(model_resp)
                model_data = {
                    "id":          raw.get("id"),
                    "name":        raw.get("name"),
                    "type":        raw.get("type"),
                    "creator":     raw.get("creator"),
                    "tags":        raw.get("tags", []),
                    "description": raw.get("description", ""),
                    "images":      raw.get("images", []),
                }

        return self._to_model_info(version_data, model_data)

    def fetch_model_versions(
        self,
        model_id: str,
        api_key: str | None = None,
    ) -> list[ModelInfo]:
        """
        Return all known versions of a CivitAI model, sorted newest-first.
        Used by the update detector.
        """
        from civitai import get_model

        model_resp = get_model(int(model_id), api_key=api_key)
        if not model_resp.get("ok"):
            return []

        raw_model = self._unwrap(model_resp)
        model_data = {
            "id":          raw_model.get("id"),
            "name":        raw_model.get("name"),
            "type":        raw_model.get("type"),
            "creator":     raw_model.get("creator"),
            "tags":        raw_model.get("tags", []),
            "description": raw_model.get("description", ""),
            "images":      raw_model.get("images", []),
        }

        versions_raw = sorted(
            raw_model.get("modelVersions", []),
            key=lambda v: v.get("id", 0),
            reverse=True,
        )
        return [self._to_model_info(v, model_data) for v in versions_raw]

    def search(
        self,
        query: str,
        limit: int = 20,
        api_key: str | None = None,
    ) -> list[ModelInfo]:
        """Search CivitAI for LoRA models matching query."""
        from civitai import search_models

        resp = search_models(query, limit=limit)
        if not resp.get("ok"):
            return []
        data = self._unwrap(resp)
        results: list[ModelInfo] = []
        for item in data.get("items", []):
            # Each item is a full model with embedded modelVersions
            model_data = {
                "id":          item.get("id"),
                "name":        item.get("name"),
                "type":        item.get("type"),
                "creator":     item.get("creator"),
                "tags":        item.get("tags", []),
                "description": item.get("description", ""),
                "images":      item.get("images", []),
            }
            versions = item.get("modelVersions", [])
            if versions:
                # Return only the latest version
                latest = sorted(versions, key=lambda v: v.get("id", 0), reverse=True)[0]
                results.append(self._to_model_info(latest, model_data))
        return results

    def get_download_url(
        self,
        model_id: str | None,
        version_id: str,
        api_key: str | None = None,
    ) -> str:
        """Build the authenticated CivitAI download URL for a version.

        ``model_id`` is accepted for interface compatibility but unused —
        CivitAI download URLs are version-scoped.
        """
        base = f"https://civitai.red/api/download/models/{version_id}"
        if api_key:
            base += f"?token={api_key}"
        return base
