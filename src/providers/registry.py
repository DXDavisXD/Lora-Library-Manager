"""
src/providers/registry.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Provider registry with multi-fallback detection.

Resolution order for ``resolve_provider()``:
  1. URL-based detection (``detect_from_url``)
  2. Stored ``provider`` field in sidecar metadata (``detect_from_metadata``)
  3. Legacy CivitAI-specific field names (``detect_from_legacy_fields``)
  4. Hash lookup via all providers that support it (requires sha256)
  5. Filename heuristic (last resort)
  6. Falls back to ``"civitai"`` — preserves existing behaviour for old sidecars.
"""

from __future__ import annotations

from providers.provider_base import ProviderBase, ModelInfo

# ── Registry ──────────────────────────────────────────────────────────────────
# Populated lazily on first access to avoid circular imports at module load time.
_registry: dict[str, ProviderBase] | None = None


def _build_registry() -> dict[str, ProviderBase]:
    from providers.civitai_provider import CivitaiProvider
    from providers.arcenciel_provider import ArcencielProvider
    providers: dict[str, ProviderBase] = {
        "civitai":    CivitaiProvider(),
        "arcenciel":  ArcencielProvider(),
    }
    return providers


def _get_registry() -> dict[str, ProviderBase]:
    global _registry
    if _registry is None:
        _registry = _build_registry()
    return _registry


def get_provider(name: str) -> ProviderBase:
    """
    Return the registered provider for *name*.
    Falls back to CivitAI if the name is unknown (defensive).
    """
    reg = _get_registry()
    if name in reg:
        return reg[name]
    # Unknown provider — log a warning and fall back to CivitAI
    import warnings
    warnings.warn(
        f"Unknown provider '{name}'. Falling back to 'civitai'.",
        stacklevel=2,
    )
    return reg["civitai"]


def list_providers() -> list[str]:
    """Return the list of all registered provider names."""
    return list(_get_registry().keys())


def get_provider_presentations() -> dict[str, dict]:
    """Return a dictionary of all registered provider presentations, serialized."""
    return {
        name: provider.presentation.to_dict()
        for name, provider in _get_registry().items()
    }


# ── Detection helpers ─────────────────────────────────────────────────────────

# URL domain → provider key mapping
_URL_DOMAIN_MAP: dict[str, str] = {
    "civitai.com":    "civitai",
    "civitai.red":    "civitai",
    "arcenciel.io":   "arcenciel",
}


def detect_from_url(url: str) -> str | None:
    """
    Fallback 1: Infer provider from a URL string.

    Returns the provider key, or ``None`` if the URL doesn't match any
    registered domain.
    """
    if not url:
        return None
    url_lower = url.lower()
    for domain, provider_key in _URL_DOMAIN_MAP.items():
        if domain in url_lower:
            return provider_key
    return None


def detect_from_metadata(meta: dict) -> str | None:
    """
    Fallback 2: Read the ``provider`` field from a stored sidecar.

    Returns the provider key if the field is present and the provider is
    registered; otherwise ``None``.
    """
    if not isinstance(meta, dict):
        return None
    provider_key = meta.get("provider")
    if not provider_key:
        return None
    reg = _get_registry()
    if provider_key in reg:
        return provider_key
    return None


def detect_from_legacy_fields(meta: dict) -> str | None:
    """
    Fallback 3: Infer ``"civitai"`` from the presence of CivitAI-specific
    field names in an old sidecar that has no ``provider`` field yet.

    Only the minimal set of unmistakably-CivitAI fields is checked here.
    """
    if not isinstance(meta, dict):
        return None
    # Unwrap if the sidecar is still in the old {ok, data} wrapper format
    data = meta
    if "ok" in meta and "data" in meta:
        data = meta.get("data") or {}

    civitai_fields = ("modelId", "id", "trainedWords", "baseModel")
    if any(f in data for f in civitai_fields):
        return "civitai"
    return None


def resolve_provider(
    meta: dict | None = None,
    url: str | None = None,
    sha256: str | None = None,
    filename: str | None = None,
) -> str:
    """
    Full 5-step provider resolution chain.

    Parameters
    ----------
    meta     : Loaded sidecar dict (``{stem}.json``), if available.
    url      : A source/download URL, if available.
    sha256   : SHA-256 hex string of the model file, if available.
    filename : Model filename, used as a last-resort heuristic.

    Returns
    -------
    str
        A registered provider key (always a non-empty string).
        Falls back to ``"civitai"`` when all steps fail.
    """
    # Step 1: URL-based
    if url:
        result = detect_from_url(url)
        if result:
            return result

    # Step 2: Stored ``provider`` field
    if meta:
        result = detect_from_metadata(meta)
        if result:
            return result

    # Step 3: Legacy CivitAI-specific field names
    if meta:
        result = detect_from_legacy_fields(meta)
        if result:
            return result

    # Step 4: Hash lookup via all capable providers
    if sha256:
        from config import get_provider_api_key
        for name, provider in _get_registry().items():
            if not provider.capabilities.hash_lookup:
                continue
            try:
                api_key = get_provider_api_key(name)
                info = provider.fetch_by_hash(sha256, api_key=api_key)
                if info is not None:
                    return name
            except Exception:
                continue

    # Step 5: Filename heuristic
    if filename:
        fname_lower = str(filename).lower()
        # Phase 2: add arcenciel filename patterns here
        if "civitai" in fname_lower:
            return "civitai"

    # Final fallback — preserves existing behaviour for all old sidecars
    return "civitai"
