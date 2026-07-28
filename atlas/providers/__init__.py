"""Provider registry.

Providers are instantiated once and shared, because the rate limiter has to be
shared: two independent instances would each think they had a full bucket and
would together exceed the configured rate.
"""

from __future__ import annotations

import threading
from urllib.parse import urlparse

from .base import ModelInfo, ProviderBase, ProviderError
from .arcenciel import ArcencielProvider
from .civitai import CivitaiProvider

_PROVIDER_TYPES: tuple[type[ProviderBase], ...] = (CivitaiProvider, ArcencielProvider)

_instances: dict[str, ProviderBase] = {}
_lock = threading.Lock()


def _ensure() -> dict[str, ProviderBase]:
    global _instances
    if _instances:
        return _instances
    with _lock:
        if not _instances:
            _instances = {cls.name: cls() for cls in _PROVIDER_TYPES}
    return _instances


def get(name: str) -> ProviderBase | None:
    return _ensure().get((name or "").strip().lower())


def all_providers() -> list[ProviderBase]:
    return list(_ensure().values())


def names() -> list[str]:
    return list(_ensure().keys())


def from_url(url: str) -> ProviderBase | None:
    """Resolve a provider from a model page or download URL."""
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return None
    if not host:
        return None
    for provider in _ensure().values():
        if host in provider.domains or any(host.endswith("." + d) for d in provider.domains):
            return provider
    return None


def hash_lookup_order() -> list[ProviderBase]:
    """Providers that support hash lookup, cheapest-first.

    CivitAI first: it has by far the widest LoRA coverage, so trying it first
    minimises the number of second requests during a bulk enrich.
    """
    return [p for p in _ensure().values() if p.supports_hash_lookup]


def presentations() -> list[dict]:
    from .. import config

    out = []
    for provider in _ensure().values():
        data = provider.presentation()
        data["key_source"] = config.secret_source(provider.name)
        data["configured"] = data["key_source"] != "unset"
        out.append(data)
    return out


__all__ = [
    "ModelInfo", "ProviderBase", "ProviderError",
    "get", "all_providers", "names", "from_url",
    "hash_lookup_order", "presentations",
]
