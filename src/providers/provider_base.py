"""
src/providers/provider_base.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Core abstractions for the multi-provider architecture.

All providers must implement ProviderBase and return ModelInfo objects.
No raw provider API JSON should ever escape the provider layer into
the rest of the application.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


# ── Capabilities ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProviderCapabilities:
    """
    Advertises which features a provider supports.

    The UI and backend must check these flags instead of comparing provider
    name strings.  Example:

        provider = registry.get_provider(meta["provider"])
        if provider.capabilities.update_check:
            versions = provider.fetch_model_versions(meta["provider_model_id"])

    Never do:
        if meta["provider"] == "civitai":  # wrong
            ...
    """
    # Can resolve a model from its SHA-256 hash
    hash_lookup: bool = False
    # Has a public search API
    search: bool = False
    # Can compare versions to detect updates
    update_check: bool = False
    # Can supply a preview image URL
    preview_download: bool = False
    # Can produce an authenticated binary download URL
    model_download: bool = False


@dataclass(frozen=True)
class ProviderPresentation:
    """
    Exposes UI presentation metadata for a provider.
    Ensures the frontend remains completely agnostic of specific provider implementations.
    """
    display_name: str
    accent_color: str
    icon: str
    website: str

    def to_dict(self) -> dict:
        return {
            "display_name": self.display_name,
            "accent_color": self.accent_color,
            "icon": self.icon,
            "website": self.website,
        }


# ── Normalised model record ───────────────────────────────────────────────────

@dataclass
class ModelInfo:
    """
    Normalised, provider-agnostic model record.

    This is the strict boundary between the provider layer and the rest of
    the application.  No raw provider API response may escape
    civitai_provider.py / arcenciel_provider.py into any other module.

    Notes
    -----
    * ``download_url`` is intentionally not stored in sidecars; download URLs
      may be ephemeral (auth tokens, CDN expiry).  Fetch at download-time.
    * Unknown / custom keys from old sidecars survive through the existing
      merge system in downloader.py, not via a ``raw`` blob here.
    """
    # ── Provider identity ────────────────────────────────────────────────────
    provider: str               # internal key, e.g. "civitai", "arcenciel"
    provider_display_name: str  # human-readable, e.g. "CivitAI", "Arcenciel"
    provider_model_id: str      # stringified to handle non-integer IDs
    provider_version_id: str

    # ── Model metadata ───────────────────────────────────────────────────────
    name: str           # model name, e.g. "My Character LoRA"
    version_name: str   # version label, e.g. "v2.0"
    creator: str
    description: str
    base_model: str     # e.g. "SDXL 1.0", "Pony", "Flux.1 D"
    trained_words: list[str] = field(default_factory=list)
    trained_words_grouped: list[str] = field(default_factory=list)
    preview_urls: list[str]  = field(default_factory=list)
    tags: list[str]          = field(default_factory=list)
    preferred_weight: float | None = None
    negative_prompt: str = ""

    # ── File identity ────────────────────────────────────────────────────────
    file_sha256: str        = ""
    original_filename: str  = ""

    # ── Transient (not persisted to sidecar) ─────────────────────────────────
    download_url: str = ""  # fetch at download time; never write to disk

    def to_sidecar_dict(self) -> dict:
        """
        Return a dict suitable for merging into a ``{stem}.json`` sidecar.
        Intentionally excludes ``download_url`` (ephemeral).
        """
        return {
            "provider":               self.provider,
            "provider_display_name":  self.provider_display_name,
            "provider_model_id":      self.provider_model_id,
            "provider_version_id":    self.provider_version_id,
            "name":                   self.name,
            "version_name":           self.version_name,
            "creator":                self.creator,
            "description":            self.description,
            "base_model":             self.base_model,
            "trained_words":          self.trained_words,
            "trainedWords":           self.trained_words_grouped,
            "tags":                   self.tags,
            "preferred_weight":       self.preferred_weight,
            "negative_prompt":        self.negative_prompt,
            "file_sha256":            self.file_sha256,
            "original_filename":      self.original_filename,
        }


# ── Abstract base provider ───────────────────────────────────────────────────

class ProviderBase(ABC):
    """
    Abstract base class that every provider must implement.

    All methods that communicate with a remote API must return
    ``ModelInfo`` objects (or lists thereof) — never raw dicts.
    """

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        """Return the feature flags for this provider."""
        ...

    @property
    @abstractmethod
    def presentation(self) -> ProviderPresentation:
        """Return the UI presentation details for this provider."""
        ...

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the internal provider key, e.g. ``"civitai"``."""
        ...

    @abstractmethod
    def get_model_page_url(self, model_id: str, version_id: str | None = None) -> str:
        """
        Return the web page URL for this model/version on the provider's website.
        """
        ...

    @abstractmethod
    def fetch_by_hash(self, sha256: str, api_key: str | None = None) -> ModelInfo | None:
        """
        Resolve a model by its SHA-256 file hash.
        Returns ``None`` when the hash is unknown or the provider does not
        support hash lookup (check ``capabilities.hash_lookup`` first).
        """
        ...

    @abstractmethod
    def fetch_by_id(
        self,
        model_id: str | None,
        version_id: str,
        api_key: str | None = None,
    ) -> ModelInfo | None:
        """
        Fetch a specific model version by provider IDs.
        Returns ``None`` on failure.
        """
        ...

    @abstractmethod
    def fetch_model_versions(
        self,
        model_id: str,
        api_key: str | None = None,
    ) -> list[ModelInfo]:
        """
        Return all known versions of a model as ``ModelInfo`` objects.
        Used by the update detector — check ``capabilities.update_check`` first.
        Returns an empty list on failure or if unsupported.
        """
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        limit: int = 20,
        api_key: str | None = None,
    ) -> list[ModelInfo]:
        """
        Search for models by keyword.
        Check ``capabilities.search`` before calling.
        Returns an empty list on failure or if unsupported.
        """
        ...

    @abstractmethod
    def get_download_url(
        self,
        model_id: str | None,
        version_id: str,
        api_key: str | None = None,
    ) -> str:
        """
        Return the authenticated download URL for a model version.

        Parameters
        ----------
        model_id   : Provider model ID.  Some providers need this to resolve
                     the download URL when no version-by-ID endpoint exists.
                     May be ``None`` when the caller does not have it (the
                     provider implementation must handle this gracefully).
        version_id : Provider version ID.
        api_key    : Optional API credential.

        Check ``capabilities.model_download`` before calling.
        Returns an empty string on failure.
        """
        ...
