"""Provider contract and shared HTTP machinery.

Everything a provider needs that is not provider-specific lives here: the
normalised :class:`ModelInfo`, a token-bucket rate limiter, and a retrying
JSON fetch that honours ``Retry-After``.

Rate limiting is not optional at this scale. Resolving metadata for 1500 models
is 1500+ API calls; without a limiter you will be soft-banned partway through
and the run dies holding a half-populated database.

Secrets are always sent as an ``Authorization: Bearer`` header, never as a
query parameter — query strings end up in access logs, proxy caches and
browser history.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any

from .. import config

USER_AGENT = f"LoRALibraryManager/{config.VERSION} (+local library manager)"


@dataclass
class ModelInfo:
    """One version of one model, normalised across providers."""

    provider: str
    provider_model_id: str = ""
    provider_version_id: str = ""
    name: str = ""
    version_name: str = ""
    creator: str = ""
    description: str = ""
    base_model: str = ""
    trained_words: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    preview_urls: list[str] = field(default_factory=list)
    preferred_weight: float | None = None
    negative_prompt: str = ""
    nsfw: bool = False
    file_sha256: str = ""
    original_filename: str = ""
    download_url: str = ""
    page_url: str = ""
    published_at: str = ""
    preview_meta: list[dict] = field(default_factory=list)   # width/height/nsfw per image
    # The complete provider payload for this version, so a rich sidecar can be
    # written and the model fully reconstructed later. Not stored in the DB.
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("raw", None)          # DB/JSON-API callers don't want the blob
        return d


class ProviderError(RuntimeError):
    """Raised for a definite failure. Carries the HTTP status when there is one."""

    def __init__(self, message: str, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class RateLimiter:
    """Token bucket, shared by every thread using a given provider."""

    def __init__(self, rate_per_sec: float, burst: int = 4):
        self.rate = max(0.1, float(rate_per_sec))
        self.burst = max(1, int(burst))
        self._tokens = float(self.burst)
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(self.burst, self._tokens + (now - self._updated) * self.rate)
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self.rate
                time.sleep(min(wait, 5.0))

    def penalise(self, seconds: float) -> None:
        """Drain the bucket after a 429 so every thread backs off together."""
        with self._lock:
            self._tokens = 0.0
            self._updated = time.monotonic() + max(0.0, seconds)


class ProviderBase:
    name = "base"
    display_name = "Base"
    website = ""
    accent = "#6366f1"
    domains: tuple[str, ...] = ()
    supports_download = False
    supports_hash_lookup = False

    def __init__(self) -> None:
        self.limiter = RateLimiter(float(config.get("provider_rate_limit") or 2.0))

    # -- HTTP ---------------------------------------------------------------

    def _headers(self, api_key: str | None) -> dict[str, str]:
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
        key = api_key if api_key is not None else config.get_secret(self.name)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def get_json(
        self,
        url: str,
        api_key: str | None = None,
        *,
        attempts: int = 4,
        timeout: float | None = None,
    ) -> Any:
        """GET JSON with backoff. Returns None on a clean 404."""
        timeout = timeout or float(config.get("request_timeout") or 30)
        delay = 1.0
        last: Exception | None = None

        for attempt in range(attempts):
            self.limiter.acquire()
            request = urllib.request.Request(url, headers=self._headers(api_key), method="GET")
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = response.read()
                if not payload:
                    return None
                return json.loads(payload.decode("utf-8", errors="replace"))
            except urllib.error.HTTPError as exc:
                last = exc
                if exc.code == 404:
                    return None
                if exc.code in (401, 403):
                    raise ProviderError(
                        f"{self.display_name} rejected the credentials (HTTP {exc.code}). "
                        f"Check ATLAS_{self.name.upper()}_API_KEY.",
                        status=exc.code,
                    ) from exc
                if exc.code == 429:
                    retry_after = _retry_after(exc.headers.get("Retry-After"), delay)
                    self.limiter.penalise(retry_after)
                    time.sleep(retry_after)
                    delay = min(delay * 2, 60.0)
                    continue
                if 500 <= exc.code < 600:
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
                    continue
                raise ProviderError(f"{self.display_name} HTTP {exc.code}", status=exc.code) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = exc
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
            except json.JSONDecodeError as exc:
                raise ProviderError(f"{self.display_name} returned malformed JSON") from exc

        raise ProviderError(
            f"{self.display_name} unreachable after {attempts} attempts: {last}", retryable=True
        )

    # -- Contract -----------------------------------------------------------

    def fetch_by_hash(self, sha256: str, api_key: str | None = None) -> ModelInfo | None:
        raise NotImplementedError

    def fetch_by_id(
        self, model_id: str | None, version_id: str, api_key: str | None = None
    ) -> ModelInfo | None:
        raise NotImplementedError

    def fetch_versions(self, model_id: str, api_key: str | None = None) -> list[ModelInfo]:
        raise NotImplementedError

    def search(self, query: str, limit: int = 20, api_key: str | None = None) -> list[ModelInfo]:
        raise NotImplementedError

    def download_url(self, model_id: str | None, version_id: str) -> str:
        raise NotImplementedError

    def download_headers(self, api_key: str | None = None) -> dict[str, str]:
        """Headers for a model-file download. Keeps the key out of the URL."""
        return self._headers(api_key)

    def presentation(self) -> dict[str, Any]:
        from .. import config

        source = config.secret_source(self.name)
        return {
            "id": self.name,
            "display_name": self.display_name,
            "website": self.website,
            "accent": self.accent,
            "supports_download": self.supports_download,
            "supports_hash_lookup": self.supports_hash_lookup,
            "key_source": source,
            "configured": source != "unset",
        }


def _retry_after(header: str | None, fallback: float) -> float:
    if not header:
        return fallback
    try:
        return max(1.0, min(float(header), 120.0))
    except ValueError:
        return fallback
