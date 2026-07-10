"""
src/arcenciel.py
~~~~~~~~~~~~~~~~
Thin Arc en Ciel API client helpers.

This module is the only place in the application that knows the raw
shape of Arcenciel HTTP responses.  All other code uses the normalised
ModelInfo objects returned by ArcencielProvider.

API base: https://arcenciel.io/api
CDN base: https://arcenciel.io/uploads
Rate limit (observed): 1 200 req / 60 s  (≈ 20 req/s)
"""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ARCENCIEL_API_BASE = "https://arcenciel.io/api"
ARCENCIEL_CDN_BASE = "https://arcenciel.io/uploads"


# ── URL helpers ───────────────────────────────────────────────────────────────

def build_api_url(endpoint: str, params: dict | None = None) -> str:
    """Construct a full Arcenciel API URL."""
    clean = endpoint.strip("/")
    url = f"{ARCENCIEL_API_BASE}/{clean}"
    if params:
        url = f"{url}?{urlencode(params)}"
    return url


def build_cdn_url(file_path: str) -> str:
    """Construct a CDN URL for a hosted model file or preview image.

    ``file_path`` is the value of ``filePath`` / ``externalDownloadUrl``
    returned by the API.  It must NOT start with a slash.
    """
    return f"{ARCENCIEL_CDN_BASE}/{file_path.lstrip('/')}"


def build_headers(api_key: str | None = None) -> dict[str, str]:
    """Return HTTP headers for an Arcenciel API request."""
    headers: dict[str, str] = {"User-Agent": "LoRA-Library-Manager/2.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


# ── Standard response envelope ────────────────────────────────────────────────

def success_response(data: dict | list) -> dict:
    return {"ok": True, "status": 200, "error": None, "data": data}


def error_response(status: int | None, error: str, message: str) -> dict:
    return {"ok": False, "status": status, "error": error, "message": message, "data": None}


def _handle_http_error(err: HTTPError) -> dict:
    if err.code == 401:
        return error_response(401, "invalid_api_key", "The Arcenciel API key is invalid.")
    if err.code == 403:
        return error_response(403, "forbidden", "Access denied by Arcenciel.")
    if err.code == 404:
        return error_response(404, "not_found", "Arcenciel returned 404 — resource not found.")
    if err.code == 429:
        return error_response(429, "rate_limited", "Arcenciel rate limit reached. Try again later.")
    return error_response(err.code, "http_error", f"Arcenciel request failed with status {err.code}.")


# ── Core HTTP fetch ───────────────────────────────────────────────────────────

def fetch_json(
    endpoint: str,
    params: dict | None = None,
    api_key: str | None = None,
) -> dict:
    """GET an Arcenciel API endpoint and return a standard response dict."""
    url = build_api_url(endpoint, params)
    request = Request(url, headers=build_headers(api_key))
    try:
        with urlopen(request, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return success_response(data)
    except HTTPError as err:
        return _handle_http_error(err)
    except URLError as err:
        return error_response(None, "network_error", f"Could not reach Arcenciel: {err.reason}")
    except TimeoutError:
        return error_response(None, "network_error", "The Arcenciel request timed out.")
    except json.JSONDecodeError:
        return error_response(None, "invalid_response", "Arcenciel returned invalid JSON.")


# ── Domain-specific helpers ───────────────────────────────────────────────────

def search_models(query: str, limit: int = 20, api_key: str | None = None) -> dict:
    """
    Search Arcenciel by text, SHA256, or AutoV3 hash.

    Endpoint: GET /api/models/search?search={query}&limit={limit}
    """
    return fetch_json("models/search", {"search": query, "limit": limit}, api_key=api_key)


def get_model(model_id: str | int, api_key: str | None = None) -> dict:
    """
    Fetch a full model object including all embedded versions.

    Endpoint: GET /api/models/{model_id}
    """
    return fetch_json(f"models/{model_id}", api_key=api_key)
