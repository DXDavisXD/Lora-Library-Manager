"""Small Civitai API client helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from config import get_civitai_api_key


CIVITAI_API_BASE = "https://civitai.com/api/v1"
CHUNK_SIZE = 1024 * 1024


# Build a complete Civitai API URL from an endpoint and query parameters.
def build_api_url(endpoint: str, params: dict[str, str | int] | None = None) -> str:
    clean_endpoint = endpoint.strip("/")
    url = f"{CIVITAI_API_BASE}/{clean_endpoint}"

    if params:
        url = f"{url}?{urlencode(params)}"

    return url


# Return the HTTP headers needed for a CivitAI API request.
def build_headers(api_key: str | None = None) -> dict[str, str]:
    headers = {"User-Agent": "LoRA-Library-Manager/0.1"}
    selected_api_key = api_key if api_key is not None else get_civitai_api_key()

    if selected_api_key:
        headers["Authorization"] = f"Bearer {selected_api_key}"

    return headers


# Build a standard response for successful CivitAI API data.
def success_response(data: dict) -> dict:
    return {
        "ok": True,
        "status": 200,
        "error": None,
        "data": data,
    }


# Build a standard response for graceful CivitAI API failures.
def error_response(status: int | None, error: str, message: str) -> dict:
    return {
        "ok": False,
        "status": status,
        "error": error,
        "message": message,
        "data": None,
    }


# Return the parsed CivitAI data payload whether it is wrapped or direct JSON.
def unwrap_civitai_metadata(civitai_metadata: dict) -> dict:
    if civitai_metadata.get("ok") is True and isinstance(civitai_metadata.get("data"), dict):
        return civitai_metadata["data"]

    return civitai_metadata


# Convert an HTTP status code into a user-readable CivitAI error response.
def handle_http_error(error: HTTPError) -> dict:
    if error.code == 404:
        return error_response(404, "not_found", "No CivitAI model was found for this hash.")

    if error.code == 401:
        return error_response(401, "invalid_api_key", "The CivitAI API key is invalid.")

    if error.code == 403:
        return error_response(403, "invalid_api_key", "The CivitAI API key was rejected.")

    if error.code == 429:
        return error_response(429, "rate_limited", "CivitAI rate limit reached. Try again later.")

    return error_response(error.code, "http_error", f"CivitAI request failed with status {error.code}.")


# Fetch JSON data from the CivitAI API and return parsed JSON or a graceful error.
def fetch_json(
    endpoint: str,
    params: dict[str, str | int] | None = None,
    api_key: str | None = None,
) -> dict:
    url = build_api_url(endpoint, params)
    request = Request(url, headers=build_headers(api_key))

    try:
        with urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            return success_response(data)
    except HTTPError as error:
        return handle_http_error(error)
    except URLError as error:
        return error_response(None, "network_error", f"Could not reach CivitAI: {error.reason}")
    except TimeoutError:
        return error_response(None, "network_error", "The CivitAI request timed out.")
    except json.JSONDecodeError:
        return error_response(None, "invalid_response", "CivitAI returned invalid JSON.")


# Compute the SHA256 hash for a safetensors file.
def compute_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()

    with file_path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(CHUNK_SIZE), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# Query CivitAI for model version metadata using a SHA256 hash.
def get_model_by_hash(file_hash: str, api_key: str | None = None) -> dict:
    clean_hash = file_hash.strip().lower()
    return fetch_json(f"model-versions/by-hash/{clean_hash}", api_key=api_key)


# Compute a safetensors file hash and fetch its matching CivitAI metadata.
def get_model_for_safetensors(file_path: Path, api_key: str | None = None) -> dict:
    if file_path.suffix.lower() != ".safetensors":
        return error_response(None, "invalid_file", "Only .safetensors files can be hashed by this helper.")

    if not file_path.exists():
        return error_response(None, "missing_file", "The safetensors file does not exist.")

    file_hash = compute_sha256(file_path)
    return get_model_by_hash(file_hash, api_key)


# Search Civitai models using a text query.
def search_models(query: str, limit: int = 20) -> dict:
    return fetch_json("models", {"query": query, "types": "LORA", "limit": limit})


# Fetch details for one Civitai model by model ID.
def get_model(model_id: int) -> dict:
    return fetch_json(f"models/{model_id}")


# Extract likely download URLs from a Civitai model response.
def extract_download_urls(model_data: dict) -> list[str]:
    urls = []

    for version in model_data.get("modelVersions", []):
        for file_info in version.get("files", []):
            download_url = file_info.get("downloadUrl")
            if download_url:
                urls.append(download_url)

    return urls
