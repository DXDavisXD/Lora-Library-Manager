"""Download helpers for LoRA model files and related assets."""

import json
import os
from pathlib import Path
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from civitai import unwrap_civitai_metadata

from config import (
    PREVIEW_EXTENSIONS,
    is_dry_run,
    is_verbose,
    log_operation,
    is_lora_file,
)

METADATA_FILENAME = "metadata.json"
DEFAULT_TIMEOUT = 60


# Guess a readable filename from a download URL.
def filename_from_url(url: str, fallback: str = "downloaded_lora") -> str:
    path = urlparse(url).path
    filename = unquote(Path(path).name)
    return filename or fallback


# Return the image extension from a URL, falling back to jpg when unknown.
def image_extension_from_url(url: str) -> str:
    extension = Path(urlparse(url).path).suffix.lower()

    if extension in PREVIEW_EXTENSIONS:
        return extension

    return ".jpg"


# Build a standard status object for downloader operations.
def make_status(status: str, path: Path | None = None, message: str = "") -> dict:
    return {
        "status": status,
        "path": str(path) if path else "",
        "message": message,
    }


# Return model files found directly inside a model folder.
def find_model_files(model_folder: Path) -> list[Path]:
    if not model_folder.exists():
        return []

    return [
        path
        for path in model_folder.iterdir()
        if is_lora_file(path)
    ]


# Return the model stem that should be used for same-basename assets.
def get_model_stem(model_folder: Path) -> str:
    model_files = find_model_files(model_folder)

    for model_file in model_files:
        if model_file.stem.casefold() == model_folder.name.casefold():
            return model_file.stem

    if model_files:
        return sorted(model_files)[0].stem

    return model_folder.name


# Return the standard same-basename preview image path for a model folder.
def preview_image_path(model_folder: Path) -> Path:
    return model_folder / f"{get_model_stem(model_folder)}.png"


# Check whether a preview image already exists in a model folder.
def preview_image_exists(model_folder: Path) -> bool:
    return preview_image_path(model_folder).exists()


# Return the metadata file path for a model folder.
def metadata_path(model_folder: Path) -> Path:
    return model_folder / METADATA_FILENAME


# Check whether a model folder already has all completed download assets.
def model_assets_complete(model_folder: Path) -> bool:
    return metadata_path(model_folder).exists() and preview_image_exists(model_folder)


# Save CivitAI metadata to metadata.json without overwriting an existing file.
def save_metadata_if_missing(model_folder: Path, civitai_metadata: dict) -> dict:
    target_path = metadata_path(model_folder)

    if target_path.exists():
        return make_status("skipped", target_path, "metadata.json already exists.")

    if is_dry_run():
        log_operation(f"[DRY RUN] Would save metadata to: {target_path}")
        return make_status("saved", target_path, "metadata.json saved (dry run).")

    model_folder.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".json.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as output_file:
            json.dump(unwrap_civitai_metadata(civitai_metadata), output_file, indent=2)
        os.replace(tmp_path, target_path)
        log_operation(f"Saved metadata: {target_path}")
        return make_status("saved", target_path, "metadata.json saved.")
    except Exception as e:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        log_operation(f"Error saving metadata to {target_path}: {e}")
        return make_status("error", target_path, str(e))


# Extract image URLs from a list of CivitAI image objects, filtering out videos.
def extract_urls_from_images(images: list) -> list[str]:
    urls = []

    for image in images:
        if isinstance(image, dict) and image.get("url"):
            # Skip videos based on metadata type
            if image.get("type") == "video":
                continue

            # Skip videos based on URL file extension
            url = image["url"]
            path = urlparse(url).path
            suffix = Path(path).suffix.lower()
            if suffix in {".mp4", ".webm", ".mkv", ".mov", ".avi"}:
                continue

            urls.append(url)

    return urls



# Find the best available preview image URL inside CivitAI metadata.
def extract_preview_url(civitai_metadata: dict) -> str:
    metadata = unwrap_civitai_metadata(civitai_metadata)
    urls = []

    if isinstance(metadata.get("images"), list):
        urls.extend(extract_urls_from_images(metadata["images"]))

    model = metadata.get("model", {})
    if isinstance(model, dict) and isinstance(model.get("images"), list):
        urls.extend(extract_urls_from_images(model["images"]))

    for version in metadata.get("modelVersions", []):
        if isinstance(version, dict) and isinstance(version.get("images"), list):
            urls.extend(extract_urls_from_images(version["images"]))

    return urls[0] if urls else ""


# Download a URL to a specific file path without overwriting an existing file.
def download_to_path(url: str, target_path: Path, retries: int = 3) -> dict:
    if target_path.exists():
        return make_status("skipped", target_path, "File already exists.")

    if is_dry_run():
        log_operation(f"[DRY RUN] Would download {url} to {target_path}")
        return make_status("downloaded", target_path, "File downloaded (dry run).")

    target_path.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "LoRA-Library-Manager/0.1"})
    last_error = ""

    tmp_path = target_path.with_suffix(".tmp")
    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
                with tmp_path.open("wb") as output_file:
                    output_file.write(response.read())

            os.replace(tmp_path, target_path)
            log_operation(f"Downloaded: {url} -> {target_path}")
            return make_status("downloaded", target_path, "File downloaded.")
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            last_error = str(error)
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

            if attempt < retries:
                sleep(attempt)

    log_operation(f"Error downloading {url} to {target_path}: {last_error}")
    return make_status("error", target_path, last_error)


# Download a URL to a destination folder and return a status object.
def download_file(
    url: str,
    destination_dir: Path,
    filename: str | None = None,
    retries: int = 3,
) -> dict:
    target_name = filename or filename_from_url(url)
    target_path = destination_dir / target_name
    return download_to_path(url, target_path, retries)


# Download several URLs into the same destination folder.
def download_many(urls: list[str], destination_dir: Path, retries: int = 3) -> list[dict]:
    downloaded_files = []

    for url in urls:
        downloaded_files.append(download_file(url, destination_dir, retries=retries))

    return downloaded_files


# Download the preview image for a model folder when no preview exists yet.
def download_preview_if_missing(model_folder: Path, civitai_metadata: dict, retries: int = 3) -> dict:
    if preview_image_exists(model_folder):
        return make_status("skipped", None, "Preview image already exists.")

    preview_url = extract_preview_url(civitai_metadata)

    if not preview_url:
        return make_status("missing", None, "No preview image URL found in CivitAI metadata.")

    target_path = preview_image_path(model_folder)
    return download_to_path(preview_url, target_path, retries)


# Download model assets from CivitAI metadata without overwriting local files.
def download_model_assets(model_folder: Path, civitai_metadata: dict, retries: int = 3) -> dict:
    return {
        "metadata": save_metadata_if_missing(model_folder, civitai_metadata),
        "preview_image": download_preview_if_missing(model_folder, civitai_metadata, retries),
    }
