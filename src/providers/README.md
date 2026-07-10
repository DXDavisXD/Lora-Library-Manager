# Providers Architecture Overview

This directory implements the multi-provider abstraction layer for the LoRA Library Manager, allowing the application to resolve and download model metadata and files from multiple sources seamlessly.

## Supported Providers

1.  **CivitAI** (Domain: `civitai.com`)
2.  **Arcenciel** (Domain: `arcenciel.io`)

---

## Provider Abstraction Overview

The architecture decouples the main application from any single provider's API structure. The system is built around three core classes/interfaces:

1.  `ProviderBase` ([provider_base.py](file:///E:/Lora-Project/Lora-Library-Manager-dev/src/providers/provider_base.py)): Abstract base class defining the contract all providers must satisfy.
2.  `ModelInfo` ([provider_base.py](file:///E:/Lora-Project/Lora-Library-Manager-dev/src/providers/provider_base.py)): Immutable dataclass representing a fully normalized model version.
3.  `ProviderRegistry` ([registry.py](file:///E:/Lora-Project/Lora-Library-Manager-dev/src/providers/registry.py)): Global dispatcher that routes operations based on provider name or URL domain matching.

---

## ModelInfo Normalized Schema

A `ModelInfo` instance represents a specific version of a model. Key fields are:

| Field | Type | Description |
| :--- | :--- | :--- |
| `provider` | `str` | Name of the provider (e.g. `'civitai'`, `'arcenciel'`) |
| `provider_model_id` | `str` | Unique model ID on the provider's platform |
| `provider_version_id` | `str` | Unique version ID on the provider's platform |
| `name` | `str` | Normalized model title |
| `version_name` | `str` | Normalized version title (e.g., `'v1.0'`) |
| `creator` | `str` | Username of the creator / uploader |
| `description` | `str` | HTML-stripped, cleaned plain text description |
| `base_model` | `str` | Standardized base model type (e.g., `'SDXL 1.0'`, `'Illustrious'`) |
| `trained_words` | `list[str]` | Deduplicated list of activation/trigger tags |
| `preview_urls` | `list[str]` | List of resolved URLs to preview images |
| `tags` | `list[str]` | Associated tags or categories |
| `preferred_weight` | `float \| None` | Recommended inference weight |
| `negative_prompt` | `str` | Recommended negative prompt |
| `file_sha256` | `str` | Clean, lowercase SHA-256 hash of the model file |
| `original_filename` | `str` | Original filename as stored on the platform |
| `download_url` | `str` | Resolved URL to download the model file |

---

## Provider Contract Expectations

All concrete implementations of `ProviderBase` must implement the following methods:

### 1. Hash Lookup
```python
def fetch_by_hash(self, sha256: str, api_key: str | None = None) -> ModelInfo | None
```
Resolves a model version using its SHA-256 hash. Must fall back to checking AutoV3/WebUI hashes if the platform supports them.

### 2. ID Lookup
```python
def fetch_by_id(self, model_id: str | None, version_id: str, api_key: str | None = None) -> ModelInfo | None
```
Retrieves a model version by its platform model ID and version ID. `model_id` is optional but should be used if the platform lacks direct version lookup endpoints (e.g. Arcenciel).

### 3. Version Enumeration
```python
def fetch_model_versions(self, model_id: str, api_key: str | None = None) -> list[ModelInfo]
```
Returns all versions of a model, sorted newest-first (used for update detection).

### 4. Text Search
```python
def search(self, query: str, limit: int = 20, api_key: str | None = None) -> list[ModelInfo]
```
Queries the platform search index and returns a list of matching models.

### 5. Download URL Resolution
```python
def get_download_url(self, model_id: str | None, version_id: str, api_key: str | None = None) -> str
```
Resolves the direct download URL for a model version.

---

## Adding a New Provider

To add support for a new provider (e.g., `'huggingface'`):

1.  **Create Client Module**: If needed, add `src/huggingface.py` for API endpoint helpers.
2.  **Implement Provider Class**: Create `src/providers/huggingface_provider.py` inheriting from `ProviderBase`.
3.  **Register the Provider**: In `src/providers/registry.py`:
    *   Import and register the class in `_REGISTRY`.
    *   Add domain mapping rules in `resolve_provider_from_url` (e.g., `huggingface.co`).
4.  **Add Contract Tests**: Create a test file `src/providers/tests/test_huggingface_provider.py` inheriting from `ProviderContractTests`.
