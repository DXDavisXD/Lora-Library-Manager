# LoRA Library Manager (Version 2.0)

A high-performance, stable command-line utility and responsive dark-themed web gallery for organizing, hashing, downloading metadata/previews from CivitAI, and building prompts for large libraries of Stable Diffusion LoRA models.

---

## Features

*   **Version Grouping**: Groups multiple versions/variants of a model under a single card, sorted by release version (CivitAI `version_id` descending, falling back to modification time). The latest version is automatically selected by default.
*   **Dynamic Version Selector**: A dropdown selector on each card allows switching between variants dynamically, instantly updating preview images, tags, trigger words, description, and copy parameters without page refreshes.
*   **Secure Model Version Deletion**: Delete unwanted variants directly from the gallery interface. Terminates model files, metadata, and preview assets securely via the backend, and automatically cleans up empty parent directories.
*   **Asynchronous Rebuilds**: Deleted items immediately fade from the gallery DOM, while a background thread updates the `gallery.html` index asynchronously to keep the UI smooth and responsive.
*   **Library Manifests (Snapshot Backup/Restore)**:
    *   **Export**: Completely self-contained, offline-friendly snapshot generation that computes SHA256 hashes and builds a full library index including personal ratings, collections, notes, and trigger words.
    *   **Import**: Restores models from a manifest, automatically resolving files. Includes **Smart Folder Matching** to download new versions directly into folders where other variants of that model are already present, and auto-runs library organization and gallery refresh on completion.
*   **Unicode Safety**: Fully compatible with international characters and Japanese directory names (like "有馬 かな" and "ツルギ") on Windows CMD/PowerShell consoles without crashing.

---

## Requirements

*   **Python**: Python 3.10 or newer.
*   **Dependencies**: Uses only Python standard library modules (`pathlib`, `shutil`, `json`, `urllib`, `time`, `hashlib`, `socketserver`, `http.server`, `webbrowser`). No external pip dependencies are required.
*   **Web Browser**: Any modern web browser supporting ES6, CSS Grid, and HTML5 `localStorage` (Chrome, Firefox, Edge, Safari).

---

## Installation & Configuration

1.  Clone or copy the application files to your local system.
2.  Open [config.py](file:///e:/Lora-Project/Lora-Library-Manager-dev/config.py) and configure your variables:
    *   `LORA_ROOT`: Absolute path to your Stable Diffusion LoRA models folder (e.g. `Path(r"E:\SD\webui-forge\models\Lora")`).
    *   `CIVITAI_API_KEY`: Your personal CivitAI API Key for accessing remote metadata.
    *   `TMP_CLEANUP_AGE_SECONDS`: Time threshold (default `3600` seconds / 1 hour) for safe startup cleanup of temporary files.

---

## Library Folder Structure

The manager organizes loose LoRA model files into clean, dedicated folders named after their official CivitAI model names:

```
LORA_ROOT/
├── Anima Turbo LoRA/
│   ├── anima-turbo-lora-v0.2.safetensors
│   ├── anima-turbo-lora-v0.2.png (local preview image)
│   └── metadata.json (CivitAI metadata)
├── Kana Arima (有馬 かな)  Tsurugi (ツルギ) - Oshi no Ko (【推しの子】)/
│   ├── kana-arima-irekomi-illustriousxl-lora-nochekaiser.safetensors
│   ├── kana-arima-irekomi-illustriousxl-lora-nochekaiser.json
│   ├── kana-arima-irekomi-illustriousxl-lora-nochekaiser.jpeg
│   ├── kana-arima-s1s2-illustriousxl-lora-nochekaiser.safetensors
│   ├── kana-arima-s1s2-illustriousxl-lora-nochekaiser.json
│   ├── kana-arima-s1s2-illustriousxl-lora-nochekaiser.png
│   └── metadata.json (shared folder-level index)
├── logs/
│   ├── 2026-06-15.log
│   └── 2026-06-16.log
└── gallery.html (generated dark-theme web preview UI)
```

---

## CLI Options & Operations

On startup, the manager automatically cleans up stale `.tmp` files (older than 1 hour) and loads your persisted preferences from `settings.json`. It displays a status header and menu:

### Header Indicators
*   **App Name**: LoRA Library Manager
*   **Library Path**: Path currently pointed to by `LORA_ROOT`.
*   **Dry Run Mode**: `ENABLED` or `DISABLED`.
*   **Verbose Mode**: `ENABLED` or `DISABLED`.

### Menu Operations
1.  **Automatic**: Performs a complete workflow: runs file organization, queries CivitAI to download missing metadata and preview images, and refreshes the web gallery.
2.  **Manual Operations Submenu**: Opens a sub-menu containing targeted manual operations:
    *   *Process Single File*: Prompts for a single `.safetensors` file path to fetch and save its CivitAI assets locally.
    *   *Organize Only*: Scans `LORA_ROOT` and moves loose `.safetensors`, `.ckpt`, `.pt`, and `.bin` models alongside their matching sidecars into structured folders, cleaning up empty folders.
    *   *Download Previews Only*: Crawls folders and downloads missing preview images from CivitAI.
    *   *Download Metadata Only*: Crawls folders and fetches missing metadata json files from CivitAI.
    *   *Generate Gallery Only*: Compiles all organized LoRA models, metadata, and preview paths into `gallery.html`.
    *   *Go Back to Main Menu*: Returns to the main menu.
3.  **Start Gallery Server**: Launches a local HTTP server on port 8000 to host the interactive web UI and handle model deletions.
4.  **Export Library Manifest**: Exports a snapshot containing metadata statistics, active ratings, user collections, notes, and hashes.
5.  **Import Library Manifest**: Imports a snapshot manifest, downloads missing models/metadata/previews, restores your sidecars, organizes the library, and refreshes the gallery.
6.  **Exit**: Safely closes the CLI menu.
7.  **Toggle Dry Run Mode**: Toggle non-destructive dry runs. In Dry Run mode, all filesystem writes and downloads are bypassed and reported without modifying files. Saves instantly to `settings.json`.
8.  **Toggle Verbose Mode**: Toggle detailed terminal log output. Saves instantly to `settings.json`.
9.  **Set CivitAI API Key**: Set or clear your CivitAI API Key.

---

## Web UI & Gallery Features

Open `gallery.html` in your web browser (or start the server using Option 7) to access the responsive dark-theme dashboard:

### 1. Multi-Version Card UI
*   **Version Dropdown**: Located directly on each card, letting you toggle between versions. Swapping a version instantly updates the active preview image, descriptions, tags, and builder hooks.
*   **Ellipsis Text Truncation**: Supports long version name strings gracefully inside a flexible card layout.
*   **Variant Deletion**: A delete button next to the dropdown deletes specific files (Safetensors, metadata, preview) from your system securely.

### 2. Collapsible Trigger Word Groups
*   **Outfit & Preset Preservations**: Instead of flattening trigger words, the gallery groups them by style presets extracted directly from CivitAI metadata (e.g. `School Uniform`, `White Dress`, `Casual`).
*   **Accordion Folders**: Groups render collapsed by default to save card space. Click any group header to expand it with a smooth animation.
*   **Copy Button per Group**: Click the inline copy icon on any group header to copy only the trigger words contained in that specific style group to your clipboard.

### 3. Interactive Prompt Builder Panel
*   **Docked Panel**: A persistent builder panel sits anchored at the bottom of the page, showing the active LoRA tokens in your prompt.
*   **Add to Prompt**: Click `[+ Add to Prompt]` on any card. It appends `<lora:filename_without_extension:selected_weight>` using the global weight selector's current value.
*   **Tag Weight Selectors**: Select Copy weights (`0.6`, `0.8`, `1.0`, `1.2`) in the top toolbar. The builder remembers the last selected weight using `localStorage`.
*   **Token Management**: Click the close `×` button on any token to remove it individually, clear the entire builder with `Clear All`, or copy the combined spaced-separated prompt string using `Copy Prompt`.

### 4. Smart Search & Filters
*   **Global Search**: Instantly searches by model name, folder name, descriptions, tags, creator usernames, and hidden trigger fields.
*   **Favorites Filter**: Persists favorited models in browser `localStorage` using their SHA256 hash. Click the favorite toggle to display only starred items.
*   **Double-click Triggers**: Double-clicking the card preview image copies all trigger words for that model version immediately.

### 5. Accessibility Support
*   **Full Keyboard Navigation**: Navigate all buttons, favorites, triggers, dropdowns, and prompt builder items using the `Tab` key.
*   **Action Triggers**: Press `Enter` or `Space` on any focused header or button to expand groups, toggle favorites, copy tags, or clear tokens.
*   **ARIA attributes**: All interactive icons, input elements, and buttons utilize semantic ARIA labels for screen readers.

### 6. Keyboard Shortcuts
*   `Ctrl+F` (or `Cmd+F`): Focuses and highlights the search input.
*   `Escape`: Instantly clears search text, blurs focus, and resets filters.

---

## Changelog

### Version 1.8.0
*   **Version Grouping**: Groups multiple versions of a model under one card, sorted by CivitAI version ID (descending) so the latest release is selected by default.
*   **Dynamic UI Dropdowns**: Switching variants dynamically updates preview images, descriptions, tags, and builder hooks in real-time.
*   **Model Deletion Support**: Integrated a delete button directly inside the web UI, securely erasing models and sidecars, automatically cleaning up parent folders, and refreshing the gallery asynchronously in a background thread.
*   **Manifest Exports & Imports**: Added offline snapshot export support and smart folder matching during import to download files directly into existing folders of the same model. Added automatic organizer and gallery refresh post-import.
*   **Unicode Safety**: Added safety checks to prevent UnicodeEncodeError crashes when handling directories with international characters in Windows consoles.
