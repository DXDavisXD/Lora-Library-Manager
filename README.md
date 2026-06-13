# LoRA Library Manager (Version 1.0)

A high-performance, stable command-line utility and responsive dark-themed web gallery for organizing, hashing, downloading metadata/previews from CivitAI, and building prompts for large libraries of Stable Diffusion LoRA models.

---

## Requirements

*   **Python**: Python 3.10 or newer.
*   **Dependencies**: Uses only Python standard library modules (`pathlib`, `shutil`, `json`, `urllib`, `time`, `hashlib`). No external pip dependencies are required.
*   **Web Browser**: Any modern web browser supporting ES6, CSS Grid, and HTML5 `localStorage` (Chrome, Firefox, Edge, Safari).

---

## Installation & Configuration

1.  Clone or copy the application files to your local system.
2.  Open [config.py](file:///e:/Lora-Project/Lora-Library-Manager%20v1/config.py) and configure your variables:
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
├── Makima (マキマ) - Chainsaw Man/
│   ├── makima.safetensors
│   ├── makima.png
│   └── metadata.json
├── logs/
│   ├── 2026-06-13.log
│   └── 2026-06-14.log
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
2.  **Manual**: Prompts for a single `.safetensors` file path to fetch and save its CivitAI assets locally.
3.  **Organize only**: Scans `LORA_ROOT` and moves loose `.safetensors`, `.ckpt`, `.pt`, and `.bin` models alongside their matching sidecars into structured folders.
4.  **Download previews**: Crawls folders and downloads missing preview images from CivitAI.
5.  **Download metadata**: Crawls folders and fetches missing `metadata.json` files from CivitAI.
6.  **Generate Gallery**: Compiles all organized LoRA models, metadata, and preview paths into `gallery.html`.
7.  **Exit**: Safely closes the CLI menu.
8.  **Toggle Dry Run Mode**: Toggle non-destructive dry runs. In Dry Run mode, all filesystem writes and downloads are bypassed and reported without modifying files. Saves instantly to `settings.json`.
9.  **Toggle Verbose Mode**: Toggle detailed terminal log output. Saves instantly to `settings.json`.

---

## Web UI & Gallery Features

Open `gallery.html` in your web browser to access the responsive dark-theme dashboard:

### 1. Collapsible Trigger Word Groups
*   **Outfit & Preset Preservations**: Instead of flattening trigger words, the gallery groups them by style presets extracted directly from CivitAI metadata (e.g. `School Uniform`, `White Dress`, `Casual`).
*   **Accordion Folders**: Groups render collapsed by default to save card space. Click any group header to expand it with a smooth animation.
*   **Copy Button per Group**: Click the inline copy icon on any group header to copy only the trigger words contained in that specific style group to your clipboard.

### 2. Interactive Prompt Builder Panel
*   **Docked Panel**: A persistent builder panel sits anchored at the bottom of the page, showing the active LoRA tokens in your prompt.
*   **Add to Prompt**: Click `[+ Add to Prompt]` on any card. It appends `<lora:filename_without_extension:selected_weight>` using the global weight selector's current value.
*   **Tag Weight Selectors**: Select Copy weights (`0.6`, `0.8`, `1.0`, `1.2`) in the top toolbar. The builder remembers the last selected weight using `localStorage`.
*   **Token Management**: Click the close `×` button on any token to remove it individually, clear the entire builder with `Clear All`, or copy the combined spaced-separated prompt string using `Copy Prompt`.

### 3. Smart Search & Filters
*   **Global Search**: Instantly searches by model name, folder name, descriptions, tags, creator usernames, and hidden trigger fields.
*   **Favorites Filter**: Persists favorited models in browser `localStorage` using their SHA256 hash (or relative path if missing). Click the favorite toggle to display only starred items. Filters work dynamically alongside text searches.
*   **Double-click Triggers**: Double-clicking the card preview image copies all trigger words for that model immediately and displays a temporary "Triggers Copied!" feedback message.

### 4. Accessibility Support
*   **Full Keyboard Navigation**: Navigate all buttons, favorites, triggers, dropdowns, and prompt builder items using the `Tab` key.
*   **Action Triggers**: Press `Enter` or `Space` on any focused header or button to expand groups, toggle favorites, copy tags, or clear tokens.
*   **ARIA attributes**: All interactive icons, input elements, and buttons utilize semantic ARIA labels for screen readers.

### 5. Keyboard Shortcuts
*   `Ctrl+F` (or `Cmd+F`): Focuses and highlights the search input.
*   `Escape`: Instantly clears search text, blurs focus, and resets filters.

---

## Example Workflow

### Scenario: Organizing and Generating a Prompt
1.  Place a fresh set of downloaded LoRA models into your `LORA_ROOT` folder.
2.  Start `main.py` and run option `1` (Automatic). The manager will:
    *   Scan and organize files into folders like `Anima Turbo LoRA/`.
    *   Compute the SHA256 hash of each safetensors file and query the CivitAI API.
    *   Download preview images and `metadata.json` files.
    *   Compile `gallery.html`.
3.  Open `gallery.html` in a web browser.
4.  Use `Ctrl+F` to search for your model (e.g. `Makima`).
5.  In the top toolbar, select weight `0.8`.
6.  Click `[+ Add to Prompt]` on the Makima card. The prompt builder panel opens displaying `<lora:makima:0.8>`.
7.  Click `Copy Prompt` to copy the ready-to-use tag string for WebUI generation.
