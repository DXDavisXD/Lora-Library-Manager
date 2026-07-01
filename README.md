# LoRA Library Manager

<p align="center">

**A modern, local-first LoRA management suite for Stable Diffusion.**

Organize • Browse • Search • Build Prompts • Backup • Migrate • Analyze

</p>

---

## 📖 Overview

Managing a growing LoRA library quickly becomes difficult.

After a few hundred models, Windows Explorer simply isn't enough:

- Finding the right LoRA takes time.
- Trigger words are scattered across CivitAI pages.
- Duplicate versions accumulate.
- Metadata becomes inconsistent.
- Preview images go missing.
- Existing folder structures become difficult to maintain.

LoRA Library Manager solves these problems by combining automatic organization, metadata management, an interactive web gallery, prompt building, diagnostics, migration utilities, and backup tools into a single local application.

The entire project runs locally. No cloud services are required.

---

# ✨ Features

## 📚 Library Management

- Automatic LoRA organization
- Manual organization tools
- Existing library import (Migration Wizard)
- Folder hierarchy analysis
- Collection management
- Duplicate detection
- Missing asset detection
- Incremental library indexing
- Smart refresh system
- SD WebUI Compatibility (Forge/A1111 sidecar parsing/merging)
- Lenient sidecar repair parser (recovers from missing commas, syntax errors)

---

## 🖼 Interactive Gallery

A modern browser-based gallery generated directly from your LoRA library.

Features include:

- Responsive masonry layout
- Multiple version support
- Version selector
- Favorites
- Ratings
- Personal notes
- Collections
- Trigger groups
- Personal trigger words
- Trigger copying
- Prompt builder integration
- Hover previews
- Search
- Base model filtering
- Collection filtering
- Sorting options

---

## 🎛 Web Management Dashboard

Access a live control panel at `http://localhost:8000/manage` to manage your library from your browser:

- **Live Logs Streaming**: Stream real-time terminal sync details directly via Server-Sent Events (SSE).
- **Library Health Dashboard**: Monitor total file count, size, missing previews, and missing metadata at a glance.
- **Sync Control Triggers**: Run Automatic Sync, Quick Refresh, Metadata/Preview Downloads, and metadata migrations with a single click.
- **Settings Configuration**: Toggle dry-run and verbose settings dynamically from the web interface.

---

## ✍ Prompt Builder

Build prompts without leaving the gallery.

Supports:

- Add LoRA
- Adjustable LoRA weights
- Prompt history
- Trigger copying
- Official trigger words
- Personal trigger words
- Multiple version support

---

## 📦 Manifest System

Export your entire library into a portable manifest.

Includes:

- Model information
- SHA256 hashes
- User data
- Ratings
- Notes
- Collections
- Personal trigger words
- Prompt presets
- Structural checksum

Importing restores:

- Models
- Metadata
- Preview images
- User data

Perfect for moving libraries between computers.

---

## 📂 Import Existing Library

Designed specifically for users with already organized libraries.

Instead of forcing a new folder structure, the manager can analyze your existing organization and import it as Collections.

This makes migrating libraries containing thousands of manually organized LoRAs significantly easier.

Current workflow:

```
Existing Library

        ↓

Analyze Folder Structure

        ↓

Generate Health Report

        ↓

Import Collections

        ↓

Keep Existing Layout Intact
```

---

## 🩺 Library Analysis

Analyze the health of your library before making changes.

Reports include:

- Models detected
- Collections detected
- Missing metadata
- Missing previews
- Duplicate SHA256 hashes
- Duplicate Version IDs
- Orphan asset folders
- Missing models
- Folder statistics
- Migration readiness

---

## 🚀 Performance

Large libraries remain responsive thanks to:

- Incremental indexing
- Library cache
- Smart refresh
- Cached metadata
- Atomic writes
- Reduced filesystem traversal
- Local HTTP server

Only modified models are reprocessed whenever possible.

---

## 🌐 Local Web Server

The built-in server powers the interactive gallery.

Features:

- Live editing
- User data updates
- Collection management
- Prompt builder
- Details modal
- Dashboard
- REST API
- Automatic gallery refresh

No internet connection is required after metadata has been downloaded.

---

# 🗂 Folder Structure

Example library:

```
LoRA/

├── Fate/
│   ├── Rin Tohsaka/
│   │   ├── Rin.safetensors
│   │   ├── metadata.json
│   │   ├── preview.png
│   │   └── user_data.json
│   │
│   └── Saber/
│       ├── Saber.safetensors
│       ├── metadata.json
│       ├── preview.png
│       └── user_data.json
│
├── collections.json
├── library_index.json
├── library_manifest.json
└── logs/
```

---

# ⚙ Core Components

| Module | Purpose |
|---------|----------|
| `main.py` | CLI entry point |
| `organizer.py` | Organizes LoRAs and folders |
| `gallery.py` | Generates the interactive gallery |
| `server.py` | Local HTTP server & API |
| `downloader.py` | Downloads metadata and preview images |
| `manifest.py` | Export & import library manifests |
| `migration.py` | Existing library migration & analysis |
| `diagnostics.py` | Library diagnostics |
| `library_index.py` | Incremental indexing & cache |
| `config.py` | Global configuration |

---

# 📸 Screenshots

*(Replace these placeholders with actual screenshots.)*

- Main Gallery
- Details Modal
- Prompt Builder
- Dashboard
- Library Analysis
- Migration Wizard
- Manifest Import
- Diagnostics

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Lora-Library-Manager.git

cd Lora-Library-Manager
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python main.py
```

---

# ⚡ Quick Start

1. Configure your LoRA root folder.
2. Run Automatic Organize.
3. Download metadata and previews.
4. Generate the gallery.
5. Launch the local server.
6. Open the gallery in your browser.
7. Start organizing using Collections, Ratings, Notes, and Prompt Builder.

---

# 💾 Data Files

The application stores additional information alongside your library.

| File | Purpose |
|------|---------|
| `metadata.json` | Downloaded CivitAI metadata |
| `user_data.json` | Personal ratings, notes, collections and triggers |
| `collections.json` | Collection definitions |
| `library_index.json` | Incremental cache |
| `library_manifest.json` | Portable backup |

---

# 🔒 Safety

The project is designed to minimize accidental data loss.

Features include:

- Atomic file writes
- Dry Run mode
- Automatic backups
- Logging
- Incremental updates
- Manifest export
- Health analysis before migration

---

# 🛣 Roadmap

### Completed

- Automatic organization
- Gallery generation
- Version grouping
- Prompt Builder
- Favorites
- Notes
- Ratings
- Collections
- Trigger groups
- Personal trigger words
- Manifest export/import
- Incremental indexing
- Local server
- Migration Wizard
- Library Analysis
- Diagnostics


---

# 🤝 Contributing

Contributions, suggestions and bug reports are always welcome.

If you have ideas for improving workflow, performance or usability, feel free to open an issue or submit a pull request.

---

## 📜 License

No license has been specified at this time.

Please do not redistribute or reuse this project's source code without permission.

A license may be added in a future release.

---

<p align="center">

Built for the Stable Diffusion community ❤️

</p>
