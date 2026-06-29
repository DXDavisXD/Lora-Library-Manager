\# LoRA Library Manager Documentation



\---



\# Table of Contents



1\. Introduction

2\. Architecture

3\. Project Structure

4\. Application Workflow

5\. Configuration

6\. Core Modules

7\. Gallery System

8\. Local Server

9\. User Data System

10\. Collections

11\. Prompt Builder

12\. Manifest System

13\. Library Index

14\. Migration System

15\. Diagnostics

16\. Data Files

17\. CLI Reference

18\. Developer Notes



\---



\# Introduction



LoRA Library Manager is a local-first management suite designed for Stable Diffusion LoRA collections.



Unlike traditional file organizers, it combines:



\- File organization

\- Metadata management

\- Interactive browsing

\- Prompt building

\- User annotations

\- Library analysis

\- Backup \& restoration



into a single application.



The application is designed to remain responsive even for libraries containing thousands of LoRAs through incremental indexing and caching.



\---



\# Architecture



```



```

&#x20;               +--------------------+

&#x20;               |      main.py       |

&#x20;               +--------------------+

&#x20;                         │

&#x20;        ┌────────────────┼────────────────┐

&#x20;        │                │                │

&#x20;        ▼                ▼                ▼

&#x20;organizer.py      downloader.py    migration.py

&#x20;        │                │                │

&#x20;        └──────────┬─────┴─────┬──────────┘

&#x20;                   ▼

&#x20;           library\_index.py

&#x20;                   │

&#x20;                   ▼

&#x20;            gallery.py

&#x20;                   │

&#x20;                   ▼

&#x20;             server.py

&#x20;                   │

&#x20;                   ▼

&#x20;              Web Browser

```



\---



\# Project Structure



```

project/



├── main.py

├── config.py

├── organizer.py

├── downloader.py

├── gallery.py

├── server.py

├── manifest.py

├── migration.py

├── diagnostics.py

├── library\_index.py

├── requirements.txt

└── README.md

```



\---



\# Application Workflow



The application is divided into several independent systems.



\## Organization



```

User



↓



Automatic Organize



↓



Scan Library



↓



Move Packages



↓



Download Metadata



↓



Download Preview



↓



Update Cache



↓



Refresh Gallery

```



\---



\## Gallery



```

Library



↓



Gallery Builder



↓



gallery.html



↓



Local Server



↓



Browser

```



\---



\## Prompt Builder



```

LoRA



↓



Version Selection



↓



Weight



↓



Trigger Words



↓



Clipboard



↓



Prompt

```



\---



\# Configuration



Configuration is stored inside `config.py`.



Key settings include:



\- LoRA root directory

\- Supported model extensions

\- Supported preview extensions

\- Logging

\- Dry Run mode

\- Verbose mode

\- Cleanup settings



\---



\# Core Modules



\---



\## main.py



Main application entry point.



Responsibilities:



\- CLI

\- Menu handling

\- Workflow orchestration

\- Startup

\- Shutdown



\---



\## organizer.py



Responsible for physical file organization.



Features:



\- Automatic organization

\- Manual organization

\- Package moves

\- Folder creation

\- Duplicate detection



\---



\## downloader.py



Handles communication with CivitAI.



Downloads:



\- metadata.json

\- preview images



Uses atomic writes to prevent corruption.



\---



\## gallery.py



Generates the interactive HTML gallery.



Responsible for:



\- Card generation

\- Version grouping

\- Search

\- Collections

\- Prompt Builder

\- Details modal

\- Gallery index



\---



\## server.py



Local HTTP server.



Provides:



\- Gallery hosting

\- REST API

\- User data editing

\- Collection editing

\- Dashboard



\---



\## manifest.py



Creates portable backups.



Supports:



\- Export

\- Import

\- User data restoration

\- Checksum validation



\---



\## migration.py



Analyzes existing libraries.



Features:



\- Folder analysis

\- Health reports

\- Collection import

\- Existing library migration



\---



\## diagnostics.py



Library diagnostics.



Reports:



\- Missing metadata

\- Missing previews

\- Duplicate hashes

\- Duplicate Version IDs

\- Missing assets

\- Orphan folders



\---



\## library\_index.py



Incremental cache system.



Stores:



\- modification times

\- metadata timestamps

\- gallery entries

\- hashes



Used to avoid rebuilding the entire library.



\---



\# Gallery System



The gallery is generated as a completely static HTML page.



Advantages:



\- Extremely fast

\- Portable

\- Works offline

\- Easy to host



Features include:



\- Search

\- Sorting

\- Favorites

\- Ratings

\- Notes

\- Collections

\- Version selector

\- Trigger groups

\- Prompt Builder

\- Hover preview

\- Base model filtering



\---



\# Details Modal



Every LoRA can be opened in an extended Details Modal.



Supports:



\- Version switching

\- Metadata

\- Description

\- Notes

\- Collections

\- Ratings

\- Trigger groups

\- Personal triggers

\- Copy tag

\- Add to prompt

\- View on CivitAI



\---



\# User Data System



Personal information is stored separately from downloaded metadata.



```

metadata.json

```



Contains:



\- Official metadata



```

user\_data.json

```



Contains:



\- Notes

\- Rating

\- Collections

\- Personal trigger words



This separation prevents updates from overwriting user changes.



\---



\# Collections



Collections are user-defined groups.



Examples:



```

Favorites



Anime



Fate



Portrait



High Quality

```



Collections are logical only.



They do not need to match the folder structure.



\---



\# Prompt Builder



Prompt Builder allows multiple LoRAs to be assembled into prompts.



Supports:



\- Adjustable weights

\- History

\- Presets

\- Trigger groups

\- Clipboard copy



\---



\# Manifest System



The Manifest System creates portable backups.



Contents:



\- Model identifiers

\- SHA256

\- Metadata

\- User data

\- Collections

\- Notes

\- Ratings

\- Prompt presets



Designed for transferring libraries between computers.



\---



\# Library Index



The incremental index avoids rebuilding the gallery every run.



Workflow:



```

Scan



↓



Compare timestamps



↓



Changed?



↓



Yes → rebuild



No → reuse cached entry

```



This dramatically reduces processing time for large libraries.



\---



\# Migration System



Designed for users with existing organized libraries.



Workflow:



```

Existing Folder Structure



↓



Analyze



↓



Health Report



↓



Import Collections



↓



Leave Folder Structure Untouched

```



The Migration Wizard never reorganizes the physical library.



Its purpose is to learn the existing organization.



\---



\# Diagnostics



Diagnostics provide insight into library health.



Checks include:



\- Missing metadata

\- Missing previews

\- Duplicate models

\- Duplicate Version IDs

\- Duplicate SHA256

\- Orphan asset folders

\- Missing models



\---



\# Data Files



| File | Purpose |

|-------|----------|

| metadata.json | Official metadata |

| user\_data.json | Personal information |

| collections.json | User collections |

| library\_index.json | Incremental cache |

| library\_manifest.json | Portable backup |



\---



\# CLI Reference



| Option | Description |

|---------|-------------|

| Automatic Organize | Organize library automatically |

| Manual Organize | Manual processing |

| Download Assets | Metadata \& previews |

| Generate Gallery | HTML gallery |

| Launch Server | Local web server |

| Analyze Library | Health report |

| Import Existing Library | Import collections from folders |

| Export Manifest | Portable backup |

| Import Manifest | Restore backup |

| Diagnostics | Library analysis |



\---



\# Developer Notes



\## Atomic Writes



All important files are written to temporary files before replacing the originals.



This minimizes corruption if the process is interrupted.



\---



\## Incremental Cache



The application avoids unnecessary work by caching gallery entries and filesystem metadata.



Only changed models are rebuilt.



\---



\## Local-First Philosophy



The application is designed to work locally.



Internet access is only required when downloading metadata from CivitAI.



All browsing, searching, editing and prompt building continue to function offline.



\---



\## Future Goals



\- Folder synchronization

\- Filesystem watcher

\- Plugin support

\- Batch editing

\- AI-assisted tagging

\- Duplicate merge assistant

\- Improved dashboard

