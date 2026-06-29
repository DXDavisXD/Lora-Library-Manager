\# Changelog



All notable changes to \*\*LoRA Library Manager\*\* are documented in this file.



This project follows a milestone-based changelog rather than strict semantic versioning while development is ongoing.



\---



\# v2.0 (Current)



This release transforms LoRA Library Manager from a simple organization utility into a complete local-first LoRA management suite.



\---



\## ✨ Added



\### Interactive Gallery



\- Modern responsive HTML gallery

\- Masonry card layout

\- Version grouping

\- Version selector

\- Base model filtering

\- Collection filtering

\- Search improvements

\- Sorting improvements

\- Favorites

\- Ratings

\- Personal notes

\- Personal trigger words

\- Trigger grouping

\- Hover previews

\- Improved Details Modal



\---



\### Prompt Builder



\- Prompt Builder panel

\- Adjustable LoRA weights

\- Prompt history

\- Trigger copying

\- Official trigger support

\- Personal trigger support

\- Copy LoRA tag

\- Add to Prompt workflow



\---



\### Collections



\- User-defined collections

\- Collection filtering

\- Collection assignment

\- Collection search

\- Collection management

\- Persistent collection storage



\---



\### User Data



Introduced `user\_data.json`.



Stores:



\- Ratings

\- Notes

\- Collections

\- Personal trigger words



Separates personal data from downloaded metadata.



\---



\### Local Server



\- Built-in HTTP server

\- REST API

\- Live editing

\- Dashboard

\- Automatic gallery refresh

\- Local-only workflow



\---



\### Manifest System



Added portable library manifests.



Supports:



\- Export

\- Import

\- SHA256 verification

\- User data restoration

\- Prompt preset support

\- Version-aware restoration



\---



\### Incremental Library Index



Introduced `library\_index.json`.



Features:



\- Incremental scanning

\- Smart refresh

\- Cached gallery entries

\- Reduced filesystem traversal

\- Faster startup

\- Improved scalability



\---



\### Migration Wizard



Added support for importing existing libraries.



Features:



\- Folder analysis

\- Health reports

\- Existing library detection

\- Collection import

\- Read-only migration workflow



Designed specifically for users with large manually organized libraries.



\---



\### Diagnostics



New diagnostics module.



Checks:



\- Missing metadata

\- Missing previews

\- Duplicate Version IDs

\- Duplicate SHA256 hashes

\- Missing models

\- Orphan asset folders



\---



\### Performance



\- Incremental gallery generation

\- Metadata caching

\- Reduced disk activity

\- Atomic writes

\- Stale temporary file cleanup

\- Startup optimizations



\---



\### CLI



Expanded command-line interface.



Added support for:



\- Automatic organization

\- Manual organization

\- Gallery generation

\- Local server

\- Diagnostics

\- Manifest export/import

\- Migration Wizard

\- Library Analysis



\---



\## 🎨 Improved



\### Gallery UI



Complete redesign.



Improvements include:



\- Responsive layout

\- Better card spacing

\- Improved Details Modal

\- Cleaner metadata display

\- Sticky action buttons

\- Better trigger organization

\- Better collection management

\- Improved version handling

\- Improved search experience



\---



\### Details Modal



Major redesign.



Added:



\- Version selector

\- Better metadata layout

\- Collapsible personal triggers

\- Better collection editor

\- Improved notes editor

\- Better trigger management

\- Sticky action area



\---



\### Backend



Refactored:



\- Modular architecture

\- Configuration handling

\- Logging

\- Atomic writes

\- Cache handling

\- File traversal



\---



\## 🔧 Changed



\- User information now stored separately from downloaded metadata.

\- Gallery now uses incremental rebuilding whenever possible.

\- Organization workflow optimized for large libraries.

\- Manifest format redesigned for portability.

\- Prompt Builder integrated directly into the gallery.

\- Library Analysis introduced before migration operations.



\---



\## 🛡 Safety



Added safeguards including:



\- Atomic file operations

\- Dry Run mode

\- Automatic backups

\- Incremental updates

\- Health analysis

\- Manifest verification



\---



\# v1.0



Initial public milestone.



Included:



\- Automatic organization

\- Metadata downloading

\- Preview downloading

\- HTML gallery generation

\- Basic search

\- Logging

\- Configuration system

\- Dry Run mode

\- Manual organization

\- Gallery generation



This release established the foundation of the project.



\---



\# v0.x



Early development builds.



Focused on:



\- Learning CivitAI APIs

\- Folder organization

\- Metadata parsing

\- Preview downloading

\- HTML generation

\- Internal testing

\- Performance experiments



These builds were never intended for public release.



\---



\# Future



Planned features include:



\- Folder synchronization from Collections

\- Real-time filesystem monitoring

\- Duplicate merge assistant

\- Batch editing

\- Plugin system

\- AI-assisted tagging

\- Improved dashboard

\- Advanced library statistics

\- Better update management



\---



\## Versioning



During active development, this project follows milestone-based releases rather than strict semantic versioning.



Major milestones represent significant architectural improvements and new capabilities rather than incremental patches.

