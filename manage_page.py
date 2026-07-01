"""Generates the self-contained HTML for the /manage web Library Manager."""

from pathlib import Path
from config import LORA_EXTENSIONS, PREVIEW_EXTENSIONS, is_lora_file


def get_library_stats(lora_root: Path) -> dict:
    """Scan the library and return quick stats for the dashboard header."""
    total_loras = 0
    missing_preview = 0
    missing_metadata = 0
    total_size_bytes = 0
    collections_set = set()

    try:
        for f in lora_root.rglob("*"):
            if not f.is_file():
                continue
            if is_lora_file(f):
                total_loras += 1
                try:
                    total_size_bytes += f.stat().st_size
                except OSError:
                    pass

                stem = f.stem
                folder = f.parent
                has_preview = any(
                    (folder / f"{stem}{ext}").exists()
                    for ext in PREVIEW_EXTENSIONS
                )
                has_metadata = (
                    (folder / f"{stem}.json").exists()
                    or (folder / "metadata.json").exists()
                )
                if not has_preview:
                    missing_preview += 1
                if not has_metadata:
                    missing_metadata += 1

        for ud in lora_root.rglob("user_data.json"):
            try:
                import json
                with ud.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                for c in data.get("collections", []):
                    if c and c.strip():
                        collections_set.add(c.strip())
            except Exception:
                pass
    except Exception:
        pass

    gb = total_size_bytes / (1024 ** 3)
    size_str = f"{gb:.2f} GB" if gb >= 1 else f"{total_size_bytes / (1024**2):.0f} MB"

    return {
        "total_loras": total_loras,
        "missing_preview": missing_preview,
        "missing_metadata": missing_metadata,
        "collections": len(collections_set),
        "size": size_str,
    }


def build_manage_page(lora_root: Path) -> str:
    """Return the full HTML string for the Library Manager interface."""
    stats = get_library_stats(lora_root)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LoRA Library Manager — Control Center</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #090a0f;
    --panel: #11131c;
    --panel-header-bg: #151824;
    --panel-body-bg: #11131c;
    --border: rgba(255, 255, 255, 0.05);
    --border-hover: rgba(99, 102, 241, 0.35);
    --accent: #6366f1;
    --accent-hover: #4f46e5;
    --accent-alpha: rgba(99, 102, 241, 0.1);
    --text: #f3f4f6;
    --text-muted: #9ca3af;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.7);
    --radius: 12px;
    --transition: all .2s cubic-bezier(0.4, 0, 0.2, 1);
  }}

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    overflow-x: hidden;
  }}

  ::-webkit-scrollbar {{
    width: 6px;
    height: 6px;
  }}
  ::-webkit-scrollbar-track {{
    background: var(--bg);
  }}
  ::-webkit-scrollbar-thumb {{
    background: #272a37;
    border-radius: 4px;
  }}
  ::-webkit-scrollbar-thumb:hover {{
    background: #3c4055;
  }}

  /* ── Top bar ── */
  .topbar {{
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 14px 32px;
    background: rgba(17, 19, 28, 0.85);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 100;
  }}
  .topbar-logo {{
    font-size: 18px;
    font-weight: 800;
    letter-spacing: -0.025em;
    background: linear-gradient(135deg, #a5b4fc, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }}
  .topbar-sub {{
    font-size: 13px;
    color: var(--text-muted);
    font-weight: 500;
  }}
  .topbar-sep {{ flex: 1; }}
  .nav-btn {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 16px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.02);
    color: var(--text-muted);
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: var(--transition);
    text-decoration: none;
  }}
  .nav-btn:hover {{
    border-color: var(--border-hover);
    color: var(--text);
    background: var(--accent-alpha);
  }}

  /* ── Main layout ── */
  main {{
    max-width: 1400px;
    width: 100%;
    margin: 0 auto;
    padding: 32px;
    display: flex;
    flex-direction: column;
    gap: 28px;
    flex: 1;
  }}

  .header-area {{
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .page-title {{
    font-size: 24px;
    font-weight: 800;
    color: var(--text);
    letter-spacing: -0.5px;
  }}
  .page-title span {{
    color: var(--text-muted);
    font-weight: 400;
    font-size: 18px;
  }}

  /* ── Stats grid ── */
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 16px;
  }}
  .stat-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    transition: var(--transition);
    box-shadow: var(--shadow);
  }}
  .stat-card:hover {{ border-color: var(--border-hover); }}
  .stat-label {{
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: var(--text-muted);
  }}
  .stat-value {{
    font-size: 32px;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
  }}
  .stat-value.warn {{ color: var(--warning); }}
  .stat-value.ok   {{ color: var(--success); }}

  /* ── Content Grid ── */
  .content-grid {{
    display: grid;
    grid-template-columns: 360px 1fr;
    gap: 28px;
    align-items: start;
  }}

  /* ── Left Column Panels ── */
  .left-col {{
    display: flex;
    flex-direction: column;
    gap: 24px;
  }}

  .panel {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow);
    display: flex;
    flex-direction: column;
  }}
  .panel-header {{
    padding: 14px 20px;
    background: var(--panel-header-bg);
    border-bottom: 1px solid var(--border);
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .panel-body {{
    padding: 18px;
    background: var(--panel-body-bg);
    display: flex;
    flex-direction: column;
    gap: 12px;
  }}

  /* Collapsible content (sub-sections) */
  .section-group {{
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 12px;
  }}
  .section-group:last-child {{ margin-bottom: 0; }}
  .section-title {{
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    padding-bottom: 4px;
    margin-bottom: 6px;
  }}

  /* ── Action buttons ── */
  .action-btn {{
    width: 100%;
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.02);
    color: var(--text);
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: var(--transition);
    text-align: left;
  }}
  .action-btn:hover:not(:disabled) {{
    border-color: var(--accent);
    background: var(--accent-alpha);
    transform: translateX(3px);
  }}
  .action-btn:disabled {{ opacity: .35; cursor: not-allowed; }}
  .action-btn .btn-icon {{ font-size: 16px; flex-shrink: 0; }}
  .action-btn .btn-label {{ flex: 1; }}
  .action-btn .btn-sub {{ font-size: 11px; font-weight: 400; color: var(--text-muted); display: block; margin-top: 1px; }}

  /* Form layouts */
  .form-row {{
    display: flex;
    gap: 8px;
    align-items: center;
    width: 100%;
  }}
  .form-input {{
    flex: 1;
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-size: 12px;
    padding: 8px 10px;
    outline: none;
    transition: var(--transition);
    font-family: inherit;
  }}
  .form-input:focus {{ border-color: var(--accent); }}
  .form-select {{
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-size: 12px;
    padding: 8px 10px;
    outline: none;
    transition: var(--transition);
    font-family: inherit;
    cursor: pointer;
  }}
  .form-select:focus {{ border-color: var(--accent); }}
  .btn-mini {{
    background: var(--accent);
    border: none;
    border-radius: 6px;
    color: white;
    font-size: 12px;
    font-weight: 600;
    padding: 8px 14px;
    cursor: pointer;
    transition: var(--transition);
    white-space: nowrap;
  }}
  .btn-mini:hover {{ background: var(--accent-hover); }}

  /* Settings panel styles */
  .settings-form {{ display: flex; flex-direction: column; gap: 14px; }}
  .setting-row {{ display: flex; flex-direction: column; gap: 5px; }}
  .setting-label {{ font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: .05em; }}
  .setting-input {{
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid var(--border);
    border-radius: 6px;
    color: var(--text);
    font-size: 12px;
    padding: 8px 10px;
    width: 100%;
    outline: none;
    transition: var(--transition);
    font-family: inherit;
  }}
  .setting-input:focus {{ border-color: var(--accent); }}
  .toggle-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid var(--border);
  }}
  .toggle-row:last-child {{ border-bottom: none; }}
  .toggle-name {{ font-size: 13px; font-weight: 600; color: var(--text); }}
  .toggle-switch {{
    position: relative;
    width: 44px;
    height: 24px;
    cursor: pointer;
  }}
  .toggle-switch input {{ opacity: 0; width: 0; height: 0; }}
  .toggle-track {{
    position: absolute;
    inset: 0;
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid var(--border);
    border-radius: 24px;
    transition: var(--transition);
  }}
  .toggle-track::after {{
    content: '';
    position: absolute;
    top: 3px; left: 3px;
    width: 16px; height: 16px;
    background: var(--text-muted);
    border-radius: 50%;
    transition: var(--transition);
  }}
  .toggle-switch input:checked + .toggle-track {{ background: var(--accent); border-color: var(--accent); }}
  .toggle-switch input:checked + .toggle-track::after {{ transform: translateX(20px); background: #fff; }}
  .btn-save {{
    padding: 9px 16px;
    background: var(--accent);
    border: none;
    border-radius: 6px;
    color: #fff;
    font-size: 12px;
    font-weight: 700;
    cursor: pointer;
    transition: var(--transition);
    align-self: flex-end;
  }}
  .btn-save:hover {{ background: var(--accent-hover); transform: translateY(-1px); }}

  /* ── Right Column Panels ── */
  .right-col {{
    display: flex;
    flex-direction: column;
    gap: 28px;
    position: sticky;
    top: 88px;
    max-height: calc(100vh - 120px);
    overflow-y: auto;
    padding-right: 4px;
  }}

  /* Active queue card */
  .active-job-container {{
    display: flex;
    flex-direction: column;
    gap: 16px;
    background: rgba(99, 102, 241, 0.03);
    border: 1px dashed rgba(99, 102, 241, 0.2);
    border-radius: 8px;
    padding: 20px;
  }}
  .active-job-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .active-job-name {{
    font-size: 15px;
    font-weight: 700;
    color: var(--text);
  }}
  .priority-badge {{
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    padding: 3px 8px;
    border-radius: 12px;
    letter-spacing: 0.05em;
  }}
  .priority-badge.interactive {{ background: rgba(245, 158, 11, 0.15); color: var(--warning); border: 1px solid rgba(245, 158, 11, 0.3); }}
  .priority-badge.normal {{ background: rgba(99, 102, 241, 0.15); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.3); }}
  .priority-badge.maintenance {{ background: rgba(156, 163, 175, 0.15); color: var(--text-muted); border: 1px solid rgba(156, 163, 175, 0.3); }}

  .progress-wrapper {{
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}
  .progress-info {{
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: var(--text-muted);
  }}
  .progress-bar-bg {{
    width: 100%;
    height: 8px;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
    border: 1px solid var(--border);
  }}
  .progress-bar-fill {{
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, #6366f1, #a5b4fc);
    border-radius: 4px;
    transition: width 0.3s ease;
  }}
  .progress-status-message {{
    font-size: 12px;
    font-weight: 500;
    color: var(--text-muted);
  }}
  .progress-active-item {{
    font-size: 11px;
    font-family: monospace;
    color: var(--text-muted);
    background: rgba(0, 0, 0, 0.15);
    padding: 4px 8px;
    border-radius: 4px;
    margin-top: 4px;
    word-break: break-all;
  }}

  /* Waiting list queue */
  .waiting-queue-list {{
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  .waiting-item {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 12px;
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border);
    border-radius: 6px;
  }}
  .waiting-name {{
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
  }}

  /* Terminals */
  .terminal-box {{
    height: 280px;
    background: #06070a;
    border: 1px solid var(--border);
    border-radius: 6px;
    font-family: monospace;
    font-size: 11px;
    line-height: 1.6;
    padding: 12px;
    overflow-y: auto;
    white-space: pre-wrap;
    color: #94a3b8;
  }}
  .terminal-box span {{ display: block; }}
  .terminal-box .err {{ color: var(--danger); }}
  .terminal-box .warn {{ color: var(--warning); }}
  .terminal-box .ok {{ color: var(--success); }}
  .terminal-box .head {{ color: var(--accent); font-weight: 700; }}

  /* Activity feed */
  .activity-feed {{
    display: flex;
    flex-direction: column;
    gap: 12px;
    max-height: 350px;
    overflow-y: auto;
    padding-right: 4px;
  }}
  .activity-item {{
    display: flex;
    gap: 12px;
    align-items: start;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
  }}
  .activity-item:last-child {{
    border-bottom: none;
    padding-bottom: 0;
  }}
  .activity-time {{
    font-size: 11px;
    color: var(--text-muted);
    font-family: monospace;
    width: 60px;
    flex-shrink: 0;
    margin-top: 2px;
  }}
  .activity-content {{
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }}
  .activity-title {{
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
  }}
  .activity-detail {{
    font-size: 11px;
    color: var(--text-muted);
  }}
  .status-dot-mini {{
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
    margin-top: 5px;
  }}
  .status-dot-mini.completed {{ background: var(--success); }}
  .status-dot-mini.failed {{ background: var(--danger); }}
  .status-dot-mini.interrupted {{ background: var(--warning); }}

  /* Advanced settings panel details */
  .settings-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }}
  @media (max-width: 768px) {{
    .settings-grid {{ grid-template-columns: 1fr; }}
  }}
  .settings-footer {{
    margin-top: 14px;
    border-top: 1px solid var(--border);
    padding-top: 12px;
    display: flex;
    justify-content: flex-end;
  }}

  /* Toast Containers */
  #toast-container {{
    position: fixed;
    bottom: 24px;
    right: 24px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    z-index: 9999;
  }}
  .toast {{
    padding: 12px 20px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    border: 1px solid transparent;
    animation: toast-slide .2s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: var(--shadow);
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  @keyframes toast-slide {{
    from {{ transform: translateY(20px); opacity: 0; }}
    to {{ transform: translateY(0); opacity: 1; }}
  }}
  .toast.success {{ background: rgba(16, 185, 129, 0.15); border-color: rgba(16, 185, 129, 0.3); color: var(--success); }}
  .toast.error {{ background: rgba(239, 68, 68, 0.15); border-color: rgba(239, 68, 68, 0.3); color: var(--danger); }}
  .toast.info {{ background: var(--accent-alpha); border-color: var(--border-hover); color: #a5b4fc; }}
  
  .btn-cancel-task {{
    background: rgba(239, 68, 68, 0.15);
    color: var(--danger);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 600;
    cursor: pointer;
    transition: var(--transition);
  }}
  .btn-cancel-task:hover {{
    background: var(--danger);
    color: white;
  }}
</style>
</head>
<body>

<div class="topbar">
  <div class="topbar-logo">LoRA<span>Manager</span></div>
  <div class="topbar-sub">Control Center</div>
  <div class="topbar-sep"></div>
  <a class="nav-btn" href="/">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>
    🖼️ Browse Gallery
  </a>
</div>

<main>
  <div class="header-area">
    <div class="page-title">Library Manager <span>/ Workspace Control</span></div>
  </div>

  <!-- Stats Grid -->
  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-label">Total LoRAs</div>
      <div class="stat-value" id="stat-total">{stats['total_loras']}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Missing Previews</div>
      <div class="stat-value {'warn' if stats['missing_preview'] > 0 else 'ok'}" id="stat-previews">{stats['missing_preview']}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Missing Metadata</div>
      <div class="stat-value {'warn' if stats['missing_metadata'] > 0 else 'ok'}" id="stat-metadata">{stats['missing_metadata']}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Collections</div>
      <div class="stat-value" id="stat-collections">{stats['collections']}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Library Size</div>
      <div class="stat-value" style="font-size:22px" id="stat-size">{stats['size']}</div>
    </div>
  </div>

  <!-- Content Grid -->
  <div class="content-grid">

    <!-- Left Column: Operations and Settings -->
    <div class="left-col">

      <!-- Workflows Section -->
      <div class="panel">
        <div class="panel-header">Library Workflows</div>
        <div class="panel-body">
          <button class="action-btn" id="btn-sync" onclick="triggerTask('sync')">
            <span class="btn-icon">🔄</span>
            <span class="btn-label">
              Full Sync
              <span class="btn-sub">Organize + Download + Rebuild</span>
            </span>
          </button>
          <button class="action-btn" id="btn-refresh" onclick="triggerTask('quick-refresh')">
            <span class="btn-icon">⚡</span>
            <span class="btn-label">
              Quick Refresh
              <span class="btn-sub">Scan changes & update gallery</span>
            </span>
          </button>
          <button class="action-btn" id="btn-organize" onclick="triggerTask('organize')">
            <span class="btn-icon">🗂️</span>
            <span class="btn-label">
              Organize Only
              <span class="btn-sub">Deduplicate folders and package files</span>
            </span>
          </button>
          <button class="action-btn" id="btn-previews" onclick="triggerTask('download-previews')">
            <span class="btn-icon">🖼️</span>
            <span class="btn-label">
              Download Previews
              <span class="btn-sub">Pull missing preview cards from CivitAI</span>
            </span>
          </button>
          <button class="action-btn" id="btn-metadata" onclick="triggerTask('download-metadata')">
            <span class="btn-icon">📄</span>
            <span class="btn-label">
              Download Metadata
              <span class="btn-sub">Fetch missing CivitAI JSON sidecars</span>
            </span>
          </button>
          <button class="action-btn" id="btn-rebuild" onclick="triggerTask('rebuild-gallery')">
            <span class="btn-icon">🏗️</span>
            <span class="btn-label">
              Rebuild Gallery
              <span class="btn-sub">Force compile gallery.html structure</span>
            </span>
          </button>
        </div>
      </div>

      <!-- Manual Operations Panel -->
      <div class="panel">
        <div class="panel-header">Manual Operations</div>
        <div class="panel-body">
          <div class="section-group">
            <div class="section-title">Process Single Model</div>
            <div class="form-row">
              <input type="text" id="manual-file-path" class="form-input" placeholder="E:\\Path\\to\\model.safetensors">
              <button class="btn-mini" onclick="processSingleFile()">Run</button>
            </div>
          </div>
          <div class="section-group" style="margin-top: 6px;">
            <div class="section-title">Process Custom Folder</div>
            <div class="form-row">
              <input type="text" id="manual-folder-path" class="form-input" placeholder="Subfolder path (relative or absolute)">
              <button class="btn-mini" onclick="processFolder()">Run</button>
            </div>
          </div>
        </div>
      </div>

      <!-- Maintenance & Diagnostics Section -->
      <div class="panel">
        <div class="panel-header">Maintenance & Diagnostics</div>
        <div class="panel-body">
          <button class="action-btn" id="btn-migrate" onclick="triggerTask('migrate-metadata')">
            <span class="btn-icon">📦</span>
            <span class="btn-label">
              Migrate Metadata
              <span class="btn-sub">Import A1111/Forge custom fields to user_data</span>
            </span>
          </button>
          <button class="action-btn" id="btn-update-check" onclick="triggerTask('civitai-update-detector')">
            <span class="btn-icon">🔔</span>
            <span class="btn-label">
              Update Checker
              <span class="btn-sub">Scan CivitAI for newer versions of LoRAs</span>
            </span>
          </button>
          <button class="action-btn" id="btn-hashes" onclick="triggerTask('generate-hashes')">
            <span class="btn-icon">🔑</span>
            <span class="btn-label">
              Generate Missing Hashes
              <span class="btn-sub">Compute SHA256 sums for loose files</span>
            </span>
          </button>

          <div class="section-group" style="margin-top: 10px;">
            <div class="section-title">Duplicate Finder</div>
            <div class="form-row">
              <select id="duplicate-strategy" class="form-select" style="flex:1">
                <option value="scan-only">Scan Only (No deletion)</option>
                <option value="keep-oldest">Keep Oldest (Delete newer duplicates)</option>
                <option value="keep-newest">Keep Newest (Delete older duplicates)</option>
              </select>
              <button class="btn-mini" onclick="runDuplicateFinder()">Scan</button>
            </div>
          </div>
        </div>
      </div>

      <!-- Migration Panel -->
      <div class="panel">
        <div class="panel-header">Migration Wizard</div>
        <div class="panel-body">
          <button class="action-btn" onclick="triggerTask('migration-preview')">
            <span class="btn-icon">🔍</span>
            <span class="btn-label">
              Scan Library Hierarchy
              <span class="btn-sub">Read folders and preview collection layout</span>
            </span>
          </button>
          <button class="action-btn" onclick="triggerTask('import-existing')">
            <span class="btn-icon">💾</span>
            <span class="btn-label">
              Import Folders as Collections
              <span class="btn-sub">Auto-assign models to collections based on directories</span>
            </span>
          </button>
        </div>
      </div>

      <!-- Backup & Restore Panel -->
      <div class="panel">
        <div class="panel-header">Backup & Recovery</div>
        <div class="panel-body">
          <button class="action-btn" onclick="triggerTask('export-manifest')">
            <span class="btn-icon">📤</span>
            <span class="btn-label">
              Export Manifest JSON
              <span class="btn-sub">Save complete library backup under backup/</span>
            </span>
          </button>
          
          <div class="section-group" style="margin-top: 8px;">
            <div class="section-title">Restore Backup</div>
            <div class="form-row" style="flex-direction:column; gap:8px; align-items:stretch;">
              <select id="backup-select" class="form-select">
                <option value="">No backups found</option>
              </select>
              <button class="btn-mini" onclick="restoreSelectedBackup()" style="align-self:flex-end">Restore selected</button>
            </div>
          </div>
        </div>
      </div>

      <!-- Settings Panel -->
      <div class="panel">
        <div class="panel-header">Settings</div>
        <div class="panel-body">
          <div class="settings-form">
            <div class="setting-row">
              <div class="setting-label">LoRA Root Path</div>
              <input type="text" class="setting-input" id="lora-root-input" placeholder="E:\\SD\\models\\Lora">
            </div>
            <div class="setting-row">
              <div class="setting-label">CivitAI API Key</div>
              <input type="password" class="setting-input" id="api-key-input" placeholder="Paste your API key…">
            </div>
            
            <div class="settings-grid">
              <div class="setting-row">
                <div class="setting-label">Download Mode</div>
                <select id="setting-download-strategy" class="form-select">
                  <option value="skip">Skip Existing</option>
                  <option value="overwrite">Overwrite/Refresh</option>
                </select>
              </div>
              <div class="setting-row">
                <div class="setting-label">Default Sort</div>
                <select id="setting-default-sort" class="form-select">
                  <option value="name">Name</option>
                  <option value="size">Size</option>
                  <option value="rating">Rating</option>
                  <option value="modified">Date Modified</option>
                </select>
              </div>
            </div>

            <div class="setting-row">
              <div class="setting-label">Default Prompt Weight</div>
              <input type="number" step="0.1" class="setting-input" id="setting-default-weight" placeholder="1.0">
            </div>

            <div class="toggle-row">
              <span class="toggle-name">Dry Run Mode</span>
              <label class="toggle-switch">
                <input type="checkbox" id="toggle-dry-run">
                <div class="toggle-track"></div>
              </label>
            </div>
            <div class="toggle-row">
              <span class="toggle-name">Verbose Logging</span>
              <label class="toggle-switch">
                <input type="checkbox" id="toggle-verbose">
                <div class="toggle-track"></div>
              </label>
            </div>
            
            <div class="settings-footer">
              <button class="btn-save" onclick="saveSettings()">Save Configuration</button>
            </div>
          </div>
        </div>
      </div>

    </div>

    <!-- Right Column: Task queue, logs and activity feed -->
    <div class="right-col">

      <!-- Active queue panel -->
      <div class="panel">
        <div class="panel-header">Active Tasks</div>
        <div class="panel-body">
          <div id="active-task-container" style="display:none;" class="active-job-container">
            <div class="active-job-header">
              <span id="active-task-name" class="active-job-name">Task Name</span>
              <div class="active-job-header-right" style="display:flex; align-items:center; gap:8px;">
                <span id="active-task-priority" class="priority-badge normal">interactive</span>
                <button id="btn-cancel-active" class="btn-cancel-task" onclick="cancelActiveTask()">Cancel</button>
              </div>
            </div>
            <div class="progress-wrapper">
              <div class="progress-info">
                <span id="active-task-percentage-text">0%</span>
                <span id="active-task-items-text">0 / 0</span>
              </div>
              <div class="progress-bar-bg">
                <div id="active-task-bar" class="progress-bar-fill"></div>
              </div>
            </div>
            <div id="active-task-message" class="progress-status-message">Starting...</div>
            <div id="active-task-item" class="progress-active-item" style="display:none;">item</div>
          </div>
          
          <div id="no-active-tasks" style="text-align:center; padding: 24px; color:var(--text-muted); font-size:13px; font-weight: 500;">
            No tasks are currently running.
          </div>

          <div style="margin-top: 14px;">
            <div class="section-title">Queue Waiting List</div>
            <div id="waiting-queue-list" class="waiting-queue-list">
              <div style="text-align:center; padding: 8px; color:var(--text-muted); font-size:12px; font-style:italic;">Queue is empty</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Live Terminal logs -->
      <div class="panel">
        <div class="panel-header">
          Console Output
          <button class="btn-clear" onclick="clearLog()" style="margin-left:auto; padding:2px 8px; font-size:11px;">Clear</button>
        </div>
        <div class="panel-body" style="padding:0;">
          <div id="terminal" class="terminal-box">
            <span class="head">Ready. Trigger an operation to start logging.</span>
          </div>
        </div>
      </div>

      <!-- Recent activity feed -->
      <div class="panel">
        <div class="panel-header">Recent Activity Log</div>
        <div class="panel-body">
          <div id="activity-feed" class="activity-feed">
            <div style="text-align:center; padding: 24px; color:var(--text-muted); font-size:13px; font-style:italic;">No historical records loaded</div>
          </div>
        </div>
      </div>

    </div>

  </div>
</main>

<div id="toast-container"></div>

<script>
  let runningTaskId = null;
  let eventSource = null;
  let pollInterval = null;

  // ── Toast Messages ─────────────────────────────────────────────────────────
  function toast(msg, type='info') {{
    const container = document.getElementById('toast-container');
    const el = document.createElement('div');
    el.className = `toast ${{type}}`;
    
    let icon = 'ℹ️';
    if (type === 'success') icon = '✅';
    if (type === 'error') icon = '❌';

    el.innerHTML = `<span>${{icon}}</span><span>${{msg}}</span>`;
    container.appendChild(el);
    setTimeout(() => el.remove(), 4000);
  }}

  // ── Terminal Log Functions ──────────────────────────────────────────────────
  function clearLog() {{
    document.getElementById('terminal').innerHTML = '<span class="head">Terminal cleared.</span>';
  }}

  function appendLog(text) {{
    const term = document.getElementById('terminal');
    const line = document.createElement('span');
    
    // Classify line styling
    const t = text.toLowerCase();
    if (t.includes('error') || t.includes('failed') || t.includes('exception')) {{
      line.className = 'err';
    }} else if (t.includes('warning') || t.includes('missing') || t.includes('skipped')) {{
      line.className = 'warn';
    }} else if (t.includes('completed') || t.includes('success') || t.includes('saved')) {{
      line.className = 'ok';
    }} else if (t.startsWith('===') || t.startsWith('---') || t.startsWith('[')) {{
      line.className = 'head';
    }}
    
    line.textContent = text;
    term.appendChild(line);
    term.scrollTop = term.scrollHeight;
  }}

  // ── API Operations ──────────────────────────────────────────────────────────
  function triggerTask(action, params = {{}}) {{
    const query = new URLSearchParams(params).toString();
    fetch(`/api/run/${{action}}?${{query}}`, {{ method: 'POST' }})
      .then(r => r.json())
      .then(data => {{
        if (data.status === 'error' || !data.ok) {{
          toast(data.message || 'Error enqueuing task', 'error');
          appendLog(`✗ Error enqueuing: ${{data.message || 'unknown error'}}`);
        }} else {{
          toast(`Enqueued: ${{action}}`, 'info');
          appendLog(`▶ Enqueued job: ${{action}} (ID: ${{data.job_id}})`);
          startLogStream();
          refreshTasks();
        }}
      }})
      .catch(err => {{
        toast('Connection failed', 'error');
        appendLog(`✗ Error: ${{err}}`);
      }});
  }}

  function cancelActiveTask() {{
    if (!runningTaskId) return;
    cancelTask(runningTaskId);
  }}

  function cancelTask(jobId) {{
    fetch(`/api/tasks/${{jobId}}/cancel`, {{ method: 'POST' }})
      .then(r => r.json())
      .then(data => {{
        if (data.ok) {{
          toast('Task cancellation requested', 'info');
          appendLog(`🛑 Cancellation requested for job: ${{jobId}}`);
          refreshTasks();
        }} else {{
          toast('Could not cancel task', 'error');
        }}
      }})
      .catch(() => toast('Network error requesting cancellation', 'error'));
  }}

  // ── Actions Helpers ──────────────────────────────────────────────────────────
  function processSingleFile() {{
    const fileInput = document.getElementById('manual-file-path');
    const path = fileInput.value.trim();
    if (!path) {{
      toast('Please enter a file path', 'error');
      return;
    }}
    triggerTask('process-single-file', {{ file_path: path }});
    fileInput.value = '';
  }}

  function processFolder() {{
    const folderInput = document.getElementById('manual-folder-path');
    const path = folderInput.value.trim();
    if (!path) {{
      toast('Please enter a folder path', 'error');
      return;
    }}
    // Process folder enqueues a quick refresh or sync for that root,
    // let's run a sync enqueued task for safety
    triggerTask('sync'); 
    folderInput.value = '';
  }}

  function runDuplicateFinder() {{
    const strategy = document.getElementById('duplicate-strategy').value;
    triggerTask('duplicate-finder', {{ strategy: strategy }});
  }}

  function restoreSelectedBackup() {{
    const backupSelect = document.getElementById('backup-select');
    const val = backupSelect.value;
    if (!val) {{
      toast('Please select a backup first', 'error');
      return;
    }}
    triggerTask('import-manifest', {{ manifest_path: `backup/${{val}}` }});
  }}

  // ── Log Streaming (SSE) ─────────────────────────────────────────────────────
  function startLogStream() {{
    if (eventSource) return; // Already running
    
    eventSource = new EventSource('/api/log-stream');
    eventSource.onmessage = (e) => {{
      if (e.data === '__DONE__') {{
        stopLogStream();
      }} else {{
        appendLog(e.data);
      }}
    }};
    eventSource.onerror = () => {{
      stopLogStream();
    }};
  }}

  function stopLogStream() {{
    if (eventSource) {{
      eventSource.close();
      eventSource = null;
    }}
  }}

  // ── Poll & Update Job Queue ─────────────────────────────────────────────────
  function refreshTasks() {{
    // 1. Fetch active tasks
    fetch('/api/tasks')
      .then(r => r.json())
      .then(activeJobs => {{
        const activeContainer = document.getElementById('active-task-container');
        const noActive = document.getElementById('no-active-tasks');
        const waitingList = document.getElementById('waiting-queue-list');
        
        let runningJob = activeJobs.find(j => j.status === 'running');
        let queuedJobs = activeJobs.filter(j => j.status === 'pending');

        // Render running task
        if (runningJob) {{
          runningTaskId = runningJob.id;
          activeContainer.style.display = 'flex';
          noActive.style.display = 'none';

          document.getElementById('active-task-name').textContent = runningJob.name;
          
          const pBadge = document.getElementById('active-task-priority');
          pBadge.className = `priority-badge ${{runningJob.priority}}`;
          pBadge.textContent = runningJob.priority;

          // Progress values
          const percent = runningJob.progress_percent || 0;
          document.getElementById('active-task-percentage-text').textContent = `${{percent}}%`;
          document.getElementById('active-task-bar').style.width = `${{percent}}%`;

          const itemsText = document.getElementById('active-task-items-text');
          if (runningJob.progress_total > 0) {{
            itemsText.style.display = '';
            itemsText.textContent = `${{runningJob.progress_current}} / ${{runningJob.progress_total}}`;
          }} else {{
            itemsText.style.display = 'none';
          }}

          document.getElementById('active-task-message').textContent = runningJob.progress_message || 'Running...';
          
          const itemEl = document.getElementById('active-task-item');
          if (runningJob.progress_item) {{
            itemEl.style.display = 'block';
            itemEl.textContent = `Active item: ${{runningJob.progress_item}}`;
          }} else {{
            itemEl.style.display = 'none';
          }}
          
          // Ensure log streaming runs
          startLogStream();
        }} else {{
          runningTaskId = null;
          activeContainer.style.display = 'none';
          noActive.style.display = 'block';
        }}

        // Render waiting list queue
        if (queuedJobs.length > 0) {{
          waitingList.innerHTML = queuedJobs.map(j => `
            <div class="waiting-item">
              <span class="waiting-name">${{j.name}}</span>
              <div style="display:flex; align-items:center; gap:8px;">
                <span class="priority-badge ${{j.priority}}">${{j.priority}}</span>
                <button class="btn-cancel-task" onclick="cancelTask('${{j.id}}')">Remove</button>
              </div>
            </div>
          `).join('');
        }} else {{
          waitingList.innerHTML = '<div style="text-align:center; padding: 8px; color:var(--text-muted); font-size:12px; font-style:italic;">Queue is empty</div>';
        }}
      }})
      .catch(() => {{}});

    // 2. Fetch completed task history
    fetch('/api/task-history')
      .then(r => r.json())
      .then(history => {{
        const feed = document.getElementById('activity-feed');
        if (history.length > 0) {{
          feed.innerHTML = history.map(h => {{
            let timeStr = 'unknown';
            if (h.end_time) {{
              // Format YYYY-MM-DDTHH:MM:SS -> HH:MM:SS
              try {{
                const parts = h.end_time.split('T');
                timeStr = parts[1].slice(0, 8);
              }} catch(e) {{
                timeStr = h.end_time;
              }}
            }}
            
            let badgeClass = 'completed';
            let statusText = 'Completed';
            if (h.status === 'failed') {{
              badgeClass = 'failed';
              statusText = `Failed: ${{h.error || 'unknown error'}}`;
            }} else if (h.status === 'interrupted') {{
              badgeClass = 'interrupted';
              statusText = 'Interrupted';
            }}

            return `
              <div class="activity-item">
                <span class="status-dot-mini ${{badgeClass}}"></span>
                <span class="activity-time">${{timeStr}}</span>
                <div class="activity-content">
                  <span class="activity-title">${{h.name}}</span>
                  <span class="activity-detail">${{statusText}}</span>
                </div>
              </div>
            `;
          }}).join('');
        }} else {{
          feed.innerHTML = '<div style="text-align:center; padding: 24px; color:var(--text-muted); font-size:13px; font-style:italic;">No historical records loaded</div>';
        }}
      }})
      .catch(() => {{}});
  }}

  // ── Settings Functions ──────────────────────────────────────────────────────
  function loadSettings() {{
    fetch('/api/settings')
      .then(r => r.json())
      .then(s => {{
        const keyInput = document.getElementById('api-key-input');
        if (s.api_key) {{
          keyInput.placeholder = s.api_key.length > 8
            ? s.api_key.slice(0,4) + '••••' + s.api_key.slice(-4)
            : '(set)';
        }} else {{
          keyInput.placeholder = 'Paste your API key…';
        }}
        document.getElementById('lora-root-input').value = s.lora_root || '';
        document.getElementById('setting-download-strategy').value = s.download_strategy || 'skip';
        document.getElementById('setting-default-sort').value = s.default_sort || 'name';
        document.getElementById('setting-default-weight').value = s.default_weight || 1.0;
        document.getElementById('toggle-dry-run').checked  = !!s.dry_run;
        document.getElementById('toggle-verbose').checked  = !!s.verbose;
      }})
      .catch(() => {{}});
  }}

  function saveSettings() {{
    const apiKey = document.getElementById('api-key-input').value.trim();
    const loraRoot = document.getElementById('lora-root-input').value.trim();
    const downloadStrategy = document.getElementById('setting-download-strategy').value;
    const defaultSort = document.getElementById('setting-default-sort').value;
    const defaultWeight = parseFloat(document.getElementById('setting-default-weight').value) || 1.0;
    const dryRun = document.getElementById('toggle-dry-run').checked;
    const verbose = document.getElementById('toggle-verbose').checked;

    const payload = {{
      dry_run: dryRun,
      verbose: verbose,
      download_strategy: downloadStrategy,
      default_sort: defaultSort,
      default_weight: defaultWeight
    }};
    if (apiKey) payload.api_key = apiKey;
    if (loraRoot) payload.lora_root = loraRoot;

    fetch('/api/settings', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify(payload)
    }})
    .then(r => r.json())
    .then(s => {{
      if (s.ok) {{
        toast('Configuration saved successfully!', 'success');
        document.getElementById('api-key-input').value = '';
        loadSettings();
        // Refresh statistics in header
        setTimeout(refreshStats, 800);
      }} else {{
        toast('Failed to save configuration', 'error');
      }}
    }})
    .catch(() => toast('Network error saving configuration', 'error'));
  }}

  // ── Load Backups Dropdown ───────────────────────────────────────────────────
  function loadBackups() {{
    fetch('/api/backups')
      .then(r => r.json())
      .then(backups => {{
        const select = document.getElementById('backup-select');
        if (backups && backups.length > 0) {{
          select.innerHTML = backups.map(b => `<option value="${{b}}">${{b}}</option>`).join('');
        }} else {{
          select.innerHTML = '<option value="">No backups found</option>';
        }}
      }})
      .catch(() => {{}});
  }}

  // ── Refresh Statistics Card Values ─────────────────────────────────────────
  function refreshStats() {{
    fetch('/api/stats')
      .then(r => r.json())
      .then(s => {{
        const setVal = (id, val) => {{
          const el = document.getElementById(id);
          if (el) el.textContent = val;
        }};
        setVal('stat-total', s.total_loras);
        
        const previewEl = document.getElementById('stat-previews');
        if (previewEl) {{
          previewEl.textContent = s.missing_preview;
          previewEl.className = `stat-value ${{s.missing_preview > 0 ? 'warn' : 'ok'}}`;
        }}
        
        const metaEl = document.getElementById('stat-metadata');
        if (metaEl) {{
          metaEl.textContent = s.missing_metadata;
          metaEl.className = `stat-value ${{s.missing_metadata > 0 ? 'warn' : 'ok'}}`;
        }}
        
        setVal('stat-collections', s.collections);
        setVal('stat-size', s.size);
      }})
      .catch(() => {{}});
  }}

  // ── Initialization ──────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {{
    loadSettings();
    loadBackups();
    refreshTasks();
    
    // Poll queue status every 1.5 seconds
    pollInterval = setInterval(refreshTasks, 1500);
  }});
</script>
</body>
</html>"""
