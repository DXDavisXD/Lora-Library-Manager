"""Generates the self-contained HTML for the /manage web dashboard."""

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
    """Return the full HTML string for the management dashboard."""
    stats = get_library_stats(lora_root)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LoRA Library Manager — Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #0d0f1a;
    --panel: #13172b;
    --panel2: #1a1f38;
    --border: rgba(99,102,241,.18);
    --border-hover: rgba(99,102,241,.45);
    --accent: #6366f1;
    --accent-alpha: rgba(99,102,241,.12);
    --text: #e2e8f0;
    --text-muted: #94a3b8;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --shadow: 0 4px 24px rgba(0,0,0,.5);
    --radius: 12px;
    --transition: all .18s ease;
  }}

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }}

  /* ── Top bar ── */
  .topbar {{
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 14px 32px;
    background: var(--panel);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 100;
  }}
  .topbar-logo {{
    font-size: 17px;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -.3px;
  }}
  .topbar-logo span {{ color: var(--accent); }}
  .topbar-sep {{ flex: 1; }}
  .btn-gallery {{
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 7px 16px;
    border: 1px solid var(--border);
    border-radius: 8px;
    background: transparent;
    color: var(--text-muted);
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: var(--transition);
    text-decoration: none;
  }}
  .btn-gallery:hover {{ border-color: var(--border-hover); color: var(--text); background: var(--accent-alpha); }}

  /* ── Main layout ── */
  main {{
    max-width: 1300px;
    width: 100%;
    margin: 0 auto;
    padding: 32px;
    display: flex;
    flex-direction: column;
    gap: 28px;
    flex: 1;
  }}

  /* ── Page title ── */
  .page-title {{
    font-size: 22px;
    font-weight: 700;
    color: var(--text);
  }}
  .page-title span {{ color: var(--accent); }}

  /* ── Stats row ── */
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
    gap: 16px;
  }}
  .stat-card {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px 20px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    transition: var(--transition);
  }}
  .stat-card:hover {{ border-color: var(--border-hover); }}
  .stat-label {{
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .07em;
    color: var(--text-muted);
  }}
  .stat-value {{
    font-size: 28px;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
  }}
  .stat-value.warn {{ color: var(--warning); }}
  .stat-value.ok   {{ color: var(--success); }}

  /* ── Two-column lower layout ── */
  .content-grid {{
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 24px;
    align-items: start;
  }}

  /* ── Section panels ── */
  .panel {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
  }}
  .panel-header {{
    padding: 14px 20px;
    border-bottom: 1px solid var(--border);
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: var(--text-muted);
  }}
  .panel-body {{ padding: 16px; display: flex; flex-direction: column; gap: 10px; }}

  /* ── Action buttons ── */
  .action-btn {{
    width: 100%;
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 14px;
    border: 1px solid var(--border);
    border-radius: 9px;
    background: var(--accent-alpha);
    color: var(--text);
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: var(--transition);
    text-align: left;
  }}
  .action-btn:hover:not(:disabled) {{
    border-color: var(--accent);
    background: rgba(99,102,241,.22);
    transform: translateX(2px);
  }}
  .action-btn:disabled {{ opacity: .45; cursor: not-allowed; }}
  .action-btn .btn-icon {{ font-size: 18px; flex-shrink: 0; }}
  .action-btn .btn-label {{ flex: 1; }}
  .action-btn .btn-sub {{ font-size: 11px; font-weight: 400; color: var(--text-muted); display: block; margin-top: 2px; }}

  .action-btn.running {{
    border-color: var(--accent);
    animation: pulse-border 1.5s ease infinite;
  }}
  @keyframes pulse-border {{
    0%,100% {{ border-color: var(--accent); }}
    50%      {{ border-color: rgba(99,102,241,.3); }}
  }}

  /* Settings form */
  .settings-form {{ display: flex; flex-direction: column; gap: 14px; }}
  .setting-row {{ display: flex; flex-direction: column; gap: 5px; }}
  .setting-label {{ font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: .05em; }}
  .setting-input {{
    background: var(--panel2);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text);
    font-size: 13px;
    padding: 9px 12px;
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
  .toggle-name {{ font-size: 13px; font-weight: 600; }}
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
    background: var(--panel2);
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
    border-radius: 8px;
    color: #fff;
    font-size: 13px;
    font-weight: 700;
    cursor: pointer;
    transition: var(--transition);
    align-self: flex-end;
  }}
  .btn-save:hover {{ opacity: .85; transform: translateY(-1px); }}

  /* ── Log panel ── */
  .log-panel {{ display: flex; flex-direction: column; height: 560px; }}
  .log-panel .panel-header {{ display: flex; align-items: center; gap: 10px; }}
  .log-status {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    background: var(--panel2);
    color: var(--text-muted);
    border: 1px solid var(--border);
    margin-left: auto;
  }}
  .log-status.running {{ color: var(--accent); border-color: var(--accent); background: var(--accent-alpha); }}
  .log-status .dot {{
    width: 7px; height: 7px;
    border-radius: 50%;
    background: currentColor;
  }}
  .log-status.running .dot {{ animation: blink .8s ease infinite alternate; }}
  @keyframes blink {{ from {{ opacity:1 }} to {{ opacity:.2 }} }}

  .log-output {{
    flex: 1;
    overflow-y: auto;
    padding: 14px 16px;
    font-family: 'SFMono-Regular', Consolas, monospace;
    font-size: 12px;
    line-height: 1.7;
    color: var(--text-muted);
    background: #0a0c16;
    white-space: pre-wrap;
    word-break: break-word;
  }}
  .log-output .log-line {{ display: block; }}
  .log-output .log-line.info  {{ color: #94a3b8; }}
  .log-output .log-line.ok    {{ color: var(--success); }}
  .log-output .log-line.warn  {{ color: var(--warning); }}
  .log-output .log-line.err   {{ color: var(--danger); }}
  .log-output .log-line.head  {{ color: var(--accent); font-weight: 700; }}

  .log-footer {{
    padding: 10px 16px;
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .btn-clear {{
    padding: 6px 14px;
    background: transparent;
    border: 1px solid var(--border);
    border-radius: 7px;
    color: var(--text-muted);
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: var(--transition);
  }}
  .btn-clear:hover {{ border-color: var(--danger); color: var(--danger); }}
  .log-tip {{ font-size: 12px; color: var(--text-muted); flex: 1; }}

  /* Toasts */
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
    padding: 12px 18px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
    border: 1px solid transparent;
    animation: slide-in .25s ease;
    max-width: 340px;
  }}
  @keyframes slide-in {{ from {{ opacity:0; transform:translateY(12px) }} to {{ opacity:1; transform:translateY(0) }} }}
  .toast.success {{ background: rgba(16,185,129,.15); border-color: rgba(16,185,129,.3); color: var(--success); }}
  .toast.error   {{ background: rgba(239,68,68,.15);  border-color: rgba(239,68,68,.3);  color: var(--danger); }}
  .toast.info    {{ background: var(--accent-alpha);   border-color: var(--border-hover); color: #a5b4fc; }}
</style>
</head>
<body>

<div class="topbar">
  <div class="topbar-logo">LoRA<span>Manager</span></div>
  <span style="font-size:13px; color:var(--text-muted); font-weight:500;">Management Dashboard</span>
  <div class="topbar-sep"></div>
  <a class="btn-gallery" href="/gallery.html">
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>
    Open Gallery
  </a>
</div>

<main>
  <div class="page-title">Dashboard <span>/ Library Overview</span></div>

  <!-- Stats -->
  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-label">Total LoRAs</div>
      <div class="stat-value">{stats['total_loras']}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Missing Previews</div>
      <div class="stat-value {'warn' if stats['missing_preview'] > 0 else 'ok'}">{stats['missing_preview']}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Missing Metadata</div>
      <div class="stat-value {'warn' if stats['missing_metadata'] > 0 else 'ok'}">{stats['missing_metadata']}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Collections</div>
      <div class="stat-value">{stats['collections']}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Library Size</div>
      <div class="stat-value" style="font-size:22px">{stats['size']}</div>
    </div>
  </div>

  <!-- Content grid -->
  <div class="content-grid">

    <!-- Left: Actions + Settings -->
    <div style="display:flex;flex-direction:column;gap:18px;">

      <!-- Quick Actions -->
      <div class="panel">
        <div class="panel-header">Quick Actions</div>
        <div class="panel-body">
          <button class="action-btn" id="btn-sync" onclick="runTask('sync')">
            <span class="btn-icon">🔄</span>
            <span class="btn-label">
              Full Sync
              <span class="btn-sub">Organize + Download + Gallery</span>
            </span>
          </button>
          <button class="action-btn" id="btn-refresh" onclick="runTask('quick-refresh')">
            <span class="btn-icon">⚡</span>
            <span class="btn-label">
              Quick Refresh
              <span class="btn-sub">Scan changes & rebuild gallery</span>
            </span>
          </button>
          <button class="action-btn" id="btn-organize" onclick="runTask('organize')">
            <span class="btn-icon">🗂️</span>
            <span class="btn-label">
              Organize Only
              <span class="btn-sub">Sort files into folders</span>
            </span>
          </button>
          <button class="action-btn" id="btn-previews" onclick="runTask('download-previews')">
            <span class="btn-icon">🖼️</span>
            <span class="btn-label">
              Download Previews
              <span class="btn-sub">Fetch missing preview images</span>
            </span>
          </button>
          <button class="action-btn" id="btn-metadata" onclick="runTask('download-metadata')">
            <span class="btn-icon">📄</span>
            <span class="btn-label">
              Download Metadata
              <span class="btn-sub">Fetch missing .json sidecars</span>
            </span>
          </button>
          <button class="action-btn" id="btn-rebuild" onclick="runTask('rebuild-gallery')">
            <span class="btn-icon">🏗️</span>
            <span class="btn-label">
              Rebuild Gallery
              <span class="btn-sub">Force regenerate gallery.html</span>
            </span>
          </button>
          <button class="action-btn" id="btn-migrate" onclick="runTask('migrate-metadata')">
            <span class="btn-icon">📦</span>
            <span class="btn-label">
              Migrate Metadata
              <span class="btn-sub">Import WebUI fields to user_data.json</span>
            </span>
          </button>
        </div>
      </div>

      <!-- Settings -->
      <div class="panel">
        <div class="panel-header">Settings</div>
        <div class="panel-body">
          <div class="settings-form" id="settings-form">
            <div class="setting-row">
              <div class="setting-label">CivitAI API Key</div>
              <input type="password" class="setting-input" id="api-key-input" placeholder="Paste your API key…">
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
            <button class="btn-save" onclick="saveSettings()">Save Settings</button>
          </div>
        </div>
      </div>

    </div>

    <!-- Right: Live log -->
    <div class="panel log-panel">
      <div class="panel-header">
        Live Log
        <span class="log-status" id="log-status">
          <span class="dot"></span>
          <span id="log-status-text">Idle</span>
        </span>
      </div>
      <div class="log-output" id="log-output">
        <span class="log-line info">Ready. Select an action on the left to begin.</span>
      </div>
      <div class="log-footer">
        <span class="log-tip" id="log-tip">No task running.</span>
        <button class="btn-clear" onclick="clearLog()">Clear Log</button>
      </div>
    </div>

  </div>
</main>

<div id="toast-container"></div>

<script>
  // ── State ──────────────────────────────────────────────────────────────────
  let taskRunning = false;
  let eventSource = null;
  const ALL_BTNS = ['btn-sync','btn-refresh','btn-organize','btn-previews','btn-metadata','btn-rebuild','btn-migrate'];

  // ── Helpers ────────────────────────────────────────────────────────────────
  function toast(msg, type='info') {{
    const el = document.createElement('div');
    el.className = `toast ${{type}}`;
    el.textContent = msg;
    document.getElementById('toast-container').appendChild(el);
    setTimeout(() => el.remove(), 3500);
  }}

  function clearLog() {{
    document.getElementById('log-output').innerHTML = '';
  }}

  function appendLog(text, cls='info') {{
    const out = document.getElementById('log-output');
    const line = document.createElement('span');
    line.className = `log-line ${{cls}}`;
    line.textContent = text;
    out.appendChild(line);
    out.scrollTop = out.scrollHeight;
  }}

  function classifyLine(text) {{
    const t = text.toLowerCase();
    if (t.includes('error') || t.includes('failed') || t.includes('exception')) return 'err';
    if (t.includes('warn') || t.includes('missing') || t.includes('skip')) return 'warn';
    if (t.includes('done') || t.includes('success') || t.includes('saved') || t.includes('complete')) return 'ok';
    if (t.startsWith('===') || t.startsWith('---') || t.startsWith('[')) return 'head';
    return 'info';
  }}

  function setRunning(running, btnId) {{
    taskRunning = running;
    const statusEl = document.getElementById('log-status');
    const statusText = document.getElementById('log-status-text');
    const tip = document.getElementById('log-tip');

    ALL_BTNS.forEach(id => {{
      const b = document.getElementById(id);
      if (b) b.disabled = running;
      if (b) b.classList.toggle('running', running && id === btnId);
    }});

    if (running) {{
      statusEl.className = 'log-status running';
      statusText.textContent = 'Running…';
      tip.textContent = 'Task in progress — please wait.';
    }} else {{
      statusEl.className = 'log-status';
      statusText.textContent = 'Idle';
      tip.textContent = 'No task running.';
    }}
  }}

  // ── Run a task via SSE ─────────────────────────────────────────────────────
  function runTask(action) {{
    if (taskRunning) {{
      toast('A task is already running. Please wait.', 'error');
      return;
    }}
    const btnId = {{
      'sync': 'btn-sync',
      'quick-refresh': 'btn-refresh',
      'organize': 'btn-organize',
      'download-previews': 'btn-previews',
      'download-metadata': 'btn-metadata',
      'rebuild-gallery': 'btn-rebuild',
      'migrate-metadata': 'btn-migrate',
    }}[action] || '';

    clearLog();
    appendLog(`▶ Starting: ${{action}}`, 'head');
    setRunning(true, btnId);

    // Kick off the task then stream logs
    fetch(`/api/run/${{action}}`, {{ method: 'POST' }})
      .then(r => r.json())
      .then(data => {{
        if (!data.ok) {{
          appendLog(`✗ Failed to start task: ${{data.message || 'unknown error'}}`, 'err');
          setRunning(false, '');
          toast('Failed to start task', 'error');
          return;
        }}
        // Open SSE stream
        if (eventSource) eventSource.close();
        eventSource = new EventSource('/api/log-stream');
        eventSource.onmessage = (e) => {{
          if (e.data === '__DONE__') {{
            eventSource.close();
            eventSource = null;
            setRunning(false, '');
            appendLog('✓ Task complete.', 'ok');
            toast('Task completed!', 'success');
            // Refresh stat cards after a moment
            setTimeout(refreshStats, 1200);
          }} else {{
            appendLog(e.data, classifyLine(e.data));
          }}
        }};
        eventSource.onerror = () => {{
          eventSource.close();
          eventSource = null;
          setRunning(false, '');
          appendLog('Connection to log stream lost.', 'warn');
        }};
      }})
      .catch(err => {{
        appendLog(`✗ Network error: ${{err}}`, 'err');
        setRunning(false, '');
        toast('Network error', 'error');
      }});
  }}

  // ── Refresh stat cards without full reload ─────────────────────────────────
  function refreshStats() {{
    fetch('/api/stats')
      .then(r => r.json())
      .then(s => {{
        const el = (id) => document.querySelector(`[data-stat="${{id}}"]`);
        const set = (id, val, warnIfPos) => {{
          const e = el(id);
          if (!e) return;
          e.textContent = val;
          if (warnIfPos !== undefined) {{
            e.className = `stat-value ${{warnIfPos && val > 0 ? 'warn' : 'ok'}}`;
          }}
        }};
        document.getElementById('stat-total').textContent = s.total_loras;
        document.getElementById('stat-previews').textContent = s.missing_preview;
        document.getElementById('stat-previews').className = `stat-value ${{s.missing_preview > 0 ? 'warn' : 'ok'}}`;
        document.getElementById('stat-metadata').textContent = s.missing_metadata;
        document.getElementById('stat-metadata').className = `stat-value ${{s.missing_metadata > 0 ? 'warn' : 'ok'}}`;
        document.getElementById('stat-collections').textContent = s.collections;
        document.getElementById('stat-size').textContent = s.size;
      }})
      .catch(() => {{}});
  }}

  // ── Settings ───────────────────────────────────────────────────────────────
  function loadSettings() {{
    fetch('/api/settings')
      .then(r => r.json())
      .then(s => {{
        const keyInput = document.getElementById('api-key-input');
        if (s.api_key) keyInput.placeholder = s.api_key.length > 8
          ? s.api_key.slice(0,4) + '••••' + s.api_key.slice(-4)
          : '(set)';
        document.getElementById('toggle-dry-run').checked  = !!s.dry_run;
        document.getElementById('toggle-verbose').checked  = !!s.verbose;
      }})
      .catch(() => {{}});
  }}

  function saveSettings() {{
    const apiKey = document.getElementById('api-key-input').value.trim();
    const dryRun = document.getElementById('toggle-dry-run').checked;
    const verbose = document.getElementById('toggle-verbose').checked;

    fetch('/api/settings', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ api_key: apiKey, dry_run: dryRun, verbose: verbose }})
    }})
    .then(r => r.json())
    .then(s => {{
      if (s.ok) {{
        toast('Settings saved!', 'success');
        document.getElementById('api-key-input').value = '';
        loadSettings();
      }} else {{
        toast('Error saving settings', 'error');
      }}
    }})
    .catch(() => toast('Network error saving settings', 'error'));
  }}

  // ── Init ───────────────────────────────────────────────────────────────────
  // Patch stat element IDs so refreshStats() can target them
  document.addEventListener('DOMContentLoaded', () => {{
    const statCards = document.querySelectorAll('.stat-card');
    const ids = ['stat-total','stat-previews','stat-metadata','stat-collections','stat-size'];
    statCards.forEach((card, i) => {{
      const val = card.querySelector('.stat-value');
      if (val && ids[i]) val.id = ids[i];
    }});
    loadSettings();
  }});
</script>
</body>
</html>"""
