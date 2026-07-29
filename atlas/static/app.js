/* Lora Atlas — client
 *
 * Two ideas carry the whole file:
 *
 * 1. Virtualised grid. Cards are a fixed size, so row/column geometry is pure
 *    arithmetic and only the visible window (plus a small overscan) is ever in
 *    the DOM. 1500 models cost the same as 50.
 * 2. No polling. One EventSource carries jobs, stats and logs. The UI is a
 *    function of pushed state.
 *
 * Rows of model data are fetched a page at a time and cached by absolute index,
 * so scrolling back never refetches.
 */

const CARD_W = 224, CARD_H = 396, GAP = 16, OVERSCAN_ROWS = 3;

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
};

/* ------------------------------------------------------------------ state */

const EMPTY_FILTERS = {
  q: '', sort: 'name', base_model: '', creator: '', collection: '', category: '', subcategory: '',
  uncategorised: false,
  favorite: false, rated: false, has_notes: false, has_triggers: false,
  meta_state: '', missing_preview: false, nsfw: '',
};

const state = {
  filters: { ...EMPTY_FILTERS },
  pageSize: 120,
  total: 0,                // rows matching the current filters
  libraryTotal: 0,         // rows in the library, for the "N of M" brand count
  items: new Map(),        // absolute index -> model
  pending: new Set(),      // page numbers in flight
  columns: 1,
  offsetX: 0,              // horizontal centring offset for the grid
  rendered: new Map(),     // absolute index -> element
  bootstrap: null,
  currentJob: null,
  followConsole: true,
  view: 'gallery',
  categories: [],          // category tree for the cascade
  uncategorised: 0,        // models no collection has claimed
};

/* -------------------------------------------------------------- transport */

async function apiCall(path, options = {}) {
  const response = await fetch(path, {
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  let payload = null;
  try { payload = await response.json(); } catch { /* empty body */ }
  if (!response.ok) {
    const message = (payload && payload.error) || `HTTP ${response.status}`;
    throw new Error(message);
  }
  return payload;
}

const apiGet = (path) => apiCall(path);
const apiPost = (path, body) =>
  apiCall(path, { method: 'POST', body: JSON.stringify(body || {}) });

/* ------------------------------------------------------------------ toast */

function toast(message, kind = 'info', sticky = false) {
  const node = el('div', `toast ${kind}`);
  node.append(el('div', 'grow', message));
  const close = el('button', null, '✕');
  close.setAttribute('aria-label', 'Dismiss');
  close.onclick = () => node.remove();
  node.append(close);
  $('toasts').append(node);
  // Errors persist until dismissed; a failure that vanishes in four seconds is
  // a failure the user never read.
  if (!sticky && kind !== 'error') setTimeout(() => node.remove(), 4200);
}

/* ------------------------------------------------------------- formatting */

const fmtBytes = (n) => {
  if (!n) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.min(units.length - 1, Math.floor(Math.log(n) / Math.log(1024)));
  return `${(n / 1024 ** i).toFixed(i ? 1 : 0)} ${units[i]}`;
};

const fmtDuration = (seconds) => {
  if (seconds == null || !isFinite(seconds)) return '';
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
};

const fmtTime = (ts) => ts ? new Date(ts * 1000).toLocaleTimeString() : '';

/* ------------------------------------------------------------ data paging */

function queryString(page) {
  const params = new URLSearchParams();
  const f = state.filters;
  if (f.q) params.set('q', f.q);
  if (f.sort) params.set('sort', f.sort);
  if (f.base_model) params.set('base_model', f.base_model);
  if (f.creator) params.set('creator', f.creator);
  if (f.collection) params.set('collection', f.collection);
  if (f.uncategorised) {
    params.set('uncategorised', '1');
  } else {
    if (f.category) params.set('category', f.category);
    if (f.subcategory) params.set('subcategory', f.subcategory);
  }
  if (f.favorite) params.set('favorite', '1');
  if (f.rated) params.set('rated', '1');
  if (f.has_notes) params.set('has_notes', '1');
  if (f.has_triggers) params.set('has_triggers', '1');
  if (f.meta_state) params.set('meta_state', f.meta_state);
  if (f.missing_preview) params.set('missing_preview', '1');
  if (f.nsfw) params.set('nsfw', f.nsfw);
  params.set('page', String(page));
  params.set('page_size', String(state.pageSize));
  return params.toString();
}

async function loadPage(page) {
  if (state.pending.has(page)) return;
  state.pending.add(page);
  try {
    const data = await apiGet(`/api/models?${queryString(page)}`);
    state.total = data.total;
    const base = (data.page - 1) * data.page_size;
    data.items.forEach((item, offset) => state.items.set(base + offset, item));
    updateResultCount();
    layout();
  } catch (err) {
    toast(`Could not load models: ${err.message}`, 'error');
  } finally {
    state.pending.delete(page);
  }
}

function resetResults() {
  state.items.clear();
  state.pending.clear();
  state.rendered.forEach((node) => node.remove());
  state.rendered.clear();
  state.total = 0;
  $('scroller').scrollTop = 0;
  loadPage(1);
}

/* ------------------------------------------------------------ virtual grid */

function measureUsable() {
  const scroller = $('scroller');
  const styles = getComputedStyle(scroller);
  return scroller.clientWidth
    - parseFloat(styles.paddingLeft) - parseFloat(styles.paddingRight);
}

function layout() {
  const usable = measureUsable();
  // The gallery may be hidden (Manage view), or not yet laid out on first
  // paint — either way it measures 0 wide and a column count from that would
  // be wrong. Skip, and let the ResizeObserver in observeGridWidth() re-run
  // this the moment the element actually has a width.
  if (usable <= 0) { state.layoutDeferred = true; return; }
  state.layoutDeferred = false;
  state.columns = Math.max(1, Math.floor((usable + GAP) / (CARD_W + GAP)));
  // Fixed-width cards rarely tile a container exactly, which leaves a ragged
  // gutter on the right that reads as "empty grid". Centre the columns so any
  // leftover splits evenly to both sides instead.
  const contentWidth = state.columns * CARD_W + (state.columns - 1) * GAP;
  state.offsetX = Math.max(0, Math.round((usable - contentWidth) / 2));
  const rows = Math.ceil(state.total / state.columns);
  $('spacer').style.height = `${Math.max(0, rows * (CARD_H + GAP) - GAP)}px`;
  $('empty').hidden = state.total !== 0;
  render();
}

/** Re-lay-out whenever the grid's width actually changes.
 *
 * This is the single mechanism behind first paint, the sidebar collapsing, and
 * window resizing — all of them are "the scroller changed width". Relying on a
 * one-shot layout at startup meant a load that raced the stylesheet left the
 * gallery permanently empty.
 */
function observeGridWidth() {
  const scroller = $('scroller');
  if (!scroller || typeof ResizeObserver === 'undefined') return;
  let lastWidth = -1;
  const observer = new ResizeObserver(() => {
    const width = scroller.clientWidth;
    if (width <= 0) return;
    if (width === lastWidth && !state.layoutDeferred) return;
    lastWidth = width;
    // The column count changes with width, so every cached card position is
    // stale — drop them and rebuild.
    state.rendered.forEach((node) => node.remove());
    state.rendered.clear();
    layout();
  });
  observer.observe(scroller);
}

function render() {
  const scroller = $('scroller');
  const grid = $('grid');
  const rowHeight = CARD_H + GAP;
  const firstRow = Math.max(0, Math.floor(scroller.scrollTop / rowHeight) - OVERSCAN_ROWS);
  const visibleRows = Math.ceil(scroller.clientHeight / rowHeight) + OVERSCAN_ROWS * 2;

  const start = firstRow * state.columns;
  const end = Math.min(state.total, (firstRow + visibleRows) * state.columns);

  // Pull in any page covering the window that we do not already hold.
  for (let index = start; index < end; index += state.pageSize) {
    if (!state.items.has(index)) loadPage(Math.floor(index / state.pageSize) + 1);
  }
  if (end > start && !state.items.has(end - 1)) {
    loadPage(Math.floor((end - 1) / state.pageSize) + 1);
  }

  const needed = new Set();
  for (let index = start; index < end; index++) {
    needed.add(index);
    if (state.rendered.has(index)) continue;
    const item = state.items.get(index);
    if (!item) continue;
    const card = buildCard(item, index);
    grid.append(card);
    state.rendered.set(index, card);
  }

  // Recycle anything that scrolled out of the window.
  state.rendered.forEach((node, index) => {
    if (!needed.has(index)) {
      node.remove();
      state.rendered.delete(index);
    }
  });
}

let renderScheduled = false;

/** Coalesce render requests to one per frame.
 *
 * requestAnimationFrame is the right primitive while the tab is visible, but it
 * is not guaranteed to run when it is not — so pick the scheduler that will
 * actually fire rather than assuming rAF always does.
 */
function scheduleRender() {
  if (renderScheduled) return;
  renderScheduled = true;
  const run = () => { renderScheduled = false; render(); };
  if (document.visibilityState === 'hidden') setTimeout(run, 16);
  else requestAnimationFrame(run);
}

function positionOf(index) {
  const row = Math.floor(index / state.columns);
  const col = index % state.columns;
  return { top: row * (CARD_H + GAP), left: (state.offsetX || 0) + col * (CARD_W + GAP) };
}

function buildCard(item, index) {
  const { top, left } = positionOf(index);
  const card = el('article', 'card');
  card.style.transform = `translate(${left}px, ${top}px)`;
  card.tabIndex = 0;
  card.setAttribute('role', 'button');
  card.setAttribute('aria-label', `${item.name}. Open details.`);

  const thumb = el('div', 'card-thumb');
  if (item.preview) {
    const img = el('img');
    img.alt = '';
    img.loading = 'lazy';
    img.decoding = 'async';
    img.src = `/media/${item.preview.split('/').map(encodeURIComponent).join('/')}`;
    img.addEventListener('load', () => img.classList.add('loaded'), { once: true });
    img.addEventListener('error', () => {
      img.remove();
      thumb.append(el('div', 'placeholder', 'unreadable'));
    }, { once: true });
    thumb.append(img);
  } else {
    thumb.append(el('div', 'placeholder', 'no preview'));
  }

  // Top-left: provider + base model. Kept to two so the corner stays readable.
  const topBadges = el('div', 'card-badges card-badges-top');
  if (item.provider) topBadges.append(el('span', `badge provider ${item.provider}`, item.provider));
  if (item.base_model) topBadges.append(el('span', 'badge', item.base_model));
  if (item.nsfw) topBadges.append(el('span', 'badge nsfw', 'NSFW'));
  thumb.append(topBadges);

  // Bottom-left: at-a-glance personal state — the indicators the flat rating
  // star made too easy to miss. Only shown when the model actually has them.
  const marks = el('div', 'card-marks');
  if (item.rating) {
    const r = el('span', 'mark mark-rating');
    r.title = `Rated ${item.rating}/5`;
    r.append(starSvg(), document.createTextNode(String(item.rating)));
    marks.append(r);
  }
  if ((item.user_tags || []).length) {
    const t = el('span', 'mark mark-tags', `🏷 ${item.user_tags.length}`);
    t.title = `${item.user_tags.length} custom tag(s)`;
    marks.append(t);
  }
  if (item.notes) {
    const n = el('span', 'mark mark-notes', '📝');
    n.title = 'Has notes';
    marks.append(n);
  }
  if ((item.personal_triggers || []).length) {
    const pt = el('span', 'mark mark-trig', `✦ ${item.personal_triggers.length}`);
    pt.title = `${item.personal_triggers.length} personal trigger(s)`;
    marks.append(pt);
  }
  if (item.meta_state !== 'complete') {
    const w = el('span', 'mark mark-warn', item.meta_state === 'missing' ? 'no meta' : 'partial');
    marks.append(w);
  }
  if (marks.childElementCount) thumb.append(marks);

  const fav = el('button', `card-fav${item.favorite ? ' on' : ''}`);
  fav.type = 'button';
  fav.setAttribute('aria-label', item.favorite ? 'Remove from favourites' : 'Add to favourites');
  fav.setAttribute('aria-pressed', String(!!item.favorite));
  fav.innerHTML = '<svg viewBox="0 0 24 24"><use href="#i-star"></use></svg>';
  fav.onclick = async (event) => {
    event.stopPropagation();
    await toggleFavourite(item, index, fav);
  };
  thumb.append(fav);

  const body = el('div', 'card-body');
  body.append(el('div', 'card-title', item.name));
  if (item.creator) {
    // A real button, not the whole card: clicking the artist name filters the
    // gallery to their other models without opening this one's detail view.
    const by = el('button', 'card-sub card-sub-creator', `by ${item.creator}`);
    by.type = 'button';
    by.title = `Show everything by ${item.creator}`;
    by.onclick = (event) => { event.stopPropagation(); applyCreatorFilter(item.creator); };
    body.append(by);
  }
  // Folder gives the organisational context the old cards had.
  if (item.folder) {
    const folder = el('div', 'card-folder', item.folder.split('/').slice(-2).join(' / '));
    folder.title = item.folder;
    body.append(folder);
  }

  const foot = el('div', 'card-foot');
  const trigCount = (item.trained_words || []).length + (item.personal_triggers || []).length;
  foot.append(el('span', 'card-foot-item', trigCount ? `${trigCount} triggers` : 'no triggers'));
  foot.append(el('span', 'card-foot-item', fmtBytes(item.size)));
  body.append(foot);

  card.append(thumb, body);
  const open = () => openDetail(item.id);
  card.onclick = open;
  card.onkeydown = (event) => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); open(); }
  };
  return card;
}

function starSvg() {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('width', '11');
  svg.setAttribute('height', '11');
  const use = document.createElementNS('http://www.w3.org/2000/svg', 'use');
  use.setAttribute('href', '#i-star');
  svg.append(use);
  return svg;
}

async function toggleFavourite(item, index, button) {
  const next = !item.favorite;
  button.classList.toggle('on', next);
  button.setAttribute('aria-pressed', String(next));
  button.setAttribute('aria-label', next ? 'Remove from favourites' : 'Add to favourites');
  try {
    await apiPost(`/api/models/${item.id}/user`, { favorite: next });
    item.favorite = next;
    state.items.set(index, item);
  } catch (err) {
    button.classList.toggle('on', !next);       // roll the optimistic update back
    toast(`Could not save: ${err.message}`, 'error');
  }
}

/* ------------------------------------------------------ detail modal (3-col) */

let lastFocused = null;

async function openDetail(id) {
  lastFocused = document.activeElement;
  $('modal-backdrop').hidden = false;
  $('detail-modal').hidden = false;
  $('dm-left').replaceChildren(el('p', 'muted', 'Loading…'));
  $('dm-middle').replaceChildren();
  $('dm-right').replaceChildren();
  $('modal-close').focus();
  document.body.classList.add('modal-open');

  try {
    const model = await apiGet(`/api/models/${id}`);
    renderDetail(model);
  } catch (err) {
    $('dm-left').replaceChildren(el('p', 'muted', `Could not load: ${err.message}`));
  }
}

function closeDetail() {
  $('detail-modal').hidden = true;
  $('modal-backdrop').hidden = true;
  document.body.classList.remove('modal-open');
  if (lastFocused && lastFocused.isConnected) lastFocused.focus();
}

function renderDetail(model) {
  $('detail-modal').setAttribute('aria-label', model.name);
  // Left column = a scrolling body + a genuinely fixed action footer that the
  // body scrolls under, so the three buttons never move.
  const body = el('div', 'dm-left-body');
  body.append(...detailLeft(model));
  $('dm-left').replaceChildren(body, detailActions(model));
  $('dm-middle').replaceChildren(...detailMiddle(model));
  $('dm-right').replaceChildren(...detailRight(model));
  body.scrollTop = 0;
}

/* --- shared bits --- */

function sectionTitle(text) {
  return el('h3', 'dm-section-title', text);
}

function copyChip(word) {
  const chip = el('button', 'chip copyable', word);
  chip.type = 'button';
  chip.title = 'Copy to clipboard';
  chip.onclick = async () => {
    try {
      await navigator.clipboard.writeText(word);
      toast(`Copied “${word}”`, 'ok');
    } catch { toast('Clipboard blocked by the browser', 'error'); }
  };
  return chip;
}

async function copyText(text, label) {
  try {
    await navigator.clipboard.writeText(text);
    toast(`Copied ${label}`, 'ok');
  } catch { toast('Clipboard blocked by the browser', 'error'); }
}

/* --- tinting the detail view from its preview --- */

function rgbToHsl(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l];
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h;
  if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
  else if (max === g) h = ((b - r) / d + 2) / 6;
  else h = ((r - g) / d + 4) / 6;
  return [h, s, l];
}

/** The most *present* colour in an image, not the average — averaging any real
 *  picture returns mud.
 *
 *  Pixels are bucketed by hue and each bucket is weighted by saturation, so a
 *  small area of strong colour outranks a large flat wash. Near-black,
 *  near-white and near-grey pixels are dropped outright: they carry no usable
 *  hue and, being most of a typical image, would win on count alone.
 *  Returns null when there is nothing colourful enough to be worth using.
 */
function dominantColour(img) {
  const N = 40;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = N;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  let data;
  try {
    ctx.drawImage(img, 0, 0, N, N);
    data = ctx.getImageData(0, 0, N, N).data;
  } catch {
    return null;                    // tainted canvas; previews are same-origin
  }

  const buckets = new Map();
  for (let i = 0; i < data.length; i += 4) {
    if (data[i + 3] < 128) continue;
    const [h, s, l] = rgbToHsl(data[i], data[i + 1], data[i + 2]);
    if (l < 0.12 || l > 0.92 || s < 0.18) continue;
    const key = Math.round(h * 24);            // 15-degree buckets
    const acc = buckets.get(key) || { w: 0, h: 0, s: 0 };
    acc.w += s; acc.h += h * s; acc.s += s * s;
    buckets.set(key, acc);
  }

  // Rank the buckets and take two. The runner-up has to sit a reasonable
  // distance around the wheel from the winner, or a single gradient — which is
  // one hue smeared across several adjacent buckets — would supply both and the
  // "second" colour would be indistinguishable from the first.
  const ranked = [...buckets.entries()]
    .map(([key, acc]) => ({ key, w: acc.w, h: acc.h / acc.w, s: acc.s / acc.w }))
    .sort((a, b) => b.w - a.w);
  if (!ranked.length || ranked[0].w < 12) return null;   // too little colour

  const primary = ranked[0];
  const apart = (a, b) => {
    const d = Math.abs(a - b);
    return Math.min(d, 1 - d);                 // hue is a circle
  };
  const secondary = ranked.slice(1).find(
    (c) => c.w >= primary.w * 0.12 && apart(c.h, primary.h) > 0.08) || null;

  return {
    h: primary.h, s: primary.s,
    h2: secondary ? secondary.h : null,
    s2: secondary ? secondary.s : null,
  };
}

/** Push a preview's dominant hue into the detail modal.
 *
 *  Only surfaces move: the panel wash, its edge, and the section labels. Body
 *  text keeps its own token, because a hue taken from an arbitrary image is not
 *  something to bet legibility on. Saturation is clamped so a violently
 *  saturated preview cannot produce a garish panel, and the lightness stays
 *  fixed per theme so contrast is unchanged whatever hue arrives.
 */
function tintDetailFromImage(img) {
  const modal = $('detail-modal');
  if (!modal) return;
  const clear = () => ['--dm-h', '--dm-s', '--dm-h2', '--dm-s2']
    .forEach((k) => modal.style.removeProperty(k));

  const colour = dominantColour(img);
  if (!colour) return clear();

  // Clamped hard on both sides. Below the floor the wash is invisible and the
  // panel may as well be neutral; above the ceiling a lurid preview would put
  // a fluorescent cast over the whole thing.
  const sat = (v) => `${Math.round(Math.min(0.72, Math.max(0.30, v)) * 100)}%`;
  modal.style.setProperty('--dm-h', String(Math.round(colour.h * 360)));
  modal.style.setProperty('--dm-s', sat(colour.s));

  // The accent hue drives the smaller elements — chips, trigger groups, the
  // rating stars. Falling back to the dominant hue rather than to neutral: a
  // single-colour image should still tint them, just without the contrast a
  // second hue would have given.
  modal.style.setProperty('--dm-h2',
    String(Math.round((colour.h2 ?? colour.h) * 360)));
  modal.style.setProperty('--dm-s2', sat(colour.s2 ?? colour.s));
}

/* --- left column: identity, image, actions --- */

function detailLeft(model) {
  const parts = [];
  parts.push(el('h2', 'dm-title', model.name));

  // Every version has its own card in the gallery, so there is nothing to
  // switch between here — just say which version you are looking at, and how
  // many others of the same model are sitting alongside it.
  const siblings = (model.versions || []).length - 1;
  if (model.version_name || siblings > 0) {
    const line = el('div', 'dm-version-static', model.version_name || 'This version');
    if (siblings > 0) {
      line.append(el('span', 'dm-version-count',
        `${siblings} other version${siblings > 1 ? 's' : ''} in your library`));
    }
    parts.push(line);
  }

  const media = el('div', 'dm-media');
  const mediaUrl = (rel, full) =>
    `/media/${rel.split('/').map(encodeURIComponent).join('/')}${full ? '?full=1' : ''}`;

  if (model.previews && model.previews.length) {
    // A tall, narrow image cannot fill a fixed-width frame, and the gap used to
    // be flat near-black down both sides. The same image, blurred and
    // over-scaled, fills it with the picture's own colour instead.
    const fill = el('div', 'dm-media-fill');
    const img = el('img');
    img.alt = '';
    img.loading = 'lazy';
    img.title = 'Open the full-size image';
    const show = (rel) => {
      const full = mediaUrl(rel, true);
      img.src = full;
      fill.style.backgroundImage = `url("${full}")`;
      img.onclick = () => window.open(full, '_blank', 'noopener');
    };
    img.addEventListener('load', () => tintDetailFromImage(img));
    show(model.previews[0]);
    media.append(fill, img);
    if (model.provider) {
      const badge = el('span', `dm-media-badge provider ${model.provider}`, model.provider);
      badge.title = `Metadata from ${model.provider}`;
      media.append(badge);
    }
    parts.push(media);

    // When a model ships several previews, offer the rest as a thumbnail strip.
    if (model.previews.length > 1) {
      const strip = el('div', 'dm-thumbs');
      model.previews.slice(0, 8).forEach((rel, i) => {
        const thumb = el('button', `dm-thumb${i === 0 ? ' on' : ''}`);
        thumb.type = 'button';
        thumb.setAttribute('aria-label', `Preview ${i + 1}`);
        const ti = el('img');
        ti.alt = '';
        ti.loading = 'lazy';
        ti.src = mediaUrl(rel, false);
        thumb.append(ti);
        thumb.onclick = () => {
          show(rel);
          [...strip.children].forEach((c) => c.classList.toggle('on', c === thumb));
        };
        strip.append(thumb);
      });
      parts.push(strip);
    }
  } else {
    media.append(el('div', 'placeholder', 'no preview'));
    parts.push(media);
  }

  // Details sit under the image (using the space the poster leaves) so the
  // middle column can give the description real room.
  parts.push(sectionTitle('Details'));
  const facts = el('div', 'dm-facts');
  const fact = (k, v) => {
    if (!v && v !== 0) return;
    const cell = el('div', 'dm-fact');
    cell.append(el('div', 'dm-fact-k', k), el('div', 'dm-fact-v', String(v)));
    return facts.append(cell);
  };
  fact('Creator', model.creator);
  fact('Base model', model.base_model);
  // Provider is a badge on the preview instead of a row here. It is one word
  // that never changes, and giving it a labelled cell of its own cost a whole
  // row of a panel that already needs scrolling.
  fact('Size', fmtBytes(model.size));
  fact('Metadata', model.meta_state);
  fact('Used', model.use_count ? `${model.use_count}×` : '');
  parts.push(facts);

  const paths = el('div', 'dm-paths');
  const pathRow = (label, value, action) => {
    if (!value) return;
    const row = el('div', 'dm-path-row');
    row.append(el('span', 'dm-fact-k', label));
    const val = el('span', 'dm-path-val', value);
    val.title = value;
    row.append(val);
    const btn = el('button', 'icon-btn', '');
    btn.type = 'button';
    btn.title = action.title;
    btn.innerHTML = action.icon;
    btn.onclick = action.run;
    row.append(btn);
    paths.append(row);
  };
  const COPY_ICON = '<svg viewBox="0 0 24 24" width="13" height="13"><use href="#i-copy"></use></svg>';
  const FOLDER_ICON = '<svg viewBox="0 0 24 24" width="14" height="14"><use href="#i-folder"></use></svg>';

  pathRow('File', model.filename, {
    title: 'Copy filename', icon: COPY_ICON,
    run: () => copyText(model.filename, 'filename'),
  });
  // Opening the folder is what you actually want from a path; copying it was
  // an extra step to get there.
  pathRow('Path', model.folder || '(library root)', {
    title: 'Show this model in your file manager', icon: FOLDER_ICON,
    run: async () => {
      try {
        await apiPost('/api/open-folder', { path: model.path });
      } catch (err) { toast(err.message, 'error'); }
    },
  });
  if (model.sha256) {
    pathRow('SHA-256', `${model.sha256.slice(0, 20)}…`, {
      title: 'Copy hash', icon: COPY_ICON,
      run: () => copyText(model.sha256, 'hash'),
    });
  }
  parts.push(paths);
  return parts;
}

/** The fixed action footer for the left column. */
function detailActions(model) {
  const actions = el('div', 'dm-actions');
  const weight = model.weight ?? model.preferred_weight ?? 1;
  const copyTag = el('button', 'btn primary block', 'Copy LoRA tag');
  copyTag.type = 'button';
  copyTag.onclick = () => copyText(`<lora:${model.filename.replace(/\.[^.]+$/, '')}:${weight}>`, 'LoRA tag');
  actions.append(copyTag);

  const trigs = [...(model.trained_words || []), ...(model.personal_triggers || [])];
  if (trigs.length) {
    const copyTrig = el('button', 'btn block', 'Copy all triggers');
    copyTrig.type = 'button';
    copyTrig.onclick = () => copyText(trigs.join(', '), 'triggers');
    actions.append(copyTrig);
  }

  const row = el('div', 'dm-action-row');
  if (model.page_url) {
    const arc = model.provider === 'arcenciel';
    const link = el('a', `btn view-on ${arc ? 'arcenciel' : 'civitai'}`, '');
    link.href = model.page_url;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.append(document.createTextNode('View on '));
    link.append(el('strong', null, arc ? 'Arcenciel' : 'CivitAI'));
    row.append(link);
  }

  const del = el('button', 'btn danger', 'Delete');
  del.type = 'button';
  del.title = 'Delete this version and its files';
  del.onclick = () => deleteModel(model);
  row.append(del);
  actions.append(row);
  return actions;
}

/** Delete one version, spelling out exactly what will go. */
async function deleteModel(model) {
  const others = (model.versions || []).length - 1;
  const lines = [
    `Delete "${model.name}"?`, '',
    `This removes ${model.filename} and its metadata and preview files.`,
  ];
  if (others > 0) {
    lines.push('', `${others} other version(s) of this model stay untouched.`);
  } else {
    lines.push('', 'Its folder will be removed if nothing else is left in it.');
  }
  lines.push('', 'This cannot be undone.');
  if (!confirm(lines.join('\n'))) return;

  try {
    const res = await apiCall(`/api/models/${model.id}`, { method: 'DELETE' });
    closeDetail();
    toast(`Deleted ${res.removed.length} file(s)`
      + (res.folder_removed ? ' and its folder' : ''), 'ok');
    refreshFacets();
    resetResults();
  } catch (err) { toast(`Could not delete: ${err.message}`, 'error'); }
}

/* --- middle column: rating, collection cascade, description (roomy) --- */

function detailMiddle(model) {
  const parts = [];

  parts.push(sectionTitle('Personal rating'));
  parts.push(ratingStars(model));

  parts.push(sectionTitle('Collection'));
  parts.push(collectionCascade(model));

  parts.push(sectionTitle('Description'));
  if (model.description_html) {
    // Already reduced to a small allowlist server-side (sidecar.clean_html):
    // no scripts, no attributes but a scheme-checked href, unknown tags
    // dropped. Rendering it as markup is what keeps the author's headings,
    // lists and line breaks — flattened to one paragraph a long description is
    // barely readable.
    const body = el('div', 'dm-description dm-rich');
    body.innerHTML = model.description_html;
    parts.push(body);
  } else if (model.description) {
    parts.push(el('div', 'dm-description', model.description.slice(0, 8000)));
  } else {
    parts.push(el('p', 'muted small', '— No description —'));
  }

  if (model.duplicates && model.duplicates.length) {
    parts.push(sectionTitle(`Duplicates (${model.duplicates.length})`));
    const list = el('ul', 'dm-dup-list');
    model.duplicates.forEach((d) => list.append(el('li', 'muted small', d.path)));
    parts.push(list);
  }
  return parts;
}

function ratingStars(model) {
  const stars = el('div', 'dm-stars');
  stars.setAttribute('role', 'group');
  stars.setAttribute('aria-label', 'Rating');
  for (let value = 1; value <= 5; value++) {
    const star = el('button', `dm-star${model.rating >= value ? ' on' : ''}`);
    star.type = 'button';
    star.setAttribute('aria-label', `${value} star${value > 1 ? 's' : ''}`);
    star.innerHTML = '<svg viewBox="0 0 24 24"><use href="#i-star"></use></svg>';
    star.onclick = async () => {
      const next = model.rating === value ? 0 : value;
      try {
        await apiPost(`/api/models/${model.id}/user`, { rating: next });
        model.rating = next || null;
        [...stars.children].forEach((c, i) => c.classList.toggle('on', i < next));
        invalidateItem(model.id, (item) => { item.rating = next || null; });
      } catch (err) { toast(err.message, 'error'); }
    };
    stars.append(star);
  }
  return stars;
}

/* --- right column: tags, notes, triggers --- */

function detailRight(model) {
  const parts = [];

  if ((model.tags || []).length) {
    parts.push(sectionTitle(`Tags (${model.tags.length})`));
    // Capped height with a fade, and a real button to lift the cap. A faint
    // "scroll for more" line was too easy to miss — a button is unmissable and
    // saves the scrolling entirely.
    const wrap = el('div', 'dm-tags-wrap');
    const box = el('div', 'dm-tags-box');       // the fade anchors to this, so
    const chips = el('div', 'chips dm-tags-scroll');  // it can never cover the button
    // Tags are for finding things, so a click searches rather than copies.
    model.tags.forEach((t) => chips.append(searchChip(t)));
    box.append(chips);
    wrap.append(box);
    // Decide after layout whether the content actually overflows.
    requestAnimationFrame(() => {
      if (chips.scrollHeight <= chips.clientHeight + 2) return;
      wrap.classList.add('has-more');
      const more = el('button', 'dm-tags-more');
      more.type = 'button';
      const paint = () => {
        const open = wrap.classList.contains('expanded');
        more.textContent = open ? 'Show fewer' : `Show all ${model.tags.length} tags`;
        more.append(el('span', 'dm-tags-more-arrow', open ? '↑' : '↓'));
      };
      more.onclick = () => { wrap.classList.toggle('expanded'); paint(); };
      paint();
      wrap.append(more);
    });
    parts.push(wrap);
  }

  parts.push(sectionTitle('Notes'));
  const notes = el('textarea', 'dm-notes');
  notes.value = model.notes || '';
  notes.placeholder = 'Your notes about this model…';
  notes.setAttribute('aria-label', 'Personal notes');
  let saveTimer;
  notes.oninput = () => {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(async () => {
      try {
        await apiPost(`/api/models/${model.id}/user`, { notes: notes.value });
        model.notes = notes.value;
        invalidateItem(model.id, (item) => { item.notes = notes.value; });
      } catch (err) { toast(`Notes not saved: ${err.message}`, 'error'); }
    }, 600);
  };
  parts.push(notes);

  if ((model.trained_words || []).length) {
    parts.push(sectionTitle(`Official triggers (${model.trained_words.length})`));
    // A big trigger set is capped and scrolls in place, so "Your triggers"
    // below stays reachable without scrolling past dozens of chips.
    const chips = el('div', 'chips dm-trigger-scroll');
    model.trained_words.forEach((w) => chips.append(copyChip(w)));
    parts.push(chips);
  }

  parts.push(sectionTitle('Your triggers'));
  parts.push(triggerGroupEditor(model));
  return parts;
}

/** Personal triggers as one editable group (comma-separated), not a wall of
 *  individual chips. Stored as a single group element. */
function triggerGroupEditor(model) {
  const wrap = el('div', 'dm-group-editor');
  const area = el('textarea', 'dm-group-input');
  area.value = (model.personal_triggers || []).join(', ');
  area.placeholder = 'Your own trigger words, comma-separated…';
  area.setAttribute('aria-label', 'Your triggers');
  area.rows = 2;

  let saveTimer;
  const save = async () => {
    const text = area.value.trim();
    const value = text ? [text] : [];   // one group element
    try {
      await apiPost(`/api/models/${model.id}/user`, { personal_triggers: value });
      model.personal_triggers = value;
      invalidateItem(model.id, (item) => { item.personal_triggers = value; });
    } catch (err) { toast(`Not saved: ${err.message}`, 'error'); }
  };
  area.oninput = () => { clearTimeout(saveTimer); saveTimer = setTimeout(save, 600); };

  const copyBtn = el('button', 'btn tiny ghost', 'Copy');
  copyBtn.type = 'button';
  copyBtn.onclick = () => copyText(area.value.trim(), 'your triggers');
  const row = el('div', 'dm-group-row');
  row.append(copyBtn);

  wrap.append(area, row);
  return wrap;
}

/** A chip that runs its text as a gallery search on click. */
function searchChip(text) {
  const chip = el('button', 'chip searchable', text);
  chip.type = 'button';
  chip.title = `Search “${text}”`;
  chip.onclick = () => {
    $('search').value = text;
    syncSearchHint();
    state.filters.q = text;
    state.filters.creator = '';
    closeDetail();
    resetResults();
  };
  return chip;
}

/** An add/remove chip list bound to one user-owned field.
 *
 * `field` is what the API expects; `readKey` is where it lives in the payload
 * when the two differ (user tags are sent as `tags` but returned as
 * `user_tags`, because provider tags occupy `tags` in the response).
 */
function editableList(model, field, placeholder, readKey = null) {
  const key = readKey || field;
  const wrap = el('div');
  const chips = el('div', 'chips');
  let values = [...(model[key] || [])];

  const persist = async () => {
    try {
      await apiPost(`/api/models/${model.id}/user`, { [field]: values });
      model[key] = [...values];
      invalidateItem(model.id, (item) => { item[key] = [...values]; });
    } catch (err) {
      toast(`Could not save: ${err.message}`, 'error');
    }
  };

  const paint = () => {
    chips.replaceChildren();
    if (!values.length) {
      chips.append(el('span', 'muted small', 'None yet.'));
    }
    values.forEach((value, index) => {
      const chip = el('span', 'chip');
      chip.append(document.createTextNode(value + ' '));
      const remove = el('button', null, '×');
      remove.type = 'button';
      remove.setAttribute('aria-label', `Remove ${value}`);
      remove.style.cssText = 'background:none;border:none;cursor:pointer;color:inherit;padding:0 2px';
      remove.onclick = async () => {
        values.splice(index, 1);
        paint();
        await persist();
      };
      chip.append(remove);
      chips.append(chip);
    });
  };

  const input = el('input');
  input.type = 'text';
  input.placeholder = placeholder;
  input.setAttribute('aria-label', placeholder);
  input.onkeydown = async (event) => {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    const value = input.value.trim();
    if (!value) return;
    if (values.some((v) => v.toLowerCase() === value.toLowerCase())) {
      input.value = '';
      return;
    }
    values.push(value);
    input.value = '';
    paint();
    await persist();
  };

  paint();
  wrap.append(chips, input);
  return wrap;
}

/** Collection assignment as a Category → Sub-category cascade.
 *
 * Mirrors the sidebar filter and matches the mental model people actually
 * have: a broad category ("Anime") and an optional sub-category ("Fate")
 * rather than a single "Anime / Fate" string. Free typing creates a new
 * category or sub-category. A "move files" toggle turns the collection into a
 * real folder on disk.
 */
/** A dropdown you can type into.
 *
 *  A native <select> cannot be searched, and with dozens of sub-categories the
 *  list ran to the bottom of the window. This opens a capped, scrollable panel
 *  with the search box already focused, so the common case is "start typing".
 *
 *  Creating is offered only when nothing matches what was typed — a permanent
 *  "＋ New…" row in the list is one more thing to scroll past, and it invites a
 *  duplicate of something already there a few rows below.
 */
function searchPicker({ placeholder, emptyLabel, options, value, onPick, onCreate }) {
  const wrap = el('div', 'picker');
  const button = el('button', 'picker-button');
  button.type = 'button';
  const caret = el('span', 'picker-caret', '▾');
  const text = el('span', 'picker-value');
  button.append(text, caret);

  const panel = el('div', 'picker-panel');
  panel.hidden = true;
  const searchRow = el('div', 'picker-search');
  const search = el('input');
  search.type = 'text';
  search.placeholder = placeholder;
  search.autocomplete = 'off';
  searchRow.append(search);
  const list = el('div', 'picker-list');
  panel.append(searchRow, list);
  wrap.append(button, panel);

  let selected = value || '';
  const paintValue = () => { text.textContent = selected || emptyLabel; };
  paintValue();

  const close = () => {
    panel.hidden = true;
    button.setAttribute('aria-expanded', 'false');
    document.removeEventListener('mousedown', onOutside, true);
  };
  const onOutside = (event) => { if (!wrap.contains(event.target)) close(); };

  const choose = (name) => { selected = name; paintValue(); close(); onPick(name); };

  const paintList = () => {
    const query = search.value.trim().toLowerCase();
    const matches = options().filter((o) => o.name.toLowerCase().includes(query));
    list.replaceChildren();

    const none = el('button', `picker-item${selected ? '' : ' on'}`);
    none.type = 'button';
    none.append(el('span', 'picker-item-name', emptyLabel));
    none.onclick = () => choose('');
    if (!query) list.append(none);

    matches.forEach((o) => {
      const row = el('button', `picker-item${o.name === selected ? ' on' : ''}`);
      row.type = 'button';
      row.append(el('span', 'picker-item-name', o.name));
      if (o.count != null) row.append(el('span', 'picker-item-count', String(o.count)));
      row.onclick = () => choose(o.name);
      list.append(row);
    });

    const typed = search.value.trim();
    const exists = matches.some((o) => o.name.toLowerCase() === typed.toLowerCase());
    if (typed && !exists && onCreate) {
      const make = el('button', 'picker-item picker-create');
      make.type = 'button';
      make.append(el('span', 'picker-item-name', `Create “${typed}”`));
      make.onclick = () => choose(typed);
      list.append(make);
    }
    if (!list.childElementCount) {
      list.append(el('div', 'picker-empty', 'Nothing matches'));
    }
  };

  button.onclick = () => {
    if (!panel.hidden) return close();
    panel.hidden = false;
    button.setAttribute('aria-expanded', 'true');
    search.value = '';
    paintList();
    search.focus();
    document.addEventListener('mousedown', onOutside, true);
  };
  search.oninput = paintList;
  search.onkeydown = (event) => {
    if (event.key === 'Escape') { event.stopPropagation(); close(); button.focus(); }
    if (event.key === 'Enter') {
      event.preventDefault();
      const first = list.querySelector('.picker-item');
      if (first) first.click();
    }
  };

  return {
    node: wrap,
    get value() { return selected; },
    set value(v) { selected = v || ''; paintValue(); },
    refresh: paintList,
  };
}

function collectionCascade(model) {
  const wrap = el('div', 'dm-collection');
  const current = el('div', 'chips');

  const paintCurrent = () => {
    current.replaceChildren();
    if (!(model.collections || []).length) {
      current.append(el('span', 'muted small', 'Not in any collection.'));
    }
    (model.collections || []).forEach((c) => {
      const chip = el('span', 'chip');
      const label = el('button', 'chip-label', c.name);
      label.type = 'button';
      label.title = 'Filter the gallery by this collection';
      label.onclick = () => {
        const [cat, ...rest] = c.name.split(' / ');
        applyCategoryFilter(cat, rest.join(' / '));
        closeDetail();
      };
      const remove = el('button', 'chip-x', '×');
      remove.type = 'button';
      remove.setAttribute('aria-label', `Remove from ${c.name}`);
      remove.onclick = async () => {
        try {
          await apiPost(`/api/collections/${c.id}/members`, { file_ids: [model.id], add: false });
          model.collections = model.collections.filter((x) => x.id !== c.id);
          paintCurrent();
          refreshFacets();
          invalidateItem(model.id, () => {});
        } catch (err) { toast(err.message, 'error'); }
      };
      chip.append(label, remove);
      current.append(chip);
    });
  };
  paintCurrent();

  // Both pickers sit on one row. Stacked, each burned a full row of the panel
  // on a control as wide as the panel itself, for names rarely longer than a
  // word — which is the space that was going to waste.
  const row = el('div', 'dm-picker-row');

  const catField = el('div', 'dm-field');
  catField.append(el('label', 'field-label', 'Category'));
  const catPicker = searchPicker({
    placeholder: 'Search, or type a new name',
    emptyLabel: 'Choose a category',
    options: () => (state.categories || []).map((c) => ({ name: c.name, count: c.count })),
    onPick: () => { subPicker.value = ''; },
    onCreate: true,
  });
  catField.append(catPicker.node);

  const subField = el('div', 'dm-field');
  subField.append(el('label', 'field-label', 'Sub-category (optional)'));
  const subPicker = searchPicker({
    placeholder: 'Search, or type a new name',
    emptyLabel: 'None (top level)',
    options: () => {
      const cat = catPicker.value.toLowerCase();
      const found = (state.categories || []).find((c) => c.name.toLowerCase() === cat);
      return (found ? found.subcategories : []).map((sc) => ({ name: sc.name, count: sc.count }));
    },
    onPick: () => {},
    onCreate: true,
  });
  subField.append(subPicker.node);
  row.append(catField, subField);

  const currentCategory = () => catPicker.value.trim();
  const currentSub = () => subPicker.value.trim();

  // Default to the model's *deepest* existing collection, so both the category
  // and its sub-category show what it's actually filed under (a model in both
  // "Anime" and "Anime / Fate" should default to Anime → Fate, not Anime → None).
  const existing = (model.collections || [])
    .slice()
    .sort((a, b) => b.name.split(' / ').length - a.name.split(' / ').length)[0];
  if (existing) {
    const [cat, ...rest] = existing.name.split(' / ');
    catPicker.value = cat;
    subPicker.value = rest.join(' / ');
  }

  // Adding to a collection is the same act as filing it into that folder, so
  // there is no opt-in checkbox and no confirmation — the files always move.
  const addBtn = el('button', 'btn primary block', 'Add & move to collection');
  addBtn.type = 'button';
  addBtn.onclick = async () => {
    const cat = currentCategory();
    if (!cat) { toast('Pick or type a category first', 'error'); return; }
    const sub = currentSub();
    const name = sub ? `${cat} / ${sub}` : cat;
    if ((model.collections || []).some((c) => c.name.toLowerCase() === name.toLowerCase())) {
      toast('Already in that collection', 'ok');
      return;
    }
    try {
      const created = await apiPost('/api/collections', { name });
      const res = await apiPost(`/api/collections/${created.id}/members`,
        { file_ids: [model.id], add: true, move_files: true });
      model.collections = [...(model.collections || []), { id: created.id, name }];
      paintCurrent();
      await refreshFacets();
      catPicker.value = '';
      subPicker.value = '';
      invalidateItem(model.id, () => {});
      toast(res.job ? `Added to ${name}; moving files…` : `Added to ${name}`, 'ok');
    } catch (err) { toast(err.message, 'error'); }
  };

  wrap.append(current, row, addBtn);
  return wrap;
}

function invalidateItem(id, mutate) {
  state.items.forEach((item, index) => {
    if (item.id !== id) return;
    mutate(item);
    const node = state.rendered.get(index);
    if (node) { node.remove(); state.rendered.delete(index); }
  });
  render();
}

/* -------------------------------------------------------------- live feed */

function connectEvents() {
  const source = new EventSource('/events');

  source.addEventListener('open', () => setStatus('idle', 'Connected'));
  source.addEventListener('error', () => setStatus('err', 'Reconnecting…'));

  source.addEventListener('job', (event) => {
    const payload = JSON.parse(event.data);
    applyJobState(payload);
  });
  source.addEventListener('hello', (event) => applyJobState(JSON.parse(event.data)));
  source.addEventListener('stats', (event) => renderStats(JSON.parse(event.data)));
  source.addEventListener('log', (event) => appendLog(JSON.parse(event.data)));
  source.addEventListener('settings', () => refreshManage());
  source.addEventListener('collections', () => refreshFacets());
  source.addEventListener('model', () => { /* card already updated optimistically */ });
  source.addEventListener('close', () => source.close());
}

function applyJobState(payload) {
  const job = payload.job || null;
  const jobs = payload.jobs || payload;
  const running = (jobs && jobs.running) || (job && job.status === 'running' ? job : null);
  state.currentJob = running;

  if (running) {
    setStatus('running', running.label);
    $('status-fill').style.width = `${running.percent}%`;
    const bits = [];
    if (running.total) bits.push(`${running.current}/${running.total}`);
    if (running.elapsed) bits.push(fmtDuration(running.elapsed));
    if (running.eta) bits.push(`~${fmtDuration(running.eta)} left`);
    $('status-meta').textContent = bits.join(' · ');
    $('status-cancel').hidden = false;
    $('status-cancel').onclick = () => apiPost(`/api/jobs/${running.id}/cancel`).catch(
      (err) => toast(err.message, 'error'));
  } else {
    // Leave the last message visible rather than snapping back to "Ready" —
    // the outcome of the job that just finished is the useful information.
    if (job && job.status !== 'running') {
      const kind = job.status === 'error' ? 'err' : 'ok';
      setStatus(kind, `${job.label}: ${job.message || job.status}`);
      $('status-fill').style.width = job.status === 'done' ? '100%' : '0%';
      $('status-meta').textContent = job.elapsed ? fmtDuration(job.elapsed) : '';
      if (job.status === 'error') toast(`${job.label} failed: ${job.error}`, 'error', true);
      if (job.status === 'done') {
        refreshFacets();
        // A scan/sync can add or remove models, so reload the visible grid —
        // otherwise newly-detected models don't appear until a manual reload.
        if (['scan', 'quick-refresh', 'full-sync', 'ingest', 'resolve',
             'resolve-force', 'organise', 'adopt-folders', 'restore-backup',
             'rebuild-gallery', 'process-new', 'group-versions'].includes(job.kind)) {
          if (state.view === 'gallery') resetResults();
        }
      }
    }
    $('status-cancel').hidden = true;
  }

  if (state.view === 'manage') renderQueue(jobs);
}

function setStatus(kind, text) {
  const dot = $('status-dot');
  dot.className = `status-dot ${kind === 'idle' ? '' : kind}`;
  $('status-text').textContent = text;
}

function appendLog(entry) {
  const console = $('console');
  const line = el('div', entry.level || 'info', entry.line);
  console.append(line);
  while (console.childElementCount > 800) console.firstElementChild.remove();
  if (state.followConsole) console.scrollTop = console.scrollHeight;
}

/* ------------------------------------------------------------ stats & UI */

function renderStats(stats) {
  const list = $('stat-list');
  const rows = [
    ['Models', stats.total, ''],
    ['Size', fmtBytes(stats.total_bytes), ''],
    ['No metadata', stats.missing_metadata, stats.missing_metadata ? 'warn' : 'ok'],
    ['Partial', stats.partial_metadata, stats.partial_metadata ? 'warn' : 'ok'],
    ['No preview', stats.missing_preview, stats.missing_preview ? 'warn' : 'ok'],
    ['Unhashed', stats.unhashed, ''],
    ['Favourites', stats.favorites, ''],
    ['Duplicates', stats.duplicate_groups, stats.duplicate_groups ? 'warn' : 'ok'],
    ['Missing files', stats.missing_files, stats.missing_files ? 'warn' : ''],
  ];
  list.replaceChildren(...rows.map(([key, value, cls]) => {
    const li = el('li');
    li.append(el('span', 'k', key), el('span', `v ${cls}`.trim(), String(value)));
    return li;
  }));
  state.libraryTotal = stats.total;
  updateResultCount();
}

/** The brand carries the count for both states.
 *
 * There used to be a bar of its own under the top bar whose entire job was to
 * repeat a number the brand was already showing. Unfiltered it reads "107
 * models"; filtered it narrows to "24 of 107", which is the only case where
 * the result count said anything the brand did not.
 */
function updateResultCount() {
  const active = Object.entries(state.filters)
    .filter(([key, value]) => value && key !== 'sort').length;
  const total = state.libraryTotal;
  const shown = state.total;
  const label = (n) => `${n.toLocaleString()} model${n === 1 ? '' : 's'}`;
  $('brand-count').textContent = (active && total && shown !== total)
    ? `${shown.toLocaleString()} of ${label(total)}`
    : label(total || shown);
  $('clear-filters').hidden = active === 0;
}

async function refreshFacets() {
  try {
    const data = await apiGet('/api/facets');
    fillSelect($('filter-base'), data.base_models.map(
      (b) => ({ value: b.value, label: `${b.value} (${b.count})` })), 'All');
    applyCategories(data.categories || [], data.uncategorised ?? 0);
    const stats = await apiGet('/api/stats');
    renderStats(stats);
  } catch { /* the strip already shows the connection state */ }
}

/** Populate the sidebar category cascade and remember the tree for the modal. */
/** The "no collection" choice is marked with a data attribute on its <option>,
 *  not a reserved value. Any string could legitimately be a category name, and
 *  a sentinel carrying an unusual character does not survive the DOM intact —
 *  so identity lives on the element, where nothing can collide with it. */
const UNCATEGORISED = 'uncategorised';

function applyCategories(categories, uncategorised = null) {
  state.categories = categories;
  if (uncategorised !== null) state.uncategorised = uncategorised;
  const catSel = $('filter-category');
  const prev = catSel.value;
  catSel.replaceChildren(new Option('All categories', ''));
  categories.forEach((c) => catSel.append(new Option(`${c.name} (${c.count})`, c.name)));
  // Last, and only when there is something in it: the pile you have not filed
  // yet is the one you actually want to open while organising.
  if (state.uncategorised) {
    const opt = new Option(`Uncategorized (${state.uncategorised})`, UNCATEGORISED);
    opt.dataset.uncategorised = '1';
    catSel.append(opt);
  }
  if ([...catSel.options].some((o) => o.value === prev)) catSel.value = prev;
  paintSubcategories();
}

/** Fill the sub-category select for the chosen category.
 *
 * The select is never hidden — it greys out instead, so the top bar keeps its
 * shape and you can see that sub-categories exist as a concept before you have
 * picked anything.
 */
function paintSubcategories() {
  const cat = state.filters.category;
  const subSel = $('filter-subcategory');
  const found = state.categories.find((c) => c.name === cat);
  const subs = found ? found.subcategories : [];
  const usable = !!cat && subs.length > 0;
  subSel.disabled = !usable;
  subSel.title = !cat ? 'Pick a category first'
    : cat === UNCATEGORISED ? 'Models in no collection have no sub-categories'
    : usable ? 'Sub-category' : `"${cat}" has no sub-categories`;
  // Rename and delete target whatever is currently selected. "Uncategorized"
  // is a query, not a collection — there is nothing on disk to rename or delete.
  const real = !!cat && cat !== UNCATEGORISED;
  $('delete-category').disabled = !real;
  $('rename-category').disabled = !real;
  if (!usable) {
    subSel.replaceChildren(new Option(
      !cat ? 'All' : cat === UNCATEGORISED ? '—' : 'No sub-categories', ''));
    subSel.value = '';
    return;
  }
  const prev = state.filters.subcategory;
  subSel.replaceChildren(new Option(`All of ${cat}`, ''));
  subs.forEach((s) => subSel.append(new Option(`${s.name} (${s.count})`, s.name)));
  if ([...subSel.options].some((o) => o.value === prev)) subSel.value = prev;
}

/** Rename the selected category or sub-category, folder on disk included. */
async function renameSelectedCategory() {
  const cat = state.filters.category;
  if (!cat) return;
  const sub = state.filters.subcategory;
  const current = sub || cat;
  const full = sub ? `${cat} / ${sub}` : cat;
  const next = prompt(
    `New name for "${full}".\n\n` +
    `The folder on disk is renamed to match, and every model in it moves with ` +
    `it. Your ratings, notes and trigger words are untouched.`, current);
  if (next === null) return;
  const trimmed = next.trim();
  if (!trimmed || trimmed === current) return;
  try {
    const res = await apiPost('/api/collections/rename-by-name',
                              { name: full, new_name: trimmed });
    toast(`Renamed to "${res.new_name}" — moved ${res.models_moved} model folder(s)`,
          'ok');
    await refreshFacets();
    applyCategoryFilter(sub ? cat : trimmed, sub ? trimmed : '');
  } catch (err) { toast(`Could not rename: ${err.message}`, 'error'); }
}

/** Delete the selected category or sub-category. Files are kept and moved up. */
async function deleteSelectedCategory() {
  const cat = state.filters.category;
  if (!cat) return;
  const sub = state.filters.subcategory;
  const name = sub ? `${cat} / ${sub}` : cat;
  const destination = sub ? `the "${cat}" folder` : 'the library root';
  const nested = sub ? '' :
    '\nAny sub-categories under it are removed too, and their model folders move up.';
  if (!confirm(
      `Delete the collection "${name}"?\n${nested}\n` +
      `No model files are deleted — their folders move to ${destination}.\n\n` +
      `Continue?`)) return;
  try {
    const res = await apiPost('/api/collections/delete-by-name', { name });
    toast(`Deleted ${res.deleted_collections.length} collection(s); ` +
          `moved ${res.models_moved} model folder(s)`, 'ok');
    applyCategoryFilter('', '');
    await refreshFacets();
  } catch (err) { toast(`Could not delete: ${err.message}`, 'error'); }
}

/** Apply a category (and optional sub-category) filter from anywhere. */
function applyCategoryFilter(category, subcategory = '') {
  state.filters.collection = '';
  state.filters.category = category || '';
  state.filters.subcategory = subcategory || '';
  $('filter-category').value = state.filters.category;
  paintSubcategories();
  if (subcategory) $('filter-subcategory').value = subcategory;
  resetResults();
}

/** Filter the gallery down to one artist's models, from a card or anywhere else.
 *  Writes "artist: name" into the search box itself rather than a separate
 *  control — clearing or editing the search clears the filter too, so there is
 *  only one place that shows (and removes) an active search or artist filter. */
function applyCreatorFilter(creator) {
  const text = creator ? `artist: ${creator}` : '';
  $('search').value = text;
  syncSearchHint();
  state.filters.q = '';
  state.filters.creator = creator || '';
  resetResults();
}

function fillSelect(select, options, emptyLabel) {
  const previous = select.value;
  select.replaceChildren();
  select.append(new Option(emptyLabel, ''));
  options.forEach((opt) => select.append(new Option(opt.label, opt.value)));
  if ([...select.options].some((o) => o.value === previous)) select.value = previous;
}

/* --------------------------------------------------------------- manage */

function renderQueue(jobs) {
  const container = $('queue-list');
  const rows = [];
  if (jobs.running) rows.push(queueRow(jobs.running, true));
  (jobs.queued || []).forEach((job) => rows.push(queueRow(job, false)));
  container.replaceChildren(...(rows.length ? rows : [el('p', 'muted', 'Idle.')]));
}

function queueRow(job, running) {
  const row = el('div', 'queue-item');
  row.append(el('span', 'grow', job.label));
  if (running && job.total) {
    row.append(el('span', 'muted', `${job.percent}%`));
  } else if (!running) {
    row.append(el('span', 'muted', 'queued'));
  }
  const cancel = el('button', 'btn tiny ghost', running ? 'Cancel' : 'Remove');
  cancel.type = 'button';
  cancel.onclick = () => apiPost(`/api/jobs/${job.id}/cancel`).catch(
    (err) => toast(err.message, 'error'));
  row.append(cancel);
  return row;
}

function renderHistory(history) {
  $('history-list').replaceChildren(...history.map((job) => {
    const li = el('li');
    li.append(
      el('span', `status ${job.status}`, job.status === 'done' ? '✓' : job.status === 'error' ? '✕' : '–'),
      el('span', 'grow', job.label),
      el('span', 'muted', fmtDuration(job.elapsed) || fmtTime(job.ended_at)),
    );
    li.title = job.message || job.error || '';
    return li;
  }));
}

function renderActions(actions, groups) {
  state.actions = actions;
  state.actionGroups = groups || [];

  // Sidebar quick actions: a curated shortlist.
  const primary = ['full-sync', 'process-new', 'rebuild-gallery', 'quick-refresh'];
  $('action-stack').replaceChildren(...primary.map((id) => {
    const spec = actions.find((a) => a.id === id);
    if (!spec) return el('span');
    const button = el('button', `btn${spec.primary ? ' primary' : ''}`, spec.label);
    button.type = 'button';
    button.title = spec.help || spec.desc || '';
    button.onclick = () => runAction(id);
    return button;
  }));

  // Manage view: grouped, with descriptions.
  const container = $('manage-actions');
  container.replaceChildren();
  (groups || []).forEach((group) => {
    const inGroup = actions.filter((a) => a.group === group.id);
    if (!inGroup.length) return;
    container.append(el('h3', 'op-group-title', group.label));
    const grid = el('div', 'op-grid');
    inGroup.forEach((spec) => {
      const card = el('button', `op-card${spec.primary ? ' primary' : ''}`);
      card.type = 'button';
      // Hover shows the long plain-English explanation, not the short blurb
      // that is already printed on the card.
      card.title = spec.help || spec.desc || spec.label;
      card.onclick = () => runAction(spec.id);
      card.append(el('span', 'op-label', spec.label));
      if (spec.desc) card.append(el('span', 'op-desc', spec.desc));
      grid.append(card);
    });
    container.append(grid);
  });
}


async function runAction(id) {
  try {
    await apiPost(`/api/run/${id}`);
  } catch (err) {
    toast(`Could not start: ${err.message}`, 'error');
  }
}

function renderProviders(providers) {
  $('provider-list').replaceChildren(...providers.map((provider) => {
    const block = el('div', 'provider-block');

    const row = el('div', 'provider-row');
    const dot = el('span', 'provider-dot');
    dot.style.background = provider.accent;
    row.append(dot, el('span', null, provider.display_name));
    const pill = el('span', `pill${provider.configured ? ' on' : ''}`,
      provider.configured ? `key: ${provider.key_source}` : 'no key');
    row.append(pill);
    block.append(row);

    // A key from the environment wins on read, so editing here would be a lie.
    if (provider.key_source === 'env') {
      block.append(el('div', 'muted small',
        `Set by ATLAS_${provider.id.toUpperCase()}_API_KEY — unset it to edit here.`));
      return block;
    }

    const editRow = el('div', 'key-row');
    const input = el('input');
    input.type = 'password';
    input.autocomplete = 'off';
    input.spellcheck = false;
    input.placeholder = provider.configured
      ? 'Paste a new key to replace…' : `Paste your ${provider.display_name} key…`;
    input.setAttribute('aria-label', `${provider.display_name} API key`);

    const save = el('button', 'btn tiny primary', provider.configured ? 'Replace' : 'Save');
    save.type = 'button';
    save.onclick = async () => {
      const key = input.value.trim();
      if (!key) { toast('Paste a key first', 'error'); return; }
      try {
        await apiPost(`/api/settings/key/${provider.id}`, { key });
        input.value = '';
        toast(`${provider.display_name} key saved`, 'ok');
        refreshManage();
      } catch (err) { toast(err.message, 'error'); }
    };
    editRow.append(input, save);

    if (provider.configured) {
      const clear = el('button', 'btn tiny ghost', 'Clear');
      clear.type = 'button';
      clear.onclick = async () => {
        if (!confirm(`Remove the saved ${provider.display_name} key?`)) return;
        try {
          await apiPost(`/api/settings/key/${provider.id}`, { key: '' });
          toast(`${provider.display_name} key removed`, 'ok');
          refreshManage();
        } catch (err) { toast(err.message, 'error'); }
      };
      editRow.append(clear);
    }
    block.append(editRow);
    return block;
  }));
}

/** Library-folder picker: browse or paste, preview, then confirm. */
function bindLibraryRoot() {
  const pathInput = $('root-path');
  const preview = $('root-preview');
  const apply = $('root-apply');

  const offer = (path, models, bytes) => {
    pathInput.value = path;
    preview.textContent = (models != null)
      ? `${models} model(s), ${fmtBytes(bytes || 0)} found here.`
      : 'Press "Use this folder" to switch.';
    apply.hidden = !path;
  };

  $('root-browse').onclick = async () => {
    try {
      const res = await apiPost('/api/settings/browse-root', {});
      if (res.path) offer(res.path, res.models, res.bytes);
      else toast('No folder chosen — you can paste a path instead', 'ok');
    } catch {
      toast('Folder picker unavailable — paste a path instead', 'ok');
      pathInput.focus();
    }
  };
  pathInput.oninput = () => {
    apply.hidden = !pathInput.value.trim();
    preview.textContent = '';
  };
  apply.onclick = async () => {
    const path = pathInput.value.trim();
    if (!path) return;
    if (!confirm(`Switch the library to:\n${path}\n\n` +
                 `The new folder gets its own index and is scanned fresh. ` +
                 `Nothing in either folder is moved or deleted.`)) return;
    try {
      await apiPost('/api/settings', { library_root: path });
      toast('Library folder changed — scanning…', 'ok');
      await apiPost('/api/run/quick-refresh');
      pathInput.value = '';
      apply.hidden = true;
      preview.textContent = '';
      refreshManage();
    } catch (err) { toast(err.message, 'error'); }
  };
}

function renderSettings(settings) {
  const editable = [
    ['hash_workers', 'Hash workers', 'number'],
    ['provider_rate_limit', 'Requests / second', 'number'],
    ['thumbnail_size', 'Thumbnail size (px)', 'number'],
    ['page_size', 'Page size', 'number'],
    ['default_weight', 'Default LoRA weight', 'number'],
  ];
  const nodes = editable.map(([key, label, type]) => {
    const wrap = el('div', 'setting');
    const id = `set-${key}`;
    const labelNode = el('label', null, label);
    labelNode.htmlFor = id;
    const input = el('input');
    input.id = id;
    input.type = type;
    input.step = 'any';
    input.value = settings[key] ?? '';
    input.onchange = async () => {
      try {
        await apiPost('/api/settings', { [key]: Number(input.value) });
        toast(`${label} saved`, 'ok');
      } catch (err) { toast(err.message, 'error'); }
    };
    wrap.append(labelNode, input);
    return wrap;
  });

  [['dry_run', 'Dry run (no writes)'], ['auto_scan_on_start', 'Scan on start'],
   ['open_browser', 'Open browser on launch']].forEach(([key, label]) => {
    const wrap = el('div', 'setting');
    const checkbox = el('input');
    checkbox.type = 'checkbox';
    checkbox.checked = !!settings[key];
    checkbox.onchange = async () => {
      try { await apiPost('/api/settings', { [key]: checkbox.checked }); }
      catch (err) { toast(err.message, 'error'); }
    };
    const wrapper = el('label', 'check');
    wrapper.append(checkbox, document.createTextNode(' ' + label));
    wrap.append(wrapper);
    nodes.push(wrap);
  });

  $('settings-grid').replaceChildren(...nodes);
}

async function refreshManage() {
  try {
    const [jobs, backups, settings] = await Promise.all([
      apiGet('/api/jobs'), apiGet('/api/backups'), apiGet('/api/settings'),
    ]);
    renderQueue(jobs);
    renderHistory(jobs.history || []);
    renderProviders(settings.providers);
    renderSettings(settings.settings);
    renderBackups(backups.backups);
    state.backupFolder = backups.folder || '';
    const bf = $('backup-folder-path');
    if (bf) bf.textContent = state.backupFolder || '—';
    // Read the live root from settings, not the cached bootstrap — it may have
    // been changed from this very panel.
    const root = $('settings-root');
    if (root) {
      root.textContent = settings.settings.library_root
        || (state.bootstrap && state.bootstrap.library_root) || '—';
    }
  } catch (err) {
    toast(`Manage view: ${err.message}`, 'error');
  }
}

function renderBackups(backups) {
  const list = $('backup-list');
  if (!backups.length) {
    list.replaceChildren(el('li', 'muted', 'No backups yet.'));
    return;
  }
  list.replaceChildren(...backups.map((entry) => {
    const li = el('li');
    const meta = el('div', 'grow');
    meta.append(el('div', null, entry.name));
    meta.append(el('div', 'muted small',
      `${fmtBytes(entry.bytes)} · ${new Date(entry.modified * 1000).toLocaleString()}`));
    li.append(meta);
    // Clicking "Use" loads it into the restore path field, so restore uses the
    // shared options below rather than a bare confirm().
    const use = el('button', 'btn tiny', 'Use');
    use.type = 'button';
    use.onclick = () => {
      $('backup-restore-path').value = entry.path;
      $('backup-restore-path').scrollIntoView({ behavior: 'smooth', block: 'center' });
      toast('Loaded into restore — set options and click Restore', 'ok');
    };
    li.append(use);
    return li;
  }));
}

async function doExportBackup() {
  try {
    await apiPost('/api/backups/export', {
      favourites_only: $('backup-favs').checked,
      include_previews: $('backup-previews').checked,
    });
    toast('Backup started', 'ok');
  } catch (err) { toast(err.message, 'error'); }
}

async function doRestoreBackup() {
  const path = $('backup-restore-path').value.trim();
  if (!path) { toast('Choose or paste a backup file first', 'error'); return; }
  const download = $('restore-download').checked;
  const warning = download
    ? '\n\nMissing models will be RE-DOWNLOADED from their provider. This can ' +
      'take a long time and use a lot of bandwidth.'
    : '\n\nOnly models already in your library are updated. Tick ' +
      '"Re-download missing models" to rebuild ones that are gone.';
  if (!confirm(`Restore from:\n${path}\n\nYour ratings, notes, tags, collections and ` +
               `folder layout are merged into the current library by file hash.` + warning)) return;
  try {
    await apiPost('/api/backups/restore', {
      path,
      restore_metadata: $('restore-metadata').checked,
      restore_previews: $('restore-previews').checked,
      download_missing: download,
    });
    toast(download ? 'Restore started — downloading missing models' : 'Restore started', 'ok');
  } catch (err) { toast(err.message, 'error'); }
}

async function pickBackupFile() {
  // The server can browse for a file via a native dialog if available;
  // otherwise fall back to manual path entry.
  try {
    const res = await apiPost('/api/backups/browse', {});
    if (res && res.path) $('backup-restore-path').value = res.path;
  } catch {
    toast('Type or paste the .llmb path in the box below', 'ok');
    $('backup-restore-path').focus();
  }
}

/* ------------------------------------------------------------------ views */

function showView(name) {
  state.view = name;
  $('view-gallery').hidden = name !== 'gallery';
  $('view-manage').hidden = name !== 'manage';
  $('nav-gallery').setAttribute('aria-pressed', String(name === 'gallery'));
  $('nav-manage').setAttribute('aria-pressed', String(name === 'manage'));
  // The result bar exists only on the gallery, so the scroller's top inset
  // changes with the view.
  setChromeMetrics();
  if (name === 'manage') {
    refreshManage();
  } else {
    // Returning to the gallery: the grid may have been laid out while hidden
    // (width 0 → 1 column) if the sidebar was toggled on another view. Rebuild
    // from scratch now that it's visible and has a real width.
    state.rendered.forEach((node) => node.remove());
    state.rendered.clear();
    layout();
  }
}

/* ------------------------------------------------------------------ theme */

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem('atlas-theme', theme);
  const button = $('theme-toggle');
  button.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`);
  button.querySelector('use').setAttribute('href', theme === 'dark' ? '#i-moon' : '#i-sun');
}

/** Dark/light and solid/glass are independent axes.
 *
 *  `solid` is the default and is not a stripped-down glass: every rule in the
 *  Transparent block is scoped to [data-surface="glass"], so on solid none of
 *  them match and there is no backdrop to re-blur. Glass costs GPU work that a
 *  5000-model library may not want to spend, which is exactly why it is a
 *  choice rather than the only option.
 */
function applySurface(surface) {
  const glass = surface === 'glass';
  document.documentElement.dataset.surface = glass ? 'glass' : 'solid';
  localStorage.setItem('atlas-surface', glass ? 'glass' : 'solid');
  const button = $('surface-toggle');
  if (button) {
    button.setAttribute('aria-pressed', String(glass));
    button.title = glass
      ? 'Transparent — surfaces you can see through. Click for the lighter solid look.'
      : 'Solid surfaces — the lightest option. Click for transparent.';
    button.setAttribute('aria-label', glass ? 'Switch to solid surfaces' : 'Switch to transparent surfaces');
  }
  // The glass layout overlays the chrome, so the scroller's top inset only
  // applies on that surface — remeasure whenever it changes.
  setChromeMetrics();
  if (state.view === 'gallery') layout();
}

/* ------------------------------------------------------------------- init */

/** Keep the "/" hint out of the way once the box has something in it. */
function syncSearchHint() {
  $('search').closest('.search-wrap')
    .classList.toggle('has-text', $('search').value.length > 0);
}

/** The search box doubles as the artist filter: typed text is a normal
 *  search unless it starts with "artist:", in which case the rest of the
 *  text is an exact creator match instead — the same syntax the artist-name
 *  button on a card writes in for you. One field, one thing to clear. */
function parseSearchInput(raw) {
  const trimmed = (raw || '').trim();
  const match = /^artist:\s*(.+)$/i.exec(trimmed);
  return match ? { q: '', creator: match[1].trim() } : { q: trimmed, creator: '' };
}

function bindFilters() {
  let debounce;
  $('search').addEventListener('input', (event) => {
    syncSearchHint();
    clearTimeout(debounce);
    debounce = setTimeout(() => {
      const parsed = parseSearchInput(event.target.value);
      state.filters.q = parsed.q;
      state.filters.creator = parsed.creator;
      resetResults();
    }, 220);
  });

  const bind = (id, key, transform = (v) => v) => {
    $(id).addEventListener('change', (event) => {
      const target = event.target;
      state.filters[key] = transform(target.type === 'checkbox' ? target.checked : target.value);
      resetResults();
    });
  };
  // Sort now lives in the top bar.
  bind('topbar-sort-select', 'sort');
  bind('filter-base', 'base_model');
  bind('filter-fav', 'favorite');
  bind('filter-rated', 'rated');
  bind('filter-notes', 'has_notes');
  bind('filter-triggers', 'has_triggers');
  bind('filter-nometa', 'meta_state', (checked) => (checked ? 'missing' : ''));
  bind('filter-nopreview', 'missing_preview');
  bind('filter-sfw', 'nsfw', (checked) => (checked ? 'hide' : ''));

  // Category cascade: choosing a category resets the sub-category and reveals
  // its options; choosing a sub-category narrows further.
  $('filter-category').addEventListener('change', (event) => {
    const option = event.target.selectedOptions[0];
    state.filters.uncategorised = option?.dataset.uncategorised === '1';
    state.filters.category = state.filters.uncategorised ? UNCATEGORISED : event.target.value;
    state.filters.subcategory = '';
    state.filters.collection = '';
    paintSubcategories();
    resetResults();
  });
  $('filter-subcategory').addEventListener('change', (event) => {
    state.filters.subcategory = event.target.value;
    resetResults();
  });

  $('clear-filters').onclick = () => {
    state.filters = { ...EMPTY_FILTERS, sort: state.filters.sort };
    $('search').value = '';
    syncSearchHint();
    $('filter-base').value = '';
    $('filter-category').value = '';
    paintSubcategories();
    ['filter-fav', 'filter-rated', 'filter-notes', 'filter-triggers',
     'filter-nometa', 'filter-nopreview', 'filter-sfw']
      .forEach((id) => { $(id).checked = false; });
    resetResults();
  };
}

function bindChrome() {
  $('nav-gallery').onclick = () => showView('gallery');
  $('nav-manage').onclick = () => showView('manage');
  $('theme-toggle').onclick = () =>
    applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
  $('surface-toggle').onclick = () =>
    applySurface(document.documentElement.dataset.surface === 'glass' ? 'solid' : 'glass');

  $('modal-close').onclick = closeDetail;
  $('modal-backdrop').onclick = closeDetail;

  // Collapsible sidebar. Persisted so it stays how the user left it.
  // No manual re-layout here: the width animation drives the ResizeObserver,
  // which rebuilds the grid as the sidebar settles.
  const setSidebar = (open) => {
    document.body.classList.toggle('sidebar-collapsed', !open);
    $('sidebar-toggle').setAttribute('aria-pressed', String(open));
    localStorage.setItem('atlas-sidebar', open ? 'open' : 'collapsed');
  };
  $('sidebar-toggle').onclick = () =>
    setSidebar(document.body.classList.contains('sidebar-collapsed'));
  if (localStorage.getItem('atlas-sidebar') === 'collapsed') {
    document.body.classList.add('sidebar-collapsed');
    $('sidebar-toggle').setAttribute('aria-pressed', 'false');
  }

  $('console-clear').onclick = () => $('console').replaceChildren();
  $('console-follow').onchange = (event) => { state.followConsole = event.target.checked; };

  $('backup-export').onclick = doExportBackup;
  $('backup-restore-run').onclick = doRestoreBackup;
  $('backup-restore-pick').onclick = pickBackupFile;
  $('delete-category').onclick = deleteSelectedCategory;
  $('rename-category').onclick = renameSelectedCategory;
  $('backup-folder-open').onclick = async () => {
    try { await apiPost('/api/open-folder', { path: state.backupFolder }); }
    catch (err) { toast(err.message, 'error'); }
  };
  $('backup-folder-pick').onclick = async () => {
    try {
      const res = await apiPost('/api/backups/browse-folder', {});
      if (!res.path) return;
      await apiPost('/api/settings', { backup_folder: res.path });
      toast('Backup folder changed', 'ok');
      refreshManage();
    } catch (err) { toast(err.message, 'error'); }
  };

  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !$('detail-modal').hidden) closeDetail();
    const typing = document.activeElement.tagName === 'INPUT'
      || document.activeElement.tagName === 'TEXTAREA';
    if (event.key === '/' && !typing) {
      event.preventDefault();
      $('search').focus();
    }
    if (event.key === '\\' && !typing) {
      event.preventDefault();
      $('sidebar-toggle').click();
    }
  });

  $('scroller').addEventListener('scroll', scheduleRender, { passive: true });

  // A hidden tab does not run requestAnimationFrame at all. Without this the
  // pending flag would never clear and the grid would stay frozen even after
  // the tab came back — so re-arm and repaint on the way in.
  document.addEventListener('visibilitychange', () => {
    renderScheduled = false;
    if (document.visibilityState === 'visible') render();
  });

  // Width changes — window resize, sidebar toggle, first paint — are all
  // handled by the ResizeObserver rather than a separate resize listener.
  observeGridWidth();
  observeTopbarBalance();
}

/** Keep the search box centred in the whole top bar, not just in whatever
 *  room is left between the brand and the nav. The sidebar toggle + brand
 *  block is wider than the nav block on the other side, so without this the
 *  search box sits visibly right of true centre. A single measured spacer
 *  before the nav evens the two flanks out; `.topbar-centre` already centres
 *  its own contents, so once the flanks match, so does the search box. */
function balanceTopbar() {
  const spacer = $('topbar-edge-spacer');
  const topbar = document.querySelector('.topbar');
  const centre = document.querySelector('.topbar-centre');
  if (!spacer || !topbar || !centre) return;
  spacer.style.width = '0px';           // reset first — the diff below must
  const tb = topbar.getBoundingClientRect();  // measure the unpadded gap
  const c = centre.getBoundingClientRect();
  if (!tb.width || !c.width) return;
  const diff = (c.left - tb.left) - (tb.right - c.right);
  spacer.style.width = `${Math.max(0, Math.round(diff))}px`;
}

/** Publish the chrome's measured heights as CSS variables.
 *
 * The top bar and status strip are translucent and overlay the content so it
 * scrolls underneath them — that movement behind the blur is the whole point of
 * the material. Which means the scroller has to reserve their height as top
 * padding, and the bar wraps to two and three rows as the window narrows, so
 * the numbers cannot be hard-coded. The CSS carries fallbacks for the first
 * frame, before this has run.
 */
function setChromeMetrics() {
  const px = (el) => (el ? Math.round(el.getBoundingClientRect().height) : 0);
  const topbar = px(document.querySelector('.topbar'));
  const status = px(document.querySelector('.status-strip'));
  const root = document.documentElement.style;
  if (topbar) root.setProperty('--topbar-h', `${topbar}px`);
  if (topbar) root.setProperty('--chrome-h', `${topbar + status}px`);
}

/** Re-run the balance and the chrome measurements whenever the bar's width
 *  changes (window resize) or the brand's own width changes (the "N models"
 *  count grows a digit). */
function observeTopbarBalance() {
  if (typeof ResizeObserver === 'undefined') return;
  const observer = new ResizeObserver(() => { balanceTopbar(); setChromeMetrics(); });
  observer.observe(document.querySelector('.topbar'));
  observer.observe(document.querySelector('.status-strip'));
  observer.observe($('brand-count'));
}

async function init() {
  applyTheme(localStorage.getItem('atlas-theme')
    || (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'));
  // Solid unless asked otherwise: the lighter surface is the one a large
  // library should get by default.
  applySurface(localStorage.getItem('atlas-surface') || 'solid');

  // The launch URL carries the session token; the server has now set it as a
  // cookie, so drop it from the address bar before it gets copied anywhere.
  if (location.search.includes('k=')) {
    history.replaceState(null, '', location.pathname);
  }

  bindFilters();
  bindChrome();
  bindLibraryRoot();

  try {
    const boot = await apiGet('/api/bootstrap');
    state.bootstrap = boot;
    state.pageSize = boot.settings.page_size || 120;
    state.filters.sort = boot.settings.default_sort || 'name';

    const SORT_LABELS = {
      name: 'Name (A–Z)', name_desc: 'Name (Z–A)', added: 'Recently added',
      oldest: 'Oldest first', size: 'Largest first', rating: 'Highest rated',
      used: 'Most used', base: 'Base model',
    };
    fillSelect($('topbar-sort-select'),
      boot.sorts.map((s) => ({ value: s, label: SORT_LABELS[s] || s })), 'Name (A–Z)');
    // The empty-label option maps to '' which the server treats as default name.
    $('topbar-sort-select').options[0].value = 'name';
    $('topbar-sort-select').value = state.filters.sort;

    renderStats(boot.stats);
    balanceTopbar();      // brand-count just got its first real text — settle once
    setChromeMetrics();   // and the bar's final height is now known
    renderActions(boot.actions, boot.action_groups);
    renderProviders(boot.providers);
    renderSettings(boot.settings);
    renderHistory(boot.history || []);
    fillSelect($('filter-base'), boot.facets.base_models.map(
      (b) => ({ value: b.value, label: `${b.value} (${b.count})` })), 'All');
    applyCategories(boot.facets.categories || [], boot.facets.uncategorised ?? 0);

    applyJobState({ jobs: boot.jobs });
    connectEvents();
    resetResults();
  } catch (err) {
    toast(`Startup failed: ${err.message}`, 'error', true);
    setStatus('err', err.message);
  }
}

init();
