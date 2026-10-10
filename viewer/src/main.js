// Start the page: read the catalog, open the clip the address names or the first one, and connect the
// controls, the keys and the pointer to the actions of app.js. The address keeps the open clip, frame,
// mode and followed region, so a view can be shared as a link.
import { loadCatalog } from './data.js';
import { initStage, invalidate, resetView, startRenderLoop } from './stage.js';
import {
  BOTH, cycleCloudShow, onChange, onError, openClip, pickNode, refitPanels, setCloudShow, setFocusNode, setFocusStage, setGraph3D,
  setHover, setIsolate, setMapBig, setPlaying, setRegionsMenu, setStatus, setStripCellHeight, setStripPhoto, setWorldView,
  seekFrame, setTrack, state, stepFrame, stripCellHeight, switchMode, syncControls, toggleRegions,
} from './app.js';
import { KEYS, CREDITS } from './help.js';
import { esc } from './format.js';

const $ = (id) => document.getElementById(id);

// ── Errors ──────────────────────────────────────────────────────────────────

let toastTimer = null;
function showError(msg) {
  console.error(msg);
  $('toast-text').textContent = msg;
  $('toast').hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $('toast').hidden = true; }, 12000);
}

// ── The address ─────────────────────────────────────────────────────────────

const NODE_KEY = /^[a-z0-9_]+:\d+$/;

/**
 * Write the open clip, frame, mode, track and followed region into the address, without a history entry.
 * The track is `both` while both tracks of the hierarchy are shown.
 */
function writeAddress() {
  if (!state.clip) return;
  const u = new URL(location.href);
  u.search = '';
  u.searchParams.set('clip', state.clip.id);
  if (state.both) u.searchParams.set('track', BOTH);
  else if (state.track) u.searchParams.set('track', state.track);
  if (state.frame) u.searchParams.set('frame', String(state.frame));
  if (state.mode === 'world') u.searchParams.set('mode', 'world');
  if (state.focusKey) u.searchParams.set('focus', state.focusKey);
  history.replaceState(null, '', u);
}

// ── Dialogs ─────────────────────────────────────────────────────────────────

function fillDialogs() {
  $('keys-body').innerHTML = KEYS.map((k) => (k.group
    ? `<div class="grp">${esc(k.group)}</div>`
    : `<span class="k">${esc(k.keys)}</span><span>${esc(k.what)}</span>`)).join('');
  $('about-body').innerHTML = CREDITS.map((g) => `<h3>${esc(g.group)}</h3><ol>${g.items.map((r) => `
    <li><b>${esc(r.name)}</b> <span class="ab-use">${esc(r.use)}</span><br>${esc(r.cite)}
      <a href="${esc(r.url)}" target="_blank" rel="noopener noreferrer">${esc(r.url.replace(/^https:\/\//, ''))} ↗</a></li>`).join('')}</ol>`).join('');
}

function openDialog(id) {
  for (const d of ['keys-overlay', 'about-overlay']) $(d).hidden = d !== id;
  $(id).querySelector('button')?.focus();
}

const dialogOpen = () => !$('keys-overlay').hidden || !$('about-overlay').hidden;
const closeDialogs = () => { $('keys-overlay').hidden = true; $('about-overlay').hidden = true; };

// ── Wiring ──────────────────────────────────────────────────────────────────

function fillPicker(catalog) {
  const sel = $('clip-select');
  sel.innerHTML = '';
  const groups = new Map();
  for (const c of catalog.clips) {
    const g = `${c.dataset_name} · ${c.procedure}`;
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(c);
  }
  for (const [label, clips] of groups) {
    const og = document.createElement('optgroup');
    og.label = label;
    for (const c of clips) {
      const o = document.createElement('option');
      o.value = c.id;
      o.textContent = `${c.video} · ${c.id.split('__').pop()} · ${c.n_frames} frames`;
      og.appendChild(o);
    }
    sel.appendChild(og);
  }
}

/** Open a clip, then the address's frame, mode, track and followed region, unless another clip is asked for. */
async function open(id, { frame = 0, mode = 'frame', track = null, focus = null } = {}) {
  $('clip-select').value = id;
  if (!await openClip(id)) {
    // The clip on screen is still the last one; the picker says so again.
    if (state.clip) $('clip-select').value = state.clip.id;
    return;
  }
  const seq = state.clipSeq;
  const current = () => state.clipSeq === seq;
  if (track === BOTH || state.clip.trackById.has(track)) setTrack(track);
  if (frame > 0 && frame < state.clip.n_frames) await seekFrame(frame);
  if (!current()) return;
  if (mode === 'world') await switchMode('world');
  if (!current()) return;
  if (focus && NODE_KEY.test(focus) && state.clip.trackById.has(focus.split(':')[0])) await setFocusNode(focus);
}

function wire() {
  $('clip-select').addEventListener('change', () => {
    // The picker lets go of the keys, so the arrows step the frames again.
    $('clip-select').blur();
    open($('clip-select').value);
  });
  for (const b of document.querySelectorAll('#mode-btns button')) {
    b.addEventListener('click', () => switchMode(b.dataset.mode));
  }
  for (const b of document.querySelectorAll('#worldview-btns button')) b.addEventListener('click', () => setWorldView(b.dataset.wv));
  for (const b of document.querySelectorAll('#show-btns button')) b.addEventListener('click', () => setCloudShow(b.dataset.show));
  $('btn-graph3d').addEventListener('click', () => setGraph3D(!state.graph3d));
  $('btn-regions').addEventListener('click', (ev) => { ev.stopPropagation(); setRegionsMenu($('regions-pop').hidden); });
  document.addEventListener('click', (ev) => {
    if (!$('regions-pop').hidden && !$('regions-menu').contains(ev.target)) setRegionsMenu(false);
  });
  $('btn-reset-view').addEventListener('click', () => resetView(state.clip?.frames[state.frame]));
  $('btn-keys').addEventListener('click', () => openDialog('keys-overlay'));
  $('btn-about').addEventListener('click', () => openDialog('about-overlay'));
  for (const id of ['keys-close', 'about-close']) $(id).addEventListener('click', closeDialogs);
  for (const id of ['keys-overlay', 'about-overlay']) {
    $(id).addEventListener('click', (ev) => { if (ev.target === ev.currentTarget) closeDialogs(); });
  }
  $('toast-close').addEventListener('click', () => { $('toast').hidden = true; });

  // Dragging the slider moves the label only; letting go loads, so a drag across a clip fetches one frame.
  const slider = $('frame-slider');
  slider.addEventListener('input', () => {
    if (state.mode === 'world') { seekFrame(Number(slider.value)); return; }
    $('frame-label').textContent = `${slider.value} / ${state.clip.n_frames - 1}`;
  });
  slider.addEventListener('change', () => seekFrame(Number(slider.value)));
  $('btn-prev').addEventListener('click', () => stepFrame(-1));
  $('btn-next').addEventListener('click', () => stepFrame(1));
  $('btn-play').addEventListener('click', () => setPlaying(!state.playing));

  for (const b of document.querySelectorAll('#fb-steps button')) {
    b.addEventListener('click', () => setFocusStage(Number(b.dataset.stage)));
  }
  $('fb-clear').addEventListener('click', () => setFocusNode(null));
  $('nf-close').addEventListener('click', () => setFocusNode(null));
  $('btn-strip-photo').addEventListener('click', () => setStripPhoto(!state.stripPhoto));
  $('btn-fuse').addEventListener('click', () => setMapBig(true));
  $('btn-map-big').addEventListener('click', () => setMapBig(!state.mapBig));
  $('mb-close').addEventListener('click', () => setMapBig(false));

  // A node clicked in a panel is followed; clicked again, it is let go.
  const panels = [$('nodelink-body'), $('strip'), $('mb-body')];
  for (const el of panels) {
    el.addEventListener('click', (ev) => {
      const key = ev.target.closest?.('[data-node-key]')?.getAttribute('data-node-key');
      if (!key) return;
      ev.stopPropagation();
      setFocusNode(key === state.focusKey ? null : key);
    }, true);
    el.addEventListener('pointermove', (ev) => setHover(ev.target.closest?.('[data-node-key]')?.getAttribute('data-node-key') ?? null));
    el.addEventListener('pointerleave', () => setHover(null));
  }

  const canvas = $('gl');
  let raf = null, lastMove = null, downAt = null;
  canvas.addEventListener('pointermove', (ev) => {
    lastMove = ev;
    if (raf) return;
    raf = requestAnimationFrame(() => {
      raf = null;
      const key = pickNode(canvas, lastMove);
      canvas.style.cursor = key ? 'pointer' : '';
      setHover(key);
    });
  });
  canvas.addEventListener('pointerleave', () => setHover(null));
  canvas.addEventListener('pointerdown', (ev) => { downAt = [ev.clientX, ev.clientY]; });
  canvas.addEventListener('pointerup', (ev) => {
    // A drag turns the view; only a click picks.
    if (!downAt || Math.hypot(ev.clientX - downAt[0], ev.clientY - downAt[1]) > 5) { downAt = null; return; }
    downAt = null;
    const key = pickNode(canvas, ev);
    if (key) setFocusNode(key === state.focusKey ? null : key);
  });

  document.addEventListener('keydown', onKey);
  initResizers();
  const ro = new ResizeObserver(() => requestAnimationFrame(refitPanels));
  for (const id of ['nodelink-body', 'mb-body', 'nf-body']) ro.observe($(id));
  // The 3D view is drawn again at its new size whenever a splitter or the window changes it.
  new ResizeObserver(invalidate).observe($('viewport'));
}

function onKey(ev) {
  if (ev.defaultPrevented || ev.metaKey || ev.ctrlKey || ev.altKey) return;
  if (ev.target.closest?.('input:not([type=range]), select, textarea')) return;
  // Space and Enter on a focused control press that control, and nothing else.
  if ((ev.key === ' ' || ev.key === 'Enter') && ev.target.closest?.('button, a, [role=button]')) return;
  // A held key repeats; each repeat of W, G or I would rebuild every cloud again.
  if (ev.repeat) { ev.preventDefault(); return; }
  if (dialogOpen()) {
    if (ev.key === 'Escape' || ev.key === '?') { closeDialogs(); ev.preventDefault(); }
    return;
  }
  if (!state.clip) return;
  const k = ev.key.length === 1 ? ev.key.toLowerCase() : ev.key;
  const actions = {
    ArrowLeft: () => stepFrame(-1),
    ArrowRight: () => stepFrame(1),
    ' ': () => setPlaying(!state.playing),
    w: () => switchMode(state.mode === 'world' ? 'frame' : 'world'),
    g: () => setGraph3D(!state.graph3d),
    r: () => resetView(state.clip.frames[state.frame]),
    p: () => state.mode === 'world' && setStripPhoto(!state.stripPhoto),
    i: () => cycleCloudShow(),
    s: () => toggleRegions(state.both ? BOTH : state.track),
    o: () => state.focusKey && state.mode === 'world' && setIsolate(!state.isolate),
    '?': () => openDialog('keys-overlay'),
    Escape: () => (state.mapBig ? setMapBig(false) : state.focusKey && setFocusNode(null)),
  };
  if (!actions[k]) return;
  ev.preventDefault();
  actions[k]();
}

/** The side splitter sets the side panel's width; the dock splitter sets the strip's cell height. */
function initResizers() {
  const drag = (handle, begin, move) => handle.addEventListener('pointerdown', (ev) => {
    ev.preventDefault();
    handle.setPointerCapture(ev.pointerId);
    handle.classList.add('dragging');
    const start = begin(ev);
    const onMove = (e) => move(e, start);
    const end = () => {
      handle.classList.remove('dragging');
      handle.removeEventListener('pointermove', onMove);
      handle.removeEventListener('pointerup', end);
      handle.removeEventListener('pointercancel', end);
    };
    handle.addEventListener('pointermove', onMove);
    handle.addEventListener('pointerup', end);
    handle.addEventListener('pointercancel', end);
  });
  drag($('split-side'), () => null, (e) => {
    const w = Math.min(window.innerWidth * 0.6, Math.max(240, e.clientX - 10));
    document.documentElement.style.setProperty('--side-w', `${w}px`);
    invalidate();
  });
  drag($('split-dock'), (ev) => ({ y: ev.clientY, h: stripCellHeight() }), (e, start) => {
    setStripCellHeight(start.h + (start.y - e.clientY));
  });
}

// ── Start ───────────────────────────────────────────────────────────────────

async function boot() {
  onError(showError);
  onChange(writeAddress);
  fillDialogs();
  if (!initStage($('gl'), $('viewport'))) {
    $('viewport').innerHTML = '<div class="no-webgl">This browser cannot draw WebGL, which the 3D view needs.</div>';
    return;
  }
  wire();
  startRenderLoop();
  let catalog;
  try {
    catalog = await loadCatalog();
  } catch (err) {
    setStatus('No catalog');
    showError(`The clips could not be listed: ${err.message}`);
    return;
  }
  if (!catalog.clips.length) { showError('The catalog lists no clips.'); return; }
  fillPicker(catalog);
  syncControls();
  const q = new URLSearchParams(location.search);
  const wanted = q.get('clip');
  const known = catalog.clips.some((c) => c.id === wanted);
  if (wanted && !known) showError(`No clip ${wanted} in this catalog; the first clip is open instead.`);
  await open(known ? wanted : catalog.clips[0].id, {
    frame: Number.parseInt(q.get('frame') ?? '0', 10) || 0,
    mode: q.get('mode') === 'world' ? 'world' : 'frame',
    track: q.get('track'),
    focus: q.get('focus'),
  });
}

boot().catch((err) => showError(`The page did not start: ${err.message}`));
