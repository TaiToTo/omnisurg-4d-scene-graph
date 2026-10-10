// Open a clip of the catalog and show its 4D scene graph: the point cloud of
// each frame, the graph in the camera's view, World mode's stack of frames,
// the band of regions through time, and one region followed through time.
//
// The URL names what is on screen, so a view can be linked: `?clip=<path>`,
// `track=<id>|both`, `mode=world`, `frame=<n>`, `focus=<track>:<id>`,
// `view=overlay`, `worldframes=<n>`, `seg=1` and `graph=1`.
import * as THREE from 'three';
import { state, trackInfo, orderTracks, onStage, cloudsOnScreen, focusFamily } from './state.js';
import { canvas, clearGroup, invalidate, makeLabelSprite, measureSceneScale, resetView, setStatus, startRenderLoop, sz } from './scene.js';
import { clearCaches, fetchJSON, glbURL, loadCatalog, loadGLB, manifestURL, overlayURL, prefetchGLB, rgbURL, temporalURL } from './data.js';
import { groupClips } from './lib/catalog.js';
import { adaptOverlays, manifestForSource } from './lib/overlays.js';
import { cameraAxes, edgeAxes } from './lib/edge_axes.js';
import { containmentPartners, topPartners } from './lib/relations.js';
import { esc } from './lib/format.js';
import { MAX_WORLD_FRAMES, nearestStacked, worldIndices } from './lib/sampling.js';
import { applyRegionHighlight, applySegBlend, segToCanvas } from './cloud.js';
import { enterWorld, exitWorld, focusWorldFrame } from './world.js';
import { ellipsesOf, renderFrameNodeLink, renderWorldNodeLink, setHoverKey } from './nodelink.js';
import { renderStrip, renderTrackingStrip, updateStripFocus } from './strip.js';
import { renderNodeFocus, updateNodeFocusCursor } from './focus_band.js';
import { buildGraph3D, pickKey, setHover3D } from './graph3d.js';
import { renderHelp } from './help.js';

const $ = (id) => document.getElementById(id);
const el = {
  clip: $('clip-select'), slider: $('frame-slider'), frameLabel: $('frame-label'), play: $('btn-play'),
  nodeLink: $('nodelink-body'), nodeLinkTitle: $('nodelink-title'), mapBig: $('mb-body'),
  strip: $('strip'), nodeFocus: $('node-focus'), nfTitle: $('nf-title'), nfHint: $('nf-hint'), nfBody: $('nf-body'),
  thumbRgb: $('thumb-rgb'), thumbSeg: $('thumb-seg'), thumbSeg2: $('thumb-seg2'),
};
const hoverPanels = [el.nodeLink, el.strip, el.mapBig];

const PLAY_MS = 700;
const HL_FOCUS = [0.0, 0.52, 0.64];
const HL_PARTNER = [0.85, 0.42, 0.0];

// ── Notices ──────────────────────────────────────────────────────────────────

/** Show a failure where a reader will see it, not only on the status line. */
function showError(msg) {
  $('toast-text').textContent = msg;
  $('toast').hidden = false;
  setStatus(msg);
}

function readStored(key) {
  try { return Number(localStorage.getItem(key)) || null; } catch { return null; }
}

function store(key, value) {
  try { localStorage.setItem(key, String(Math.round(value))); } catch { /* a page without storage keeps the size for this visit */ }
}

// ── Tracks ───────────────────────────────────────────────────────────────────

const semanticTrack = () => state.clip.tracks.find((t) => trackInfo(t).semantic) ?? null;
const proposedTrack = () => state.clip.tracks.find((t) => !trackInfo(t).semantic) ?? null;
const painted = () => (state.segVisible ? state.tracks : []);

/** Return the tracks of a selection: a track id, or 'both' for an annotated and an automatic track with their containment. */
function tracksFor(sel) {
  if (sel === 'both' && state.clip.hierarchy) {
    const pair = [proposedTrack(), semanticTrack()].filter(Boolean);
    if (pair.length === 2) return pair;
  }
  if (state.clip.tracks.includes(sel)) return [sel];
  return [proposedTrack() ?? state.clip.tracks[0]];
}

const selectionOf = () => (state.tracks.length > 1 ? 'both' : state.tracks[0]);

function renderRegionsMenu() {
  const pop = $('regions-pop');
  pop.innerHTML = '';
  const items = state.clip.tracks.map((t) => ({ sel: t, short: trackInfo(t).short, label: trackInfo(t).label }));
  if (tracksFor('both').length === 2) {
    items.push({ sel: 'both', short: 'Both', label: 'Annotation and automatic regions, with which contains which' });
  }
  for (const it of items) {
    const b = document.createElement('button');
    b.type = 'button';
    b.dataset.sel = it.sel;
    b.setAttribute('role', 'menuitemradio');
    b.innerHTML = `${esc(it.short)}<small>${esc(it.label)}</small>`;
    b.addEventListener('click', () => { setRegionsMenu(false); setTracks(it.sel); });
    pop.appendChild(b);
  }
  const paint = document.createElement('button');
  paint.type = 'button';
  paint.id = 'rp-paint';
  paint.setAttribute('role', 'menuitemcheckbox');
  paint.addEventListener('click', () => { setRegionsMenu(false); setSegVisible(!state.segVisible); });
  pop.appendChild(paint);
  syncRegionsMenu();
}

function syncRegionsMenu() {
  const sel = selectionOf();
  $('regions-now').textContent = `${sel === 'both' ? 'Both' : trackInfo(sel).short}${state.segVisible ? ' ●' : ''}`;
  $('btn-regions').classList.toggle('on', state.segVisible);
  for (const b of $('regions-pop').querySelectorAll('button[data-sel]')) {
    b.classList.toggle('active', b.dataset.sel === sel);
    b.setAttribute('aria-checked', String(b.dataset.sel === sel));
  }
  const paint = $('rp-paint');
  if (paint) {
    paint.innerHTML = `${state.segVisible ? '✓ ' : ''}Paint the regions on the cloud<small>S</small>`;
    paint.setAttribute('aria-checked', String(state.segVisible));
  }
}

function setRegionsMenu(open) {
  $('regions-pop').hidden = !open;
  $('btn-regions').setAttribute('aria-expanded', String(open));
}

/** Show a selection's graph and regions; the painting on the cloud follows. */
function setTracks(sel) {
  state.tracks = tracksFor(sel);
  clearEventHighlight(false);
  syncRegionsMenu();
  repaintClouds();
  if (state.mode === 'world') rerenderStrip();
  else renderPerFrameStrip();
  updateThumbs();
  renderNodeLinkPanel();
}

/** Paint a track's regions on the cloud, or take them off when that track is already painted alone. */
function toggleRegions(track) {
  if (state.segVisible && state.tracks.length === 1 && state.tracks[0] === track) {
    setSegVisible(false);
    return;
  }
  if (selectionOf() !== track) setTracks(track);
  setSegVisible(true);
}

function setSegVisible(on) {
  state.segVisible = !!on;
  syncRegionsMenu();
  repaintClouds();
  if (state.mode === 'frame') renderPerFrameStrip();
  setStatus(state.segVisible ? `Regions on the cloud: ${state.tracks.map((t) => trackInfo(t).label).join(' and ')}`
    : 'Regions off: the cloud in its own colours');
}

/** Repaint every cloud on screen: region colours, then the followed region's highlight, then the graph. */
function repaintClouds() {
  markFollowStage();
  for (const fr of cloudsOnScreen()) applySegBlend(fr.group, fr.segByTrack, painted());
  applyFocusHighlight();
  buildGraph3D();
  invalidate();
}

// ── Opening a clip ───────────────────────────────────────────────────────────

function renderClipOptions() {
  el.clip.innerHTML = '';
  for (const ds of groupClips(state.catalog)) {
    for (const g of ds.groups) {
      const og = document.createElement('optgroup');
      og.label = `${ds.dataset} — ${g.label}`;
      for (const c of g.clips) {
        const partial = c.tracks.some((t) => c.coverage[t] < c.frames);
        const opt = new Option(`${c.label}${partial ? '  (some frames without regions)' : ''}`, c.path);
        og.appendChild(opt);
      }
      el.clip.appendChild(og);
    }
  }
}

// Name only the clip in the URL: the rest of a link described the clip it was made on.
function setURLClip(path) {
  const u = new URL(location.href);
  u.search = new URLSearchParams({ clip: path }).toString();
  history.replaceState(null, '', u);
}

/**
 * Open a clip on its first frame, in per-frame mode, showing its automatic track.
 *
 * A clip opened while another is still loading wins: every await re-checks.
 * `keepURL` leaves the URL as it is, for the link the page was opened with.
 */
async function loadClip(path, { keepURL = false } = {}) {
  const seq = ++state.clipSeq;
  const stale = () => state.clipSeq !== seq;
  const clip = state.catalog.byPath.get(path);
  state.loadSeq++;
  setPlaying(false);

  // Forget the previous clip.
  clearCaches();
  exitWorld();
  Object.assign(state, {
    clip, manifest: null, geoManifest: null, mode: 'frame', frame: 0, maxFrame: clip.frames - 1,
    focusNode: null, focusLabel: null, focusStage: 0, isolateFocus: false, nfTg: null, nfNodeId: null,
    nfEvents: [], selectedEvent: null, trackingStrip: null, segByTrack: {}, graphByTrack: {}, hierarchy: null,
  });
  nfAxes = null;
  state.tracks = tracksFor(proposedTrack() ?? clip.tracks[0]);
  clearGroup(state.highlightGroup);
  el.nodeFocus.hidden = true;
  el.nfBody.innerHTML = '';
  setMapBig(false);
  syncModeUI();
  syncFocusBar();
  renderRegionsMenu();
  el.clip.value = path;
  if (!keepURL) setURLClip(path);
  renderPerFrameStrip();

  // Read the manifest and check it against the catalog.
  setStatus(`Loading ${clip.label}…`);
  const manifest = await fetchJSON(manifestURL(path));
  if (stale()) return;
  if (!manifest?.frames) { showError(`${clip.label}: frame_manifest.json did not load`); return; }
  const n = manifest.n_frames ?? manifest.frames.length;
  if (n !== clip.frames || manifest.frames.length !== clip.frames) {
    showError(`${clip.label}: the manifest has ${n} frames and the catalog says ${clip.frames}`);
    return;
  }
  try {
    state.geoManifest = manifestForSource(manifest, clip.geometry);
  } catch (err) {
    showError(`${clip.label}: ${err.message}`);
    return;
  }
  state.manifest = manifest;
  el.slider.max = String(state.maxFrame);

  // Draw the first frame and fill the band.
  await loadFrame(0);
  if (stale()) return;
  resetView();
  loadTrackingStrip();
}

// ── One frame ────────────────────────────────────────────────────────────────

// The frame last drawn; a failed load puts the controls back on it.
let drawnFrame = 0;

/** Load and draw frame `i` in per-frame mode: its cloud, overlays, panels and graph. */
async function loadFrame(i) {
  const seq = ++state.loadSeq;
  const stale = () => state.loadSeq !== seq || state.mode !== 'frame';
  const clip = state.clip;
  state.frame = i;
  syncFrameUI();

  // Fetch the cloud, once more on a failure, and the overlays of every track.
  let gltf;
  try {
    gltf = await loadGLB(glbURL(clip.path, i, clip.geometry));
  } catch {
    try {
      gltf = await loadGLB(glbURL(clip.path, i, clip.geometry));
    } catch {
      if (stale()) return;
      state.frame = drawnFrame;
      syncFrameUI();
      showError(`Frame ${i} of ${clip.label}: the point cloud did not load`);
      return;
    }
  }
  const [segs, graphs, hierarchy] = await Promise.all([
    Promise.all(clip.tracks.map((t) => fetchJSON(overlayURL(clip.path, 'seg_frame', i, t)))),
    Promise.all(clip.tracks.map((t) => fetchJSON(overlayURL(clip.path, 'graph_frame', i, t)))),
    clip.hierarchy ? fetchJSON(overlayURL(clip.path, 'hierarchy_frame', i)) : null,
  ]);
  if (stale()) return;

  // Put the cloud on stage and fit the overlays to it.
  clearGroup(state.pointsGroup);
  const group = new THREE.Group();
  gltf.scene.traverse((obj) => {
    if (!obj.isPoints) return;
    const clone = obj.clone();
    clone.userData.sharedGeometry = true;
    clone.material = obj.material.clone();
    group.add(clone);
  });
  state.pointsGroup.add(group);
  const fit = adaptOverlays({
    segByTrack: Object.fromEntries(clip.tracks.map((t, k) => [t, segs[k]])),
    graphByTrack: Object.fromEntries(clip.tracks.map((t, k) => [t, graphs[k]])),
    group, manifest: state.manifest, source: clip.geometry,
  });
  Object.assign(state, { segByTrack: fit.segByTrack, graphByTrack: fit.graphByTrack, hierarchy, segAligned: fit.aligned });

  // Paint, highlight and annotate it, then the panels.
  measureSceneScale();
  applySegBlend(group, state.segByTrack, painted());
  applyFocusHighlight();
  buildGraph3D();
  updateThumbs();
  renderNodeLinkPanel();
  if (i < state.maxFrame) prefetchGLB(glbURL(clip.path, i + 1, clip.geometry));
  setStatus(`Frame ${i} of ${state.maxFrame} — ${clip.label}`
    + (state.segAligned ? '' : ' — the overlays could not be fitted to this cloud'));
  drawnFrame = i;
  invalidate();
}

// ── Side panels ──────────────────────────────────────────────────────────────

function setThumbBadge(figure, seg, track) {
  figure.querySelector('.cell-badge')?.remove();
  if (!seg?.provenance) return;
  const info = trackInfo(track);
  const anchor = seg.provenance.stage === 'anchor';
  const badge = document.createElement('div');
  badge.className = `cell-badge thumb-badge ${anchor ? (info.semantic ? 'badge-gt' : 'badge-sam') : 'badge-tracked'}`;
  badge.textContent = anchor ? info.anchorBadge : info.trackedBadge;
  figure.appendChild(badge);
}

function renderLegend(seg) {
  const legend = $('thumb-legend');
  const present = ellipsesOf(seg);
  const items = (seg?.classes || []).filter((c) => c.id !== 0 && present?.has(c.id) && Array.isArray(c.color));
  legend.innerHTML = items.map((c) => {
    const rgb = c.color.map((v) => Math.round(v * 255)).join(',');
    return `<span class="legend-item"><i style="background: rgb(${rgb})"></i>${esc(c.name ?? `class ${c.id}`)}</span>`;
  }).join('');
  legend.hidden = !items.length;
}

/** Draw the frame on stage's image and masks; a followed node's family is emphasised. */
function updateThumbs() {
  const stage = onStage();
  if (!stage) return;
  el.thumbRgb.src = rgbURL(state.clip.path, stage.i);
  const fam = focusFamily();
  const shown = orderTracks(state.tracks);
  const figs = [[el.thumbSeg, $('thumb-seg-fig'), $('thumb-seg-cap')], [el.thumbSeg2, $('thumb-seg2-fig'), $('thumb-seg2-cap')]];
  figs.forEach(([cv, fig, cap], k) => {
    const track = shown[k];
    fig.hidden = !track;
    if (!track) return;
    segToCanvas(stage.segByTrack[track], cv, fam[track]?.size ? fam[track] : null);
    cap.textContent = trackInfo(track).label;
    setThumbBadge(fig, stage.segByTrack[track], track);
  });
  const sem = shown.find((t) => trackInfo(t).semantic);
  renderLegend(sem ? stage.segByTrack[sem] : null);
}

function renderNodeLinkPanel() {
  if (state.mode === 'world' && state.world) {
    el.nodeLinkTitle.textContent = 'Scene graph — every stacked frame, one camera';
    const opts = {
      world: state.world, refFrame: state.worldFocus, tracks: state.tracks, manifest: state.geoManifest,
      hoverKey: state.hoverKey, focusKey: state.focusNode, equalize: state.worldView === 'overlay',
    };
    renderWorldNodeLink(el.nodeLink, opts);
    if (state.mapBig) {
      $('mb-note').textContent = `All ${state.world.frames.length} stacked frames, projected into the camera of frame ${state.worldFocus}: `
        + 'a dot is a region in one frame, a line follows it through time, the shapes are the regions of the frame in focus.';
      renderWorldNodeLink(el.mapBig, opts);
    }
  } else {
    el.nodeLinkTitle.textContent = 'Scene graph — camera view';
    renderFrameNodeLink(el.nodeLink, {
      graphByTrack: state.graphByTrack, segByTrack: state.segByTrack, tracks: state.tracks,
      hierarchy: state.hierarchy, hoverKey: state.hoverKey, focusKey: state.focusNode,
    });
  }
}

/** Show the World map large, over the 3D view. */
function setMapBig(on) {
  state.mapBig = !!on && state.mode === 'world' && !!state.world;
  $('map-big').hidden = !state.mapBig;
  if (state.mapBig) renderNodeLinkPanel();
  else el.mapBig.innerHTML = '';
}

// ── The band under the 3D view ───────────────────────────────────────────────

let stripCellH = readStored('viewer.stripCellH') ?? (window.innerHeight >= 950 ? 96 : window.innerHeight >= 800 ? 72 : 60);

/**
 * Load the per-frame band: every track's masks on the sampled frames.
 *
 * Each track's seed frame is added to the sample, since where a track started is what the band shows.
 */
async function loadTrackingStrip() {
  const clip = state.clip;
  const row = async (i) => {
    const segs = await Promise.all(clip.tracks.map((t) => fetchJSON(overlayURL(clip.path, 'seg_frame', i, t))));
    return { i, segByTrack: Object.fromEntries(clip.tracks.map((t, k) => [t, segs[k]])) };
  };
  const indices = worldIndices(state.maxFrame, state.worldFrames);
  let frames = await Promise.all(indices.map(row));
  if (state.clip !== clip) return;
  const seeds = new Set();
  for (const fr of frames) {
    for (const t of clip.tracks) {
      const af = fr.segByTrack[t]?.provenance?.anchor_frame;
      if (Number.isInteger(af) && af >= 0 && af <= state.maxFrame && !indices.includes(af)) seeds.add(af);
    }
  }
  if (seeds.size) {
    frames = [...frames, ...await Promise.all([...seeds].map(row))].sort((a, b) => a.i - b.i);
    if (state.clip !== clip) return;
  }
  state.trackingStrip = { indices: frames.map((f) => f.i), frames };
  renderPerFrameStrip();
}

function renderPerFrameStrip() {
  if (state.mode !== 'frame') return;
  if (!state.trackingStrip) {
    el.strip.classList.remove('tstrip');
    el.strip.innerHTML = '<div class="strip-empty">Loading the tracked regions…</div>';
    return;
  }
  renderTrackingStrip(el.strip, {
    frames: state.trackingStrip.frames, tracks: state.clip.tracks, painted: painted(),
    focusFrame: nearestStacked(state.frame, state.trackingStrip.indices),
    onSeek: seekFrame, onPickTrack: toggleRegions, cellH: stripCellH,
  });
}

/** Return the neighbour whose edge with the followed node is read out, at the relations step. */
function partnerKey() {
  if (!state.focusNode || state.focusStage < 3 || !state.nfTg) return null;
  const other = topPartners(state.nfTg, state.nfNodeId, 1)[0]?.[0];
  return other == null ? null : `${state.focusNode.split(':')[0]}:${other}`;
}

function rerenderStrip() {
  if (state.mode === 'world' && state.world) {
    renderStrip(el.strip, {
      world: state.world, tracks: state.tracks, focusFrame: state.worldFocus, manifest: state.manifest,
      onSeek: seekFrame, focusKey: state.focusNode, partnerKey: partnerKey(), cellH: stripCellH,
      clipPath: state.stripPhoto ? state.clip.path : null,
    });
  } else {
    renderPerFrameStrip();
  }
}

function setStripPhoto(on) {
  state.stripPhoto = !!on;
  $('btn-strip-photo').classList.toggle('active', state.stripPhoto);
  rerenderStrip();
}

// ── Modes and frames ─────────────────────────────────────────────────────────

function syncModeUI() {
  document.body.classList.toggle('mode-world', state.mode === 'world');
  for (const b of document.querySelectorAll('#mode-btns button')) b.classList.toggle('active', b.dataset.mode === state.mode);
  $('worldview-btns').hidden = state.mode !== 'world';
}

/** Switch between per-frame mode and World mode, rebuilding the scene. */
async function switchMode(mode) {
  if (mode === state.mode || !state.manifest) return;
  clearEventHighlight(false);
  state.mode = mode;
  state.loadSeq++;
  if (mode !== 'world') setMapBig(false);
  syncModeUI();
  markFollowStage();
  if (mode === 'world') {
    const world = await enterWorld();
    if (!world || state.mode !== 'world') return;
    state.world = world;
    state.worldFocus = focusWorldFrame(world, state.frame, state.worldView);
    markFollowStage();
    repaintClouds();
    rerenderStrip();
    updateThumbs();
    renderNodeLinkPanel();
  } else {
    exitWorld();
    renderPerFrameStrip();
    await loadFrame(state.frame);
  }
  resetView();
}

/** Move to frame `f`: World mode moves its spotlight, per-frame mode loads the frame. */
function seekFrame(f) {
  if (!suppressEventClear) clearEventHighlight();
  state.frame = Math.min(state.maxFrame, Math.max(0, f));
  syncFrameUI();
  if (state.mode === 'world') return applyWorldFocus();
  return loadFrame(state.frame);
}

function stepFrame(delta) {
  const next = Math.min(state.maxFrame, Math.max(0, state.frame + delta));
  if (next !== state.frame) seekFrame(next);
}

/**
 * Mark the stacked frame in focus as the one that keeps its whole scene while a region is followed.
 *
 * The camera marks step aside then too: framed on one region, they cross the foreground.
 */
function markFollowStage() {
  const following = state.isolateFocus && state.mode === 'world';
  state.cameraVizGroup.visible = !following;
  for (const fr of state.world?.frames ?? []) fr.group.userData.isoContext = following && fr.i === state.worldFocus;
}

function applyWorldFocus() {
  if (!state.world) return;
  const focus = focusWorldFrame(state.world, state.frame, state.worldView);
  if (focus === state.worldFocus) return;
  state.worldFocus = focus;
  if (state.isolateFocus && state.focusNode) {
    // The old stage goes back to the region alone as its highlight clears; the new one gets its scene.
    markFollowStage();
    applyFocusHighlight();
  }
  buildGraph3D();
  updateStripFocus(el.strip, focus);
  updateThumbs();
  renderNodeLinkPanel();
}

function setWorldView(view) {
  state.worldView = view;
  for (const b of document.querySelectorAll('#worldview-btns button')) b.classList.toggle('active', b.dataset.wv === view);
  if (state.world) {
    focusWorldFrame(state.world, state.frame, view);
    renderNodeLinkPanel();
  }
}

function setGraph3D(on) {
  state.graph3d = !!on;
  $('btn-graph3d').classList.toggle('on', state.graph3d);
  buildGraph3D();
}

function syncFrameUI() {
  el.slider.value = String(state.frame);
  el.frameLabel.textContent = `${state.frame} / ${state.maxFrame}`;
  updateNodeFocusCursor(el.nfBody, state.frame, state.maxFrame);
  if (state.mode === 'frame' && state.trackingStrip) updateStripFocus(el.strip, nearestStacked(state.frame, state.trackingStrip.indices));
}

let playTimer = null;

function setPlaying(on) {
  state.playing = !!on;
  el.play.textContent = state.playing ? '⏸ Pause' : '▶︎ Play';
  el.play.classList.toggle('active', state.playing);
  clearInterval(playTimer);
  playTimer = state.playing
    ? setInterval(() => (state.frame >= state.maxFrame ? seekFrame(0) : stepFrame(1)), PLAY_MS)
    : null;
}

// ── Following one region ─────────────────────────────────────────────────────

/** Return a function naming the annotated region that contains an automatic node, from the hierarchy on stage. */
function anatomyLookup(track) {
  const gt = semanticTrack();
  if (!gt || track === gt) return () => null;
  const stage = onStage();
  const gtLabel = new Map((stage?.graphByTrack?.[gt]?.nodes || []).map((n) => [n.id, n.label]));
  return (id) => {
    for (const e of stage?.hierarchy?.edges || []) {
      if (e.relation === 'contains' && e.dst === `${track}:${id}`) return gtLabel.get(Number(e.src.split(':')[1])) ?? null;
    }
    return null;
  };
}

function syncFocusBar() {
  const on = !!state.focusNode;
  $('fb-hint').hidden = on;
  $('fb-on').hidden = !on;
  if (on) {
    const [track, id] = state.focusNode.split(':');
    $('fb-name').textContent = state.focusLabel ?? `region ${id}`;
    $('fb-track').textContent = `· ${trackInfo(track).short}`;
  }
  for (const b of document.querySelectorAll('#fb-steps button')) {
    const n = Number(b.dataset.stage);
    b.classList.toggle('active', n === state.focusStage);
    b.classList.toggle('done', n < state.focusStage);
  }
}

/**
 * Follow a region, or stop following one (`key` null).
 *
 * Picking a region marks it (step 1); its motion and its relations are the
 * next steps, asked for from the bar on the 3D view. A link may name the step.
 */
async function setFocusNode(key, { stage = null } = {}) {
  clearEventHighlight(false);
  const had = !!state.focusNode;
  const wasIsolated = state.isolateFocus;
  state.focusNode = key;
  state.focusLabel = null;
  if (!key) {
    state.focusStage = 0;
    state.isolateFocus = false;
    if (followPlaying) { setPlaying(false); followPlaying = false; }
  } else if (stage) {
    state.focusStage = stage;
  } else if (!had) {
    state.focusStage = 1;
  }
  syncFocusBar();
  if (wasIsolated && !state.isolateFocus && state.world) focusWorldFrame(state.world, state.frame, state.worldView);
  // Isolated frames hold the old region's points only, so they are rebuilt; otherwise the highlight moves.
  if (wasIsolated || state.isolateFocus) repaintClouds();
  else { applyFocusHighlight(); buildGraph3D(); }
  updateThumbs();
  if (!key) {
    Object.assign(state, { nfTg: null, nfNodeId: null, nfEvents: [] });
    el.nodeFocus.hidden = true;
    el.nfBody.innerHTML = '';
    renderNodeLinkPanel();
    rerenderStrip();
    // Framed on one region, the view would stay a close-up of empty space.
    if (wasIsolated) resetView();
    setStatus('Stopped following the region');
    return;
  }

  // Read the node's record through time.
  const [track, idStr] = key.split(':');
  const tg = await fetchJSON(temporalURL(state.clip.path, track));
  if (state.focusNode !== key) return;
  state.nfTg = tg;
  state.nfNodeId = Number(idStr);
  const label = tg?.nodes?.find((n) => n.id === state.nfNodeId)?.label ?? `region ${idStr}`;
  const anat = anatomyLookup(track)(state.nfNodeId);
  state.focusLabel = anat && anat !== label ? `${label} (in ${anat})` : label;
  syncFocusBar();
  const other = topPartners(tg, state.nfNodeId, 1)[0]?.[0];
  const otherLabel = tg?.nodes?.find((n) => n.id === other)?.label ?? `region ${other}`;
  el.nfTitle.textContent = other == null ? `What the graph holds about ${label}` : `What the edge ${label} → ${otherLabel} holds`;
  el.nfHint.textContent = other == null ? 'Esc stops following'
    : `where ${label} lies from ${otherLabel}, on three axes; the graph keeps the largest, coloured, as its word · click a numbered event to see it on the cloud`;
  if (state.focusStage >= 3) {
    el.nodeFocus.hidden = false;   // before drawing: a hidden band measures zero width
    renderNfBand();
  }
  renderNodeLinkPanel();
  rerenderStrip();
}

let followPlaying = false;   // the motion step started the playback

/**
 * Take the followed region to step `n`: 1 the region, 2 its motion in World mode, 3 its relations.
 */
async function setFocusStage(n) {
  if (!state.focusNode || ![1, 2, 3].includes(n)) return;
  state.focusStage = n;
  syncFocusBar();
  if (n >= 2 && state.mode !== 'world') {
    await switchMode('world');
    if (state.mode !== 'world' || state.focusStage !== n) return;
  }
  if (state.mode === 'world' && state.isolateFocus !== (n >= 2)) setIsolateFocus(n >= 2);
  if (n === 2 && !state.playing) { setPlaying(true); followPlaying = true; }
  else if (n !== 2 && followPlaying) { setPlaying(false); followPlaying = false; }
  const band = n >= 3 && !!state.nfTg;
  el.nodeFocus.hidden = !band;
  if (band) renderNfBand();
  else clearEventHighlight(false);
  rerenderStrip();
}

/** Keep only the followed region's points in the stacked frames, or show them whole again. */
function setIsolateFocus(on) {
  state.isolateFocus = !!on && !!state.focusNode && state.mode === 'world';
  if (state.world) focusWorldFrame(state.world, state.frame, state.worldView);
  repaintClouds();
  resetView();
  if (!state.isolateFocus) { setStatus('Showing every region'); return; }
  const [track, idStr] = state.focusNode.split(':');
  const id = Number(idStr);
  const ok = state.world.frames.filter((fr) => fr.segByTrack?.[track]?.vertex_seg?.includes?.(id)).length;
  setStatus(`Following ${state.focusLabel ?? `region ${idStr}`} through time — found in ${ok} of ${state.world.frames.length} stacked frames`);
}

// The edge read out at the relations step, as its three numbers per frame.
let nfAxes = null;

async function loadEdgeAxes(key) {
  const [track, pair] = key.split(':');
  const [a, b] = pair.split('>').map(Number);
  const present = (id) => new Set(state.nfTg.nodes.find((n) => n.id === id)?.present_frames ?? []);
  const inB = present(b);
  const frames = [...present(a)].filter((f) => inB.has(f) && f <= state.maxFrame).sort((p, q) => p - q);
  const graphs = await Promise.all(frames.map((f) => fetchJSON(overlayURL(state.clip.path, 'graph_frame', f, track))));
  // The frame's own camera, the one the exporter chose the words in.
  const data = edgeAxes(frames.map((f, k) => {
    const fr = state.manifest?.frames?.[f];
    return { f, graph: graphs[k], cam: cameraAxes(fr?.camera_forward_glb, fr?.camera_up_glb) };
  }), a, b);
  if (data.wrong || !data.checked) {
    // Axes that do not give the stored words are not this edge's: show the words only.
    console.warn(`edge axes: ${data.wrong} of ${data.checked} stored words not reproduced for ${key}`);
    data.byFrame = new Map();
  }
  return data;
}

function renderNfBand() {
  if (!state.focusNode || !state.nfTg) return;
  const other = topPartners(state.nfTg, state.nfNodeId, 1)[0]?.[0];
  const key = other == null ? null : `${state.focusNode}>${other}`;
  if (key && nfAxes?.key !== key) {
    const entry = { key, data: null };
    nfAxes = entry;
    loadEdgeAxes(key).then((data) => {
      if (nfAxes !== entry) return;
      entry.data = data;
      if (!el.nodeFocus.hidden) renderNfBand();
    });
  }
  const ev = renderNodeFocus(el.nfBody, {
    tg: state.nfTg, nodeId: state.nfNodeId, maxFrame: state.maxFrame, frame: state.frame,
    onSeek: seekFrame, onEvent: showEvent, selectedEvent: state.selectedEvent,
    anatomyOf: anatomyLookup(state.focusNode.split(':')[0]), axes: nfAxes?.key === key ? nfAxes.data : null,
  });
  if (ev === false) {
    el.nfBody.innerHTML = '<div class="strip-empty">No record through time for this region.</div>';
    state.nfEvents = [];
  } else {
    state.nfEvents = ev;
  }
}

// The clouds that carry the region highlight, repainted plain when it clears.
let highlighted = [];

function clearFocusHighlight() {
  for (const fr of highlighted) applySegBlend(fr.group, fr.segByTrack, painted());
  highlighted = [];
}

/**
 * Paint the followed region and its containment partners on every cloud on screen.
 *
 * While a region is followed through a World stack, the frame in focus keeps
 * its scene, faded, around the lightly tinted region. A selected event's own
 * highlight wins.
 */
function applyFocusHighlight() {
  clearFocusHighlight();
  if (!state.focusNode || state.selectedEvent != null) return;
  if (state.isolateFocus && state.mode === 'world') {
    const fr = onStage();
    if (fr && applyRegionHighlight(fr.group, fr.segByTrack, new Map([[state.focusNode, HL_FOCUS]]), { mix: 0.12, dim: 0.85 })) highlighted.push(fr);
    return;
  }
  for (const fr of cloudsOnScreen()) {
    const colors = new Map([[state.focusNode, HL_FOCUS]]);
    for (const pk of containmentPartners(state.focusNode, fr.hierarchy)) colors.set(pk, HL_PARTNER);
    if (applyRegionHighlight(fr.group, fr.segByTrack, colors, { mix: 0.88, dim: 0.12 })) highlighted.push(fr);
  }
}

let suppressEventClear = false;

/** Jump to event `idx` of the band and show it on the cloud. */
async function showEvent(idx) {
  const ev = state.nfEvents?.[idx];
  if (!ev || !state.focusNode) return;
  clearEventHighlight(false);
  state.selectedEvent = idx;
  suppressEventClear = true;
  await seekFrame(ev.f);
  suppressEventClear = false;
  applyEventHighlight();
  renderNfBand();
}

function applyEventHighlight() {
  const ev = state.nfEvents?.[state.selectedEvent];
  const stage = onStage();
  if (!ev || !state.focusNode || !stage) return;
  clearFocusHighlight();
  const [track] = state.focusNode.split(':');
  const colors = new Map([[state.focusNode, HL_FOCUS]]);
  if (ev.kind === 'rel') colors.set(`${track}:${ev.partner}`, HL_PARTNER);
  if (applyRegionHighlight(stage.group, stage.segByTrack, colors)) highlighted.push(stage);
  clearGroup(state.highlightGroup);
  if (ev.kind !== 'rel') return;
  const nodes = stage.graphByTrack?.[track]?.nodes || [];
  const na = nodes.find((n) => n.id === state.nfNodeId), nb = nodes.find((n) => n.id === ev.partner);
  if (na?.pos && nb?.pos) {
    addConnector(na.pos.map((v, k) => v + stage.offset[k]), nb.pos.map((v, k) => v + stage.offset[k]),
      na.label ?? `region ${state.nfNodeId}`, nb.label ?? `region ${ev.partner}`);
  }
}

// A rod between two node centroids, with a named ball at each end.
function addConnector(a, b, labelA, labelB) {
  const from = new THREE.Vector3(...a), to = new THREE.Vector3(...b);
  const dir = new THREE.Vector3().subVectors(to, from);
  const len = dir.length();
  if (len < 1e-6) return;
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(sz(0.0035), sz(0.0035), len, 8),
    new THREE.MeshBasicMaterial({ color: 0x334050, transparent: true, opacity: 0.85, depthTest: false }));
  shaft.position.copy(from).addScaledVector(dir.clone().normalize(), len / 2);
  shaft.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  shaft.renderOrder = 90;
  state.highlightGroup.add(shaft);
  for (const [p, c, text] of [[from, 0x0899b4, labelA], [to, 0xe08a00, labelB]]) {
    const ball = new THREE.Mesh(new THREE.SphereGeometry(sz(0.009), 12, 12), new THREE.MeshBasicMaterial({ color: c, depthTest: false }));
    ball.position.copy(p);
    ball.renderOrder = 91;
    state.highlightGroup.add(ball);
    const label = makeLabelSprite(text, `#${c.toString(16).padStart(6, '0')}`, sz(0.045));
    label.position.copy(p).add(new THREE.Vector3(0, sz(0.045), 0));
    state.highlightGroup.add(label);
  }
  invalidate();
}

function clearEventHighlight(redrawBand = true) {
  if (state.selectedEvent == null) return;
  state.selectedEvent = null;
  clearGroup(state.highlightGroup);
  applyFocusHighlight();
  if (redrawBand && state.focusNode && !el.nodeFocus.hidden) renderNfBand();
}

// ── Pointer on the cloud and the panels ──────────────────────────────────────

function setHover(key) {
  if (key === state.hoverKey) return;
  state.hoverKey = key;
  setHoverKey(hoverPanels, key);
  setHover3D(key);
  canvas.style.cursor = key ? 'pointer' : '';
  invalidate();
}

function wirePointer() {
  let raf = null, lastEv = null, downAt = null;
  canvas.addEventListener('pointermove', (ev) => {
    lastEv = ev;
    if (!raf) raf = requestAnimationFrame(() => { raf = null; setHover(pickKey(lastEv)); });
  });
  canvas.addEventListener('pointerleave', () => setHover(null));
  canvas.addEventListener('pointerdown', (ev) => { downAt = [ev.clientX, ev.clientY]; });
  canvas.addEventListener('pointerup', (ev) => {
    if (!downAt) return;
    const moved = Math.hypot(ev.clientX - downAt[0], ev.clientY - downAt[1]);
    downAt = null;
    if (moved > 5) return;   // a drag of the view, not a click
    const key = pickKey(ev);
    if (key) setFocusNode(key === state.focusNode ? null : key);
  });
  for (const panel of hoverPanels) {
    // Capture phase: a click on a region inside a plate must not also seek the plate's frame.
    panel.addEventListener('click', (ev) => {
      const key = ev.target.closest?.('[data-node-key]')?.getAttribute('data-node-key');
      if (!key) return;
      ev.stopPropagation();
      setFocusNode(key === state.focusNode ? null : key);
    }, true);
    panel.addEventListener('pointermove', (ev) => {
      const key = ev.target.closest?.('[data-node-key]')?.getAttribute('data-node-key') || null;
      if (key !== state.hoverKey) { state.hoverKey = key; setHoverKey(hoverPanels, key); }
    });
    panel.addEventListener('pointerleave', () => { if (state.hoverKey) { state.hoverKey = null; setHoverKey(hoverPanels, null); } });
  }
}

// ── Controls ─────────────────────────────────────────────────────────────────

function setHelp(open) {
  $('help-overlay').hidden = !open;
  if (open) $('help-close').focus();
}

function wireControls() {
  el.clip.addEventListener('change', () => loadClip(el.clip.value));
  for (const b of document.querySelectorAll('#mode-btns button')) b.addEventListener('click', () => switchMode(b.dataset.mode));
  for (const b of document.querySelectorAll('#worldview-btns button')) b.addEventListener('click', () => setWorldView(b.dataset.wv));
  $('btn-regions').addEventListener('click', (ev) => { ev.stopPropagation(); setRegionsMenu($('regions-pop').hidden); });
  document.addEventListener('click', (ev) => { if (!$('regions-pop').hidden && !$('regions-menu').contains(ev.target)) setRegionsMenu(false); });
  $('btn-graph3d').addEventListener('click', () => setGraph3D(!state.graph3d));
  $('btn-reset-view').addEventListener('click', () => resetView());
  $('btn-help').addEventListener('click', () => setHelp(true));
  $('help-close').addEventListener('click', () => setHelp(false));
  $('help-overlay').addEventListener('click', (ev) => { if (ev.target === $('help-overlay')) setHelp(false); });
  $('toast-close').addEventListener('click', () => { $('toast').hidden = true; });
  $('btn-strip-photo').addEventListener('click', () => setStripPhoto(!state.stripPhoto));
  $('btn-map-big').addEventListener('click', () => setMapBig(!state.mapBig));
  $('btn-fuse').addEventListener('click', () => setMapBig(true));
  $('mb-close').addEventListener('click', () => setMapBig(false));
  $('nf-close').addEventListener('click', () => setFocusNode(null));
  $('fb-clear').addEventListener('click', () => setFocusNode(null));
  for (const b of document.querySelectorAll('#fb-steps button')) b.addEventListener('click', () => setFocusStage(Number(b.dataset.stage)));
  for (const b of document.querySelectorAll('.card-toggle')) {
    b.addEventListener('click', () => {
      const card = $(b.dataset.panel);
      card.classList.toggle('collapsed');
      b.textContent = card.classList.contains('collapsed') ? '+' : '–';
      invalidate();
    });
  }

  // Dragging the slider moves the label; per-frame mode loads on release, so a
  // drag across the clip does not ask for every frame on the way.
  el.slider.addEventListener('input', () => {
    if (state.mode === 'world') { seekFrame(Number(el.slider.value)); return; }
    state.frame = Number(el.slider.value);
    syncFrameUI();
  });
  el.slider.addEventListener('change', () => seekFrame(Number(el.slider.value)));
  $('btn-prev').addEventListener('click', () => stepFrame(-1));
  $('btn-next').addEventListener('click', () => stepFrame(1));
  el.play.addEventListener('click', () => setPlaying(!state.playing));

  document.addEventListener('keydown', (ev) => {
    const tag = ev.target.tagName;
    if ((tag === 'INPUT' && ev.target.type !== 'range') || tag === 'SELECT' || tag === 'TEXTAREA') return;
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (!$('help-overlay').hidden) {
      if (ev.key === 'Escape' || ev.key === '?') { setHelp(false); ev.preventDefault(); }
      return;
    }
    // A held key repeats; every repeat of W or G would rebuild the scene again.
    if (ev.repeat) { ev.preventDefault(); return; }
    const k = ev.key.length === 1 ? ev.key.toLowerCase() : ev.key;
    const actions = {
      ArrowLeft: () => stepFrame(-1),
      ArrowRight: () => stepFrame(1),
      ' ': () => setPlaying(!state.playing),
      w: () => switchMode(state.mode === 'world' ? 'frame' : 'world'),
      g: () => setGraph3D(!state.graph3d),
      s: () => setSegVisible(!state.segVisible),
      o: () => state.focusNode && state.mode === 'world' && setIsolateFocus(!state.isolateFocus),
      p: () => state.mode === 'world' && setStripPhoto(!state.stripPhoto),
      r: () => resetView(),
      '?': () => setHelp(true),
      Escape: () => (state.mapBig ? setMapBig(false) : state.focusNode && setFocusNode(null)),
    };
    if (!actions[k] || !state.clip) return;
    ev.preventDefault();
    actions[k]();
  });
}

// Drag the side splitter to size the side column, the band's to size its cells; both are remembered.
function wireResizers() {
  const drag = (handle, onMove, onEnd) => handle.addEventListener('pointerdown', (ev) => {
    ev.preventDefault();
    handle.setPointerCapture(ev.pointerId);
    handle.classList.add('dragging');
    const start = { x: ev.clientX, y: ev.clientY };
    const move = (e) => onMove(e, start);
    const end = () => {
      handle.classList.remove('dragging');
      handle.removeEventListener('pointermove', move);
      handle.removeEventListener('pointerup', end);
      handle.removeEventListener('pointercancel', end);
      onEnd();
    };
    handle.addEventListener('pointermove', move);
    handle.addEventListener('pointerup', end);
    handle.addEventListener('pointercancel', end);
  });
  const sideWidth = (x) => Math.min(window.innerWidth * 0.6, Math.max(240, x - 10));
  let sideW = readStored('viewer.sideW');
  if (sideW) document.documentElement.style.setProperty('--side-w', `${sideWidth(sideW)}px`);
  drag($('split-side'), (e) => {
    sideW = sideWidth(e.clientX);
    document.documentElement.style.setProperty('--side-w', `${sideW}px`);
    invalidate();
  }, () => sideW && store('viewer.sideW', sideW));
  let startH = stripCellH, raf = null;
  drag($('split-dock'), (e, start) => {
    stripCellH = Math.min(220, Math.max(56, startH + (start.y - e.clientY)));
    if (!raf) raf = requestAnimationFrame(() => { raf = null; rerenderStrip(); });
  }, () => { startH = stripCellH; store('viewer.stripCellH', stripCellH); });
}

// Redraw the SVG panels when their boxes change size.
function wireRefit() {
  const last = new Map();
  let raf = null;
  const dirty = new Set();
  const ro = new ResizeObserver((entries) => {
    for (const e of entries) {
      const size = `${Math.round(e.contentRect.width)}x${Math.round(e.contentRect.height)}`;
      if (last.get(e.target) === size || !e.contentRect.width || !e.contentRect.height) continue;
      last.set(e.target, size);
      dirty.add(e.target);
    }
    if (dirty.size && !raf) {
      raf = requestAnimationFrame(() => {
        raf = null;
        if (state.manifest && (dirty.has(el.nodeLink) || dirty.has(el.mapBig))) renderNodeLinkPanel();
        if (dirty.has(el.nfBody) && !el.nodeFocus.hidden) renderNfBand();
        dirty.clear();
      });
    }
  });
  for (const e of [el.nodeLink, el.mapBig, el.nfBody]) ro.observe(e);
}

// ── Start ────────────────────────────────────────────────────────────────────

/** Load the catalog, open the clip the URL names (else the first), and apply the rest of the URL. */
async function boot() {
  renderHelp($('help-body'));
  wireControls();
  wireResizers();
  wireRefit();
  wirePointer();
  startRenderLoop();
  try {
    state.catalog = await loadCatalog();
  } catch (err) {
    showError(`The catalog did not load: ${err.message}`);
    return;
  }
  renderClipOptions();
  const q = new URLSearchParams(location.search);
  const wanted = q.get('clip');
  if (wanted && !state.catalog.byPath.has(wanted)) showError(`No clip ${wanted} in the catalog; opening the first one`);
  if (q.get('view') === 'overlay') setWorldView('overlay');
  const wf = Number(q.get('worldframes'));
  if (Number.isInteger(wf) && wf >= 2 && wf <= MAX_WORLD_FRAMES) state.worldFrames = wf;
  await loadClip(state.catalog.byPath.has(wanted) ? wanted : state.catalog.clips[0].path, { keepURL: state.catalog.byPath.has(wanted) });
  if (!state.manifest) return;
  const track = q.get('track');
  if (track && (track === 'both' || state.clip.tracks.includes(track))) setTracks(track);
  if (q.get('seg') === '1') setSegVisible(true);
  if (q.get('graph') === '1') setGraph3D(true);
  const frame = Number(q.get('frame'));
  if (Number.isInteger(frame) && frame > 0 && frame <= state.maxFrame) await seekFrame(frame);
  if (q.get('mode') === 'world') await switchMode('world');
  const focus = q.get('focus') || '';
  if (/^[a-z0-9_]+:\d+$/.test(focus) && state.clip.tracks.includes(focus.split(':')[0])) await setFocusNode(focus, { stage: 3 });
}

boot().catch((err) => {
  console.error(err);
  showError(`The viewer stopped: ${err.message}`);
});
