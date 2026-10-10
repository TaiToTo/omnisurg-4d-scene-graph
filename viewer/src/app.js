// Run the page: open a clip, step through its frames, switch between one frame and the world stack, and
// follow a region through time in three steps (where it is, how it moves, how its edge to a neighbour
// changes). Every panel is drawn from `state`; an action changes the state, then redraws what it touched.
import * as THREE from 'three';
import {
  clearCaches, fetchGraph, fetchHierarchy, fetchRegions, fetchTemporal, frameImageURL, loadClip, loadCloud, prefetchCloud,
} from './data.js';
import { camera, clearGroup, groups, invalidate, measureSceneScale, onSceneScale, resetView } from './stage.js';
import { applyCloudFilter, applyRegionHighlight, applySegBlend, regionsToCanvas } from './cloud_paint.js';
import { cameraInCloudFrame, INSTRUMENT_HALO_PX, MP_GRAZING_DEG, MP_REL_THRESH } from './depth_filter.js';
import { srcVertexIndex } from './vertex_filter.js';
import { enterWorld, exitWorld, focusWorldFrame, nearestStacked, worldIndices } from './world.js';
import { buildGraph3D, drawEventConnector, setHover3D } from './graph3d.js';
import { renderFrameNodeLink, renderWorldNodeLink, setHoverKey } from './node_link.js';
import { renderTrackStrip, renderWorldStrip, updateStripStage } from './strip.js';
import { renderFocusBand, topPartners, updateFocusCursor } from './focus_band.js';
import { cameraAxes, edgeAxes } from './edge_axes.js';
import { esc, nodeKey, parseNodeKey } from './format.js';

const $ = (id) => document.getElementById(id);
const PLAY_MS = 700;
const FOCUS_RGB = [0.0, 0.52, 0.64];
const PARTNER_RGB = [0.85, 0.42, 0.0];
const SHOWS = ['all', 'tissue', 'tools'];

export const state = {
  clip: null,
  frame: 0,
  track: null,            // the track whose regions and graph the panels show
  painted: false,         // its regions are painted on the cloud
  mode: 'frame',          // 'frame' or 'world'
  worldView: 'spotlight', // 'spotlight' or 'overlay'
  show: 'all',            // 'all', 'tissue' or 'tools'
  graph3d: false,
  stripPhoto: false,
  playing: false,
  // The frame on screen in per-frame mode.
  regionsByTrack: {},
  graphByTrack: {},
  hierarchy: null,
  trackStrip: null,
  world: null,
  stageFrame: 0,
  // The followed region.
  focusKey: null,
  focusLabel: null,
  focusStage: 0,
  isolate: false,
  temporal: null,
  events: [],
  selectedEvent: null,
  edge: null,
  hoverKey: null,
  mapBig: false,
  // Tokens that let a load notice the page has moved on.
  loadSeq: 0,
  clipSeq: 0,
};

let playToken = 0;
let drawnFrame = 0;
// The view is framed on the first frame a clip draws, which may not be its first frame.
let needsFraming = false;
let followPlaying = false;
let litGroups = [];
let stripCellH = 72;
let report = (msg) => console.error(msg);
let changed = () => {};
let changePending = false;

/** Say a failure where a reader sees it; `onError` is the page's error surface. */
export function onError(fn) { report = fn; }

/** Call `fn` once after any change of clip, frame, mode or followed region. */
export function onChange(fn) { changed = fn; }

function noteChange() {
  if (changePending) return;
  changePending = true;
  queueMicrotask(() => { changePending = false; changed(); });
}

export function setStatus(msg) {
  const el = $('status');
  if (el) el.textContent = msg;
}

// ── What the clip offers ────────────────────────────────────────────────────

const trackInfo = (id) => state.clip?.trackById.get(id);

/** The track that names instrument classes on this clip, or null. */
function instrumentFilter() {
  const t = state.clip?.tracks.find((x) => x.instrument_ids.length);
  return t ? { track: t.id, ids: t.instrument_ids } : null;
}

/** What the cloud hides now. */
function view(context = false) {
  return { show: state.show, instruments: instrumentFilter(), isolate: isolating() && !context ? state.focusKey : null };
}

const paintedTracks = () => (state.painted && state.track ? [state.track] : []);

function paint(group, regionsByTrack, stats = null) {
  applySegBlend(group, regionsByTrack, paintedTracks(), view(), stats);
}

/** The frame on stage, in either mode: its index, cloud group, regions, graphs, hierarchy and offset. */
function stageEntry() {
  if (state.mode === 'world' && state.world) return state.world.frames.find((f) => f.i === state.stageFrame) ?? null;
  const group = groups.points.children[0];
  return group ? { i: state.frame, group, regionsByTrack: state.regionsByTrack, graphByTrack: state.graphByTrack,
    hierarchy: state.hierarchy, offset: [0, 0, 0] } : null;
}

/** A node's name, with the class of the region that contains it where the hierarchy says. */
function labelOf(key) {
  const { track, id } = parseNodeKey(key);
  const name = state.temporal?.nodes?.find((n) => n.id === id)?.label
    ?? stageEntry()?.graphByTrack[track]?.nodes?.find((n) => n.id === id)?.label ?? `region ${id}`;
  const container = containerOf(key);
  return container && container !== name ? `${name} (in ${container})` : name;
}

/** The name of the annotated region that contains region `key`, from the stage frame's hierarchy. */
function containerOf(key) {
  const st = stageEntry();
  for (const e of st?.hierarchy?.edges || []) {
    if (e.relation !== 'contains' || e.dst !== key) continue;
    const { track, id } = parseNodeKey(e.src);
    return st.graphByTrack[track]?.nodes?.find((n) => n.id === id)?.label ?? null;
  }
  return null;
}

// ── Opening a clip and a frame ──────────────────────────────────────────────

/**
 * Open a clip on its first frame, in per-frame mode, with nothing followed. The clip on screen stays
 * until the new one's record has loaded.
 *
 * @returns {Promise<boolean>} false when the record did not load or another clip was asked for meanwhile.
 */
export async function openClip(id) {
  const seq = ++state.clipSeq;
  const stale = () => state.clipSeq !== seq;
  setStatus(`Loading ${id}…`);
  let clip;
  try {
    clip = await loadClip(id);
  } catch (err) {
    if (!stale()) report(`${id} did not load: ${err.message}`);
    return false;
  }
  if (stale()) return false;

  // Take the last clip off the page.
  state.loadSeq++;
  setPlaying(false);
  exitWorld();
  clearGroup(groups.points);
  clearGroup(groups.highlight);
  clearGroup(groups.hover);
  clearCaches();
  Object.assign(state, {
    clip, world: null, mode: 'frame', focusKey: null, focusLabel: null, focusStage: 0, isolate: false,
    temporal: null, events: [], selectedEvent: null, edge: null, mapBig: false, trackStrip: null, frame: 0,
    regionsByTrack: {}, graphByTrack: {}, hierarchy: null, hoverKey: null,
  });
  drawnFrame = 0;
  needsFraming = true;
  document.body.classList.remove('mode-world');
  $('node-focus').hidden = true;
  $('map-big').hidden = true;

  // Set the controls for this clip, then draw its first frame and its strip.
  // The strip holds a row per track; with three rows it would crowd the 3D view off a laptop's screen.
  stripCellH = Math.round((window.innerHeight >= 950 ? 72 : 60) * (clip.tracks.length > 2 ? 0.8 : 1));
  // The page opens on the first track that is not an annotation, the method the viewer is about.
  state.track = (clip.tracks.find((t) => !t.semantic) ?? clip.tracks[0])?.id ?? null;
  if (state.show === 'tools' && !instrumentFilter()) state.show = 'all';
  $('frame-slider').max = String(clip.n_frames - 1);
  syncControls();
  renderClipInfo();
  renderTrackStrip_();
  loadTrackStrip();
  await loadFrame(0);
  return !stale();
}

/** Draw frame `i` in per-frame mode: its cloud, its regions on it, its graph and the panels. */
export async function loadFrame(i) {
  const seq = ++state.loadSeq;
  const stale = () => state.loadSeq !== seq || state.mode !== 'frame';
  const clip = state.clip;
  state.frame = i;
  syncFrameUI();
  const tracks = clip.tracks.map((t) => t.id);
  let gltf, regions, graphs, hierarchy;
  try {
    [gltf, regions, graphs, hierarchy] = await Promise.all([
      loadCloud(clip, i),
      Promise.all(tracks.map((t) => fetchRegions(clip, t, i))),
      Promise.all(tracks.map((t) => fetchGraph(clip, t, i))),
      fetchHierarchy(clip, i),
    ]);
  } catch (err) {
    if (stale()) return false;
    // Put the controls back on the frame still drawn, so they never name a frame the view does not show.
    state.frame = drawnFrame;
    syncFrameUI();
    setPlaying(false);
    report(`Frame ${i} did not load: ${err.message}`);
    return false;
  }
  if (stale()) return false;
  state.regionsByTrack = Object.fromEntries(tracks.map((t, k) => [t, regions[k]]));
  state.graphByTrack = Object.fromEntries(tracks.map((t, k) => [t, graphs[k]]));
  state.hierarchy = hierarchy;
  clearGroup(groups.points);
  const group = new THREE.Group();
  gltf.scene.traverse((obj) => {
    if (!obj.isPoints) return;
    const clone = obj.clone();
    clone.userData.sharedGeometry = true;
    clone.material = obj.material.clone();
    group.add(clone);
  });
  group.userData.cam = cameraInCloudFrame(clip.frames[i]);
  group.userData.grid = clip.geometry.grid;
  groups.points.add(group);
  measureSceneScale();
  litGroups = [];
  paint(group, state.regionsByTrack);
  applyFocusHighlight();
  rebuildGraph3D();
  updateThumbs();
  renderNodeLinkPanel();
  prefetchCloud(clip, i + 1);
  setStatus(`Frame ${i} of ${clip.n_frames - 1} · ${clip.geometry.name} reconstruction`);
  drawnFrame = i;
  if (needsFraming) { resetView(clip.frames[i]); needsFraming = false; }
  invalidate();
  return true;
}

/** Go to frame `f`: load it in per-frame mode, or put the nearest stacked frame on stage in world mode. */
export function seekFrame(f, { keepEvent = false } = {}) {
  if (!state.clip) return undefined;
  if (!keepEvent) clearEventHighlight();
  state.frame = Math.min(state.clip.n_frames - 1, Math.max(0, f));
  syncFrameUI();
  if (state.mode === 'world') return applyWorldStage();
  return loadFrame(state.frame);
}

export function stepFrame(delta) {
  const next = Math.min(state.clip.n_frames - 1, Math.max(0, state.frame + delta));
  if (next !== state.frame) seekFrame(next);
}

export function setPlaying(on) {
  state.playing = !!on && !!state.clip;
  const token = ++playToken;
  if (state.playing) play(token);
  const b = $('btn-play');
  if (b) {
    b.textContent = state.playing ? '⏸ Pause' : '▶ Play';
    b.classList.toggle('active', state.playing);
    b.setAttribute('aria-pressed', String(state.playing));
  }
}

/**
 * Step through the frames, one every PLAY_MS at most. Each step waits for its frame, so on a slow link
 * the clip plays slower rather than asking for frames it never draws. The last frame wraps to the first.
 */
async function play(token) {
  const playing = () => state.playing && token === playToken;
  let t0 = performance.now();
  while (playing()) {
    const wait = PLAY_MS - (performance.now() - t0);
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    if (!playing()) return;
    t0 = performance.now();
    await seekFrame(state.frame >= state.clip.n_frames - 1 ? 0 : state.frame + 1);
  }
}

function syncFrameUI() {
  if (!state.clip) return;
  noteChange();
  $('frame-slider').value = String(state.frame);
  const t = state.clip.frames[state.frame]?.time_s;
  $('frame-label').textContent = `${state.frame} / ${state.clip.n_frames - 1}${Number.isFinite(t) ? ` · ${t.toFixed(1)} s` : ''}`;
  updateFocusCursor($('nf-body'), state.frame, state.clip.n_frames - 1);
  if (state.mode === 'frame' && state.trackStrip) updateStripStage($('strip'), nearestStacked(state.frame, state.trackStrip.indices));
}

// ── World mode ──────────────────────────────────────────────────────────────

/** Switch between one frame at a time and the stack of frames in one space. */
export async function switchMode(mode) {
  if (!state.clip || mode === state.mode) return;
  clearEventHighlight();
  state.mode = mode;
  state.loadSeq++;
  document.body.classList.toggle('mode-world', mode === 'world');
  if (mode !== 'world') setMapBig(false);
  syncControls();
  if (mode === 'world') {
    const seq = state.loadSeq;
    setStatus('Stacking the frames in one space…');
    let world;
    try {
      world = await enterWorld(state.clip, (g, r) => paint(g, r), () => state.loadSeq !== seq || state.mode !== 'world');
    } catch (err) {
      report(`World mode could not load: ${err.message}`);
      state.mode = 'frame';
      document.body.classList.remove('mode-world');
      syncControls();
      return;
    }
    if (!world) return;
    state.world = world;
    state.stageFrame = focusWorldFrame(world, state.frame, state.worldView, isolating());
    markIsolationStage();
    litGroups = [];
    repaintAll();
    rerenderStrip();
    updateThumbs();
    renderNodeLinkPanel();
    setStatus(`World mode: ${world.indices.length} of ${state.clip.n_frames} frames stacked`);
  } else {
    // Isolation is a view of the stack; one frame shows the region in its scene.
    state.isolate = false;
    if (state.focusStage === 2) state.focusStage = 1;
    if (followPlaying) { setPlaying(false); followPlaying = false; }
    exitWorld();
    state.world = null;
    groups.cameraViz.visible = true;
    syncFocusBar();
    renderTrackStrip_();
    await loadFrame(state.frame);
  }
  needsFraming = false;
  resetView(state.clip.frames[state.frame], { withCameraMarks: !isolating() });
}

const isolating = () => state.isolate && !!state.focusKey && state.mode === 'world';

/** Put the stacked frame nearest the current one on stage, and redraw what depends on it. */
function applyWorldStage() {
  if (!state.world) return;
  const stage = focusWorldFrame(state.world, state.frame, state.worldView, isolating());
  if (stage === state.stageFrame) return;
  state.stageFrame = stage;
  if (isolating()) {
    markIsolationStage();
    repaintAll();
  } else {
    rebuildGraph3D();
  }
  updateStripStage($('strip'), stage);
  updateThumbs();
  renderNodeLinkPanel();
}

/** While a region is isolated, the frame on stage keeps its whole cloud and the camera marks step aside. */
function markIsolationStage() {
  groups.cameraViz.visible = !isolating();
  for (const fr of state.world?.frames ?? []) fr.group.userData.isoContext = isolating() && fr.i === state.stageFrame;
}

export function setWorldView(v) {
  state.worldView = v;
  syncControls();
  if (state.world) {
    focusWorldFrame(state.world, state.frame, v, isolating());
    renderNodeLinkPanel();
  }
}

// ── Painting the clouds ─────────────────────────────────────────────────────

/** Repaint every cloud on stage, then the highlight and the graph on it. */
function repaintAll(stats = null) {
  litGroups = [];
  if (state.mode === 'world' && state.world) {
    for (const fr of state.world.frames) paint(fr.group, fr.regionsByTrack, fr.i === state.stageFrame ? stats : null);
  } else {
    const g = groups.points.children[0];
    if (g) paint(g, state.regionsByTrack, stats);
  }
  applyFocusHighlight();
  rebuildGraph3D();
  invalidate();
}

/** Paint the track's regions on the cloud, or, when they are already on it, take them off. */
export function toggleRegions(track) {
  if (state.painted && state.track === track) {
    state.painted = false;
  } else {
    state.track = track;
    state.painted = true;
  }
  afterTrackChange();
  setStatus(state.painted ? `Regions on the cloud: ${trackInfo(track).name}` : 'Regions off: the cloud as reconstructed');
}

/** Show another track in the panels without painting it. */
export function setTrack(track) {
  if (!trackInfo(track) || track === state.track) return;
  state.track = track;
  afterTrackChange();
}

function afterTrackChange() {
  // A followed region of another track is no longer on any panel.
  if (state.focusKey && parseNodeKey(state.focusKey).track !== state.track) setFocusNode(null);
  syncControls();
  repaintAll();
  updateThumbs();
  renderNodeLinkPanel();
  rerenderStrip();
}

/** Which points the cloud shows: every point, the tissue only, or the instruments only. */
export function setCloudShow(show) {
  if (!SHOWS.includes(show) || (show === 'tools' && !instrumentFilter())) return;
  state.show = show;
  const st = {};
  repaintAll(st);
  syncControls();
  setStatus(showStatus(show, st));
}

export function cycleCloudShow() {
  const can = SHOWS.filter((s) => s !== 'tools' || instrumentFilter());
  setCloudShow(can[(can.indexOf(state.show) + 1) % can.length]);
}

/** Say how many points a view hides on the frame on stage, by reason. */
export function showStatus(show, st) {
  if (show === 'all') return 'Every reconstructed point';
  const total = st.total ?? 0;
  if (!total) return 'Nothing drawn to filter';
  const where = state.mode === 'world' ? ' on the frame on stage' : '';
  const by = instrumentFilter() ? `${trackInfo(instrumentFilter().track).short} classes` : '';
  if (show === 'tools') {
    if (st.unlabeled) return `Instruments: nothing shown${where}; this frame has no ${by} to find them by`;
    const kept = total - Object.entries(st).filter(([k]) => !['total', 'unlabeled'].includes(k)).reduce((s, [, v]) => s + v, 0);
    return `Instruments: ${kept.toLocaleString()} of ${total.toLocaleString()} points kept${where}, by the ${by}`;
  }
  const parts = [];
  if (st.instruments) parts.push(`${st.instruments.toLocaleString()} instrument points`);
  if (st.depthStep) parts.push(`${st.depthStep.toLocaleString()} at depth steps over ${(100 * MP_REL_THRESH).toFixed(0)} %`);
  if (st.edgeOn) parts.push(`${st.edgeOn.toLocaleString()} on surfaces seen more edge-on than ${MP_GRAZING_DEG}°`);
  if (st.toolHalo) parts.push(`${st.toolHalo.toLocaleString()} within ${INSTRUMENT_HALO_PX} px of an instrument`);
  const dropped = (st.instruments ?? 0) + (st.depthStep ?? 0) + (st.edgeOn ?? 0) + (st.toolHalo ?? 0);
  return `Tissue: ${dropped.toLocaleString()} of ${total.toLocaleString()} points hidden${where}`
    + (parts.length ? ` (${parts.join(', ')})` : '')
    + (!instrumentFilter() ? '; no track here tells instruments from tissue, so they stay' : '')
    + (st.unlabeled ? `; this frame has no ${by}, so its instruments stay` : '');
}

// ── Following a region ──────────────────────────────────────────────────────

/**
 * Follow region `key` (a node key), or stop following with null. A new region starts at the first step;
 * moving from one region to another keeps the step.
 */
export async function setFocusNode(key, { stage = null } = {}) {
  if (key && !trackInfo(parseNodeKey(key).track)) return;
  clearEventHighlight();
  if (key && parseNodeKey(key).track !== state.track) {
    state.track = parseNodeKey(key).track;
    syncControls();
  }
  const had = !!state.focusKey;
  const wasIsolating = isolating();
  state.focusKey = key;
  state.focusLabel = null;
  state.edge = null;
  if (!key) {
    state.focusStage = 0;
    state.isolate = false;
    if (followPlaying) { setPlaying(false); followPlaying = false; }
  } else if (stage) state.focusStage = stage;
  else if (!had) state.focusStage = 1;
  if (wasIsolating && !isolating() && state.world) {
    markIsolationStage();
    focusWorldFrame(state.world, state.frame, state.worldView, false);
  }
  repaintAll();
  updateThumbs();
  syncFocusBar();
  if (key) {
    const { track } = parseNodeKey(key);
    let tg = null;
    try {
      tg = await fetchTemporal(state.clip, track);
    } catch (err) {
      report(`The graph through time did not load: ${err.message}`);
    }
    if (state.focusKey !== key) return;
    state.temporal = tg;
    state.focusLabel = labelOf(key);
    syncFocusBar();
    renderBand();
  } else {
    state.temporal = null;
    state.events = [];
    $('node-focus').hidden = true;
    $('nf-body').innerHTML = '';
  }
  renderNodeLinkPanel();
  rerenderStrip();
}

/**
 * Take the followed region to step `n`: 1 marks it, 2 shows it moving (world mode, the region alone in
 * each frame, played), 3 reads out its edge to the neighbour it changes against most.
 */
export async function setFocusStage(n) {
  if (!state.focusKey || ![1, 2, 3].includes(n)) return;
  state.focusStage = n;
  syncFocusBar();
  if (n >= 2 && state.mode !== 'world') {
    await switchMode('world');
    if (state.mode !== 'world' || state.focusStage !== n) return;
  }
  if (state.mode === 'world' && state.isolate !== (n >= 2)) setIsolate(n >= 2);
  // The motion is seen by playing it; leaving the step stops what the step started.
  if (n === 2 && !state.playing) { setPlaying(true); followPlaying = true; }
  else if (n !== 2 && followPlaying) { setPlaying(false); followPlaying = false; }
  renderBand();
  rerenderStrip();
  if (n === 1) setStatus('The region, marked where it is');
  if (n === 3) setStatus('Its edge to the neighbour it changes against most, through time');
}

/** Hide every point but the followed region's, in every stacked frame but the one on stage. */
export function setIsolate(on) {
  state.isolate = !!on && !!state.focusKey;
  if (state.world) {
    markIsolationStage();
    focusWorldFrame(state.world, state.frame, state.worldView, isolating());
  }
  repaintAll();
  resetView(state.clip.frames[state.frame], { withCameraMarks: !isolating() });
  syncFocusBar();
  if (!state.isolate) { setStatus('Every region'); return; }
  const { track, id } = parseNodeKey(state.focusKey);
  const frames = state.world ? state.world.frames : [{ regionsByTrack: state.regionsByTrack }];
  const ok = frames.filter((fr) => fr.regionsByTrack[track]?.labels.includes(id)).length;
  // The frames with no label for the region stay whole; a reader is told how many.
  setStatus(`${labelOf(state.focusKey)}, alone in ${ok} of ${frames.length} stacked frames`
    + (ok < frames.length ? '; the others do not have it and stay whole' : ''));
}

/** The neighbour whose edge is read out at the third step, as a node key, or null. */
function partnerKey() {
  if (!state.focusKey || state.focusStage < 3 || !state.temporal) return null;
  const { track, id } = parseNodeKey(state.focusKey);
  const other = topPartners(state.temporal, id)[0]?.[0];
  return other == null ? null : nodeKey(track, other);
}

/** The edge's three axes per frame, from each frame's graph in the camera the relations were chosen in. */
async function loadEdgeAxes(key, other) {
  const { track, id } = parseNodeKey(key);
  const present = (nid) => new Set(state.temporal.nodes.find((n) => n.id === nid)?.present_frames ?? []);
  const inOther = present(other);
  const frames = [...present(id)].filter((f) => inOther.has(f) && f < state.clip.n_frames).sort((a, b) => a - b);
  const graphs = await Promise.all(frames.map((f) => fetchGraph(state.clip, track, f)));
  const data = edgeAxes(frames.map((f, k) => {
    const fr = state.clip.frames[f];
    const g = graphs[k];
    return {
      f,
      graph: g && { ...g, nodes: g.nodes.map((n) => ({ ...n, pos: n.graph_pos ?? n.pos })) },
      cam: cameraAxes(fr.graph_camera_forward ?? fr.camera_forward, fr.graph_camera_up ?? fr.camera_up),
    };
  }), id, other);
  // Axes that do not give the stored relations are not the edge; the band then shows the relations alone.
  if (data.wrong || !data.checked) data.byFrame = new Map();
  return data;
}

function renderBand() {
  const el = $('nf-body');
  const show = state.focusStage >= 3 && !!state.temporal && !!state.focusKey;
  $('node-focus').hidden = !show;
  if (!show) { state.events = []; clearGroup(groups.highlight); return; }
  const { id } = parseNodeKey(state.focusKey);
  const partner = partnerKey();
  const { track } = parseNodeKey(state.focusKey);
  if (partner && state.edge?.key !== `${state.focusKey}>${partner}`) {
    const entry = { key: `${state.focusKey}>${partner}`, data: null };
    state.edge = entry;
    loadEdgeAxes(state.focusKey, parseNodeKey(partner).id).catch((err) => {
      setStatus(`The edge's axes did not load (${err.message}); the band shows its relations alone`);
      return { byFrame: new Map(), checked: 0, wrong: 0 };
    }).then((data) => {
      if (state.edge !== entry) return;
      entry.data = data;
      renderBand();
    });
  }
  const partnerName = partner ? labelOf(partner) : null;
  $('nf-title').textContent = partner ? `The edge ${state.focusLabel ?? labelOf(state.focusKey)} → ${partnerName}`
    : `What the graph holds about ${state.focusLabel ?? labelOf(state.focusKey)}`;
  $('nf-hint').textContent = partner
    ? `where it is from ${partnerName} on three axes; the graph keeps the largest, in colour. Click a numbered event to see it on the cloud.`
    : 'Esc stops following.';
  const ev = renderFocusBand(el, {
    tg: state.temporal, nodeId: id, lastFrame: state.clip.n_frames - 1, frame: state.frame,
    onFrame: (f) => seekFrame(f), onEvent: showEvent, selectedEvent: state.selectedEvent,
    labelOf: (nid) => labelOf(nodeKey(track, nid)), axes: state.edge?.data ?? null,
  });
  state.events = ev || [];
  if (ev === false) el.innerHTML = '<div class="strip-empty">The graph through time has no record of this region.</div>';
}

// ── Highlights on the cloud ─────────────────────────────────────────────────

/** Light the followed region and the regions the hierarchy pairs it with, on every cloud on stage. */
function applyFocusHighlight() {
  for (const g of litGroups) {
    const fr = state.world?.frames.find((f) => f.group === g);
    paint(g, fr ? fr.regionsByTrack : state.regionsByTrack);
  }
  litGroups = [];
  if (!state.focusKey || state.selectedEvent != null) return;
  if (isolating()) {
    // The stage frame keeps its scene, pale, with the region in nearly its own colours; the path is the accent.
    const fr = stageEntry();
    if (fr && applyRegionHighlight(fr.group, fr.regionsByTrack, new Map([[state.focusKey, FOCUS_RGB]]), view(true), { mix: 0.12, dim: 0.85 })) {
      litGroups.push(fr.group);
    }
    return;
  }
  const targets = state.mode === 'world' && state.world ? state.world.frames : [stageEntry()].filter(Boolean);
  for (const fr of targets) {
    const keys = new Map([[state.focusKey, FOCUS_RGB]]);
    for (const e of fr.hierarchy?.edges || []) {
      if (e.relation !== 'contains') continue;
      if (e.src === state.focusKey) keys.set(e.dst, PARTNER_RGB);
      else if (e.dst === state.focusKey) keys.set(e.src, PARTNER_RGB);
    }
    if (applyRegionHighlight(fr.group, fr.regionsByTrack, keys, view(), { mix: 0.88, dim: 0.12 })) litGroups.push(fr.group);
  }
}

/**
 * Go to event `idx` of the band, light its regions and join their centroids. The world stack holds a
 * sample of the frames, so an event on a frame it lacks is shown in per-frame mode.
 */
async function showEvent(idx) {
  const ev = state.events[idx];
  const key = state.focusKey;
  if (!ev || !key) return;
  clearEventHighlight();
  if (state.mode === 'world' && !state.world?.indices.includes(ev.f)) {
    await switchMode('frame');
    if (state.focusKey !== key) return;
  }
  state.selectedEvent = idx;
  await seekFrame(ev.f, { keepEvent: true });
  const fr = stageEntry();
  if (state.focusKey !== key || state.selectedEvent !== idx || fr?.i !== ev.f) return;
  const { track, id } = parseNodeKey(key);
  const keys = new Map([[key, FOCUS_RGB]]);
  if (ev.kind === 'rel') keys.set(nodeKey(track, ev.partner), PARTNER_RGB);
  for (const g of litGroups) paint(g, state.world?.frames.find((f) => f.group === g)?.regionsByTrack ?? state.regionsByTrack);
  litGroups = [];
  applyRegionHighlight(fr.group, fr.regionsByTrack, keys, view());
  litGroups.push(fr.group);
  const nodes = fr.graphByTrack[track]?.nodes || [];
  const a = nodes.find((n) => n.id === id), b = ev.kind === 'rel' ? nodes.find((n) => n.id === ev.partner) : null;
  if (a?.pos && b?.pos) {
    drawEventConnector(a.pos.map((v, k) => v + fr.offset[k]), b.pos.map((v, k) => v + fr.offset[k]),
      labelOf(key), labelOf(nodeKey(track, ev.partner)));
  }
  renderBand();
}

function clearEventHighlight() {
  if (state.selectedEvent == null) return;
  state.selectedEvent = null;
  clearGroup(groups.highlight);
  applyFocusHighlight();
  renderBand();
}

function rebuildGraph3D() {
  buildGraph3D({ on: state.graph3d, track: state.track, stage: stageEntry(), world: state.mode === 'world' ? state.world : null,
    focusKey: state.focusKey });
  // The outline borrows the shapes just replaced, so it is drawn again on the new ones.
  setHover3D(state.graph3d ? state.hoverKey : null);
}

export function setGraph3D(on) {
  state.graph3d = !!on;
  syncControls();
  rebuildGraph3D();
}

// ── The panels ──────────────────────────────────────────────────────────────

/** The frame as an image, and with the shown track's regions over it, and the regions' legend. */
function updateThumbs() {
  const st = stageEntry();
  if (!state.clip || !st) return;
  $('thumb-rgb').src = frameImageURL(state.clip, st.i);
  $('thumb-rgb2').src = frameImageURL(state.clip, st.i);
  const regions = st.regionsByTrack[state.track];
  const focus = new Set();
  if (state.focusKey) {
    const { track, id } = parseNodeKey(state.focusKey);
    if (track === state.track) focus.add(id);
  }
  regionsToCanvas(regions, $('thumb-seg'), focus);
  const info = trackInfo(state.track);
  $('thumb-seg-cap').textContent = info ? info.name : 'Regions';
  const badge = $('thumb-badge');
  badge.hidden = !regions;
  if (regions && info) {
    const anchor = regions.stage === 'anchor';
    badge.className = `cell-badge ${anchor ? (info.semantic ? 'badge-gt' : 'badge-sam') : 'badge-tracked'}`;
    badge.textContent = anchor ? info.anchor_badge : info.tracked_badge;
  }
  const legend = $('thumb-legend');
  const present = regions ? new Set(regions.labels) : new Set();
  const items = info?.semantic && regions ? regions.classes.filter((c) => present.has(c.id)) : [];
  legend.hidden = !items.length;
  legend.innerHTML = items.map((c) => `<span class="legend-item"><i style="background: rgb(${c.color.map((v) => Math.round(v * 255)).join(',')})"></i>${esc(c.name)}</span>`).join('');
}

export function renderNodeLinkPanel() {
  if (!state.clip) return;
  const title = $('nodelink-title');
  if (state.mode === 'world' && state.world) {
    title.textContent = 'Scene graph · all stacked frames, in one camera';
    const opts = { world: state.world, refFrame: state.stageFrame, track: state.track, frames: state.clip.frames,
      hoverKey: state.hoverKey, focusKey: state.focusKey, equalize: state.worldView === 'overlay' };
    renderWorldNodeLink($('nodelink-body'), opts);
    if (state.mapBig) {
      $('mb-note').textContent = `The ${state.world.frames.length} stacked frames, projected into the camera of frame `
        + `${state.stageFrame}: a dot is a region in one frame, a line follows it through time, and the shapes are the regions of frame ${state.stageFrame}.`;
      renderWorldNodeLink($('mb-body'), opts);
    }
  } else {
    title.textContent = 'Scene graph · camera view';
    renderFrameNodeLink($('nodelink-body'), { graph: state.graphByTrack[state.track], regions: state.regionsByTrack[state.track],
      track: state.track, hoverKey: state.hoverKey, focusKey: state.focusKey });
  }
}

/** Show the world mode's scene graph large, over the 3D view, or put it back. */
export function setMapBig(on) {
  state.mapBig = !!on && state.mode === 'world' && !!state.world;
  $('map-big').hidden = !state.mapBig;
  if (state.mapBig) renderNodeLinkPanel();
  else $('mb-body').innerHTML = '';
}

/** Load the regions of every track on the sampled frames, for the per-frame strip. */
async function loadTrackStrip() {
  const clip = state.clip;
  const tracks = clip.tracks.map((t) => t.id);
  const row = async (i) => {
    const r = await Promise.all(tracks.map((t) => fetchRegions(clip, t, i)));
    return { i, regionsByTrack: Object.fromEntries(tracks.map((t, k) => [t, r[k]])) };
  };
  let indices = worldIndices(clip.n_frames - 1);
  let frames;
  try {
    frames = await Promise.all(indices.map(row));
    // A track's seed frame is always a column: it is where the track's regions were made.
    const anchors = new Set(frames.flatMap((fr) => Object.values(fr.regionsByTrack))
      .map((r) => r?.anchorFrame).filter((f) => Number.isInteger(f) && f < clip.n_frames && !indices.includes(f)));
    if (anchors.size) frames = [...frames, ...await Promise.all([...anchors].map(row))].sort((a, b) => a.i - b.i);
  } catch (err) {
    if (state.clip === clip) report(`The regions through time did not load: ${err.message}`);
    return;
  }
  if (state.clip !== clip) return;
  indices = frames.map((f) => f.i);
  state.trackStrip = { indices, frames };
  renderTrackStrip_();
}

function renderTrackStrip_() {
  if (state.mode !== 'frame') return;
  const el = $('strip');
  if (!state.trackStrip) {
    el.classList.remove('tstrip');
    el.innerHTML = '<div class="strip-empty">Loading the regions through time…</div>';
    return;
  }
  renderTrackStrip(el, {
    frames: state.trackStrip.frames, tracks: state.clip.tracks, painted: paintedTracks(),
    stageFrame: nearestStacked(state.frame, state.trackStrip.indices), imageURL: (i) => frameImageURL(state.clip, i),
    onFrame: (i) => seekFrame(i), onPickTrack: toggleRegions, cellH: stripCellH,
  });
}

function rerenderStrip() {
  if (state.mode === 'world' && state.world) {
    renderWorldStrip($('strip'), state.world, state.track, trackInfo(state.track), state.stageFrame, state.clip.frames,
      (i) => seekFrame(i), {
        focusKey: state.focusKey, partnerKey: partnerKey(), cellH: Math.round(stripCellH * 1.3),
        imageURL: state.stripPhoto ? (i) => frameImageURL(state.clip, i) : null,
      });
  } else {
    renderTrackStrip_();
  }
}

export function setStripPhoto(on) {
  state.stripPhoto = !!on;
  syncControls();
  rerenderStrip();
}

/** Resize the strip's cells and draw it again. */
export function setStripCellHeight(h) {
  stripCellH = Math.min(180, Math.max(48, Math.round(h)));
  rerenderStrip();
  return stripCellH;
}

export const stripCellHeight = () => stripCellH;

/** Redraw the panels whose size changed. */
export function refitPanels() {
  renderNodeLinkPanel();
  if (state.focusStage >= 3) renderBand();
}

// ── Hovering and picking on the cloud ───────────────────────────────────────

const raycaster = new THREE.Raycaster();
const PICK_RADIUS = 0.006;
raycaster.params.Points.threshold = PICK_RADIUS;
onSceneScale((s) => { raycaster.params.Points.threshold = PICK_RADIUS * s; });

/** The node under the pointer: a drawn node, or the region of the point under it. */
export function pickNode(canvas, ev) {
  const st = stageEntry();
  if (!st) return null;
  const r = canvas.getBoundingClientRect();
  raycaster.setFromCamera(new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1), camera);
  // The node shapes, not the threads between frames: a line is hit from far around it.
  const targets = state.graph3d ? groups.graph3d.children.filter((o) => o.isMesh && o.userData.nodeKey) : [];
  st.group.traverse((o) => { if (o.isPoints) targets.push(o); });
  for (const hit of raycaster.intersectObjects(targets, false)) {
    if (hit.object.userData.nodeKey) return hit.object.userData.nodeKey;
    if (hit.index === undefined) continue;
    const i = srcVertexIndex(hit.object, hit.index);
    const id = i >= 0 ? st.regionsByTrack[state.track]?.labels[i] : 0;
    if (id) return nodeKey(state.track, id);
  }
  return null;
}

/** Glow node `key` in every panel and outline it in the 3D view. */
export function setHover(key) {
  if (key === state.hoverKey) return;
  state.hoverKey = key;
  setHoverKey([$('nodelink-body'), $('strip'), $('mb-body')], key);
  setHover3D(state.graph3d ? key : null);
  invalidate();
}

// ── Controls ────────────────────────────────────────────────────────────────

/** Make every control say what the state is. */
export function syncControls() {
  const clip = state.clip;
  for (const b of document.querySelectorAll('#mode-btns button')) b.classList.toggle('active', b.dataset.mode === state.mode);
  for (const b of document.querySelectorAll('#worldview-btns button')) b.classList.toggle('active', b.dataset.wv === state.worldView);
  $('worldview-btns').hidden = state.mode !== 'world';
  const canTools = !!instrumentFilter();
  for (const b of document.querySelectorAll('#show-btns button')) {
    b.classList.toggle('active', b.dataset.show === state.show);
    if (b.dataset.show === 'tools') {
      b.disabled = !canTools;
      b.title = canTools ? 'Only the points of instrument classes' : 'No track of this clip tells instruments from tissue';
    }
  }
  $('btn-graph3d').classList.toggle('active', state.graph3d);
  $('btn-graph3d').setAttribute('aria-pressed', String(state.graph3d));
  $('btn-strip-photo').classList.toggle('active', state.stripPhoto);
  $('btn-strip-photo').hidden = state.mode !== 'world';
  const pop = $('regions-pop');
  pop.innerHTML = '';
  for (const t of clip?.tracks ?? []) {
    const b = document.createElement('button');
    b.type = 'button';
    b.setAttribute('role', 'menuitemcheckbox');
    const on = state.painted && state.track === t.id;
    b.setAttribute('aria-checked', String(on));
    b.className = on ? 'active' : (state.track === t.id ? 'shown' : '');
    b.innerHTML = `${esc(t.short)}<small>${esc(t.name)}</small>`;
    b.addEventListener('click', () => { setRegionsMenu(false); toggleRegions(t.id); });
    pop.appendChild(b);
  }
  $('regions-now').textContent = state.painted ? trackInfo(state.track)?.short ?? 'on' : 'off';
  $('btn-regions').classList.toggle('active', state.painted);
  syncFocusBar();
}

export function setRegionsMenu(open) {
  $('regions-pop').hidden = !open;
  $('btn-regions').setAttribute('aria-expanded', String(open));
}

function syncFocusBar() {
  noteChange();
  const on = !!state.focusKey;
  $('fb-hint').hidden = on;
  $('fb-on').hidden = !on;
  if (on) {
    $('fb-name').textContent = state.focusLabel ?? labelOf(state.focusKey);
    $('fb-track').textContent = `· ${trackInfo(parseNodeKey(state.focusKey).track)?.short ?? ''}`;
  }
  for (const b of document.querySelectorAll('#fb-steps button')) {
    const n = Number(b.dataset.stage);
    b.classList.toggle('active', n === state.focusStage);
    b.classList.toggle('done', n < state.focusStage);
  }
}

/** The clip's place in its dataset, and a link to the video it was cut from. */
function renderClipInfo() {
  const c = state.clip;
  const el = $('clip-info');
  el.textContent = `${c.dataset_name} · ${c.procedure} · ${c.video} · ${c.n_frames} frames · ${c.geometry.name}`;
  if (c.source_url && /^https:\/\/www\.youtube\.com\/watch\?v=[\w-]+$/.test(c.source_url)) {
    const a = document.createElement('a');
    a.href = c.source_url;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.textContent = 'source video ↗';
    el.append(' · ', a);
  }
}
