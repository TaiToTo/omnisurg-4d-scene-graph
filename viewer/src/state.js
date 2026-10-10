// Hold what the page shows: the clip, the frame, the mode, the tracks, the
// followed region, and the Three.js groups the scene draws.
import * as THREE from 'three';
import { containmentPartners } from './lib/relations.js';

export const state = {
  catalog: null,           // parseCatalog output
  clip: null,              // the open clip's catalog entry
  manifest: null,          // frame_manifest.json, as fetched
  geoManifest: null,       // the manifest with the clip's geometry source's poses in place
  frame: 0,
  maxFrame: 0,
  mode: 'frame',           // 'frame' | 'world'
  worldFrames: null,       // how many clouds World stacks; null keeps the default
  worldView: 'focus',      // 'focus' spotlights one stacked frame; 'overlay' shows all alike

  // Which tracks the graph shows; the first is the primary one. Never empty
  // once a clip is open, since every catalog clip has a track.
  tracks: [],
  segVisible: false,       // region colours painted on the cloud
  graph3d: false,          // the scene graph drawn on the cloud
  stripPhoto: false,       // camera frames under the World band's regions
  playing: false,

  // The frame on screen in per-frame mode.
  segByTrack: {},
  graphByTrack: {},
  hierarchy: null,
  segAligned: true,        // false when an overlay could not be fitted to the cloud
  trackingStrip: null,     // per-frame band: { indices, frames }

  // World mode, built by world.js.
  world: null,             // { indices, origin, frames: [{i, offset, group, segByTrack, graphByTrack, hierarchy}] }
  worldFocus: 0,           // the stacked frame in the spotlight

  // Following one region.
  hoverKey: null,          // 'track:id' under the pointer
  focusNode: null,         // 'track:id' followed, or null
  focusLabel: null,
  focusStage: 0,           // 1 the region · 2 its motion in 3D · 3 its relations
  isolateFocus: false,     // only the followed region's points in the stacked frames
  nfTg: null,              // the followed node's temporal graph
  nfNodeId: null,
  nfEvents: [],
  selectedEvent: null,
  mapBig: false,

  sceneScale: 1,           // size of the 3D annotations relative to the reference cloud
  loadSeq: 0,              // bumped by every frame load; a stale load stops
  clipSeq: 0,              // bumped by every clip load

  pointsGroup: new THREE.Group(),
  cameraVizGroup: new THREE.Group(),
  highlightGroup: new THREE.Group(),
  graph3dGroup: new THREE.Group(),
};

/** Return the track descriptor of `id` from the open catalog. */
export function trackInfo(id) {
  return state.catalog?.tracks.get(id) ?? {
    id, label: id, short: id, semantic: false, seeded: false, anchorBadge: 'ANCHOR', trackedBadge: 'tracked',
  };
}

/** Return the track ids in drawing order: semantic tracks first, so the others land on top. */
export function orderTracks(ids) {
  return [...ids].sort((a, b) => Number(trackInfo(b).semantic) - Number(trackInfo(a).semantic));
}

/**
 * Return what is on stage: the per-frame cloud, or the stacked frame in the spotlight.
 *
 * @returns {?{i: number, group: object, segByTrack: object, graphByTrack: object, hierarchy: ?object, offset: number[]}}
 */
export function onStage() {
  if (state.mode === 'world' && state.world) {
    return state.world.frames.find((f) => f.i === state.worldFocus) ?? null;
  }
  const group = state.pointsGroup.children[0];
  return group ? {
    i: state.frame, group, segByTrack: state.segByTrack, graphByTrack: state.graphByTrack,
    hierarchy: state.hierarchy, offset: [0, 0, 0],
  } : null;
}

/**
 * Return the followed node's family as track → ids: the node itself and its containment partners on stage.
 *
 * An annotated region's family is the automatic regions it contains; an
 * automatic region's family is the annotated region containing it.
 */
export function focusFamily() {
  const out = Object.fromEntries((state.clip?.tracks ?? []).map((t) => [t, new Set()]));
  if (!state.focusNode) return out;
  for (const key of [state.focusNode, ...containmentPartners(state.focusNode, onStage()?.hierarchy)]) {
    const [t, id] = key.split(':');
    out[t]?.add(Number(id));
  }
  return out;
}

/** Return every cloud on screen with its overlays: one per-frame cloud, or every stacked frame. */
export function cloudsOnScreen() {
  if (state.mode === 'world' && state.world) return state.world.frames;
  const s = onStage();
  return s ? [s] : [];
}
