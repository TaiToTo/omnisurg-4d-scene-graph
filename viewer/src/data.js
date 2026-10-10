// Fetch the catalog and a clip's files, and keep recent ones in memory.
//
// Every file lives under the data root: `data/` beside the page, or
// `VITE_DATA_ROOT` set at build time. A clip's files are named as the
// export wrote them; `viewer/README.md` lists them.
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { parseCatalog } from './lib/catalog.js';
import { DEFAULT_GEOMETRY } from './lib/overlays.js';

const DATA_ROOT = (import.meta.env?.VITE_DATA_ROOT || 'data/').replace(/\/?$/, '/');

const pad = (n, w) => String(n).padStart(w, '0');

/** Return the URL of a clip's file, from its catalog path and the file's path inside the clip. */
export function clipURL(clipPath, rel) {
  return `${DATA_ROOT}${clipPath}/${rel}`;
}

export const manifestURL = (clip) => clipURL(clip, 'frame_manifest.json');
export const rgbURL = (clip, i) => clipURL(clip, `input_images/${pad(i, 6)}.png`);
export const temporalURL = (clip, track) => clipURL(clip, `pc_vis/temporal_graph__${track}.json`);

/** Return the URL of a per-frame overlay: `seg_frame`, `graph_frame`, or `hierarchy_frame` without a track. */
export function overlayURL(clip, kind, i, track = null) {
  return clipURL(clip, `pc_vis/${kind}_${pad(i, 4)}${track ? `__${track}` : ''}.json`);
}

/** Return the URL of a frame's point cloud; the default source's files carry no suffix. */
export function glbURL(clip, i, source) {
  const suffix = source && source !== DEFAULT_GEOMETRY ? `__${source}` : '';
  return clipURL(clip, `pc_vis/frame_${pad(i, 4)}${suffix}.glb`);
}

/**
 * Fetch and validate the catalog.
 *
 * @throws {Error} when the catalog cannot be fetched or does not hold what the viewer needs.
 */
export async function loadCatalog() {
  const url = `${DATA_ROOT}catalog.json`;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`catalog: ${url} answered ${r.status}`);
  return parseCatalog(await r.json());
}

// Recent JSON files, by URL, as promises so concurrent asks share one fetch.
// Two bounds, because a segmentation frame parses to about 25 MB and a graph
// frame to well under 0.1 MB. The heavy bound holds World mode's stack of
// segmentations for two tracks, so re-entering World does not fetch again.
const MAX_HEAVY = 32;
const MAX_LIGHT = 256;
const isHeavy = (url) => url.includes('/seg_frame_');
const jsonCache = new Map();

function evictJSON(heavy) {
  let n = 0;
  for (const url of jsonCache.keys()) if (isHeavy(url) === heavy) n++;
  const bound = heavy ? MAX_HEAVY : MAX_LIGHT;
  for (const url of jsonCache.keys()) {
    if (n <= bound) return;
    if (isHeavy(url) === heavy) { jsonCache.delete(url); n--; }
  }
}

/**
 * Fetch a JSON file, or null when it is absent.
 *
 * A failure is not kept, so the next ask fetches again.
 */
export function fetchJSON(url) {
  let p = jsonCache.get(url);
  if (p) {
    jsonCache.delete(url);
    jsonCache.set(url, p);
    return p;
  }
  p = fetch(url)
    .then(async (r) => (r.ok ? r.json() : null))
    .catch((err) => {
      // A file that exists and does not parse is a fault, not an absent overlay.
      console.error(`could not read ${url}:`, err);
      return null;
    })
    .then((data) => {
      if (data === null && jsonCache.get(url) === p) jsonCache.delete(url);
      return data;
    });
  jsonCache.set(url, p);
  evictJSON(isHeavy(url));
  return p;
}

// Recent point clouds. A cloud on screen is a clone sharing the cached
// geometry, so only an evicted entry frees its geometry.
const MAX_GLB = 24;
const glbCache = new Map();
const gltfLoader = new GLTFLoader();

function disposeGLTF(gltf) {
  gltf?.scene?.traverse((obj) => {
    obj.geometry?.dispose();
    for (const m of [obj.material].flat()) m?.dispose();
  });
}

/** Load a GLB file; a failure is not kept. */
export function loadGLB(url) {
  let p = glbCache.get(url);
  if (p) {
    glbCache.delete(url);
    glbCache.set(url, p);
    return p;
  }
  p = gltfLoader.loadAsync(url);
  p.catch(() => { if (glbCache.get(url) === p) glbCache.delete(url); });
  glbCache.set(url, p);
  if (glbCache.size > MAX_GLB) {
    const oldest = glbCache.keys().next().value;
    glbCache.get(oldest).then(disposeGLTF).catch(() => {});
    glbCache.delete(oldest);
  }
  return p;
}

/** Forget every cached file, when another clip opens. */
export function clearCaches() {
  jsonCache.clear();
  for (const p of glbCache.values()) p.then(disposeGLTF).catch(() => {});
  glbCache.clear();
}

/** Load the next frame's cloud while the page is idle. */
export function prefetchGLB(url) {
  const idle = window.requestIdleCallback || ((fn) => setTimeout(fn, 200));
  idle(() => { loadGLB(url).catch(() => {}); });
}
