// Read the bundle `python -m pipeline.viewer_bundle` writes: the catalog, a clip's record, and its frames,
// clouds, regions and scene graphs. A file the clip's record lists is fetched once and kept in a bounded
// cache. A listed file that does not load is an error, never an empty frame.
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

/** The bundle format this page reads; a bundle of another format is refused. */
export const FORMAT = 1;

// A clip id names a directory of the bundle, so it may not climb out of it.
const SAFE_ID = /^[A-Za-z0-9][A-Za-z0-9_.-]*$/;
// The bounds hold what world mode and the strip read at once: a dozen frames' regions of each track.
const JSON_ENTRIES = 256;
const REGION_ENTRIES = 96;
const CLOUD_ENTRIES = 24;

/** The bundle's base URL: VITE_DATA_URL when the page was built with one, else ./data/ beside the page. */
export function dataBase() {
  const base = import.meta.env?.VITE_DATA_URL || 'data/';
  const page = globalThis.document?.baseURI ?? 'http://localhost/';
  return new URL(base.endsWith('/') ? base : `${base}/`, page);
}

/** Frame index as the bundle's four-digit file name. */
export function frameName(i) {
  return String(i).padStart(4, '0');
}

/** Refuse a record of another format, or one that is not an object. */
export function checkFormat(obj, what) {
  if (!obj || typeof obj !== 'object' || obj.format !== FORMAT) {
    throw new Error(`${what} is not a bundle of format ${FORMAT} (format ${obj?.format ?? 'missing'}); `
      + 'publish it again with this version of pipeline.viewer_bundle');
  }
  return obj;
}

/** Refuse a clip id that is not a plain directory name. */
export function checkClipId(id) {
  if (typeof id !== 'string' || !SAFE_ID.test(id)) throw new Error(`not a clip id: ${JSON.stringify(id)}`);
  return id;
}

/**
 * Expand run-length coded labels, `[value, count, ...]`, into one label per point.
 *
 * @param {number[]} runs
 * @param {number} n  the number of points the labels must cover.
 * @returns {Uint16Array|Int32Array} 16-bit when every label fits, which halves the memory a frame holds.
 * @throws when the runs do not cover exactly `n` points: labels shifted by one point paint every region
 *         in the wrong place, and look plausible doing it.
 */
export function decodeRuns(runs, n) {
  if (!Array.isArray(runs) || runs.length % 2) throw new Error('labels are not value/count pairs');
  let small = true;
  for (let k = 0; k < runs.length; k += 2) if (!(runs[k] >= 0 && runs[k] < 65536)) small = false;
  const out = small ? new Uint16Array(n) : new Int32Array(n);
  let at = 0;
  for (let k = 0; k < runs.length; k += 2) {
    const value = runs[k], count = runs[k + 1];
    if (!Number.isInteger(value) || !Number.isInteger(count) || count < 0 || at + count > n) {
      throw new Error(`labels run past the ${n} points they label`);
    }
    out.fill(value, at, at + count);
    at += count;
  }
  if (at !== n) throw new Error(`labels cover ${at} of ${n} points`);
  return out;
}

/** A least-recently-used map; `onEvict` frees what falls out. */
class LRU {
  constructor(max, onEvict = () => {}) { this.max = max; this.onEvict = onEvict; this.map = new Map(); }

  get(key) {
    const v = this.map.get(key);
    if (v !== undefined) { this.map.delete(key); this.map.set(key, v); }
    return v;
  }

  set(key, value) {
    this.map.set(key, value);
    while (this.map.size > this.max) {
      const [oldest, v] = this.map.entries().next().value;
      this.map.delete(oldest);
      this.onEvict(v);
    }
  }

  delete(key) { this.map.delete(key); }

  clear() { for (const v of this.map.values()) this.onEvict(v); this.map.clear(); }
}

const jsonCache = new LRU(JSON_ENTRIES);
const regionCache = new LRU(REGION_ENTRIES);

/**
 * Fetch and parse one file of the bundle, through the cache. `parse` turns the JSON into what the
 * caller keeps; the same object comes back while it stays cached, so a cache keyed on it works.
 * A failure is not cached: the next call asks the network again.
 */
function cachedJSON(path, parse = (x) => x, cache = jsonCache) {
  const url = new URL(path, dataBase()).href;
  let p = cache.get(url);
  if (!p) {
    p = fetch(url).then(async (r) => {
      if (!r.ok) throw new Error(`could not load ${path} (HTTP ${r.status})`);
      return parse(await r.json());
    });
    p.catch(() => { if (cache.get(url) === p) cache.delete(url); });
    cache.set(url, p);
  }
  return p;
}

/** The catalog: every clip of the bundle, as `{id, dataset_name, procedure, video, n_frames, ...}`. */
export async function loadCatalog() {
  const cat = checkFormat(await cachedJSON('catalog.json'), 'catalog.json');
  if (!Array.isArray(cat.clips)) throw new Error('catalog.json lists no clips');
  for (const c of cat.clips) checkClipId(c.id);
  return cat;
}

/** A clip's record, `clip.json`, with its tracks also by id. */
export async function loadClip(id) {
  checkClipId(id);
  return cachedJSON(`${id}/clip.json`, (rec) => {
    checkFormat(rec, `${id}/clip.json`);
    if (rec.id !== id) throw new Error(`${id}/clip.json describes ${rec.id}`);
    return { ...rec, trackById: new Map(rec.tracks.map((t) => [t.id, t])) };
  });
}

/**
 * One track's regions on frame `i`: the label of every point of the cloud, its classes, and whether the
 * frame is where the regions were made (`anchor`) or carried to (`propagated`). Null on a frame the track
 * does not cover.
 *
 * @returns {Promise<?{grid:number[], labels:Uint16Array|Int32Array, classes:object[], classById:Map, stage:string,
 *           anchorFrame:?number, track:string}>}
 */
export function fetchRegions(clip, track, i) {
  const t = clip.trackById.get(track);
  if (!t?.region_frames.includes(i)) return Promise.resolve(null);
  return cachedJSON(`${clip.id}/regions/${track}/${frameName(i)}.json`, (r) => {
    const [w, h] = r.grid;
    if (w !== clip.geometry.grid[0] || h !== clip.geometry.grid[1]) {
      throw new Error(`${clip.id} frame ${i}: ${track} regions are on a ${w}x${h} grid, the cloud on another`);
    }
    return {
      grid: [w, h], labels: decodeRuns(r.runs, w * h), classes: r.classes,
      classById: new Map(r.classes.map((c) => [c.id, c])),
      stage: r.stage, anchorFrame: Number.isInteger(r.anchor_frame) ? r.anchor_frame : null, track,
    };
  }, regionCache);
}

/** One track's scene graph on frame `i`, or null on a frame it has none. */
export function fetchGraph(clip, track, i) {
  const t = clip.trackById.get(track);
  if (!t?.graph_frames.includes(i)) return Promise.resolve(null);
  return cachedJSON(`${clip.id}/graphs/${track}/${frameName(i)}.json`);
}

/** One track's scene graph through time, or null when the bundle has none. */
export function fetchTemporal(clip, track) {
  if (!clip.trackById.get(track)?.temporal) return Promise.resolve(null);
  return cachedJSON(`${clip.id}/graphs/${track}/temporal.json`);
}

/** Which regions of one track contain which of another's on frame `i`, or null. */
export function fetchHierarchy(clip, i) {
  if (!clip.hierarchy?.frames.includes(i)) return Promise.resolve(null);
  return cachedJSON(`${clip.id}/hierarchy/${frameName(i)}.json`);
}

/** The URL of frame `i` as an image. */
export function frameImageURL(clip, i) {
  return new URL(`${clip.id}/frames/${frameName(i)}.jpg`, dataBase()).href;
}

function disposeGLTF(p) {
  p.then((gltf) => gltf.scene.traverse((o) => {
    o.geometry?.dispose();
    if (o.material) [].concat(o.material).forEach((m) => m.dispose());
  })).catch(() => {});
}

const cloudCache = new LRU(CLOUD_ENTRIES, disposeGLTF);
const gltfLoader = new GLTFLoader();

/** Frame `i`'s point cloud, as a parsed glTF. The caller clones it; the cache owns its geometry. */
export function loadCloud(clip, i) {
  const url = new URL(`${clip.id}/clouds/${frameName(i)}.glb`, dataBase()).href;
  let p = cloudCache.get(url);
  if (!p) {
    p = gltfLoader.loadAsync(url);
    p.catch(() => { if (cloudCache.get(url) === p) cloudCache.delete(url); });
    cloudCache.set(url, p);
  }
  return p;
}

/** Load the next frame's cloud while the page is idle, so stepping forward does not wait. */
export function prefetchCloud(clip, i) {
  if (i >= clip.n_frames) return;
  const idle = globalThis.requestIdleCallback ?? ((fn) => setTimeout(fn, 200));
  idle(() => { loadCloud(clip, i).catch(() => {}); });
}

/** Drop every cached file: a clip's files are not read again once another clip is open. */
export function clearCaches() {
  jsonCache.clear();
  regionCache.clear();
  cloudCache.clear();
}
