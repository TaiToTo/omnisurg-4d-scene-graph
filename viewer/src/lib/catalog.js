// Read the catalog the viewer opens: which clips there are, which tracks each
// one has, and how each track is named.
//
// `python -m pipeline.viewer_catalog` writes the catalog. The viewer refuses a
// catalog that does not hold what it needs, rather than draw half a clip:
// every clip path and track id below becomes part of a URL or a node key.

/** The format string the catalog builder writes, and the only one read here. */
export const CATALOG_FORMAT = 'omnisurg-viewer-catalog/1';

const PATH_RE = /^[A-Za-z0-9_-][A-Za-z0-9._-]*(\/[A-Za-z0-9_-][A-Za-z0-9._-]*)*$/;
const ID_RE = /^[a-z0-9_]+$/;

function fail(msg) {
  throw new Error(`catalog: ${msg}`);
}

function text(v, where) {
  if (typeof v !== 'string' || !v.trim()) fail(`${where} is not a non-empty string`);
  return v;
}

function parseTrack(id, t) {
  if (!ID_RE.test(id)) fail(`track id ${JSON.stringify(id)} is not lower-case letters, digits and _`);
  if (!t || typeof t !== 'object') fail(`track ${id} is not an object`);
  for (const key of ['semantic', 'seeded']) {
    if (typeof t[key] !== 'boolean') fail(`track ${id}: ${key} is not a boolean`);
  }
  return {
    id,
    label: text(t.label, `track ${id}: label`),
    short: text(t.short, `track ${id}: short`),
    semantic: t.semantic,
    seeded: t.seeded,
    anchorBadge: text(t.anchor_badge, `track ${id}: anchor_badge`),
    trackedBadge: text(t.tracked_badge, `track ${id}: tracked_badge`),
  };
}

function parseClip(c, k, tracks) {
  const where = `clip ${k}`;
  if (!c || typeof c !== 'object') fail(`${where} is not an object`);
  if (typeof c.path !== 'string' || !PATH_RE.test(c.path)) fail(`${where}: path ${JSON.stringify(c.path)} is not a relative path of plain names`);
  if (!Number.isInteger(c.frames) || c.frames < 1) fail(`${c.path}: frames is not a positive integer`);
  if (typeof c.geometry !== 'string' || !ID_RE.test(c.geometry)) fail(`${c.path}: geometry is not an id`);
  if (typeof c.hierarchy !== 'boolean') fail(`${c.path}: hierarchy is not a boolean`);
  if (!Array.isArray(c.tracks) || !c.tracks.length) fail(`${c.path}: has no track`);
  for (const t of c.tracks) if (!tracks.has(t)) fail(`${c.path}: track ${JSON.stringify(t)} is not in the track table`);
  if (new Set(c.tracks).size !== c.tracks.length) fail(`${c.path}: lists a track twice`);
  const coverage = {};
  for (const t of c.tracks) {
    const n = c.coverage?.[t];
    if (!Number.isInteger(n) || n < 1 || n > c.frames) fail(`${c.path}: coverage of ${t} is not between 1 and ${c.frames}`);
    coverage[t] = n;
  }
  return {
    path: c.path,
    dataset: text(c.dataset, `${c.path}: dataset`),
    group: text(c.group, `${c.path}: group`),
    label: text(c.label, `${c.path}: label`),
    frames: c.frames,
    geometry: c.geometry,
    hierarchy: c.hierarchy,
    tracks: [...c.tracks],
    coverage,
  };
}

/**
 * Validate a parsed catalog and return it in the viewer's shape.
 *
 * @param {object} json  the parsed `catalog.json`.
 * @returns {{tracks: Map<string, object>, clips: object[], byPath: Map<string, object>}}
 * @throws {Error} on any field the viewer cannot use.
 */
export function parseCatalog(json) {
  if (!json || json.format !== CATALOG_FORMAT) fail(`format is not ${CATALOG_FORMAT}`);
  if (!json.tracks || typeof json.tracks !== 'object') fail('has no track table');
  const tracks = new Map(Object.entries(json.tracks).map(([id, t]) => [id, parseTrack(id, t)]));
  if (!Array.isArray(json.clips) || !json.clips.length) fail('has no clip');
  const clips = json.clips.map((c, k) => parseClip(c, k, tracks));
  const byPath = new Map();
  for (const c of clips) {
    if (byPath.has(c.path)) fail(`${c.path} is listed twice`);
    byPath.set(c.path, c);
  }
  return { tracks, clips, byPath };
}

/** Group the clips for the picker: dataset, then group, in catalog order. */
export function groupClips(catalog) {
  const datasets = new Map();
  for (const c of catalog.clips) {
    if (!datasets.has(c.dataset)) datasets.set(c.dataset, new Map());
    const groups = datasets.get(c.dataset);
    if (!groups.has(c.group)) groups.set(c.group, []);
    groups.get(c.group).push(c);
  }
  return [...datasets].map(([dataset, groups]) => ({
    dataset, groups: [...groups].map(([label, clips]) => ({ label, clips })),
  }));
}
