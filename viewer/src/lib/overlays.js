// Fit a frame's segmentation and graph to the point cloud on screen.
//
// The overlays were exported against the default depth model's cloud
// (`vertex_seg[i]` labels its vertex i; a node's `pos` is in its frame). A
// cloud of another model has another grid and another scale, so on it the
// labels are resampled and the node geometry is read off the cloud again.
// Labels, colours, `area_frac` and the edges stay as exported.
import { isLabelArray, regionNodeGeometry, regraphFromCloud, remapLabels } from './cloud_geometry.js';

/** The depth model whose files carry no `__<source>` suffix and whose cloud the overlays index. */
export const DEFAULT_GEOMETRY = 'da3';

/**
 * Return the pixel grid a geometry source's clouds were built on.
 *
 * @returns {?{w: number, h: number}} null when the manifest does not say.
 */
export function sourceGrid(manifest, source) {
  if (!manifest) return null;
  if (!source || source === DEFAULT_GEOMETRY) {
    const s = manifest.depth_info?.depth_shape;
    return Array.isArray(s) && s.length === 3 ? { w: s[2], h: s[1] } : null;
  }
  const r = manifest.geometry_sources?.[source]?.resolution;
  return Array.isArray(r) && r.length === 2 ? { w: r[0], h: r[1] } : null;
}

/**
 * Return the manifest with a geometry source's per-frame centroid and camera in the frame's own fields.
 *
 * Everything that places a cloud reads `glb_centroid` and `camera_*_glb`; the
 * other sources keep theirs under `frames[i].geometry_sources[source]`.
 *
 * @returns {object} a new manifest; the input is not mutated.
 * @throws {Error} when a frame lacks the source's entry: its cloud would be
 *   placed with another model's pose.
 */
export function manifestForSource(manifest, source) {
  if (!source || source === DEFAULT_GEOMETRY) return manifest;
  return {
    ...manifest,
    frames: manifest.frames.map((fr, i) => {
      const geo = fr.geometry_sources?.[source];
      if (!geo) throw new Error(`frame_manifest.json: frame ${i} has no ${source} pose`);
      return { ...fr, ...geo };
    }),
  };
}

const alignCache = new WeakMap();

/**
 * Return a seg whose `vertex_seg` indexes a cloud on `grid`.
 *
 * Only `vertex_seg` changes; the 2D fields keep describing the image. Returns
 * null when the labels cannot be put on the grid honestly: the grid is
 * unknown, or the labels are not the whole source grid.
 */
export function alignSeg(seg, grid) {
  if (!seg || !isLabelArray(seg.vertex_seg)) return seg;
  if (!(grid?.w > 0 && grid?.h > 0)) return null;
  if (seg.vertex_seg.length === grid.w * grid.h) return seg;
  const key = `${grid.w}x${grid.h}`;
  let byGrid = alignCache.get(seg);
  if (!byGrid) { byGrid = new Map(); alignCache.set(seg, byGrid); }
  if (!byGrid.has(key)) {
    const labels = remapLabels(seg.vertex_seg, seg.width, seg.height, grid.w, grid.h);
    byGrid.set(key, labels ? { ...seg, vertex_seg: labels } : null);
  }
  return byGrid.get(key);
}

function positionAttrOf(group) {
  let attr = null;
  group?.traverse?.((o) => { if (!attr && o.isPoints) attr = o.geometry.attributes.position; });
  return attr;
}

/**
 * Fit one frame's overlays to the cloud in `group`.
 *
 * @param {object} p
 * @param {object} p.segByTrack    track → seg_frame JSON.
 * @param {object} p.graphByTrack  track → graph_frame JSON.
 * @param {object} p.group         the Three.js group holding the frame's cloud.
 * @param {object} p.manifest      frame_manifest.json.
 * @param {string} p.source        the geometry source of the cloud.
 * @returns {{segByTrack: object, graphByTrack: object, aligned: boolean}} `aligned` is
 *   false when a track had overlays that could not be put on this cloud; that
 *   track's graph is then empty rather than drawn at the other cloud's coordinates.
 */
export function adaptOverlays({ segByTrack, graphByTrack, group, manifest, source }) {
  if (!source || source === DEFAULT_GEOMETRY) return { segByTrack, graphByTrack, aligned: true };

  // Resample each track's labels onto this cloud's grid.
  const grid = sourceGrid(manifest, source);
  const seg = {};
  let aligned = true;
  for (const [track, s] of Object.entries(segByTrack || {})) {
    seg[track] = alignSeg(s, grid);
    if (s && !seg[track]) aligned = false;
  }

  // Read each track's node geometry off the cloud.
  const pos = positionAttrOf(group);
  const graphs = {};
  for (const [track, graph] of Object.entries(graphByTrack || {})) {
    if (!graph) { graphs[track] = graph; continue; }
    const labels = seg[track]?.vertex_seg;
    graphs[track] = labels && pos
      ? regraphFromCloud(graph, regionNodeGeometry(pos, labels))
      : { ...graph, nodes: [], edges: [] };
    if (graph.nodes?.length && !graphs[track].nodes.length) aligned = false;
  }
  return { segByTrack: seg, graphByTrack: graphs, aligned };
}
