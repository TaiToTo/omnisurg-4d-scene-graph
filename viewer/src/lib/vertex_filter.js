// Drop vertices from a point cloud, and translate between the compacted copy
// and the full cloud.
//
// Following one region renders a compacted copy of the cloud, while every
// per-vertex array of the overlays (`vertex_seg` above all) indexes the full
// one. Anything that crosses between the two goes through this module.
import * as THREE from 'three';

/**
 * Copy `geom`, keeping only the vertices where `mask[i] === 0`.
 *
 * Raw values are copied through each attribute's stride and offset, so an
 * interleaved attribute works and a normalised Uint8 colour keeps its 0–255
 * encoding. The copy's `userData.srcIndex` maps a kept index to the original;
 * read it through `srcVertexIndex`.
 */
export function filteredGeometry(geom, mask) {
  const keep = [];
  for (let i = 0; i < mask.length; i++) if (!mask[i]) keep.push(i);
  const out = new THREE.BufferGeometry();
  out.userData.srcIndex = Int32Array.from(keep);
  for (const [name, attr] of Object.entries(geom.attributes)) {
    const size = attr.itemSize;
    const interleaved = attr.isInterleavedBufferAttribute;
    const src = interleaved ? attr.data.array : attr.array;
    const stride = interleaved ? attr.data.stride : size;
    const base = interleaved ? attr.offset : 0;
    const dst = new src.constructor(keep.length * size);
    for (let k = 0; k < keep.length; k++) {
      const from = keep[k] * stride + base;
      for (let c = 0; c < size; c++) dst[k * size + c] = src[from + c];
    }
    out.setAttribute(name, new THREE.BufferAttribute(dst, size, attr.normalized));
  }
  return out;
}

/**
 * Return the full-cloud index of a raycast hit on `obj`.
 *
 * @returns {number} the original vertex index, or -1 when the hit cannot be mapped.
 */
export function srcVertexIndex(obj, idx) {
  if (!Number.isInteger(idx) || idx < 0 || !obj?.geometry) return -1;
  const map = obj.geometry.userData?.srcIndex;
  if (!map) return idx;
  return idx < map.length ? map[idx] : -1;
}

/**
 * Return the mask that drops every vertex not labelled `id`, 1 meaning drop.
 *
 * @param {ArrayLike<number>} labels  `vertex_seg` of the cloud.
 * @param {number} count  vertices in the full cloud.
 * @returns {?Uint8Array} null when the labels do not fit the cloud: which
 *   vertices are the region is then unknown, and the frame is left whole rather
 *   than shown empty. A frame whose labels fit and hold no vertex of `id` drops
 *   everything, since the region is absent there.
 */
export function isolationMask(labels, count, id) {
  if (!(Array.isArray(labels) || ArrayBuffer.isView(labels)) || labels.length !== count) return null;
  const mask = new Uint8Array(count);
  for (let i = 0; i < count; i++) if (labels[i] !== id) mask[i] = 1;
  return mask;
}

/** Return the full geometry behind a Points object, whose vertex order matches `vertex_seg`. */
export function fullGeometryOf(obj) {
  return obj?.userData?.fullGeometry ?? obj?.geometry ?? null;
}
