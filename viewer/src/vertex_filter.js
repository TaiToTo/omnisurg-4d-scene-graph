// Drop points from a cloud's geometry, and map a point of the copy back to the full cloud. The copy
// renumbers the points, while the labels index the full cloud, so the two are kept in one module.
import * as THREE from 'three';

/**
 * Copy every attribute of `geom`, keeping only vertices where mask[i] === 0.
 *
 * Reads through the RAW backing array with stride/offset so this works for
 * both plain and INTERLEAVED attributes, and copies raw values so a normalized
 * Uint8 colour attribute (what these GLBs carry) keeps its 0–255 encoding —
 * going through getX/setX would silently rescale it. The result is always
 * de-interleaved, which is fine: the filtered copy is only ever rendered,
 * never blended into (blending happens on the full geometry).
 *
 * The copy carries `userData.srcIndex`, filtered index to original index. Read it through
 * srcVertexIndex; never index the labels with a raw hit index.
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
 * A raycast hit index → index into the UNFILTERED vertex set.
 *
 * `hit.index` counts the points of the geometry on the object, which may be a filtered copy: past the
 * first dropped point the raw index names another point than the one clicked.
 *
 * @returns {number} original index, or -1 when the hit cannot be mapped.
 */
export function srcVertexIndex(obj, idx) {
  if (!Number.isInteger(idx) || idx < 0) return -1;
  if (!obj?.geometry) return -1;
  const map = obj.geometry.userData?.srcIndex;
  if (!map) return idx;
  return idx < map.length ? map[idx] : -1;
}

/**
 * The full geometry behind a Points, whose point order the labels follow; the live geometry when no
 * filter has run on it.
 */
export function fullGeometryOf(obj) {
  return obj?.userData?.fullGeometry ?? obj?.geometry ?? null;
}
