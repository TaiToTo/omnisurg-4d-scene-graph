// Trace the outline of each region of a segmentation frame, concave parts
// included, and simplify it.
//
// The band of the graph through time draws these outlines: unlike a convex
// hull, they follow a grasper's jaws or a fold of tissue.
import { classIdRaster } from './regions.js';

const DIRS = [[-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1]];

/**
 * Trace the outer boundary of class `id` on a grid sampled every `step` pixels.
 *
 * The trace starts at the longest row run of the class, which skips specks
 * without labelling components, and walks the Moore neighbourhood clockwise.
 * Holes and detached blobs are left out: the outline is drawn, not measured.
 *
 * @param {Int32Array} ids  class-id raster, `w` by `h`.
 * @returns {Array<[number, number]>} the closed outline in raster pixels, or [].
 */
export function traceOutline(ids, w, h, id, step = 2) {
  const gw = Math.ceil(w / step), gh = Math.ceil(h / step);
  const at = (gx, gy) => gx >= 0 && gy >= 0 && gx < gw && gy < gh && ids[(gy * step) * w + gx * step] === id;

  // Start on the longest row run.
  let sx = -1, sy = -1, bestRun = 0;
  for (let gy = 0; gy < gh; gy++) {
    let run = 0, runStart = -1;
    for (let gx = 0; gx <= gw; gx++) {
      if (at(gx, gy)) {
        if (run === 0) runStart = gx;
        run++;
      } else if (run > 0) {
        if (run > bestRun) { bestRun = run; sx = runStart; sy = gy; }
        run = 0;
      }
    }
  }
  if (sx < 0) return [];

  // Walk the boundary until it closes.
  const out = [];
  let cx = sx, cy = sy, backtrack = 0;
  for (let n = 0; n < 4 * gw * gh; n++) {
    out.push([Math.min(cx * step, w - 1), Math.min(cy * step, h - 1)]);
    let found = -1;
    for (let k = 1; k <= 8; k++) {
      const d = (backtrack + k) % 8;
      if (at(cx + DIRS[d][0], cy + DIRS[d][1])) { found = d; break; }
    }
    if (found < 0) break;
    cx += DIRS[found][0];
    cy += DIRS[found][1];
    backtrack = (found + 5) % 8;
    if (cx === sx && cy === sy) break;
  }
  return out.length >= 3 ? out : [];
}

/** Simplify a closed polygon with Ramer–Douglas–Peucker at tolerance `eps`. */
export function simplifyRDP(pts, eps = 2.5) {
  if (!Array.isArray(pts) || pts.length <= 4) return (pts || []).slice();
  const keep = new Uint8Array(pts.length);
  keep[0] = 1;
  keep[pts.length - 1] = 1;
  const stack = [[0, pts.length - 1]];
  while (stack.length) {
    const [a, b] = stack.pop();
    const [ax, ay] = pts[a], [bx, by] = pts[b];
    const dx = bx - ax, dy = by - ay;
    const len = Math.hypot(dx, dy) || 1e-9;
    let worst = -1, worstD = eps;
    for (let i = a + 1; i < b; i++) {
      const d = Math.abs(dy * pts[i][0] - dx * pts[i][1] + bx * ay - by * ax) / len;
      if (d > worstD) { worstD = d; worst = i; }
    }
    if (worst > 0) {
      keep[worst] = 1;
      stack.push([a, worst], [worst, b]);
    }
  }
  return pts.filter((_, i) => keep[i]);
}

/** Return class id → simplified outline (image pixels) for every region of `seg`. */
export function regionOutlines(seg, step = 2, eps = 2.5) {
  const out = new Map();
  const r = classIdRaster(seg);
  if (!r) return out;
  for (const cls of seg.classes || []) {
    if (cls.id === 0) continue;
    const poly = simplifyRDP(traceOutline(r.ids, r.w, r.h, cls.id, step), eps);
    if (poly.length >= 3) out.set(cls.id, poly);
  }
  return out;
}
