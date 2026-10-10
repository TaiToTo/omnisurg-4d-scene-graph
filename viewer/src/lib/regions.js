// Fit each region of a segmentation frame with an ellipse and a convex hull,
// in image pixels.
//
// A region's mask is the camera's view of its 3D surface, so its image-space
// spread is the region's projected footprint. The 2D scene-graph map draws
// each node there. The class-id raster every fit starts from is built once
// per frame and kept for as long as the frame's JSON is.

const rasterCache = new WeakMap();

/**
 * Return the class id of every pixel of a `seg_frame`, 0 for background.
 *
 * A pixel's class is found by its exact palette colour in `seg_colors`; a
 * colour no class has is background.
 *
 * @param {{width: number, height: number, seg_colors: number[][], classes: Array}} seg
 * @returns {?{ids: Int32Array, w: number, h: number}} null without a raster.
 */
export function classIdRaster(seg) {
  if (!seg || !Array.isArray(seg.seg_colors) || !seg.width || !seg.height) return null;
  const hit = rasterCache.get(seg);
  if (hit) return hit;
  const colorToId = new Map();
  for (const cls of seg.classes || []) {
    if (cls.id === 0 || !Array.isArray(cls.color)) continue;
    colorToId.set(colorKey(cls.color), cls.id);
  }
  const ids = new Int32Array(seg.width * seg.height);
  const n = Math.min(ids.length, seg.seg_colors.length);
  for (let i = 0; i < n; i++) ids[i] = colorToId.get(colorKey(seg.seg_colors[i])) ?? 0;
  const raster = { ids, w: seg.width, h: seg.height };
  rasterCache.set(seg, raster);
  return raster;
}

function colorKey(c) {
  return (Math.round(c[0] * 255) << 16) | (Math.round(c[1] * 255) << 8) | Math.round(c[2] * 255);
}

/**
 * Turn a region's pixel moments into an ellipse of the region's area.
 *
 * The axes come from the covariance; they are scaled so that the ellipse's
 * area equals the pixel count, which keeps a thin or scattered region from
 * ballooning. Fewer than 12 pixels give a small dot.
 *
 * @param {{n, sx, sy, sxx, syy, sxy}} m
 * @returns {?{cx, cy, a, b, theta, n}} centre, semi-axes and major-axis angle (radians).
 */
export function fitEllipse(m) {
  if (!m || m.n < 1) return null;
  const mx = m.sx / m.n, my = m.sy / m.n;
  if (m.n < 12) return { cx: mx, cy: my, a: 4, b: 4, theta: 0, n: m.n };
  const cxx = m.sxx / m.n - mx * mx;
  const cyy = m.syy / m.n - my * my;
  const cxy = m.sxy / m.n - mx * my;
  const tr = cxx + cyy, det = cxx * cyy - cxy * cxy;
  const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
  const l1 = Math.max(0.25, tr / 2 + disc);
  const l2 = Math.max(0.25, tr / 2 - disc);
  const s = Math.sqrt(m.n / (Math.PI * Math.sqrt(l1 * l2)));
  return {
    cx: mx, cy: my,
    a: s * Math.sqrt(l1),
    b: s * Math.sqrt(l2),
    theta: 0.5 * Math.atan2(2 * cxy, cxx - cyy),
    n: m.n,
  };
}

/** Return class id → ellipse (image pixels) for every region of `seg`. */
export function regionEllipses(seg) {
  const out = new Map();
  const r = classIdRaster(seg);
  if (!r) return out;
  const moments = new Map();
  for (let i = 0; i < r.ids.length; i++) {
    const cid = r.ids[i];
    if (!cid) continue;
    let m = moments.get(cid);
    if (!m) { m = { n: 0, sx: 0, sy: 0, sxx: 0, syy: 0, sxy: 0 }; moments.set(cid, m); }
    const x = i % r.w, y = (i / r.w) | 0;
    m.n++; m.sx += x; m.sy += y; m.sxx += x * x; m.syy += y * y; m.sxy += x * y;
  }
  for (const [cid, m] of moments) out.set(cid, fitEllipse(m));
  return out;
}

/** Return the convex hull of `pts`, counter-clockwise (Andrew's monotone chain). */
export function convexHull2D(pts) {
  if (!Array.isArray(pts) || pts.length < 3) return (pts || []).slice();
  const p = pts.slice().sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [];
  for (const q of p) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop();
    lower.push(q);
  }
  const upper = [];
  for (let i = p.length - 1; i >= 0; i--) {
    const q = p[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop();
    upper.push(q);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

/**
 * Return class id → convex hull polygon (image pixels) for every region of `seg`.
 *
 * Only a scanline's leftmost and rightmost region pixel can be a hull vertex,
 * so the hull is taken over those alone.
 */
export function regionHulls(seg) {
  const out = new Map();
  const r = classIdRaster(seg);
  if (!r) return out;
  const rows = new Map();
  for (let i = 0; i < r.ids.length; i++) {
    const cid = r.ids[i];
    if (!cid) continue;
    let e = rows.get(cid);
    if (!e) { e = { minX: new Int32Array(r.h).fill(r.w), maxX: new Int32Array(r.h).fill(-1) }; rows.set(cid, e); }
    const x = i % r.w, y = (i / r.w) | 0;
    if (x < e.minX[y]) e.minX[y] = x;
    if (x > e.maxX[y]) e.maxX[y] = x;
  }
  for (const [cid, e] of rows) {
    const pts = [];
    for (let y = 0; y < r.h; y++) {
      if (e.maxX[y] < 0) continue;
      pts.push([e.minX[y], y]);
      if (e.maxX[y] !== e.minX[y]) pts.push([e.maxX[y], y]);
    }
    const hull = convexHull2D(pts);
    if (hull.length >= 3) out.set(cid, hull);
  }
  return out;
}
