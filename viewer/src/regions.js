// Each region's shape in the image, from one frame's labels: an ellipse of equal area, the convex hull,
// and the traced outline. The 2D graph places its nodes by them and the strip draws them. Results are
// cached per regions object, which the data cache returns unchanged while a frame stays loaded.

/**
 * The ellipse of a region's pixel moments: centred on its centroid, oriented by its covariance, and
 * scaled so its area equals the region's pixel count. A region of under 12 pixels is a small dot.
 *
 * @param {{n:number, sx:number, sy:number, sxx:number, syy:number, sxy:number}} m
 * @returns {?{cx:number, cy:number, a:number, b:number, theta:number, n:number}} semi-axes a >= b, and
 *          the major axis' angle in radians.
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
  return { cx: mx, cy: my, a: s * Math.sqrt(l1), b: s * Math.sqrt(l2), theta: 0.5 * Math.atan2(2 * cxy, cxx - cyy), n: m.n };
}

/** Region id to its ellipse (`fitEllipse`), in image pixels. Background, label 0, has none. */
export function regionEllipses({ grid: [w], labels }) {
  const moments = new Map();
  for (let i = 0; i < labels.length; i++) {
    const id = labels[i];
    if (!id) continue;
    let m = moments.get(id);
    if (!m) { m = { n: 0, sx: 0, sy: 0, sxx: 0, syy: 0, sxy: 0 }; moments.set(id, m); }
    const x = i % w, y = (i / w) | 0;
    m.n++; m.sx += x; m.sy += y; m.sxx += x * x; m.syy += y * y; m.sxy += x * y;
  }
  const out = new Map();
  for (const [id, m] of moments) out.set(id, fitEllipse(m));
  return out;
}

/** The convex hull of a point set, counter-clockwise (Andrew's monotone chain). */
export function convexHull2D(pts) {
  if (!Array.isArray(pts) || pts.length < 3) return (pts || []).slice();
  const p = pts.slice().sort((a, b) => (a[0] - b[0]) || (a[1] - b[1]));
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const half = (seq) => {
    const out = [];
    for (const q of seq) {
      while (out.length >= 2 && cross(out[out.length - 2], out[out.length - 1], q) <= 0) out.pop();
      out.push(q);
    }
    out.pop();
    return out;
  };
  return half(p).concat(half(p.slice().reverse()));
}

/** Region id to the convex hull of its pixels, in image pixels. */
export function regionHulls({ grid: [w, h], labels }) {
  // Only the leftmost and rightmost pixel of a region on each row can be a hull vertex.
  const rows = new Map();
  for (let i = 0; i < labels.length; i++) {
    const id = labels[i];
    if (!id) continue;
    let r = rows.get(id);
    if (!r) { r = { minX: new Int32Array(h).fill(w), maxX: new Int32Array(h).fill(-1) }; rows.set(id, r); }
    const x = i % w, y = (i / w) | 0;
    if (x < r.minX[y]) r.minX[y] = x;
    if (x > r.maxX[y]) r.maxX[y] = x;
  }
  const out = new Map();
  for (const [id, r] of rows) {
    const pts = [];
    for (let y = 0; y < h; y++) {
      if (r.maxX[y] < 0) continue;
      pts.push([r.minX[y], y]);
      if (r.maxX[y] !== r.minX[y]) pts.push([r.maxX[y], y]);
    }
    const hull = convexHull2D(pts);
    if (hull.length >= 3) out.set(id, hull);
  }
  return out;
}

/**
 * Trace the outer boundary of one region's largest row run, on a grid sampled every `step` pixels
 * (Moore-neighbour tracing, clockwise). Holes and detached pieces are not traced.
 *
 * @returns {Array<[number, number]>} the outline in image pixels, at least 3 points, or [].
 */
export function traceOutline(labels, w, h, id, step = 2) {
  const gw = Math.ceil(w / step), gh = Math.ceil(h / step);
  const at = (gx, gy) => gx >= 0 && gy >= 0 && gx < gw && gy < gh && labels[(gy * step) * w + gx * step] === id;

  // Start on the left end of the longest row run, which skips one-pixel specks without labelling components.
  let sx = -1, sy = -1, best = 0;
  for (let gy = 0; gy < gh; gy++) {
    let run = 0, start = -1;
    for (let gx = 0; gx <= gw; gx++) {
      if (at(gx, gy)) {
        if (run === 0) start = gx;
        run++;
      } else if (run > 0) {
        if (run > best) { best = run; sx = start; sy = gy; }
        run = 0;
      }
    }
  }
  if (sx < 0) return [];

  // Neighbours clockwise from the west.
  const DIRS = [[-1, 0], [-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1]];
  const out = [];
  let cx = sx, cy = sy, back = 0;
  for (let n = 0; n < 4 * gw * gh; n++) {
    out.push([Math.min(cx * step, w - 1), Math.min(cy * step, h - 1)]);
    let found = -1;
    for (let k = 1; k <= 8; k++) {
      const d = (back + k) % 8;
      if (at(cx + DIRS[d][0], cy + DIRS[d][1])) { found = d; break; }
    }
    if (found < 0) break;
    cx += DIRS[found][0];
    cy += DIRS[found][1];
    back = (found + 5) % 8;
    if (cx === sx && cy === sy) break;
  }
  return out.length >= 3 ? out : [];
}

/** Simplify a closed polygon (Ramer-Douglas-Peucker): drop the points within `eps` of the line kept. */
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

/** Region id to its simplified outline, in image pixels. */
export function regionOutlines({ grid: [w, h], labels, classes }, step = 2, eps = 3) {
  const out = new Map();
  for (const { id } of classes) {
    if (!id) continue;
    const poly = simplifyRDP(traceOutline(labels, w, h, id, step), eps);
    if (poly.length >= 3) out.set(id, poly);
  }
  return out;
}

function cachedBy(fn) {
  const cache = new WeakMap();
  return (regions) => {
    if (!regions) return null;
    let v = cache.get(regions);
    if (!v) { v = fn(regions); cache.set(regions, v); }
    return v;
  };
}

/** The cached forms of the three shapes: computed once per frame's regions. */
export const ellipsesOf = cachedBy(regionEllipses);
export const hullsOf = cachedBy(regionHulls);
export const outlinesOf = cachedBy(regionOutlines);
