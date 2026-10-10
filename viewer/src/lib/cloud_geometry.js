// Carry a segmentation's per-vertex labels onto a point cloud reconstructed on
// another pixel grid, and read each region's centroid and axes off a cloud.
//
// The overlays are exported against one depth model's cloud. A second model
// reconstructs the same frame on its own grid. Both grids are dense row-major
// rasters of one image, so labels transfer by nearest-neighbour resampling in
// normalised image coordinates.

/**
 * Return whether `v` is a usable per-vertex label array.
 *
 * A typed array counts: resampled labels are an `Int32Array`, and
 * `Array.isArray` is false for it.
 */
export function isLabelArray(v) {
  return Array.isArray(v) || ArrayBuffer.isView(v);
}

/**
 * Map each destination pixel to the source pixel under its centre: `out[dst] = src`.
 *
 * @returns {?Int32Array} null when either grid is empty.
 */
export function gridRemapIndex(srcW, srcH, dstW, dstH) {
  if (!(srcW > 0 && srcH > 0 && dstW > 0 && dstH > 0)) return null;
  const rows = new Int32Array(dstH);
  for (let r = 0; r < dstH; r++) rows[r] = Math.min(srcH - 1, Math.floor(((r + 0.5) * srcH) / dstH));
  const cols = new Int32Array(dstW);
  for (let c = 0; c < dstW; c++) cols[c] = Math.min(srcW - 1, Math.floor(((c + 0.5) * srcW) / dstW));
  const out = new Int32Array(dstW * dstH);
  for (let r = 0, k = 0; r < dstH; r++) {
    const base = rows[r] * srcW;
    for (let c = 0; c < dstW; c++, k++) out[k] = base + cols[c];
  }
  return out;
}

/**
 * Resample per-vertex labels from one dense grid onto another.
 *
 * Refuses (null) a label array that is not the whole source grid: such a
 * cloud dropped vertices, and which ones cannot be recovered.
 *
 * @returns {?ArrayLike<number>} labels on the destination grid; the input itself when the grids match.
 */
export function remapLabels(labels, srcW, srcH, dstW, dstH) {
  if (!isLabelArray(labels) || labels.length !== srcW * srcH) return null;
  if (srcW === dstW && srcH === dstH) return labels;
  const map = gridRemapIndex(srcW, srcH, dstW, dstH);
  if (!map) return null;
  const out = new Int32Array(map.length);
  for (let i = 0; i < map.length; i++) out[i] = labels[map[i]];
  return out;
}

// Cyclic Jacobi rather than the closed-form cubic: a region's covariance is
// often near-planar, and the cubic loses its eigenvectors exactly there.
const JACOBI_SWEEPS = 12;

/**
 * Decompose a symmetric 3×3 matrix.
 *
 * @param {number[][]} m  not mutated.
 * @returns {{values: number[], vectors: number[][]}} eigenvalues descending, unit eigenvectors as rows.
 */
export function eigenSym3(m) {
  const a = m.map((row) => row.slice());
  const v = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const PAIRS = [[0, 1], [0, 2], [1, 2]];
  for (let sweep = 0; sweep < JACOBI_SWEEPS; sweep++) {
    let off = 0;
    for (const [p, q] of PAIRS) off += a[p][q] * a[p][q];
    if (off < 1e-24) break;
    for (const [p, q] of PAIRS) {
      if (Math.abs(a[p][q]) < 1e-18) continue;
      const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < 3; k++) {
        const akp = a[k][p], akq = a[k][q];
        a[k][p] = c * akp - s * akq;
        a[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = a[p][k], aqk = a[q][k];
        a[p][k] = c * apk - s * aqk;
        a[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < 3; k++) {
        const vkp = v[k][p], vkq = v[k][q];
        v[k][p] = c * vkp - s * vkq;
        v[k][q] = s * vkp + c * vkq;
      }
    }
  }
  const order = [0, 1, 2].sort((i, j) => a[j][j] - a[i][i]);
  return {
    values: order.map((i) => a[i][i]),
    vectors: order.map((i) => [v[0][i], v[1][i], v[2][i]]),
  };
}

// A drawn axis's half-length in standard deviations. Only the ratio of the
// three axes reaches the screen, so the exporter's percentile width and this
// give the same glyph.
const SIGMA_TO_HALF = 2.0;

/**
 * Read each region's centroid and principal axes off a cloud in one pass.
 *
 * @param {{count: number, getX: Function, getY: Function, getZ: Function}} posAttr  positions.
 * @param {ArrayLike<number>} labels  region id per vertex, 0 for background; `posAttr.count` long.
 * @returns {Map<number, {pos: number[], axes: number[][], axes_radius: number[], vertex_count: number}>}
 *   empty when the labels do not fit the cloud.
 */
export function regionNodeGeometry(posAttr, labels) {
  const out = new Map();
  const n = posAttr?.count ?? 0;
  if (!n || !isLabelArray(labels) || labels.length !== n) return out;

  // Accumulate count, sums and products per region.
  const acc = new Map();
  for (let i = 0; i < n; i++) {
    const id = labels[i];
    if (!id) continue;
    let a = acc.get(id);
    if (!a) { a = new Float64Array(10); acc.set(id, a); }
    const x = posAttr.getX(i), y = posAttr.getY(i), z = posAttr.getZ(i);
    a[0] += 1;
    a[1] += x; a[2] += y; a[3] += z;
    a[4] += x * x; a[5] += y * y; a[6] += z * z;
    a[7] += x * y; a[8] += x * z; a[9] += y * z;
  }

  // Turn the moments into a centroid and axes.
  for (const [id, a] of acc) {
    const c = a[0];
    const mx = a[1] / c, my = a[2] / c, mz = a[3] / c;
    if (c < 3) {
      out.set(id, { pos: [mx, my, mz], axes: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], axes_radius: [0, 0, 0], vertex_count: c });
      continue;
    }
    const cov = [
      [a[4] / c - mx * mx, a[7] / c - mx * my, a[8] / c - mx * mz],
      [a[7] / c - mx * my, a[5] / c - my * my, a[9] / c - my * mz],
      [a[8] / c - mx * mz, a[9] / c - my * mz, a[6] / c - mz * mz],
    ];
    const { values, vectors } = eigenSym3(cov);
    out.set(id, {
      pos: [mx, my, mz],
      axes: vectors,
      axes_radius: values.map((l) => SIGMA_TO_HALF * Math.sqrt(Math.max(l, 0))),
      vertex_count: c,
    });
  }
  return out;
}

/**
 * Return a `graph_frame` whose node geometry is the one read off the cloud.
 *
 * Label, colour, `area_frac` and the relation edges are kept. A node with no
 * vertex on this cloud is dropped with its edges, and an empty `geom` drops
 * every node: a node drawn at the other reconstruction's coordinates looks
 * right and is wrong.
 *
 * @returns {?object} a new graph; the input is not mutated.
 */
export function regraphFromCloud(graph, geom) {
  if (!graph?.nodes) return graph;
  if (!geom?.size) return { ...graph, nodes: [], edges: [] };
  const nodes = [];
  for (const n of graph.nodes) {
    const g = geom.get(n.id);
    if (g) nodes.push({ ...n, pos: g.pos, axes: g.axes, axes_radius: g.axes_radius });
  }
  const have = new Set(nodes.map((n) => n.id));
  return { ...graph, nodes, edges: (graph.edges || []).filter((e) => have.has(e.src) && have.has(e.dst)) };
}
