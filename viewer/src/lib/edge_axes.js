// Recover the three numbers an edge's relation word was chosen from.
//
// The exporter stores one word per spatial edge: the largest signed component
// of the 3D offset between the two nodes in the frame's camera (+X right, +Y
// up in the image, +Z toward the camera). The other two components hold too.
// The band draws all three, and re-checks every stored word against them.

/** The six words as (axis, sign): side, height, depth toward the camera. */
export const AXES = [
  ['right', 0, 1], ['left', 0, -1],
  ['above', 1, 1], ['below', 1, -1],
  ['front', 2, 1], ['behind', 2, -1],
];

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/** Return the camera's right, up and forward in glTF coordinates, or null without a pose. */
export function cameraAxes(forward, up) {
  if (!Array.isArray(forward) || !Array.isArray(up)) return null;
  return { right: cross(forward, up), up, forward };
}

/** Return `p - q` as [side, height, depth], positive to the right, above and toward the camera. */
export function offsetIn(p, q, cam) {
  const v = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
  return [dot(v, cam.right), dot(v, cam.up), -dot(v, cam.forward)];
}

/** Return the word the exporter keeps for an offset: its largest signed component. */
export function keptWord(c) {
  let best = null, val = -Infinity;
  for (const [w, ax, sign] of AXES) {
    if (c[ax] * sign > val) { val = c[ax] * sign; best = w; }
  }
  return best;
}

/**
 * Compute, per frame, the offset of `focusId` from `otherId`, and check every spatial edge's stored word.
 *
 * @param {Array<{f: number, graph: object, cam: object}>} frames  graph_frame JSON and the frame's `cameraAxes`.
 * @returns {{byFrame: Map<number, {c: number[], word: ?string}>, checked: number, wrong: number}}
 */
export function edgeAxes(frames, focusId, otherId) {
  const byFrame = new Map();
  let checked = 0, wrong = 0;
  for (const { f, graph, cam } of frames) {
    if (!graph?.nodes || !cam) continue;
    const pos = new Map(graph.nodes.map((n) => [n.id, n.pos]));
    let word = null;
    for (const e of graph.edges || []) {
      if (e.edge_type !== 'spatial') continue;
      const p = pos.get(e.src), q = pos.get(e.dst);
      if (!p || !q) continue;
      checked++;
      if (keptWord(offsetIn(p, q, cam)) !== e.relation) wrong++;
      if (e.src === focusId && e.dst === otherId) word = e.relation;
    }
    const p = pos.get(focusId), q = pos.get(otherId);
    if (p && q) byFrame.set(f, { c: offsetIn(p, q, cam), word });
  }
  return { byFrame, checked, wrong };
}
