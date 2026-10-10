/**
 * An edge of the scene graph as the three numbers its relation was chosen from.
 *
 * A spatial edge stores one word (right, left, above, below, front, behind): the largest of the six signed
 * components of the 3D offset between the two nodes, in the frame's camera, +X right, +Y up and +Z toward
 * the camera. The other two components are dropped. With the camera of the frame the graph was built in,
 * every stored word is reproduced; the caller checks each edge it reads and draws nothing on a mismatch.
 */

/** The six words, as (axis, sign): side, height, depth toward the camera. */
export const AXES = [
  ['right', 0, 1], ['left', 0, -1],
  ['above', 1, 1], ['below', 1, -1],
  ['front', 2, 1], ['behind', 2, -1],
];

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/** The camera's right / up / forward in glTF coordinates, or null without a pose. */
export function cameraAxes(forward, up) {
  if (!Array.isArray(forward) || !Array.isArray(up)) return null;
  return { right: cross(forward, up), up, forward };
}

/** `p - q` as [side, height, depth] (+ = right / above / toward the camera). */
export function offsetIn(p, q, cam) {
  const v = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
  return [dot(v, cam.right), dot(v, cam.up), -dot(v, cam.forward)];
}

/** The word the exporter keeps for an offset: the largest signed component. */
export function keptWord(c) {
  let best = null, val = -Infinity;
  for (const [w, ax, sign] of AXES) {
    if (c[ax] * sign > val) { val = c[ax] * sign; best = w; }
  }
  return best;
}

/**
 * Per frame, the offset of `focusId` from `otherId`, and a check of every
 * spatial edge in those frames against its stored word.
 *
 * @param {Array<{f:number, graph:object, cam:object}>} frames  graph_frame JSON
 *        and the frame's cameraAxes, for the frames both nodes are in.
 * @returns {{byFrame: Map<number, {c:number[], word:?string}>, checked:number, wrong:number}}
 */
export function edgeAxes(frames, focusId, otherId) {
  const byFrame = new Map();
  let checked = 0, wrong = 0;
  for (const { f, graph, cam } of frames) {
    if (!graph?.nodes || !cam) continue;
    const pos = new Map(graph.nodes.map(n => [n.id, n.pos]));
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
