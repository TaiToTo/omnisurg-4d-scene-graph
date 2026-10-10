// Name the colours of the scene graph's relations, and read one node's
// relations to its neighbours out of a temporal graph.
//
// The band under the 3D view states this as fact ("obj 1 is left of obj 6 on
// frames 12 to 40"), so the reading keeps spatial edges only and picks one
// relation per frame by a fixed order.

/** Colour of each relation word, shared by the 2D map, the band and the 3D arcs. */
export const RELATION_COLORS = {
  right: 0xff4444,
  left: 0x44aaff,
  above: 0x44ff44,
  below: 0xffaa00,
  front: 0xff44ff,
  behind: 0xffff44,
};

/** The word for the same offset seen from the other node. */
export const MIRROR_REL = {
  left: 'right', right: 'left', above: 'below', below: 'above', front: 'behind', behind: 'front',
};

// Which word a frame keeps when a pair holds two: depth, then height, then side.
const REL_ORDER = { front: 0, behind: 1, above: 2, below: 3, right: 4, left: 5 };

/**
 * Collect, per neighbour of `nodeId`, its spatial relation on each frame and the changes of it.
 *
 * Relations are mirrored when `nodeId` is the edge's `dst`, so every word reads
 * from `nodeId`'s side. A gap in a pair's frames is not a change: the pair was
 * absent there.
 *
 * @param {{relations?: Array}} tg  temporal_graph JSON.
 * @param {number} nodeId
 * @returns {Map<number, {weight: number, relByFrame: Map<number, string>,
 *   changes: Array<{f: number, from: string, to: string}>}>} keyed by neighbour id;
 *   `weight` counts the frames the pair is related on.
 */
export function partnerRelations(tg, nodeId) {
  const rank = (rel) => REL_ORDER[rel] ?? 9;
  const byPartner = new Map();
  for (const r of tg?.relations || []) {
    if (r.edge_type !== 'spatial') continue;
    if (r.src !== nodeId && r.dst !== nodeId) continue;
    const other = r.src === nodeId ? r.dst : r.src;
    let e = byPartner.get(other);
    if (!e) {
      e = { weight: 0, relByFrame: new Map(), changes: [] };
      byPartner.set(other, e);
    }
    e.weight += (r.frames || []).length;
    const rel = r.src === nodeId ? r.relation : (MIRROR_REL[r.relation] ?? r.relation);
    for (const f of r.frames || []) {
      const cur = e.relByFrame.get(f);
      // Strictly better only, so the result does not depend on the order of `relations`.
      if (cur === undefined || rank(rel) < rank(cur)) e.relByFrame.set(f, rel);
    }
  }
  for (const e of byPartner.values()) {
    const fr = [...e.relByFrame.keys()].sort((a, b) => a - b);
    for (let k = 1; k < fr.length; k++) {
      const from = e.relByFrame.get(fr[k - 1]), to = e.relByFrame.get(fr[k]);
      if (fr[k] === fr[k - 1] + 1 && from !== to) e.changes.push({ f: fr[k], from, to });
    }
  }
  return byPartner;
}

/**
 * Return the `k` neighbours whose edge with `nodeId` changes most, best first.
 *
 * Ties go to the neighbour related on more frames, then to the lower id.
 *
 * @returns {Array<[number, object]>} `[id, partnerRelations entry]` pairs.
 */
export function topPartners(tg, nodeId, k = 1) {
  return [...partnerRelations(tg, nodeId).entries()]
    .sort((a, b) => (b[1].changes.length - a[1].changes.length)
      || (b[1].weight - a[1].weight) || (a[0] - b[0]))
    .slice(0, k);
}

/** Collapse a frame → word map into maximal runs `{rel, a, b}` of consecutive frames. */
export function relationRuns(relByFrame) {
  const runs = [];
  for (const f of [...relByFrame.keys()].sort((a, b) => a - b)) {
    const rel = relByFrame.get(f);
    const last = runs[runs.length - 1];
    if (last && last.rel === rel && f === last.b + 1) last.b = f;
    else runs.push({ rel, a: f, b: f });
  }
  return runs;
}

/** Collapse a list of frames into maximal runs `[first, last]` of consecutive frames. */
export function frameRuns(frames) {
  const runs = [];
  for (const f of [...frames].sort((a, b) => a - b)) {
    const last = runs[runs.length - 1];
    if (last && f === last[1] + 1) last[1] = f;
    else runs.push([f, f]);
  }
  return runs;
}

/**
 * List the events of one node: entering and leaving the view, and each change of its edge to a neighbour.
 *
 * @param {object} node      a temporal_graph node (`present_frames`).
 * @param {Array<[number, object]>} partners  `topPartners` output.
 * @param {number} maxFrame  the clip's last frame index.
 * @param {function(number): string} labelOf  a node id's display name.
 * @returns {Array<{f: number, kind: string, partner?: number, label: string}>} sorted by frame.
 */
export function nodeEvents(node, partners, maxFrame, labelOf) {
  const events = [];
  for (const [a, b] of frameRuns(node?.present_frames || [])) {
    if (a > 0) events.push({ f: a, kind: 'enter', label: 'enters view' });
    if (b < maxFrame) events.push({ f: b + 1, kind: 'exit', label: 'leaves view' });
  }
  for (const [other, e] of partners) {
    for (const c of e.changes) {
      events.push({ f: c.f, kind: 'rel', partner: other, label: `vs ${labelOf(other)}: ${c.from} → ${c.to}` });
    }
  }
  return events.sort((a, b) => a.f - b.f);
}

/**
 * Return the keys related to `key` by a `contains` edge of a hierarchy frame.
 *
 * A hierarchy frame relates an annotated region to the automatic regions it
 * contains, in keys `track:id`.
 */
export function containmentPartners(key, hierarchy) {
  const out = [];
  for (const e of hierarchy?.edges || []) {
    if (e.relation !== 'contains') continue;
    if (e.src === key) out.push(e.dst);
    else if (e.dst === key) out.push(e.src);
  }
  return out;
}
