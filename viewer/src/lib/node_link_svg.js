// Draw a scene graph's nodes and edges as an SVG string.
//
// No DOM and no state: the caller passes the nodes, their positions and what
// to emphasise, and delegates clicks on the `data-node-key` attribute every
// node carries. A node's key is `track:id`, the same key the 3D view and the
// bands use.
import { RELATION_COLORS } from './relations.js';
import { esc, hexColor } from './format.js';

const DIM_OPACITY = 0.18;
// About the width of one character of an 11 px label, to keep labels inside the panel.
const LABEL_CHAR_W = 6;
const NODE_R = 6;
const DEFAULT_TINT = 0x12c2e9;

function nodeFill(n, tint) {
  if (Array.isArray(n.color) && n.color.length >= 3) {
    const to255 = (v) => (typeof v === 'number' && v <= 1 ? Math.round(v * 255) : Math.round(v));
    return `rgb(${to255(n.color[0])},${to255(n.color[1])},${to255(n.color[2])})`;
  }
  return hexColor(tint);
}

// A schematic node is an ellipse whose aspect follows the region's footprint
// (`extent_3d`), its area held at the base radius's.
function nodeRadii(n, r) {
  const e = n.extent_3d;
  let asp = 1.4;
  if (Array.isArray(e) && e.length >= 2 && e[0] > 0 && e[1] > 0) asp = e[0] / e[1];
  else if (typeof n.elongation === 'number' && n.elongation >= 1) asp = n.elongation;
  const f = Math.sqrt(Math.min(3, Math.max(1 / 3, asp)));
  return [r * f, r / f];
}

/**
 * Return a label's SVG text beside a node: right of it, or left of it where it would leave the panel.
 *
 * @param {number} x  the node's centre.
 * @param {number} rx  the node's half-width.
 */
export function nodeLabel(label, x, y, rx, width, opacity) {
  const right = x + rx + 2 + label.length * LABEL_CHAR_W <= width;
  const lx = right ? x + rx + 2 : x - rx - 2;
  return `<text class="nl-label" x="${lx.toFixed(1)}" y="${(y + 3).toFixed(1)}"${right ? '' : ' text-anchor="end"'} `
    + `opacity="${opacity}">${esc(label)}</text>`;
}

/**
 * Render a node-link SVG.
 *
 * @param {Array} nodes  graph nodes; each may carry `key` (`track:id`) and `_track`.
 * @param {Array} edges  `{src, dst, relation}`; `contains` draws as a dashed link.
 * @param {Map} positions  node id → [x, y].
 * @param {object} [opts]
 * @param {number} [opts.width]
 * @param {number} [opts.height]
 * @param {string} [opts.track]  the track of a node without `_track`.
 * @param {?Set<string>} [opts.brushKeys]  keys kept bright; null dims nothing.
 * @param {?string} [opts.hoverKey]  the key that gets the hover glow.
 * @param {boolean} [opts.showLabels]
 * @param {?Map} [opts.projById]  node id → `{rx, ry, rot}` or `{poly}` (offsets from the node).
 * @param {function(string): number} [opts.tint]  a track's colour for a node without one.
 * @returns {string}
 */
export function renderNodeLink(nodes, edges, positions, opts = {}) {
  const {
    width = 320, height = 250, track = '', brushKeys = null, hoverKey = null,
    showLabels = false, projById = null, tint = () => DEFAULT_TINT,
  } = opts;
  const open = `<svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" class="nl-svg" xmlns="http://www.w3.org/2000/svg">`;
  const list = (nodes || []).filter((n) => n && n.id != null && positions?.has(n.id));
  if (!list.length) {
    return `${open}<text x="${width / 2}" y="${height / 2}" text-anchor="middle" class="nl-empty-text">No graph</text></svg>`;
  }
  const keyById = new Map(list.map((n) => [n.id, n.key ?? `${track}:${n.origId ?? n.id}`]));
  const dimmed = (key) => (brushKeys ? !brushKeys.has(key) : false);

  // Edges under the nodes; an edge fades when either end is dimmed.
  let edgeSvg = '';
  for (const e of edges || []) {
    const sp = positions.get(e.src), dp = positions.get(e.dst);
    if (!sp || !dp) continue;
    const faded = dimmed(keyById.get(e.src) ?? `${track}:${e.src}`) || dimmed(keyById.get(e.dst) ?? `${track}:${e.dst}`);
    const line = `<line x1="${sp[0].toFixed(1)}" y1="${sp[1].toFixed(1)}" x2="${dp[0].toFixed(1)}" y2="${dp[1].toFixed(1)}"`;
    if (e.relation === 'contains') {
      edgeSvg += `${line} stroke="#9aa6b6" stroke-width="1.6" stroke-dasharray="3 2" opacity="${faded ? DIM_OPACITY : 0.55}"/>`;
    } else {
      const col = hexColor(RELATION_COLORS[e.relation] ?? 0x888888);
      edgeSvg += `${line} stroke="${col}" stroke-width="1.4" opacity="${faded ? DIM_OPACITY : 0.7}"/>`;
    }
  }

  // Nodes: the region's hull polygon or ellipse when projected, a schematic ellipse otherwise.
  let nodeSvg = '';
  for (const n of list) {
    const [x, y] = positions.get(n.id);
    const key = keyById.get(n.id);
    const op = dimmed(key) ? DIM_OPACITY : 1;
    const cls = `nl-node${hoverKey && key === hoverKey ? ' nl-hover' : ''}`;
    const fill = nodeFill(n, tint(n._track ?? track));
    const p = projById?.get(n.id);
    const title = `<title>${esc(n.label ?? key)}</title>`;
    let rx, ry, rot = 0;
    if (p) { rx = p.rx; ry = p.ry; rot = p.rot || 0; } else { [rx, ry] = nodeRadii(n, NODE_R); }
    if (Array.isArray(p?.poly) && p.poly.length >= 3) {
      const pts = p.poly.map(([dx, dy]) => `${(x + dx).toFixed(1)},${(y + dy).toFixed(1)}`).join(' ');
      nodeSvg += `<polygon class="${cls}" data-node-key="${esc(key)}" points="${pts}" fill="${fill}" `
        + `opacity="${op}" fill-opacity="0.5" stroke="${fill}" stroke-opacity="${op}">${title}</polygon>`;
    } else {
      const xf = rot ? ` transform="rotate(${rot.toFixed(1)} ${x.toFixed(1)} ${y.toFixed(1)})"` : '';
      nodeSvg += `<ellipse class="${cls}" data-node-key="${esc(key)}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" `
        + `rx="${rx.toFixed(1)}" ry="${ry.toFixed(1)}"${xf} fill="${fill}" opacity="${op}" `
        + `fill-opacity="${p ? 0.55 : 1}" stroke="${fill}" stroke-opacity="${op}">${title}</ellipse>`;
    }
    if (showLabels && n.label) nodeSvg += nodeLabel(String(n.label), x, y, rx, width, op);
  }
  return `${open}${edgeSvg}${nodeSvg}</svg>`;
}
