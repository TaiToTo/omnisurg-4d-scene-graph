// Draw the scene graph as a 2D picture of the camera's view: each node at its region's centroid in the
// image, drawn as the region's convex hull, with the graph's edges between them. Two tracks draw
// together with the hierarchy's `contains` edges between them. In world mode every stacked frame's nodes
// are projected into the camera of the frame on stage, as dots that a line follows through time.
import { ellipsesOf, hullsOf } from './regions.js';
import { RELATION_COLORS, esc, hexColor, nodeKey, rgbToHex } from './format.js';

const PAD = 18;
const GLYPH_SCALE = 0.55;
// A frame's spatial edges hold every ordered pair of nodes; past this many the lines say nothing.
const MAX_RELATION_EDGES = 30;
const DIM_OPACITY = 0.18;
// About the width of one character of an 11 px label, to keep a label inside the panel.
const LABEL_CHAR_W = 6;

const fill = (n) => hexColor(rgbToHex(n.color));

/**
 * Place nodes at their regions' centroids, fitted into a W x H panel with the image's aspect, and give
 * each a glyph: the region's hull, or its ellipse when the hull has under three points.
 *
 * @param {object[]} nodes  graph nodes; `n.id` is the region id in `regions`.
 * @param {object} regions  the frame's regions.
 * @param {?{x0:number, y0:number, x1:number, y1:number}} [bounds]  image area to fit, the image by default.
 * @returns {?{positions:Map, glyphs:Map, scale:number, offX:number, offY:number}} null when no node has
 *          a region on this frame.
 */
export function projectNodes(nodes, regions, W, H, bounds = null) {
  const ells = ellipsesOf(regions);
  const hits = nodes.filter((n) => ells?.has(n.id));
  if (!hits.length) return null;
  const [imgW, imgH] = regions.grid;
  const b = bounds || { x0: 0, y0: 0, x1: imgW, y1: imgH };
  const scale = Math.min((W - 2 * PAD) / (b.x1 - b.x0), (H - 2 * PAD) / (b.y1 - b.y0));
  const offX = (W - (b.x1 - b.x0) * scale) / 2 - b.x0 * scale;
  const offY = (H - (b.y1 - b.y0) * scale) / 2 - b.y0 * scale;
  const maxR = 0.16 * Math.min(W, H);
  const hulls = hullsOf(regions);
  const positions = new Map(), glyphs = new Map();
  for (const n of hits) {
    const e = ells.get(n.id);
    positions.set(n.id, [offX + e.cx * scale, offY + e.cy * scale]);
    const poly = hulls.get(n.id);
    if (poly && poly.length >= 3) {
      let rx = 0, ry = 0;
      const off = poly.map(([px, py]) => {
        const dx = (px - e.cx) * scale * GLYPH_SCALE, dy = (py - e.cy) * scale * GLYPH_SCALE;
        rx = Math.max(rx, Math.abs(dx)); ry = Math.max(ry, Math.abs(dy));
        return [dx, dy];
      });
      // A glyph larger than the cap shrinks as a whole, keeping its shape.
      const k = Math.max(rx, ry) > maxR ? maxR / Math.max(rx, ry) : 1;
      glyphs.set(n.id, { poly: off.map(([dx, dy]) => [dx * k, dy * k]), rx: rx * k, ry: ry * k });
    } else {
      glyphs.set(n.id, {
        rx: Math.min(maxR, Math.max(3, e.a * scale * GLYPH_SCALE)),
        ry: Math.min(maxR, Math.max(3, e.b * scale * GLYPH_SCALE)),
        rot: e.theta * 180 / Math.PI,
      });
    }
  }
  return { positions, glyphs, scale, offX, offY };
}

function glyphSVG(n, key, [x, y], g, { opacity = 1, hover = false } = {}) {
  const cls = `nl-node${hover ? ' nl-hover' : ''}`;
  const title = `<title>${esc(n.label ?? key)}</title>`;
  if (g.poly) {
    const pts = g.poly.map(([dx, dy]) => `${(x + dx).toFixed(1)},${(y + dy).toFixed(1)}`).join(' ');
    return `<polygon class="${cls}" data-node-key="${esc(key)}" points="${pts}" fill="${fill(n)}" opacity="${opacity}" `
      + `fill-opacity="0.5" stroke="${fill(n)}">${title}</polygon>`;
  }
  const rot = g.rot ? ` transform="rotate(${g.rot.toFixed(1)} ${x.toFixed(1)} ${y.toFixed(1)})"` : '';
  return `<ellipse class="${cls}" data-node-key="${esc(key)}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" `
    + `rx="${g.rx.toFixed(1)}" ry="${g.ry.toFixed(1)}"${rot} fill="${fill(n)}" opacity="${opacity}" fill-opacity="0.55" `
    + `stroke="${fill(n)}">${title}</ellipse>`;
}

/** A node's label beside it: to its right, or to its left where it would leave the panel. */
export function labelSVG(label, [x, y], rx, width, opacity) {
  const right = x + rx + 2 + label.length * LABEL_CHAR_W <= width;
  return `<text class="nl-label" x="${(right ? x + rx + 2 : x - rx - 2).toFixed(1)}" y="${(y + 3).toFixed(1)}"`
    + `${right ? '' : ' text-anchor="end"'} opacity="${opacity}">${esc(label)}</text>`;
}

/** The keys a `contains` edge of the hierarchy joins to `key`. */
export function containmentPartners(key, hierarchy) {
  const out = [];
  for (const e of hierarchy?.edges || []) {
    if (e.relation !== 'contains') continue;
    if (e.src === key) out.push(e.dst);
    else if (e.dst === key) out.push(e.src);
  }
  return out;
}

/**
 * Place every shown track's nodes, keyed `track:id`. The tracks' regions share the cloud's grid, so one
 * fit holds them all.
 *
 * @returns {?{positions:Map, glyphs:Map, nodes:Array<{n:object, key:string, track:string}>, scale:number,
 *          offX:number, offY:number}} null when no node has a region on this frame.
 */
export function projectTracks(tracks, graphByTrack, regionsByTrack, W, H, bounds = null) {
  const positions = new Map(), glyphs = new Map(), nodes = [];
  let fit = null;
  for (const track of tracks) {
    const graph = graphByTrack[track], regions = regionsByTrack[track];
    const proj = graph?.nodes?.length && regions ? projectNodes(graph.nodes, regions, W, H, bounds) : null;
    if (!proj) continue;
    fit = fit ?? proj;
    for (const n of graph.nodes) {
      if (!proj.positions.has(n.id)) continue;
      const key = nodeKey(track, n.id);
      positions.set(key, proj.positions.get(n.id));
      glyphs.set(key, proj.glyphs.get(n.id));
      nodes.push({ n, key, track });
    }
  }
  return fit && { positions, glyphs, nodes, scale: fit.scale, offX: fit.offX, offY: fit.offY };
}

function emptyPanel(el, msg) {
  el.innerHTML = `<div class="nl-empty">${esc(msg)}</div>`;
}

/**
 * Draw one frame's scene graph: one track's nodes and its relation edges, or two tracks' nodes and the
 * hierarchy's `contains` edges between them.
 *
 * @param {HTMLElement} el
 * @param {object} p
 * @param {string[]} p.tracks  the tracks shown; the first is drawn first, under the others.
 * @param {object} p.graphByTrack  the frame's scene graph per track.
 * @param {object} p.regionsByTrack  the frame's regions per track.
 * @param {?object} p.hierarchy  the frame's hierarchy, read when two tracks are shown.
 * @param {?string} p.hoverKey  the node key to glow.
 * @param {?string} p.focusKey  the followed node: the others dim, but for the regions it is paired with.
 */
export function renderFrameNodeLink(el, { tracks, graphByTrack, regionsByTrack, hierarchy = null, hoverKey = null, focusKey = null }) {
  const W = el.clientWidth || 320, H = el.clientHeight || 260;
  if (!tracks.some((t) => graphByTrack[t]?.nodes?.length)) { emptyPanel(el, 'No scene graph on this frame.'); return; }
  const proj = projectTracks(tracks, graphByTrack, regionsByTrack, W, H);
  if (!proj) { emptyPanel(el, 'No regions on this frame to place the graph by.'); return; }
  const bright = focusKey ? new Set([focusKey, ...containmentPartners(focusKey, hierarchy)]) : null;
  const dimmed = (key) => !!bright && !bright.has(key);

  // The edges: one track's relations while they are few, else the containment between two tracks.
  const edges = [];
  if (tracks.length === 1) {
    const raw = graphByTrack[tracks[0]]?.edges || [];
    if (raw.length <= MAX_RELATION_EDGES) {
      for (const e of raw) edges.push({ a: nodeKey(tracks[0], e.src), b: nodeKey(tracks[0], e.dst), relation: e.relation });
    }
  } else {
    for (const e of hierarchy?.edges || []) if (e.relation === 'contains') edges.push({ a: e.src, b: e.dst, relation: 'contains' });
  }
  const svg = [`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" class="nl-svg" xmlns="http://www.w3.org/2000/svg">`];
  for (const { a, b, relation } of edges) {
    const pa = proj.positions.get(a), pb = proj.positions.get(b);
    if (!pa || !pb) continue;
    const faded = dimmed(a) && dimmed(b);
    const line = `<line x1="${pa[0].toFixed(1)}" y1="${pa[1].toFixed(1)}" x2="${pb[0].toFixed(1)}" y2="${pb[1].toFixed(1)}"`;
    svg.push(relation === 'contains'
      ? `${line} stroke="#7d8898" stroke-width="1.4" stroke-dasharray="3 2" opacity="${faded ? DIM_OPACITY : 0.6}"/>`
      : `${line} stroke="${hexColor(RELATION_COLORS[relation] ?? 0x888888)}" stroke-width="1.4" opacity="${faded ? DIM_OPACITY : 0.7}"/>`);
  }

  // The nodes, track by track, and their labels.
  for (const { n, key } of proj.nodes) {
    const pos = proj.positions.get(key), g = proj.glyphs.get(key);
    const opacity = dimmed(key) ? DIM_OPACITY : 1;
    svg.push(glyphSVG(n, key, pos, g, { opacity, hover: key === hoverKey }));
    if (n.label) svg.push(labelSVG(String(n.label), pos, g.rx, W, opacity));
  }
  svg.push('</svg>');
  el.innerHTML = svg.join('');
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(...a) || 1; return a.map((v) => v / l); };

/** Least squares y = a x + b; null when x does not spread. */
export function fitLinear(xs, ys) {
  const n = xs.length;
  if (n < 2) return null;
  const mx = xs.reduce((s, v) => s + v, 0) / n, my = ys.reduce((s, v) => s + v, 0) / n;
  let sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sxx += (xs[i] - mx) ** 2; sxy += (xs[i] - mx) * (ys[i] - my); }
  if (sxx < 1e-9) return null;
  const a = sxy / sxx;
  return { a, b: my - a * mx };
}

/**
 * Project points of the world stack into the camera of one frame, as normalised rays (X, Y): X to the
 * camera's right and Y up, both divided by the depth along its view. Points behind it are marked.
 *
 * @param {number[][]} points  world coordinates.
 * @param {{camera_position:number[], camera_forward:number[], camera_up:number[]}} frame  a clip.json frame.
 * @param {number[]} origin  the world origin: the centroid of the first stacked frame.
 */
export function projectToCamera(points, frame, origin) {
  const camPos = sub(frame.camera_position, origin);
  const fwd = norm(frame.camera_forward);
  const right = norm(cross(frame.camera_forward, frame.camera_up));
  const up = cross(right, fwd);
  return points.map((p) => {
    const d = sub(p, camPos);
    const z = dot(d, fwd);
    return z > 1e-6 ? { X: dot(d, right) / z, Y: dot(d, up) / z, behind: false } : { X: 0, Y: 0, behind: true };
  });
}

/**
 * Draw every stacked frame's nodes in the camera of the frame on stage. The record has no intrinsics, so
 * the map from rays to image pixels is fitted, per axis, to the stage frame's own region centroids: that
 * frame's nodes land where the per-frame view draws them.
 *
 * @param {HTMLElement} el
 * @param {object} p
 * @param {object} p.world  the world stack: {origin, indices, frames: [{i, offset, regionsByTrack, graphByTrack}]}.
 * @param {number} p.refFrame  the frame on stage, one of world.indices.
 * @param {string[]} p.tracks  the tracks shown.
 * @param {object[]} p.frames  clip.json's frames.
 * @param {boolean} [p.equalize]  draw every frame's dots alike, rather than fading with time from the stage.
 */
export function renderWorldNodeLink(el, { world, refFrame, tracks, frames, hoverKey = null, focusKey = null, equalize = false }) {
  const W = el.clientWidth || 320, H = el.clientHeight || 260;
  const dimFor = (key, op) => (focusKey && key !== focusKey ? Math.min(op, 0.15) : op);
  const ref = world.frames.find((f) => f.i === refFrame);
  const shown = tracks.filter((t) => ref?.regionsByTrack[t] && ref?.graphByTrack[t]?.nodes?.length);
  if (!shown.length) { emptyPanel(el, 'No scene graph on the frame on stage.'); return; }
  const [imgW, imgH] = ref.regionsByTrack[shown[0]].grid;

  // Every stacked node of every shown track, in world coordinates, as a ray of the stage frame's camera.
  const nodes = [];
  for (const fr of world.frames) {
    for (const track of shown) {
      for (const n of fr.graphByTrack[track]?.nodes || []) {
        if (!Array.isArray(n.pos)) continue;
        nodes.push({ n, track, frameId: fr.i, key: nodeKey(track, n.id), isRef: fr.i === refFrame,
          p: [n.pos[0] + fr.offset[0], n.pos[1] + fr.offset[1], n.pos[2] + fr.offset[2]] });
      }
    }
  }
  const rays = projectToCamera(nodes.map((r) => r.p), frames[refFrame], world.origin);
  nodes.forEach((r, k) => Object.assign(r, rays[k]));

  // Fit rays to pixels on the stage frame's region centroids; a nominal 60-degree view when they are too few.
  const ellOf = (r) => ellipsesOf(ref.regionsByTrack[r.track])?.get(r.n.id);
  const fitPts = nodes.filter((r) => r.isRef && !r.behind && ellOf(r));
  const fx = fitLinear(fitPts.map((r) => r.X), fitPts.map((r) => ellOf(r).cx));
  const fy = fitLinear(fitPts.map((r) => r.Y), fitPts.map((r) => ellOf(r).cy));
  const f0 = imgW / (2 * Math.tan(Math.PI / 6));
  const mapU = fx ? (X) => fx.a * X + fx.b : (X) => f0 * X + imgW / 2;
  const mapV = fy ? (Y) => fy.a * Y + fy.b : (Y) => -f0 * Y + imgH / 2;

  // Widen the image's area on each side until nine in ten projected nodes fit, by at most MARGIN_MAX;
  // the rest are drawn clamped to the edge, and nodes behind the camera are counted, not drawn.
  const MARGIN_MIN = 0.12, MARGIN_MAX = 0.45, COVER = 0.9;
  const front = nodes.filter((r) => !r.behind).map((r) => ({ u: mapU(r.X), v: mapV(r.Y) }));
  const margin = (vals) => {
    const s = vals.sort((a, c) => a - c);
    return Math.min(MARGIN_MAX, Math.max(MARGIN_MIN, s.length ? s[Math.floor(COVER * (s.length - 1))] : 0));
  };
  const b = {
    x0: -imgW * margin(front.map((p) => Math.max(0, -p.u / imgW))),
    x1: imgW * (1 + margin(front.map((p) => Math.max(0, (p.u - imgW) / imgW)))),
    y0: -imgH * margin(front.map((p) => Math.max(0, -p.v / imgH))),
    y1: imgH * (1 + margin(front.map((p) => Math.max(0, (p.v - imgH) / imgH)))),
  };
  const proj = projectTracks(shown, ref.graphByTrack, ref.regionsByTrack, W, H, b);
  if (!proj) { emptyPanel(el, 'No regions on the frame on stage to place the graph by.'); return; }
  const toPanel = (u, v) => [proj.offX + u * proj.scale, proj.offY + v * proj.scale];
  let behind = 0, clampedCount = 0;
  const pts = [];
  for (const r of nodes) {
    if (r.behind) { behind++; continue; }
    const u = mapU(r.X), v = mapV(r.Y);
    const clamped = u < b.x0 || u > b.x1 || v < b.y0 || v > b.y1;
    if (clamped) clampedCount++;
    pts.push({ ...r, clamped, panel: toPanel(Math.min(b.x1, Math.max(b.x0, u)), Math.min(b.y1, Math.max(b.y0, v))) });
  }

  const svg = [`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" class="nl-svg" xmlns="http://www.w3.org/2000/svg">`];
  const [x0, y0] = toPanel(0, 0), [x1, y1] = toPanel(imgW, imgH);
  svg.push(`<rect x="${x0.toFixed(1)}" y="${y0.toFixed(1)}" width="${(x1 - x0).toFixed(1)}" height="${(y1 - y0).toFixed(1)}" `
    + 'fill="none" stroke="#263047" stroke-width="1"/>');

  // One line per node through its frames in time order; a clamped position would draw a path it never took.
  const order = new Map(world.indices.map((f, k) => [f, k]));
  const byKey = new Map();
  for (const p of pts) {
    if (p.clamped) continue;
    if (!byKey.has(p.key)) byKey.set(p.key, []);
    byKey.get(p.key).push(p);
  }
  for (const [key, list] of byKey) {
    if (list.length < 2) continue;
    list.sort((a, c) => order.get(a.frameId) - order.get(c.frameId));
    svg.push(`<polyline points="${list.map((p) => p.panel.map((v) => v.toFixed(1)).join(',')).join(' ')}" fill="none" `
      + `stroke="${fill(list[0].n)}" stroke-width="1.2" opacity="${dimFor(key, 0.4)}" data-node-key="${esc(key)}"/>`);
  }
  const span = Math.max(1, world.indices.length - 1);
  const refOrder = order.get(refFrame) ?? 0;
  for (const p of pts) {
    if (p.isRef) continue;
    const dist = Math.abs((order.get(p.frameId) ?? 0) - refOrder) / span;
    let op = dimFor(p.key, equalize ? 0.8 : 0.85 - 0.6 * dist);
    if (p.clamped) op *= 0.45;
    svg.push(`<circle class="nl-node${p.key === hoverKey ? ' nl-hover' : ''}" data-node-key="${esc(p.key)}" `
      + `cx="${p.panel[0].toFixed(1)}" cy="${p.panel[1].toFixed(1)}" r="${p.clamped ? 2.4 : 3.2}" fill="${fill(p.n)}" `
      + `opacity="${op.toFixed(2)}"><title>${esc(p.n.label ?? p.key)}, frame ${p.frameId}`
      + `${p.clamped ? ', outside the view and drawn at its edge' : ''}</title></circle>`);
  }
  for (const { n, key } of proj.nodes) {
    const pos = proj.positions.get(key), g = proj.glyphs.get(key);
    const op = dimFor(key, 1);
    svg.push(glyphSVG(n, key, pos, g, { opacity: op, hover: key === hoverKey }));
    if (n.label) svg.push(labelSVG(String(n.label), pos, g.rx, W, op));
  }
  svg.push(`<text class="nl-ref-note" x="${(x0 + 4).toFixed(1)}" y="${(y0 + 12).toFixed(1)}">camera of frame ${refFrame}</text>`);
  const notes = [];
  if (clampedCount) notes.push(`${clampedCount} at the edge`);
  if (behind) notes.push(`${behind} behind the camera`);
  if (notes.length) {
    svg.push(`<text class="nl-badge" x="${(x1 - 4).toFixed(1)}" y="${(y1 - 6).toFixed(1)}" text-anchor="end">${notes.join(' · ')}</text>`);
  }
  svg.push('</svg>');
  el.innerHTML = svg.join('');
}

/** Glow every element of these containers that draws the node `key`, and no other. */
export function setHoverKey(containers, key) {
  for (const el of containers) {
    for (const node of el?.querySelectorAll('[data-node-key]') ?? []) {
      node.classList.toggle('nl-hover', !!key && node.getAttribute('data-node-key') === key);
    }
  }
}
