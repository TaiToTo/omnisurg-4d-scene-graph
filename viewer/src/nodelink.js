// Draw the 2D scene-graph map: one frame's graph in its camera view, or in
// World mode every stacked frame's nodes projected into one frame's camera.
//
// A node sits at its region's centroid in the image and is drawn as the
// region's hull. The manifest has the camera's pose and no intrinsics, so the
// World projection is calibrated per reference frame: a per-axis linear map
// that takes that frame's own projected nodes onto its region centroids.
import { regionEllipses, regionHulls } from './lib/regions.js';
import { renderNodeLink } from './lib/node_link_svg.js';
import { esc, hexColor, rgbToHex } from './lib/format.js';
import { containmentPartners } from './lib/relations.js';
import { orderTracks, trackInfo } from './state.js';

const PAD = 18;
const GLYPH_SCALE = 0.55;
// A frame's graph holds nearly every ordered pair of nodes; drawn as lines,
// more than this many is a hairball.
export const MAX_RELATION_EDGES = 30;

const ellCache = new WeakMap();
const hullCache = new WeakMap();

/** Return class id → ellipse for a seg frame, computed once per frame object. */
export function ellipsesOf(seg) {
  if (!seg || !Array.isArray(seg.seg_colors)) return null;
  if (!ellCache.has(seg)) ellCache.set(seg, regionEllipses(seg));
  return ellCache.get(seg);
}

function hullsOf(seg) {
  if (!seg || !Array.isArray(seg.seg_colors)) return null;
  if (!hullCache.has(seg)) hullCache.set(seg, regionHulls(seg));
  return hullCache.get(seg);
}

/** Return a track's colour for nodes that carry none: amber for annotation, cyan otherwise. */
export function trackTint(track) {
  return trackInfo(track).semantic ? 0xff8c00 : 0x12c2e9;
}

function nodeColor(n, track) {
  return hexColor(rgbToHex(n.color, trackTint(track)));
}

/** Return the nodes of every displayed track, keyed `track:id`. */
function keyedNodes(graphByTrack, tracks) {
  const nodes = [];
  for (const track of tracks) {
    for (const n of graphByTrack[track]?.nodes || []) {
      nodes.push({ ...n, id: `${track}:${n.id}`, key: `${track}:${n.id}`, origId: n.id, _track: track });
    }
  }
  return nodes;
}

/**
 * Place nodes at their regions' centroids, mapped from image pixels into the panel.
 *
 * `bounds` widens the mapped area beyond the image, so projections outside it stay visible.
 *
 * @returns {?{positions: Map, projById: Map, scale: number, offX: number, offY: number}}
 *   null when no node has a region.
 */
function projectNodes(nodes, segByTrack, W, H, bounds = null) {
  let imgW = 0, imgH = 0;
  const hits = [];
  for (const n of nodes) {
    const seg = segByTrack[n._track];
    const e = ellipsesOf(seg)?.get(n.origId);
    if (!e) continue;
    imgW = imgW || seg.width;
    imgH = imgH || seg.height;
    hits.push({ n, e, seg });
  }
  if (!hits.length) return null;
  const b = bounds || { x0: 0, y0: 0, x1: imgW, y1: imgH };
  const scale = Math.min((W - 2 * PAD) / (b.x1 - b.x0), (H - 2 * PAD) / (b.y1 - b.y0));
  const offX = (W - (b.x1 - b.x0) * scale) / 2 - b.x0 * scale;
  const offY = (H - (b.y1 - b.y0) * scale) / 2 - b.y0 * scale;
  const maxR = 0.16 * Math.min(W, H);
  const positions = new Map(), projById = new Map();
  for (const { n, e, seg } of hits) {
    positions.set(n.id, [offX + e.cx * scale, offY + e.cy * scale]);
    const poly = hullsOf(seg)?.get(n.origId);
    if (poly?.length >= 3) {
      let rx = 0, ry = 0;
      const off = poly.map(([px, py]) => {
        const dx = (px - e.cx) * scale * GLYPH_SCALE, dy = (py - e.cy) * scale * GLYPH_SCALE;
        rx = Math.max(rx, Math.abs(dx));
        ry = Math.max(ry, Math.abs(dy));
        return [dx, dy];
      });
      const k = Math.max(rx, ry) > maxR ? maxR / Math.max(rx, ry) : 1;
      projById.set(n.id, { poly: off.map(([dx, dy]) => [dx * k, dy * k]), rx: rx * k, ry: ry * k });
    } else {
      projById.set(n.id, {
        rx: Math.min(maxR, Math.max(3, e.a * scale * GLYPH_SCALE)),
        ry: Math.min(maxR, Math.max(3, e.b * scale * GLYPH_SCALE)),
        rot: e.theta * 180 / Math.PI,
      });
    }
  }
  return { positions, projById, scale, offX, offY };
}

function emptyPanel(el, msg) {
  el.innerHTML = `<div class="nl-empty">${esc(msg)}</div>`;
}

/**
 * Draw one frame's graph in its camera view.
 *
 * Two tracks draw together with the hierarchy's `contains` edges between
 * them; one track draws its own relation edges when they are few.
 */
export function renderFrameNodeLink(el, { graphByTrack, segByTrack, tracks, hierarchy, hoverKey, focusKey }) {
  const W = el.clientWidth || 320, H = el.clientHeight || 260;
  const nodes = keyedNodes(graphByTrack, orderTracks(tracks));
  if (!nodes.length) { emptyPanel(el, 'No scene graph on this frame.'); return; }
  let edges;
  if (tracks.length > 1) {
    const have = new Set(nodes.map((n) => n.id));
    edges = (hierarchy?.edges || []).filter((e) => e.relation === 'contains' && have.has(e.src) && have.has(e.dst));
  } else {
    const raw = graphByTrack[tracks[0]]?.edges || [];
    edges = raw.length <= MAX_RELATION_EDGES
      ? raw.map((e) => ({ ...e, src: `${tracks[0]}:${e.src}`, dst: `${tracks[0]}:${e.dst}` })) : [];
  }
  const proj = projectNodes(nodes, segByTrack, W, H);
  if (!proj) { emptyPanel(el, 'No segmentation on this frame to place the graph on.'); return; }
  // The followed node keeps its containment partners bright too.
  const brush = focusKey ? new Set([focusKey, ...containmentPartners(focusKey, hierarchy)]) : null;
  el.innerHTML = renderNodeLink(nodes, edges, proj.positions, {
    width: W, height: H, projById: proj.projById, showLabels: true, hoverKey, brushKeys: brush, tint: trackTint,
  });
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => { const l = Math.hypot(...a) || 1; return a.map((v) => v / l); };

/** Fit y = a·x + b by least squares; null when x has no spread. */
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
 * Draw every stacked frame's nodes in the camera of the frame in focus.
 *
 * The reference frame's nodes are hull glyphs; the other frames' are dots
 * faded by their distance in time (equal in Overlay), joined per node by a
 * trail. A projection outside the widened view is clamped to its edge and
 * counted; a node behind the camera is dropped and counted.
 */
export function renderWorldNodeLink(el, { world, refFrame, tracks, manifest, hoverKey, focusKey, equalize = false }) {
  const dimFor = (key, op) => (focusKey && key !== focusKey ? Math.min(op, 0.15) : op);
  const W = el.clientWidth || 320, H = el.clientHeight || 260;
  const ref = world.frames.find((f) => f.i === refFrame);
  const meta = manifest?.frames?.[refFrame];
  if (!ref || !meta?.camera_pos_glb || !meta?.camera_forward_glb || !meta?.camera_up_glb) {
    emptyPanel(el, 'No camera pose for the frame in focus.');
    return;
  }
  const refNodes = keyedNodes(ref.graphByTrack, orderTracks(tracks));
  const seg0 = tracks.map((t) => ref.segByTrack[t]).find((s) => s?.width && s?.height);
  if (!seg0) { emptyPanel(el, 'No segmentation on the frame in focus to anchor the projection.'); return; }
  const imgW = seg0.width, imgH = seg0.height;

  // Project every stacked node onto the reference camera's normalised image plane.
  const camPos = sub(meta.camera_pos_glb, world.origin);
  const fwd = norm(meta.camera_forward_glb);
  const right = norm(cross(meta.camera_forward_glb, meta.camera_up_glb));
  const up = cross(right, fwd);
  const rays = [];
  for (const fr of world.frames) {
    for (const track of tracks) {
      for (const n of fr.graphByTrack[track]?.nodes || []) {
        if (!Array.isArray(n.pos)) continue;
        const d = sub([n.pos[0] + fr.offset[0], n.pos[1] + fr.offset[1], n.pos[2] + fr.offset[2]], camPos);
        const z = dot(d, fwd);
        rays.push({
          key: `${track}:${n.id}`, track, id: n.id, frameId: fr.i, behind: z <= 1e-6,
          X: z > 1e-6 ? dot(d, right) / z : 0, Y: z > 1e-6 ? dot(d, up) / z : 0,
          color: nodeColor(n, track), label: n.label ?? '', isRef: fr.i === refFrame,
        });
      }
    }
  }

  // Calibrate the plane against the reference frame's own region centroids.
  const xs = [], us = [], ys = [], vs = [];
  for (const r of rays) {
    if (!r.isRef || r.behind) continue;
    const e = ellipsesOf(ref.segByTrack[r.track])?.get(r.id);
    if (!e) continue;
    xs.push(r.X); us.push(e.cx); ys.push(r.Y); vs.push(e.cy);
  }
  const fx = fitLinear(xs, us), fy = fitLinear(ys, vs);
  // Too few nodes to fit: a nominal 60° field of view keeps the view drawable.
  const f0 = imgW / (2 * Math.tan(Math.PI / 6));
  const mapU = fx ? (X) => fx.a * X + fx.b : (X) => f0 * X + imgW / 2;
  const mapV = fy ? (Y) => fy.a * Y + fy.b : (Y) => -f0 * Y + imgH / 2;

  // Widen the image rectangle per side until 90 % of the projections fit, within limits.
  const MARGIN_MIN = 0.12, MARGIN_MAX = 0.45, COVER = 0.9;
  const front = rays.filter((r) => !r.behind).map((r) => ({ u: mapU(r.X), v: mapV(r.Y) }));
  const margin = (vals) => {
    const s = vals.sort((a, c) => a - c);
    const need = s.length ? s[Math.floor(COVER * (s.length - 1))] : 0;
    return Math.min(MARGIN_MAX, Math.max(MARGIN_MIN, need));
  };
  const b = {
    x0: -imgW * margin(front.map((p) => Math.max(0, -p.u / imgW))),
    x1: imgW * (1 + margin(front.map((p) => Math.max(0, (p.u - imgW) / imgW)))),
    y0: -imgH * margin(front.map((p) => Math.max(0, -p.v / imgH))),
    y1: imgH * (1 + margin(front.map((p) => Math.max(0, (p.v - imgH) / imgH)))),
  };
  let behind = 0;
  const mapped = [];
  for (const r of rays) {
    if (r.behind) { behind++; continue; }
    const u = mapU(r.X), v = mapV(r.Y);
    mapped.push({ ...r, clamped: u < b.x0 || u > b.x1 || v < b.y0 || v > b.y1,
      u: Math.min(b.x1, Math.max(b.x0, u)), v: Math.min(b.y1, Math.max(b.y0, v)) });
  }

  // Place the reference glyphs with the same transform as the dots.
  const proj = projectNodes(refNodes, ref.segByTrack, W, H, b);
  if (!proj) { emptyPanel(el, 'No segmentation on the frame in focus to anchor the projection.'); return; }
  const toPanel = (u, v) => [proj.offX + u * proj.scale, proj.offY + v * proj.scale];
  for (const m of mapped) m.panel = toPanel(m.u, m.v);

  // Draw the image frame, the trails, the other frames' dots, then the reference glyphs.
  const svg = [`<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" class="nl-svg" xmlns="http://www.w3.org/2000/svg">`];
  const [x0, y0] = toPanel(0, 0), [x1, y1] = toPanel(imgW, imgH);
  svg.push(`<rect x="${x0.toFixed(1)}" y="${y0.toFixed(1)}" width="${(x1 - x0).toFixed(1)}" height="${(y1 - y0).toFixed(1)}" fill="none" stroke="#263047" stroke-width="1"/>`);
  const order = new Map(world.indices.map((f, k) => [f, k]));
  const byKey = new Map();
  for (const p of mapped) {
    if (p.clamped) continue;   // a clamped position would fake the path
    if (!byKey.has(p.key)) byKey.set(p.key, []);
    byKey.get(p.key).push(p);
  }
  for (const [key, list] of byKey) {
    if (list.length < 2) continue;
    list.sort((a, c) => order.get(a.frameId) - order.get(c.frameId));
    const d = list.map((p) => `${p.panel[0].toFixed(1)},${p.panel[1].toFixed(1)}`).join(' ');
    svg.push(`<polyline points="${d}" fill="none" stroke="${list[0].color}" stroke-width="1.2" opacity="${dimFor(key, 0.4)}" data-node-key="${esc(key)}"/>`);
  }
  const span = Math.max(1, world.indices.length - 1);
  const refOrder = order.get(refFrame) ?? 0;
  for (const p of mapped) {
    if (p.isRef) continue;
    const dist = Math.abs((order.get(p.frameId) ?? 0) - refOrder) / span;
    let op = dimFor(p.key, equalize ? 0.8 : 0.85 - 0.6 * dist);
    if (p.clamped) op *= 0.45;
    const hov = hoverKey && p.key === hoverKey ? ' nl-hover' : '';
    svg.push(`<circle class="nl-node${hov}" data-node-key="${esc(p.key)}" cx="${p.panel[0].toFixed(1)}" cy="${p.panel[1].toFixed(1)}" `
      + `r="${p.clamped ? 2.4 : 3.2}" fill="${p.color}" opacity="${op.toFixed(2)}">`
      + `<title>${esc(p.label)} at frame ${p.frameId}${p.clamped ? ' (outside the view, drawn at its edge)' : ''}</title></circle>`);
  }
  for (const n of refNodes) {
    const pos = proj.positions.get(n.id), g = proj.projById.get(n.id);
    if (!pos || !g) continue;
    const fill = nodeColor(n, n._track);
    const hov = hoverKey && n.key === hoverKey ? ' nl-hover' : '';
    const op = dimFor(n.key, 1);
    const title = `<title>${esc(n.label ?? n.key)}</title>`;
    if (g.poly) {
      const d = g.poly.map(([dx, dy]) => `${(pos[0] + dx).toFixed(1)},${(pos[1] + dy).toFixed(1)}`).join(' ');
      svg.push(`<polygon class="nl-node${hov}" data-node-key="${esc(n.key)}" points="${d}" opacity="${op}" fill="${fill}" fill-opacity="0.5" stroke="${fill}">${title}</polygon>`);
    } else {
      const xf = g.rot ? ` transform="rotate(${g.rot.toFixed(1)} ${pos[0].toFixed(1)} ${pos[1].toFixed(1)})"` : '';
      svg.push(`<ellipse class="nl-node${hov}" data-node-key="${esc(n.key)}" cx="${pos[0].toFixed(1)}" cy="${pos[1].toFixed(1)}" rx="${g.rx.toFixed(1)}" ry="${g.ry.toFixed(1)}"${xf} opacity="${op}" fill="${fill}" fill-opacity="0.55" stroke="${fill}">${title}</ellipse>`);
    }
    if (n.label) svg.push(`<text class="nl-label" x="${(pos[0] + g.rx + 2).toFixed(1)}" y="${(pos[1] + 3).toFixed(1)}" opacity="${op}">${esc(n.label)}</text>`);
  }
  svg.push(`<text class="nl-ref-note" x="${(x0 + 4).toFixed(1)}" y="${(y0 + 12).toFixed(1)}">reference: frame ${refFrame}</text>`);
  const clamped = mapped.filter((m) => m.clamped).length;
  const notes = [clamped && `${clamped} at edge`, behind && `${behind} behind camera`].filter(Boolean);
  if (notes.length) svg.push(`<text class="nl-badge" x="${(x1 - 4).toFixed(1)}" y="${(y1 - 6).toFixed(1)}" text-anchor="end">${notes.join(' · ')}</text>`);
  svg.push('</svg>');
  el.innerHTML = svg.join('');
}

/** Toggle the hover glow on every element of `containers` keyed `key`. */
export function setHoverKey(containers, key) {
  for (const el of containers) {
    for (const node of el?.querySelectorAll('[data-node-key]') ?? []) {
      node.classList.toggle('nl-hover', !!key && node.getAttribute('data-node-key') === key);
    }
  }
}
