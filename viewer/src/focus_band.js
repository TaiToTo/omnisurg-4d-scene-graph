// Draw what the graph through time holds about the followed node: its edge to the neighbour it changes
// against most, frame by frame, and the numbered events (the node enters or leaves the view, the edge's
// relation changes). With the edge's three axes the band draws each axis as a row; without them, the
// stored relation. Frame f sits at the same x in every row, and a click on the band asks for a frame.
import { RELATION_COLORS, MIRROR_REL, esc, hexColor } from './format.js';

const ROW_H = 16, TOP = 6, EDGE_H = 22, AXIS_H = 30, AXIS_GAP = 4;
const AXIS_L = 118, AXIS_R = 30;

function axisX(f, W, n) {
  return AXIS_L + (f / Math.max(1, n - 1)) * (W - AXIS_L - AXIS_R);
}

/**
 * The neighbours of `nodeId` in a graph through time, the one whose relation to it changes most first,
 * ties by how many frames they share. A relation is read from `nodeId`'s side: "nodeId is <relation> of".
 *
 * @returns {Array<[number, {relByFrame:Map<number,string>, changes:Array<{f,from,to}>, frames:number}]>}
 */
export function topPartners(tg, nodeId, k = 1) {
  const byPartner = new Map();
  for (const r of tg?.relations || []) {
    if (r.edge_type && r.edge_type !== 'spatial') continue;
    if (r.src !== nodeId && r.dst !== nodeId) continue;
    const other = r.src === nodeId ? r.dst : r.src;
    if (!byPartner.has(other)) byPartner.set(other, { relByFrame: new Map(), changes: [], frames: 0 });
    const e = byPartner.get(other);
    e.frames += (r.frames || []).length;
    const rel = r.src === nodeId ? r.relation : (MIRROR_REL[r.relation] ?? r.relation);
    for (const f of r.frames || []) if (Number.isInteger(f)) e.relByFrame.set(f, rel);
  }
  for (const e of byPartner.values()) {
    const fr = [...e.relByFrame.keys()].sort((a, b) => a - b);
    for (let j = 1; j < fr.length; j++) {
      if (fr[j] === fr[j - 1] + 1 && e.relByFrame.get(fr[j]) !== e.relByFrame.get(fr[j - 1])) {
        e.changes.push({ f: fr[j], from: e.relByFrame.get(fr[j - 1]), to: e.relByFrame.get(fr[j]) });
      }
    }
  }
  return [...byPartner.entries()]
    .sort((a, b) => (b[1].changes.length - a[1].changes.length) || (b[1].frames - a[1].frames))
    .slice(0, k);
}

/** Runs of consecutive frames, as [first, last] pairs. */
export function runsOf(frames) {
  const runs = [];
  for (const f of frames.filter(Number.isInteger).sort((a, b) => a - b)) {
    const last = runs[runs.length - 1];
    if (last && f === last[1] + 1) last[1] = f;
    else runs.push([f, f]);
  }
  return runs;
}

/**
 * The events of a node, in time order: where it enters and leaves the view, and where its relation to
 * each given neighbour changes.
 *
 * @returns {Array<{f:number, kind:'enter'|'exit'|'rel', partner?:number, label:string}>}
 */
export function nodeEvents(node, partners, lastFrame, labelOf) {
  const events = [];
  for (const [a, b] of runsOf(node.present_frames || [])) {
    if (a > 0) events.push({ f: a, kind: 'enter', label: 'enters the view' });
    if (b < lastFrame) events.push({ f: b + 1, kind: 'exit', label: 'leaves the view' });
  }
  for (const [other, e] of partners) {
    for (const c of e.changes) events.push({ f: c.f, kind: 'rel', partner: other, label: `to ${labelOf(other)}: ${c.from} → ${c.to}` });
  }
  return events.sort((a, b) => a.f - b.f);
}

/**
 * Draw the band for node `nodeId` of graph through time `tg`.
 *
 * @param {HTMLElement} el
 * @param {object} p
 * @param {(f:number) => void} p.onFrame
 * @param {?(k:number) => void} p.onEvent  a click on event `k`.
 * @param {?number} p.selectedEvent  the event whose regions are lit on the cloud.
 * @param {(id:number) => string} p.labelOf  a node's name.
 * @param {?{byFrame: Map<number, {c:number[], word:?string}>}} p.axes  the edge's three axes, from edge_axes.
 * @returns {Array|false} the events drawn, or false when the node is not in the graph.
 */
export function renderFocusBand(el, { tg, nodeId, lastFrame, frame, onFrame, onEvent = null, selectedEvent = null, labelOf, axes = null }) {
  const node = tg?.nodes?.find((n) => n.id === nodeId);
  if (!node) { el.innerHTML = ''; return false; }
  const n = lastFrame + 1;
  const W = Math.max(300, Math.floor(el.getBoundingClientRect().width || 0) - 2 || 800);
  const x = (f) => axisX(f, W, n);
  const partners = topPartners(tg, nodeId);
  const events = nodeEvents(node, partners, lastFrame, labelOf);
  const byAxis = partners.length && axes?.byFrame?.size ? axes.byFrame : null;
  const edgeH = !partners.length ? 0 : byAxis ? 3 * (AXIS_H + AXIS_GAP) : EDGE_H;
  const H = TOP + edgeH + ROW_H + 14;
  const parts = [`<svg viewBox="0 0 ${W} ${H}" height="${H}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">`];
  let y = TOP;
  const rowLabel = (text, h = ROW_H) => parts.push(`<text class="nf-label" x="${AXIS_L - 8}" y="${y + h / 2 + 3.5}" text-anchor="end">${esc(text)}</text>`);

  if (byAxis) {
    // One scale for the three rows, since which axis is largest is the reading; set by the 90th
    // percentile so one far frame does not flatten the rest.
    const mags = [...byAxis.values()].flatMap((v) => v.c.map(Math.abs)).sort((a, b) => a - b);
    const vmax = Math.max(1e-9, mags[Math.floor(0.9 * (mags.length - 1))]);
    const step = x(1) - x(0);
    const colW = Math.max(2, step * 0.72);
    // Each frame's column is centred on the frame, where the cursor and the events are.
    const colX = (f) => x(f) - colW / 2;
    const half = AXIS_H / 2 - 1;
    for (const [ax, pos, neg] of [[0, 'right', 'left'], [1, 'above', 'below'], [2, 'front', 'behind']]) {
      const mid = y + AXIS_H / 2;
      parts.push(`<text class="nf-axis" x="${AXIS_L - 8}" y="${y + 10}" text-anchor="end">${pos} ▲</text>`
        + `<text class="nf-axis" x="${AXIS_L - 8}" y="${y + AXIS_H - 3}" text-anchor="end">${neg} ▼</text>`
        + `<line x1="${AXIS_L}" y1="${mid}" x2="${x(lastFrame).toFixed(1)}" y2="${mid}" stroke="#c8ced6" stroke-width="0.8"/>`);
      let run = null;
      const flush = () => {
        if (run && run.w > 26) parts.push(`<text class="nf-seg" x="${(run.x + 2).toFixed(1)}" y="${(run.up ? y + 8 : y + AXIS_H - 1).toFixed(1)}">${esc(run.word)}</text>`);
        run = null;
      };
      for (let f = 0; f <= lastFrame; f++) {
        const v = byAxis.get(f);
        if (!v) { flush(); continue; }
        const val = v.c[ax];
        const word = val >= 0 ? pos : neg;
        const kept = v.word === word;
        const h = Math.max(1, Math.min(1, Math.abs(val) / vmax) * half);
        parts.push(`<rect x="${colX(f).toFixed(1)}" y="${(val >= 0 ? mid - h : mid).toFixed(1)}" width="${colW.toFixed(1)}" `
          + `height="${h.toFixed(1)}" fill="${kept ? hexColor(RELATION_COLORS[word] ?? 0x888888) : '#c3c9d1'}">`
          + `<title>frame ${f}: ${esc(word)} ${Math.abs(val / vmax).toFixed(2)}${kept ? ', the relation the graph keeps' : ''}</title></rect>`);
        if (kept && run?.word === word) run.w = colX(f) + colW - run.x;
        else { flush(); if (kept) run = { word, x: colX(f), w: colW, up: val >= 0 }; }
      }
      flush();
      y += AXIS_H + AXIS_GAP;
    }
  } else {
    for (const [other, e] of partners) {
      rowLabel(`→ ${labelOf(other)}`, EDGE_H);
      for (const [rel, frames] of groupByRelation(e.relByFrame)) {
        for (const [a, b] of runsOf(frames)) {
          const w = Math.max(2, x(b) - x(a) + 2);
          parts.push(`<rect x="${x(a).toFixed(1)}" y="${y + 3}" width="${w.toFixed(1)}" height="${EDGE_H - 6}" rx="3" `
            + `fill="${hexColor(RELATION_COLORS[rel] ?? 0x888888)}" fill-opacity="0.8"><title>${esc(labelOf(nodeId))} is ${esc(rel)} `
            + `of ${esc(labelOf(other))}, frames ${esc(a)} to ${esc(b)}</title></rect>`);
          if (w > 30) parts.push(`<text class="nf-seg" x="${(x(a) + 4).toFixed(1)}" y="${y + EDGE_H / 2 + 3.5}">${esc(rel)}</text>`);
        }
      }
      y += EDGE_H;
    }
  }

  // The events, numbered in time order, each with a faint guide up through the rows above it.
  rowLabel('events');
  let lastX = -Infinity, stagger = false;
  events.forEach((ev, k) => {
    const cx = x(ev.f);
    const col = ev.kind === 'rel' ? '#e08a00' : ev.kind === 'enter' ? '#0899b4' : '#93a0b4';
    stagger = cx - lastX < 14 ? !stagger : false;
    lastX = cx;
    const cy = y + ROW_H / 2 + (stagger ? 5 : -1);
    parts.push(`<line x1="${cx.toFixed(1)}" y1="${TOP}" x2="${cx.toFixed(1)}" y2="${(cy - 6).toFixed(1)}" stroke="${col}" `
      + 'stroke-width="1" stroke-opacity="0.45" stroke-dasharray="2 2"/>');
    const ring = k === selectedEvent ? `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="8.5" fill="none" stroke="#1d2530" stroke-width="1.6"/>` : '';
    parts.push(`<g class="nf-event" data-idx="${k}" tabindex="0" role="button">${ring}<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="6" fill="${col}"/>`
      + `<text class="nf-ev-num" x="${cx.toFixed(1)}" y="${(cy + 2.7).toFixed(1)}" text-anchor="middle">${k + 1}</text>`
      + `<title>(${k + 1}) ${esc(ev.label)}, frame ${esc(ev.f)}; click to light it on the cloud</title></g>`);
  });
  if (!events.length) parts.push(`<text class="nf-val" x="${AXIS_L}" y="${y + ROW_H - 5}">none in this clip</text>`);
  y += ROW_H;
  parts.push(`<line class="nf-cursor" x1="${x(frame).toFixed(1)}" y1="2" x2="${x(frame).toFixed(1)}" y2="${(H - 12).toFixed(1)}" stroke="#1d2530" stroke-width="1.5"/>`
    + `<text class="tl-axis" x="${AXIS_L}" y="${H - 2}">0</text>`
    + `<text class="tl-axis" x="${x(lastFrame).toFixed(1)}" y="${H - 2}" text-anchor="end">${lastFrame}</text></svg>`);
  el.innerHTML = parts.join('');

  const svg = el.firstElementChild;
  const seek = (ev) => {
    const r = svg.getBoundingClientRect();
    const px = (ev.clientX - r.left) / r.width * W;
    onFrame(Math.min(lastFrame, Math.max(0, Math.round(((px - AXIS_L) / (W - AXIS_L - AXIS_R)) * (n - 1)))));
  };
  svg.addEventListener('pointerdown', (ev) => {
    if (ev.target.closest?.('.nf-event')) return;
    svg.setPointerCapture(ev.pointerId);
    seek(ev);
  });
  svg.addEventListener('pointermove', (ev) => { if (ev.buttons && !ev.target.closest?.('.nf-event')) seek(ev); });
  if (onEvent) {
    for (const g of el.querySelectorAll('.nf-event')) {
      const pick = () => onEvent(Number(g.dataset.idx));
      g.addEventListener('click', pick);
      g.addEventListener('keydown', (ev) => {
        if (ev.key !== 'Enter' && ev.key !== ' ') return;
        ev.preventDefault();
        ev.stopPropagation();
        pick();
      });
    }
  }
  return events;
}

function groupByRelation(relByFrame) {
  const out = new Map();
  for (const [f, rel] of relByFrame) {
    if (!out.has(rel)) out.set(rel, []);
    out.get(rel).push(f);
  }
  return out;
}

/** Move the band's frame cursor without drawing it again. */
export function updateFocusCursor(el, frame, lastFrame) {
  const svg = el.firstElementChild;
  const cursor = svg?.querySelector('.nf-cursor');
  if (!cursor) return;
  const xf = axisX(frame, svg.viewBox.baseVal.width, lastFrame + 1).toFixed(1);
  cursor.setAttribute('x1', xf);
  cursor.setAttribute('x2', xf);
}
