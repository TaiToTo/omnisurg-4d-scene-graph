// Draw what the graph holds about the followed node: its edge to the
// neighbour it changes against most, through time, and its numbered events.
//
// With the edge's three numbers (`lib/edge_axes.js`) the edge is three rows,
// side, height and depth; without them, one row of the stored word. Frame f
// sits at the same x in every row, with the playhead over all of them.
import { esc, hexColor } from './lib/format.js';
import { RELATION_COLORS, nodeEvents, relationRuns, topPartners } from './lib/relations.js';

const AXIS_L = 118, AXIS_R = 30;
const ROW_H = 16, TOP = 6, EDGE_H = 22, AXIS_H = 30, AXIS_GAP = 4;
const EVENT_COLORS = { rel: '#e08a00', enter: '#0899b4', exit: '#93a0b4' };

function axisX(f, W, n) {
  return AXIS_L + (f / Math.max(1, n - 1)) * (W - AXIS_L - AXIS_R);
}

function axisSeek(svg, ev, W, n, maxFrame, onSeek) {
  const r = svg.getBoundingClientRect();
  const px = (ev.clientX - r.left) / r.width * W;
  onSeek(Math.min(maxFrame, Math.max(0, Math.round(((px - AXIS_L) / (W - AXIS_L - AXIS_R)) * (n - 1)))));
}

/**
 * Draw the band for `nodeId` of `tg`, and return its events.
 *
 * @param {HTMLElement} el
 * @param {object} p
 * @param {object} p.tg  the temporal graph of the node's track.
 * @param {number} p.nodeId
 * @param {number} p.maxFrame
 * @param {number} p.frame  the playhead.
 * @param {function(number)} p.onSeek
 * @param {function(number)} p.onEvent  a click on event chip k.
 * @param {?number} p.selectedEvent  the chip drawn ringed.
 * @param {function(number): ?string} p.anatomyOf  a node's annotated class, if the hierarchy knows it.
 * @param {?object} p.axes  `edgeAxes` output for the edge, or null.
 * @returns {Array|false} the events drawn, or false when the node is not in the graph.
 */
export function renderNodeFocus(el, { tg, nodeId, maxFrame, frame, onSeek, onEvent, selectedEvent, anatomyOf, axes }) {
  const node = tg?.nodes?.find((m) => m.id === nodeId);
  if (!node) { el.innerHTML = ''; return false; }
  const n = maxFrame + 1;
  const W = Math.max(300, Math.floor(el.getBoundingClientRect().width || el.clientWidth || 800) - 2);
  const x = (f) => axisX(f, W, n);
  const labelOf = (id) => {
    const base = tg.nodes.find((m) => m.id === id)?.label ?? `node ${id}`;
    const anat = anatomyOf?.(id);
    return anat ? `${base} · ${anat.length > 14 ? `${anat.slice(0, 13)}…` : anat}` : base;
  };
  const partners = topPartners(tg, nodeId);
  const events = nodeEvents(node, partners, maxFrame, labelOf);
  const byAxis = partners.length && axes?.byFrame?.size ? axes.byFrame : null;
  const edgeH = !partners.length ? 0 : byAxis ? 3 * (AXIS_H + AXIS_GAP) : EDGE_H;
  const H = TOP + edgeH + ROW_H + 14;
  const parts = [`<svg viewBox="0 0 ${W} ${H}" height="${H}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">`];
  let y = TOP;
  const rowLabel = (t, h = ROW_H) => parts.push(`<text class="nf-label" x="${AXIS_L - 8}" y="${y + h / 2 + 3.5}" text-anchor="end">${esc(t)}</text>`);

  // The edge as three axes: + above a row's line, − below; coloured where the axis is the word kept.
  if (byAxis) {
    // One scale for the three rows, set by the 90th percentile.
    const mags = [...byAxis.values()].flatMap((v) => v.c.map(Math.abs)).sort((a, b) => a - b);
    const vmax = Math.max(1e-9, mags[Math.floor(0.9 * (mags.length - 1))]);
    const step = x(1) - x(0);
    const colW = Math.max(2, step * 0.72);
    const colX = (f) => x(f) + step * 0.14;
    const half = AXIS_H / 2 - 1;
    for (const [ax, pos, neg] of [[0, 'right', 'left'], [1, 'above', 'below'], [2, 'front', 'behind']]) {
      const mid = y + AXIS_H / 2;
      parts.push(`<text class="nf-axis" x="${AXIS_L - 8}" y="${y + 10}" text-anchor="end">${pos} ▲</text>`
        + `<text class="nf-axis" x="${AXIS_L - 8}" y="${y + AXIS_H - 3}" text-anchor="end">${neg} ▼</text>`
        + `<line x1="${AXIS_L}" y1="${mid}" x2="${x(maxFrame).toFixed(1)}" y2="${mid}" stroke="#c8ced6" stroke-width="0.8"/>`);
      let run = null;
      const flush = () => {
        if (run && run.w > 26) parts.push(`<text class="nf-seg" x="${(run.x + 2).toFixed(1)}" y="${(run.up ? y + 8 : y + AXIS_H - 1).toFixed(1)}">${esc(run.word)}</text>`);
        run = null;
      };
      for (let f = 0; f <= maxFrame; f++) {
        const v = byAxis.get(f);
        if (!v) { flush(); continue; }
        const val = v.c[ax];
        const word = val >= 0 ? pos : neg;
        const kept = v.word === word;
        const h = Math.max(1, Math.min(1, Math.abs(val) / vmax) * half);
        const col = kept ? hexColor(RELATION_COLORS[word] ?? 0x888888) : '#c3c9d1';
        parts.push(`<rect x="${colX(f).toFixed(1)}" y="${(val >= 0 ? mid - h : mid).toFixed(1)}" width="${colW.toFixed(1)}" height="${h.toFixed(1)}" fill="${col}">`
          + `<title>f${f}: ${esc(word)} ${Math.abs(val / vmax).toFixed(2)}${kept ? ' (the word the graph keeps)' : ''}</title></rect>`);
        if (kept && run?.word === word) run.w = colX(f) + colW - run.x;
        else { flush(); if (kept) run = { word, x: colX(f), w: colW, up: val >= 0 }; }
      }
      flush();
      y += AXIS_H + AXIS_GAP;
    }
  } else {
    // The edge as its stored word, run by run.
    for (const [other, e] of partners) {
      rowLabel(`→ ${labelOf(other)}`, EDGE_H);
      for (const { rel, a, b } of relationRuns(e.relByFrame)) {
        const wSeg = Math.max(2, x(b) - x(a) + 2);
        parts.push(`<rect x="${x(a).toFixed(1)}" y="${y + 3}" width="${wSeg.toFixed(1)}" height="${EDGE_H - 6}" rx="3" `
          + `fill="${hexColor(RELATION_COLORS[rel] ?? 0x888888)}" fill-opacity="0.8">`
          + `<title>${esc(labelOf(nodeId))} is ${esc(rel)} of ${esc(labelOf(other))} (frames ${a}–${b})</title></rect>`);
        if (wSeg > 30) parts.push(`<text class="nf-seg" x="${(x(a) + 4).toFixed(1)}" y="${y + EDGE_H / 2 + 3.5}">${esc(rel)}</text>`);
      }
      y += EDGE_H;
    }
  }

  // The events, numbered in time; chips closer than a radius alternate up and down.
  rowLabel('events');
  let lastX = -Infinity, stagger = false;
  events.forEach((ev, k) => {
    const cx = x(ev.f);
    const col = EVENT_COLORS[ev.kind];
    stagger = cx - lastX < 14 ? !stagger : false;
    lastX = cx;
    const cy = y + ROW_H / 2 + (stagger ? 5 : -1);
    const ring = k === selectedEvent ? `<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="8.5" fill="none" stroke="#1d2530" stroke-width="1.6"/>` : '';
    parts.push(`<line x1="${cx.toFixed(1)}" y1="${TOP}" x2="${cx.toFixed(1)}" y2="${(cy - 6).toFixed(1)}" stroke="${col}" stroke-width="1" stroke-opacity="0.45" stroke-dasharray="2 2"/>`
      + `<g class="nf-event" data-idx="${k}">${ring}<circle cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="6" fill="${col}"/>`
      + `<text class="nf-ev-num" x="${cx.toFixed(1)}" y="${(cy + 2.7).toFixed(1)}" text-anchor="middle">${k + 1}</text>`
      + `<title>(${k + 1}) ${esc(ev.label)} at f${ev.f}; click to show it on the cloud</title></g>`);
  });
  if (!events.length) parts.push(`<text class="nf-val" x="${AXIS_L}" y="${y + ROW_H - 5}">none in this clip</text>`);
  y += ROW_H;

  // The playhead and the frame axis.
  parts.push(`<line class="nf-cursor" x1="${x(frame).toFixed(1)}" y1="2" x2="${x(frame).toFixed(1)}" y2="${(H - 12).toFixed(1)}" stroke="#1d2530" stroke-width="1.5"/>`
    + `<text class="tl-axis" x="${AXIS_L}" y="${H - 2}">0</text>`
    + `<text class="tl-axis" x="${x(maxFrame).toFixed(1)}" y="${H - 2}" text-anchor="end">${maxFrame}</text></svg>`);
  el.innerHTML = parts.join('');

  const svg = el.firstElementChild;
  svg.addEventListener('pointerdown', (ev) => {
    if (ev.target.closest?.('.nf-event')) return;
    svg.setPointerCapture(ev.pointerId);
    axisSeek(svg, ev, W, n, maxFrame, onSeek);
  });
  svg.addEventListener('pointermove', (ev) => { if (ev.buttons) axisSeek(svg, ev, W, n, maxFrame, onSeek); });
  for (const g of el.querySelectorAll('.nf-event')) g.addEventListener('click', () => onEvent(Number(g.dataset.idx)));
  return events;
}

/** Move the band's playhead to `frame` without redrawing it. */
export function updateNodeFocusCursor(el, frame, maxFrame) {
  const svg = el.firstElementChild;
  const cursor = svg?.querySelector('.nf-cursor');
  if (!cursor) return;
  const xf = axisX(frame, svg.viewBox.baseVal.width, maxFrame + 1).toFixed(1);
  cursor.setAttribute('x1', xf);
  cursor.setAttribute('x2', xf);
}
