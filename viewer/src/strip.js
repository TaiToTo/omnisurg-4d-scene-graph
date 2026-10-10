// Draw the band under the 3D view: in World mode the graph through time, one
// slanted plate per stacked frame with its regions' outlines and the same
// node threaded from plate to plate; in per-frame mode one row of masks per
// track across the sampled frames, the frames each track started from marked.
//
// A click on a plate or a cell seeks that frame. A click on a region carries
// `data-node-key` and is taken by the page, which follows that region.
import { esc, cssColor } from './lib/format.js';
import { regionOutlines } from './lib/contour.js';
import { ellipsesOf } from './nodelink.js';
import { segToCanvas } from './cloud.js';
import { rgbURL } from './data.js';
import { orderTracks, trackInfo } from './state.js';

const ACCENT = '#0899b4';
const PARTNER = '#d97706';

const outlineCache = new WeakMap();
function outlinesOf(seg) {
  if (!seg || !Array.isArray(seg.seg_colors)) return null;
  if (!outlineCache.has(seg)) outlineCache.set(seg, regionOutlines(seg, 2, 3));
  return outlineCache.get(seg);
}

// A plate is a parallelogram, so a frame's image maps onto it with an affine
// SVG transform; the image box is 100 units so the image is rasterised at a
// usable size before the transform scales it.
const SLAB_QUAD = [[0, 10.2], [202.9, 0], [229.6, 85.9], [26.6, 96.1]];
const SLAB_VB = { x: -3, y: -3, w: 236, h: 102 };
const SLAB_STEP = 216;
const SLAB_CAP_H = 18;
const SLAB_AXIS_H = 14;
const SLAB_IMG_BOX = 100;

function slabMap(u, v) {
  const [A, B, C, D] = SLAB_QUAD;
  return [0, 1].map((k) => (1 - u) * (1 - v) * A[k] + u * (1 - v) * B[k] + u * v * C[k] + (1 - u) * v * D[k]);
}

function slabImageMatrix(dx) {
  const [A, B, , D] = SLAB_QUAD;
  const k = SLAB_IMG_BOX;
  return [(B[0] - A[0]) / k, (B[1] - A[1]) / k, (D[0] - A[0]) / k, (D[1] - A[1]) / k, A[0] + dx, A[1]]
    .map((v) => v.toFixed(4)).join(' ');
}

function lighten(color, k = 0.62) {
  if (!Array.isArray(color)) return '#dddddd';
  return `rgb(${color.map((c) => Math.round(c * 255 * (1 - k) + 255 * k)).join(',')})`;
}

const nameOf = (seg, cid) => (seg.classes || []).find((c) => c.id === cid)?.name ?? `region ${cid}`;
// Name what a plate's frame was for the tracks shown: the seed of a seeded track, else an annotated frame.
function anchorNote(fr, tracks) {
  const anchors = tracks.filter((t) => fr.segByTrack?.[t]?.provenance?.stage === 'anchor');
  if (anchors.some((t) => trackInfo(t).seeded)) return ' · seed';
  return anchors.length ? ' · annotated' : '';
}

// One plate's SVG at horizontal offset dx; fills `centroids` ('track:id' → [x, y]) for the threads.
function slabParts(fr, tracks, { dx, isFocus, caption, centroids, photo, focusKey, partnerKey }) {
  const parts = [];
  const at = ([x, y]) => `${(x + dx).toFixed(1)},${y.toFixed(1)}`;
  parts.push(`<polygon class="slab-plate" data-frame="${fr.i}" points="${SLAB_QUAD.map(at).join(' ')}" fill="#f7f8fa" fill-opacity="0.96" `
    + `stroke="${isFocus ? ACCENT : '#c8ced6'}" stroke-width="${isFocus ? 2.2 : 1}"/>`);
  if (photo) {
    parts.push(`<image class="slab-photo" data-frame="${fr.i}" href="${esc(photo)}" x="0" y="0" width="${SLAB_IMG_BOX}" height="${SLAB_IMG_BOX}" `
      + `preserveAspectRatio="none" transform="matrix(${slabImageMatrix(dx)})"/>`);
  }
  const mapPt = (x, y, w, h) => { const [px, py] = slabMap(x / w, y / h); return [px + dx, py]; };

  // The regions; with a region followed, it and its partner stay in colour.
  const chips = [];
  const chipAt = new Map();
  for (const track of orderTracks(tracks)) {
    const seg = fr.segByTrack[track];
    const outlines = outlinesOf(seg);
    if (!outlines) continue;
    const clsById = new Map((seg.classes || []).map((c) => [c.id, c]));
    const ells = ellipsesOf(seg);
    for (const [cid, poly] of outlines) {
      const key = `${track}:${cid}`;
      const cls = clsById.get(cid);
      const stroke = cssColor(cls?.color);
      const focus = focusKey === key, partner = partnerKey === key;
      const lit = !focusKey || focus || partner;
      const fillOpacity = photo ? (focusKey ? (lit ? 0.5 : 0.06) : 0.2) : (focusKey ? (focus ? 0.55 : partner ? 0.6 : 0.75) : 0.5);
      const fill = focus && !photo ? stroke : (photo || lit ? lighten(cls?.color) : '#e9e5dc');
      const line = photo || lit ? stroke : '#b7b1a5';
      const pts = poly.map(([x, y]) => mapPt(x, y, seg.width, seg.height).map((v) => v.toFixed(1)).join(',')).join(' ');
      parts.push(`<polygon points="${pts}" fill="${fill}" fill-opacity="${fillOpacity}" stroke="${line}" stroke-opacity="${photo && !lit ? 0.3 : 1}" `
        + `stroke-width="${focus ? 2 : partner ? 1.4 : 0.9}" stroke-linejoin="round" data-node-key="${esc(key)}">`
        + `<title>${esc(nameOf(seg, cid))} at frame ${fr.i}</title></polygon>`);
      const e = ells?.get(cid);
      if (!e) continue;
      const c = mapPt(e.cx, e.cy, seg.width, seg.height);
      centroids.set(key, c);
      chipAt.set(key, c);
      // A numbered chip per node of an instance track; an annotation reads by its colours.
      if (!trackInfo(track).semantic) {
        const ring = focus ? ACCENT : partner ? PARTNER : '#4a5260';
        chips.push(`<g class="slab-chip" data-node-key="${esc(key)}" opacity="${lit ? 1 : 0.55}">`
          + `<circle cx="${c[0].toFixed(1)}" cy="${c[1].toFixed(1)}" r="${focus ? 7 : 6}" fill="#fbfaf7" stroke="${ring}" stroke-width="${focus || partner ? 1.8 : 0.9}"/>`
          + `<text class="slab-id" x="${c[0].toFixed(1)}" y="${(c[1] + 3).toFixed(1)}">${esc(cid)}</text></g>`);
      }
    }
  }
  const a = focusKey && chipAt.get(focusKey), b = partnerKey && chipAt.get(partnerKey);
  if (a && b) parts.push(`<line x1="${a[0].toFixed(1)}" y1="${a[1].toFixed(1)}" x2="${b[0].toFixed(1)}" y2="${b[1].toFixed(1)}" stroke="#4a5260" stroke-width="1.1"/>`);
  parts.push(...chips);
  parts.push(`<text class="slab-cap${isFocus ? ' on' : ''}" data-frame="${fr.i}" x="${(dx + SLAB_VB.w / 2 - 3).toFixed(1)}" `
    + `y="${SLAB_VB.y + SLAB_VB.h + 12}" text-anchor="middle">${esc(caption)}</text>`);
  return parts;
}

/**
 * Draw the graph through time for a World stack.
 *
 * @param {HTMLElement} el
 * @param {object} p
 * @param {object} p.world  the stack.
 * @param {string[]} p.tracks  displayed tracks.
 * @param {number} p.focusFrame  the stacked frame in focus.
 * @param {object} p.manifest  frame_manifest.json, for the timestamps.
 * @param {function(number)} p.onSeek
 * @param {?string} [p.focusKey]  the followed region, whose thread stays lit.
 * @param {?string} [p.partnerKey]  its neighbour whose edge is read out.
 * @param {?string} [p.clipPath]  set to lay each frame's image under its plate.
 * @param {number} [p.cellH]  plate height in pixels.
 */
export function renderStrip(el, { world, tracks, focusFrame, manifest, onSeek, focusKey = null, partnerKey = null, clipPath = null, cellH = 96 }) {
  el.classList.remove('tstrip');
  el.innerHTML = '';
  const frames = world.frames.filter((fr) => tracks.some((t) => fr.segByTrack[t]?.width));
  if (!frames.length) return;
  const vbW = (frames.length - 1) * SLAB_STEP + SLAB_VB.w;
  const vbH = SLAB_VB.h + SLAB_CAP_H + SLAB_AXIS_H;
  const pxH = Math.round(cellH * vbH / SLAB_VB.h);
  const parts = [`<svg viewBox="${SLAB_VB.x} ${SLAB_VB.y} ${vbW} ${vbH}" width="${Math.round(pxH * vbW / vbH)}" height="${pxH}" xmlns="http://www.w3.org/2000/svg">`];

  // One plate per frame.
  const perSlab = frames.map((fr, k) => {
    const t = manifest?.frames?.[fr.i]?.timestamp_sec;
    const caption = `f${fr.i}${Number.isFinite(t) ? ` · ${t.toFixed(1)} s` : ''}${anchorNote(fr, tracks)}`;
    const centroids = new Map();
    parts.push(...slabParts(fr, tracks, {
      dx: k * SLAB_STEP, isFocus: fr.i === focusFrame, caption, centroids, focusKey, partnerKey,
      photo: clipPath ? rgbURL(clipPath, fr.i) : null,
    }));
    return centroids;
  });

  // Threads: the same node from plate to plate; with a region followed, its thread and its partner's only.
  for (let k = 0; k + 1 < perSlab.length; k++) {
    for (const [key, a] of perSlab[k]) {
      const b = perSlab[k + 1].get(key);
      if (!b) continue;
      const focus = key === focusKey, partner = key === partnerKey;
      if (focusKey && !focus && !partner) continue;
      parts.push(`<line x1="${a[0].toFixed(1)}" y1="${a[1].toFixed(1)}" x2="${b[0].toFixed(1)}" y2="${b[1].toFixed(1)}" `
        + `stroke="${focus ? ACCENT : partner ? PARTNER : '#93a0b4'}" stroke-width="${focus || partner ? 1.4 : 1}" `
        + `stroke-opacity="${focus || partner ? 0.85 : 0.35}" stroke-dasharray="2 3" data-node-key="${esc(key)}"/>`);
    }
  }

  // The time arrow under the captions.
  const ay = SLAB_VB.y + SLAB_VB.h + SLAB_CAP_H + 6;
  const ax1 = SLAB_VB.x + vbW - 6;
  parts.push(`<line x1="${SLAB_VB.x + 30}" y1="${ay}" x2="${ax1}" y2="${ay}" stroke="#9aa3ae" stroke-width="1"/>`
    + `<path d="M${ax1 - 6},${ay - 3.5} L${ax1},${ay} L${ax1 - 6},${ay + 3.5}" fill="none" stroke="#9aa3ae" stroke-width="1"/>`
    + `<text class="slab-time" x="${SLAB_VB.x + 2}" y="${ay + 3.5}">time</text></svg>`);
  el.innerHTML = parts.join('');
  el.firstElementChild.addEventListener('click', (ev) => {
    const f = ev.target.closest?.('[data-frame]')?.getAttribute('data-frame');
    if (f != null) onSeek(Number(f));
  });
}

function stripCell(fr, focusFrame, onSeek) {
  const cell = document.createElement('div');
  cell.className = `strip-cell${fr.i === focusFrame ? ' focus' : ''}`;
  cell.dataset.frame = String(fr.i);
  cell.addEventListener('click', () => onSeek(fr.i));
  return cell;
}

function cellLabel(text) {
  const label = document.createElement('div');
  label.className = 'cell-label';
  label.textContent = text;
  return label;
}

/**
 * Draw one row of masks per track across the sampled frames (per-frame mode).
 *
 * A frame a track has no mask for keeps its column, marked "not annotated", so
 * the rows stay aligned frame under frame.
 *
 * @param {HTMLElement} el
 * @param {object} p
 * @param {Array<{i: number, segByTrack: object}>} p.frames
 * @param {string[]} p.tracks
 * @param {number} p.focusFrame
 * @param {function(number)} p.onSeek
 * @param {string[]} p.painted  tracks painted on the cloud; their rows are marked.
 * @param {function(string)} p.onPickTrack  a click on a row heading.
 * @param {number} [p.cellH]
 */
export function renderTrackingStrip(el, { frames, tracks, focusFrame, onSeek, painted, onPickTrack, cellH = 72 }) {
  el.classList.add('tstrip');
  el.innerHTML = '';
  const labels = document.createElement('div');
  labels.className = 'tstrip-labels';
  const scroll = document.createElement('div');
  scroll.className = 'tstrip-scroll';
  const drawTracks = orderTracks(tracks);
  let aspect = 16 / 9;
  for (const fr of frames) {
    const s = drawTracks.map((t) => fr.segByTrack[t]).find((v) => v?.width && v?.height);
    if (s) { aspect = s.width / s.height; break; }
  }
  for (const track of drawTracks) {
    const info = trackInfo(track);
    const on = painted.includes(track);

    // The row heading: the track, what its marks mean, and whether it is on the cloud.
    const label = document.createElement('div');
    label.className = `tstrip-label ${on ? 'on' : 'off'}`;
    label.style.height = `${cellH + 18}px`;
    label.title = on ? 'Take these regions off the 3D view' : 'Paint these regions on the 3D view';
    label.innerHTML = `${esc(info.label)}<span>${esc(`${info.anchorBadge}S → ${info.trackedBadge}s`)}</span>`
      + `<em>${on ? '● on the 3D view' : 'show on the 3D view'}</em>`;
    label.addEventListener('click', () => onPickTrack(track));
    labels.appendChild(label);

    // The cells.
    const cells = document.createElement('div');
    cells.className = `tstrip-cells${painted.length && !on ? ' off' : ''}`;
    for (const fr of frames) {
      const cell = stripCell(fr, focusFrame, onSeek);
      const seg = fr.segByTrack[track];
      if (!seg?.width) {
        cell.classList.add('strip-gap');
        const mark = document.createElement('div');
        mark.className = 'gap-mark';
        mark.style.height = `${cellH}px`;
        mark.style.width = `${Math.round(cellH * aspect)}px`;
        mark.textContent = 'not annotated';
        cell.append(mark, cellLabel(`f${fr.i}`));
      } else {
        const anchor = seg.provenance?.stage === 'anchor';
        if (anchor) cell.classList.add(info.semantic ? 'anchor-gt' : 'anchor-sam');
        const canvas = document.createElement('canvas');
        segToCanvas(seg, canvas);
        canvas.style.height = `${cellH}px`;
        canvas.style.width = `${Math.round(cellH * seg.width / seg.height)}px`;
        const badge = document.createElement('div');
        badge.className = `cell-badge ${anchor ? (info.semantic ? 'badge-gt' : 'badge-sam') : 'badge-tracked'}`;
        badge.textContent = anchor ? info.anchorBadge : info.trackedBadge;
        cell.append(canvas, badge, cellLabel(`f${fr.i}`));
      }
      cells.appendChild(cell);
    }
    scroll.appendChild(cells);
  }
  el.append(labels, scroll);
}

/** Move the focus mark of either band to `focusFrame` without redrawing it. */
export function updateStripFocus(el, focusFrame) {
  for (const cell of el.querySelectorAll('.strip-cell')) cell.classList.toggle('focus', Number(cell.dataset.frame) === focusFrame);
  for (const p of el.querySelectorAll('.slab-plate')) {
    const on = Number(p.getAttribute('data-frame')) === focusFrame;
    p.setAttribute('stroke', on ? ACCENT : '#c8ced6');
    p.setAttribute('stroke-width', on ? '2.2' : '1');
  }
  for (const c of el.querySelectorAll('text.slab-cap')) c.classList.toggle('on', Number(c.getAttribute('data-frame')) === focusFrame);
}
