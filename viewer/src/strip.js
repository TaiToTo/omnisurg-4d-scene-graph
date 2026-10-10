// Draw the band under the 3D view. In world mode it is the graph through time: one slanted plate per
// stacked frame with the regions' outlines, a numbered chip per node, and a dotted line joining a node
// from plate to plate. In per-frame mode it is one row of region masks per track, frame under frame.
// A click on a plate or a cell asks for that frame; a region carries its node key for focus and hover.
import { ellipsesOf, outlinesOf } from './regions.js';
import { regionsToCanvas } from './cloud_paint.js';
import { esc, hexColor, lighten, nodeKey, rgbToHex } from './format.js';

const DEFAULT_CELL_H = 96;
// The plate: a parallelogram 230 x 96 units, which an SVG matrix maps the frame image onto exactly.
const SLAB_QUAD = [[0, 10.2], [202.9, 0], [229.6, 85.9], [26.6, 96.1]];
const SLAB_VB = { x: -3, y: -3, w: 236, h: 102 };
const SLAB_IMG_BOX = 100;
// Plates overlap a little: each starts before the last one ends.
const SLAB_STEP = 216;
const SLAB_CAP_H = 18;
const SLAB_AXIS_H = 14;
const QUIET_FILL = '#e9e5dc';
const QUIET_STROKE = '#b7b1a5';
const FOCUS = '#0899b4';
const PARTNER = '#d97706';

/** Map a point of the image, as fractions (u, v) of its width and height, onto the plate. */
export function slabMap(u, v) {
  const [A, B, C, D] = SLAB_QUAD;
  return [
    (1 - u) * (1 - v) * A[0] + u * (1 - v) * B[0] + u * v * C[0] + (1 - u) * v * D[0],
    (1 - u) * (1 - v) * A[1] + u * (1 - v) * B[1] + u * v * C[1] + (1 - u) * v * D[1],
  ];
}

function slabImageMatrix(dx) {
  const [A, B, , D] = SLAB_QUAD;
  const k = SLAB_IMG_BOX;
  return [(B[0] - A[0]) / k, (B[1] - A[1]) / k, (D[0] - A[0]) / k, (D[1] - A[1]) / k, A[0] + dx, A[1]]
    .map((v) => v.toFixed(4)).join(' ');
}

const nameOf = (regions, id) => regions.classById.get(id)?.name ?? `region ${id}`;

/**
 * One plate's SVG at horizontal offset dx. Fills `centroids` (node key to plate point) for the lines
 * between plates.
 */
function slabParts(fr, track, info, opts, dx, isStage, caption, centroids) {
  const { focusKey, partnerKey, imageURL } = opts;
  const parts = [];
  const at = ([x, y]) => `${(x + dx).toFixed(1)},${y.toFixed(1)}`;
  parts.push(`<polygon class="slab-plate" data-frame="${fr.i}" points="${SLAB_QUAD.map(at).join(' ')}" `
    + `fill="#f7f8fa" fill-opacity="0.96" stroke="${isStage ? FOCUS : '#c8ced6'}" stroke-width="${isStage ? 2.2 : 1}"/>`);
  if (imageURL) {
    parts.push(`<image class="slab-photo" data-frame="${fr.i}" href="${esc(imageURL(fr.i))}" x="0" y="0" `
      + `width="${SLAB_IMG_BOX}" height="${SLAB_IMG_BOX}" preserveAspectRatio="none" transform="matrix(${slabImageMatrix(dx)})"/>`);
  }
  const regions = fr.regionsByTrack[track];
  const outlines = outlinesOf(regions);
  const ells = ellipsesOf(regions);
  const chips = [];
  const chipAt = new Map();
  if (outlines) {
    const [w, h] = regions.grid;
    const mapPt = (x, y) => { const [px, py] = slabMap(x / w, y / h); return [px + dx, py]; };
    for (const [id, poly] of outlines) {
      const key = nodeKey(track, id);
      const cls = regions.classById.get(id);
      const stroke = hexColor(rgbToHex(cls?.color));
      const isFocus = key === focusKey, isPartner = key === partnerKey;
      const lit = !focusKey || isFocus || isPartner;
      // Over the photo a solid fill hides the tissue; the outline and the chip already place the region.
      const fo = imageURL ? (focusKey ? (lit ? 0.5 : 0.06) : 0.2) : (focusKey ? (isFocus ? 0.85 : isPartner ? 0.6 : 0.75) : 0.5);
      const fillCol = isFocus && !imageURL ? stroke : (imageURL || lit ? lighten(cls?.color, 0.62) : QUIET_FILL);
      parts.push(`<polygon points="${poly.map(([x, y]) => mapPt(x, y).map((v) => v.toFixed(1)).join(',')).join(' ')}" `
        + `fill="${fillCol}" fill-opacity="${isFocus && !imageURL ? 0.55 : fo}" `
        + `stroke="${imageURL || lit ? stroke : QUIET_STROKE}" stroke-opacity="${imageURL && !lit ? 0.3 : 1}" `
        + `stroke-width="${isFocus ? 2 : isPartner ? 1.4 : 0.9}" stroke-linejoin="round" data-node-key="${esc(key)}">`
        + `<title>${esc(nameOf(regions, id))}, frame ${fr.i}</title></polygon>`);
      const e = ells.get(id);
      if (!e) continue;
      const c = mapPt(e.cx, e.cy);
      centroids.set(key, c);
      chipAt.set(key, c);
      // A numbered chip for an object; a class is read by its colour, where a number is noise.
      if (!info.semantic) {
        const ring = isFocus ? FOCUS : isPartner ? PARTNER : '#4a5260';
        chips.push(`<g class="slab-chip" data-node-key="${esc(key)}" opacity="${lit ? 1 : 0.55}">`
          + `<circle cx="${c[0].toFixed(1)}" cy="${c[1].toFixed(1)}" r="${isFocus ? 7 : 6}" fill="#fbfaf7" `
          + `stroke="${ring}" stroke-width="${isFocus || isPartner ? 1.8 : 0.9}"/>`
          + `<text class="slab-id" x="${c[0].toFixed(1)}" y="${(c[1] + 3).toFixed(1)}">${id}</text></g>`);
      }
    }
  }
  // The edge read out at the relations step: the followed node to its neighbour.
  const a = focusKey && chipAt.get(focusKey), b = partnerKey && chipAt.get(partnerKey);
  if (a && b) {
    parts.push(`<line x1="${a[0].toFixed(1)}" y1="${a[1].toFixed(1)}" x2="${b[0].toFixed(1)}" y2="${b[1].toFixed(1)}" stroke="#4a5260" stroke-width="1.1"/>`);
  }
  parts.push(...chips);
  parts.push(`<text class="slab-cap${isStage ? ' on' : ''}" data-frame="${fr.i}" x="${(dx + SLAB_VB.w / 2 - 3).toFixed(1)}" `
    + `y="${SLAB_VB.y + SLAB_VB.h + 12}" text-anchor="middle">${esc(caption)}</text>`);
  return parts;
}

/** A plate's caption: its frame, its time, and whether the track's regions were made on it. */
export function slabCaption(i, timeS, isAnchor) {
  return `f${i}${Number.isFinite(timeS) ? ` · ${timeS.toFixed(1)} s` : ''}${isAnchor ? ' · seed' : ''}`;
}

/**
 * Draw the graph through time: one plate per stacked frame that has the track's regions.
 *
 * @param {HTMLElement} el
 * @param {object} world  the world stack.
 * @param {string} track
 * @param {object} info  the track's record in clip.json.
 * @param {number} stageFrame  the frame on stage.
 * @param {object[]} frames  clip.json's frames.
 * @param {(i:number) => void} onFrame
 * @param {object} [opts]
 * @param {?string} [opts.focusKey]  the followed node, whose line stays lit.
 * @param {?string} [opts.partnerKey]  the neighbour whose edge is read out.
 * @param {number} [opts.cellH]  plate height in pixels.
 * @param {?(i:number) => string} [opts.imageURL]  lay each frame's image on its plate.
 */
export function renderWorldStrip(el, world, track, info, stageFrame, frames, onFrame, opts = {}) {
  const { focusKey = null, partnerKey = null, cellH = DEFAULT_CELL_H } = opts;
  el.classList.remove('tstrip');
  el.innerHTML = '';
  const plates = world.frames.filter((fr) => fr.regionsByTrack[track]);
  if (!plates.length) { el.innerHTML = '<div class="strip-empty">This track has no regions on the stacked frames.</div>'; return; }
  const vbW = (plates.length - 1) * SLAB_STEP + SLAB_VB.w;
  const vbH = SLAB_VB.h + SLAB_CAP_H + SLAB_AXIS_H;
  const pxH = Math.round(cellH * vbH / SLAB_VB.h);
  const pxW = Math.round(pxH * vbW / vbH);
  const parts = [`<svg viewBox="${SLAB_VB.x} ${SLAB_VB.y} ${vbW} ${vbH}" width="${pxW}" height="${pxH}" xmlns="http://www.w3.org/2000/svg">`];
  const perSlab = [];
  plates.forEach((fr, k) => {
    const centroids = new Map();
    const cap = slabCaption(fr.i, frames[fr.i]?.time_s, fr.regionsByTrack[track]?.stage === 'anchor');
    parts.push(...slabParts(fr, track, info, { ...opts, focusKey, partnerKey }, k * SLAB_STEP, fr.i === stageFrame, cap, centroids));
    perSlab.push(centroids);
  });
  // The same node on consecutive plates, joined by a dotted line; with a node followed, only its line
  // and its neighbour's.
  for (let k = 0; k + 1 < perSlab.length; k++) {
    for (const [key, a] of perSlab[k]) {
      const b = perSlab[k + 1].get(key);
      if (!b) continue;
      const isFocus = key === focusKey, isPartner = key === partnerKey;
      if (focusKey && !isFocus && !isPartner) continue;
      parts.push(`<line x1="${a[0].toFixed(1)}" y1="${a[1].toFixed(1)}" x2="${b[0].toFixed(1)}" y2="${b[1].toFixed(1)}" `
        + `stroke="${isFocus ? FOCUS : isPartner ? PARTNER : '#93a0b4'}" stroke-width="${isFocus || isPartner ? 1.4 : 1}" `
        + `stroke-opacity="${isFocus || isPartner ? 0.85 : 0.35}" stroke-dasharray="2 3" data-node-key="${esc(key)}"/>`);
    }
  }
  const ay = SLAB_VB.y + SLAB_VB.h + SLAB_CAP_H + 6;
  const ax1 = SLAB_VB.x + vbW - 6;
  parts.push(`<line x1="${SLAB_VB.x + 30}" y1="${ay}" x2="${ax1}" y2="${ay}" stroke="#9aa3ae" stroke-width="1"/>`
    + `<path d="M${ax1 - 6},${ay - 3.5} L${ax1},${ay} L${ax1 - 6},${ay + 3.5}" fill="none" stroke="#9aa3ae" stroke-width="1"/>`
    + `<text class="slab-time" x="${SLAB_VB.x + 2}" y="${ay + 3.5}">time</text></svg>`);
  el.innerHTML = parts.join('');
  el.firstElementChild.addEventListener('click', (ev) => {
    const f = ev.target.closest?.('[data-frame]')?.getAttribute('data-frame');
    if (f != null) onFrame(Number(f));
  });
}

/**
 * Draw one row of region masks per track over the sampled frames, frame under frame. A frame a track
 * has no regions on keeps its column, marked as such. The heading of a row paints that track on the
 * cloud, or takes it off.
 *
 * @param {HTMLElement} el
 * @param {object} p
 * @param {Array<{i:number, regionsByTrack:object}>} p.frames
 * @param {object[]} p.tracks  the tracks' records in clip.json.
 * @param {string[]} p.painted  the tracks painted on the cloud.
 * @param {number} p.stageFrame
 * @param {(i:number) => string} p.imageURL
 * @param {(i:number) => void} p.onFrame
 * @param {(track:string) => void} p.onPickTrack
 */
export function renderTrackStrip(el, { frames, tracks, painted, stageFrame, imageURL, onFrame, onPickTrack, cellH = 72 }) {
  el.classList.add('tstrip');
  el.innerHTML = '';
  const labels = document.createElement('div');
  labels.className = 'tstrip-labels';
  const scroll = document.createElement('div');
  scroll.className = 'tstrip-scroll';
  let aspect = 16 / 9;
  const any = frames.flatMap((fr) => Object.values(fr.regionsByTrack)).find(Boolean);
  if (any) aspect = any.grid[0] / any.grid[1];
  const cellW = Math.round(cellH * aspect);
  for (const t of tracks) {
    const on = painted.includes(t.id);
    const label = document.createElement('button');
    label.type = 'button';
    label.className = `tstrip-label ${on ? 'on' : 'off'}`;
    label.style.height = `${cellH + 18}px`;
    label.innerHTML = `${esc(t.name)}<span>${esc(t.anchor_badge.toLowerCase())}s → ${esc(t.tracked_badge)}s</span>`
      + `<em>${on ? 'on the 3D view · hide' : 'show on the 3D view'}</em>`;
    label.title = on ? 'Take these regions off the 3D view' : 'Paint these regions on the 3D view';
    label.addEventListener('click', () => onPickTrack(t.id));
    labels.appendChild(label);

    const row = document.createElement('div');
    row.className = `tstrip-cells${painted.length && !on ? ' off' : ''}`;
    for (const fr of frames) {
      const regions = fr.regionsByTrack[t.id];
      const cell = document.createElement('div');
      cell.dataset.frame = String(fr.i);
      cell.addEventListener('click', () => onFrame(fr.i));
      const box = document.createElement('div');
      box.className = 'cell-box';
      box.style.width = `${cellW}px`;
      box.style.height = `${cellH}px`;
      if (!regions) {
        cell.className = `strip-cell strip-gap${fr.i === stageFrame ? ' focus' : ''}`;
        box.textContent = 'no regions';
      } else {
        const anchor = regions.stage === 'anchor';
        cell.className = `strip-cell${anchor ? (t.semantic ? ' anchor-gt' : ' anchor-sam') : ''}${fr.i === stageFrame ? ' focus' : ''}`;
        const img = document.createElement('img');
        img.src = imageURL(fr.i);
        img.alt = '';
        img.loading = 'lazy';
        const canvas = document.createElement('canvas');
        regionsToCanvas(regions, canvas);
        const badge = document.createElement('div');
        badge.className = `cell-badge ${anchor ? (t.semantic ? 'badge-gt' : 'badge-sam') : 'badge-tracked'}`;
        badge.textContent = anchor ? t.anchor_badge : t.tracked_badge;
        box.append(img, canvas, badge);
      }
      const cap = document.createElement('div');
      cap.className = 'cell-label';
      cap.textContent = `f${fr.i}`;
      cell.append(box, cap);
      row.appendChild(cell);
    }
    scroll.appendChild(row);
  }
  el.append(labels, scroll);
}

/** Move the stage mark to another frame, without drawing the band again. */
export function updateStripStage(el, stageFrame) {
  for (const cell of el.querySelectorAll('.strip-cell')) cell.classList.toggle('focus', Number(cell.dataset.frame) === stageFrame);
  for (const p of el.querySelectorAll('.slab-plate')) {
    const on = Number(p.getAttribute('data-frame')) === stageFrame;
    p.setAttribute('stroke', on ? FOCUS : '#c8ced6');
    p.setAttribute('stroke-width', on ? '2.2' : '1');
  }
  for (const c of el.querySelectorAll('text.slab-cap')) c.classList.toggle('on', Number(c.getAttribute('data-frame')) === stageFrame);
}
