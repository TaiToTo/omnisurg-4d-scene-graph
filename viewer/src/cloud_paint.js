// Paint a frame's regions on its point cloud, light the followed regions, and hide the points a view
// leaves out. Clones of a cloud share the cached geometry, so the cloud's own colours are kept on the
// geometry once and every repaint starts from them. A view that hides points draws a filtered copy and
// keeps the full geometry beside it, whose point order the labels follow.
import { filteredGeometry } from './vertex_filter.js';
import { axialDepth, flyingPixelMask, grazingMask, silhouetteHalo } from './depth_filter.js';
import { invalidate } from './stage.js';

// How strongly a region's colour covers the cloud's own.
const SEG_ALPHA = 0.45;
// Region colours are moved this far toward white before they are painted, so they read softly on the page.
const PASTEL = 0.35;
const pastel = (v) => v * (1 - PASTEL) + PASTEL;

function eachPoints(group, fn) {
  group.traverse((obj) => { if (obj.isPoints) fn(obj); });
}

/** Put a Points back on its full geometry, freeing any filtered copy. */
function useFullGeometry(obj) {
  const full = obj.userData.fullGeometry;
  if (!full) { obj.userData.fullGeometry = obj.geometry; return; }
  if (obj.geometry !== full) {
    obj.geometry.dispose();
    obj.geometry = full;
    obj.userData.sharedGeometry = true;
  }
}

// The flying pixels depend on the cloud alone, so they are found once per cached geometry.
const flyingCache = new WeakMap();

function flyingPixelsFor(geom, cam, grid) {
  if (flyingCache.has(geom)) return flyingCache.get(geom);
  const pos = geom.attributes.position;
  let mask = null;
  if (pos && !pos.isInterleavedBufferAttribute) {
    const z = axialDepth(pos.array, cam.pos, cam.fwd);
    const step = flyingPixelMask(z, grid[0], grid[1]);
    mask = step && { step, edgeOn: grazingMask(pos.array, cam.pos, z, grid[0], grid[1]) };
  }
  flyingCache.set(geom, mask);
  return mask;
}

function labelsFor(regionsByTrack, track, count) {
  const r = track ? regionsByTrack?.[track] : null;
  return r && r.labels.length === count ? r.labels : null;
}

/**
 * Which points to hide, 1 to hide, for every reason the view has; null when it hides none.
 *
 * @param {object} regionsByTrack  track to its regions on this frame.
 * @param {number} count  the points of the full cloud.
 * @param {object} view
 * @param {'all'|'tissue'|'tools'} view.show  every point; the tissue without instruments and without
 *        flying pixels; or the instruments alone.
 * @param {?{track:string, ids:number[]}} view.instruments  the track whose classes tell instruments apart.
 * @param {?string} view.isolate  a node key: hide every point but that region's.
 * @param {object} [extra]
 * @param {object} [extra.geom]  the full geometry, for the flying pixels.
 * @param {?{pos:number[], fwd:number[]}} [extra.cam]  the camera in the cloud's coordinates.
 * @param {?number[]} [extra.grid]  the cloud's pixel grid [w, h].
 * @param {object} [extra.stats]  filled with the count of points each reason hides, and `total`.
 * @param {boolean} [extra.context]  the frame on stage while a region is isolated: it keeps its points.
 */
export function cloudDropMask(regionsByTrack, count, view, { geom = null, cam = null, grid = null, stats = null, context = false } = {}) {
  const mask = new Uint8Array(count);
  let n = 0;
  const drop = (i, why) => { if (!mask[i]) { mask[i] = 1; n++; if (stats) stats[why] = (stats[why] ?? 0) + 1; } };
  if (stats) stats.total = (stats.total ?? 0) + count;

  const { show, instruments } = view;
  const filter = show === 'all' ? null : instruments;
  const vs = filter ? labelsFor(regionsByTrack, filter.track, count) : null;
  const inst = vs ? new Set(filter.ids) : null;
  // Instrument classes this frame has no labels for: 'tools' then shows nothing rather than every point
  // under a button that promises the instruments alone, and the status line says why.
  const unlabeled = !!filter && !vs;
  if (unlabeled && stats) stats.unlabeled = (stats.unlabeled ?? 0) + 1;

  if (show === 'tissue' && vs) for (let i = 0; i < count; i++) if (inst.has(vs[i])) drop(i, 'instruments');
  if (show === 'tools' && vs) for (let i = 0; i < count; i++) if (!inst.has(vs[i])) drop(i, 'tissue');
  if (show === 'tools' && unlabeled) for (let i = 0; i < count; i++) drop(i, 'tissue');

  if (show !== 'all') {
    const fp = cam && geom && grid ? flyingPixelsFor(geom, cam, grid) : null;
    if (fp) {
      for (let i = 0; i < count; i++) if (fp.step[i]) drop(i, 'depthStep');
      if (fp.edgeOn) for (let i = 0; i < count; i++) if (fp.edgeOn[i]) drop(i, 'edgeOn');
    }
    if (show === 'tissue' && vs && grid) {
      const halo = silhouetteHalo(vs, filter.ids, grid[0], grid[1]);
      if (halo) for (let i = 0; i < count; i++) if (halo[i]) drop(i, 'toolHalo');
    }
  }

  // Isolation keeps one region. A frame without labels for its track stays whole: hiding every point
  // would look like the region vanished, and the caller reports how many frames could be isolated.
  if (view.isolate && !context) {
    const at = view.isolate.lastIndexOf(':');
    const keep = labelsFor(regionsByTrack, view.isolate.slice(0, at), count);
    const id = Number(view.isolate.slice(at + 1));
    if (keep) for (let i = 0; i < count; i++) if (keep[i] !== id) drop(i, 'isolation');
  }
  return n ? mask : null;
}

/**
 * Hide the points the view leaves out, on a cloud already painted. The group carries the frame's camera
 * (`userData.cam`), its pixel grid (`userData.grid`) and whether it is the isolated stage
 * (`userData.isoContext`).
 */
export function applyCloudFilter(group, regionsByTrack, view, stats = null) {
  eachPoints(group, (obj) => {
    useFullGeometry(obj);
    const full = obj.userData.fullGeometry;
    const mask = cloudDropMask(regionsByTrack, full.attributes.position.count, view, {
      geom: full, cam: group.userData.cam ?? null, grid: group.userData.grid ?? null, stats,
      context: group.userData.isoContext === true,
    });
    if (!mask) return;
    obj.geometry = filteredGeometry(full, mask);
    obj.userData.sharedGeometry = false;
  });
  invalidate();
}

function originalColors(obj) {
  const attr = obj.geometry.attributes.color;
  if (attr && !obj.geometry.userData.origColors) obj.geometry.userData.origColors = attr.array.slice();
  return attr;
}

/**
 * Paint the regions of `tracks` over the cloud's own colours, then hide what the view leaves out.
 * Several tracks paint in turn, each over the last. A track whose labels do not fit the cloud is skipped.
 */
export function applySegBlend(group, regionsByTrack, tracks, view, stats = null) {
  eachPoints(group, useFullGeometry);
  eachPoints(group, (obj) => {
    const attr = originalColors(obj);
    if (!attr) return;
    attr.array.set(obj.geometry.userData.origColors);
    for (const track of tracks) {
      const r = regionsByTrack[track];
      if (!r || r.labels.length !== attr.count) continue;
      const palette = new Map(r.classes.map((c) => [c.id, c.color.map(pastel)]));
      for (let i = 0; i < attr.count; i++) {
        const c = palette.get(r.labels[i]);
        if (!c) continue;
        attr.setXYZ(i, attr.getX(i) * (1 - SEG_ALPHA) + c[0] * SEG_ALPHA,
          attr.getY(i) * (1 - SEG_ALPHA) + c[1] * SEG_ALPHA, attr.getZ(i) * (1 - SEG_ALPHA) + c[2] * SEG_ALPHA);
      }
    }
    attr.needsUpdate = true;
  });
  applyCloudFilter(group, regionsByTrack, view, stats);
}

/**
 * Light some regions and pale the rest: a lit point moves `mix` of the way to its key's colour, any
 * other point `dim` of the way to white, from the cloud's own colours.
 *
 * @param {Map<string, number[]>} keyColors  node key to a 0-1 RGB colour.
 * @returns {boolean} whether any key had labels on this frame; a region can be absent from a frame.
 */
export function applyRegionHighlight(group, regionsByTrack, keyColors, view, { mix = 0.65, dim = 0.12 } = {}) {
  const sel = [];
  for (const [key, color] of keyColors) {
    const at = key.lastIndexOf(':');
    const r = regionsByTrack[key.slice(0, at)];
    if (r) sel.push({ labels: r.labels, id: Number(key.slice(at + 1)), color });
  }
  if (!sel.length) return false;
  eachPoints(group, useFullGeometry);
  eachPoints(group, (obj) => {
    const attr = originalColors(obj);
    if (!attr) return;
    // The saved colours are the attribute's raw array: 0-255 bytes when it is normalized Uint8, and the
    // whole shared buffer when the attribute is interleaved with others.
    const orig = obj.geometry.userData.origColors;
    const scale = orig.BYTES_PER_ELEMENT === 1 ? 1 / 255 : 1;
    const stride = attr.isInterleavedBufferAttribute ? attr.data.stride : attr.itemSize;
    const base = attr.isInterleavedBufferAttribute ? attr.offset : 0;
    for (let i = 0; i < attr.count; i++) {
      let hl = null;
      for (const s of sel) if (s.labels.length === attr.count && s.labels[i] === s.id) { hl = s.color; break; }
      const o = (k) => orig[i * stride + base + k] * scale;
      if (hl) attr.setXYZ(i, o(0) * (1 - mix) + hl[0] * mix, o(1) * (1 - mix) + hl[1] * mix, o(2) * (1 - mix) + hl[2] * mix);
      else attr.setXYZ(i, o(0) * (1 - dim) + dim, o(1) * (1 - dim) + dim, o(2) * (1 - dim) + dim);
    }
    attr.needsUpdate = true;
  });
  applyCloudFilter(group, regionsByTrack, view);
  return true;
}

/**
 * Draw a frame's regions into a transparent canvas laid over the frame's image: each region in its
 * colour, or, with `focusIds`, those regions strong and a light veil over the rest.
 *
 * @param {?object} regions  the frame's regions; null clears the canvas.
 * @param {HTMLCanvasElement} canvas
 * @param {?Set<number>} [focusIds]
 */
export function regionsToCanvas(regions, canvas, focusIds = null) {
  if (!regions) {
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  const [w, h] = regions.grid;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(w, h);
  const rgba = new Map(regions.classes.map((c) => [c.id, c.color.map((v) => Math.round(pastel(v) * 255))]));
  const focus = focusIds?.size ? focusIds : null;
  const { labels } = regions;
  for (let i = 0; i < labels.length; i++) {
    const c = rgba.get(labels[i]);
    let a = 0;
    let col = [255, 255, 255];
    if (focus) {
      if (c && focus.has(labels[i])) { col = c; a = 0.75; } else a = 0.12;
    } else if (c) { col = c; a = 0.6; }
    img.data[i * 4] = col[0];
    img.data[i * 4 + 1] = col[1];
    img.data[i * 4 + 2] = col[2];
    img.data[i * 4 + 3] = Math.round(a * 255);
  }
  ctx.putImageData(img, 0, 0);
}
