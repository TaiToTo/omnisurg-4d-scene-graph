// Paint region colours and highlights on a point cloud, keep only the
// followed region's points, and draw a segmentation frame into a canvas.
//
// A cloud on screen shares its geometry with the GLB cache, so painting writes
// the shared colour attribute: the original colours are kept once on the
// geometry and restored before every repaint. Dropping vertices renders a
// compacted copy; the full geometry stays on `userData.fullGeometry`.
import { state } from './state.js';
import { invalidate } from './scene.js';
import { isLabelArray } from './lib/cloud_geometry.js';
import { pastel01, segImagePixels } from './lib/seg_image.js';
import { filteredGeometry, isolationMask } from './lib/vertex_filter.js';

// How much of a region's colour covers the cloud's own colour.
const SEG_ALPHA = 0.45;

function eachPoints(group, fn) {
  group.traverse((obj) => { if (obj.isPoints) fn(obj); });
}

// Put a Points object back on its full geometry, freeing a compacted copy.
function useFullGeometry(obj) {
  const full = obj.userData.fullGeometry;
  if (!full) { obj.userData.fullGeometry = obj.geometry; return; }
  if (obj.geometry !== full) {
    obj.geometry.dispose();
    obj.geometry = full;
    obj.userData.sharedGeometry = true;
  }
}

/**
 * Keep only the followed region's points in a stacked frame, or put the frame back whole.
 *
 * The frame on stage while a region is followed keeps every point
 * (`userData.isoContext`): there the rest is faded by colour instead, so the
 * region is seen in its scene.
 */
function applyIsolation(group, segByTrack) {
  const on = state.isolateFocus && state.focusNode && !group.userData.isoContext;
  const [track, idStr] = on ? state.focusNode.split(':') : [];
  eachPoints(group, (obj) => {
    useFullGeometry(obj);
    if (!on) return;
    const full = obj.userData.fullGeometry;
    const mask = isolationMask(segByTrack?.[track]?.vertex_seg, full.attributes.position.count, Number(idStr));
    if (!mask) return;
    obj.geometry = filteredGeometry(full, mask);
    obj.userData.sharedGeometry = false;
  });
}

function originalColors(obj) {
  const colorAttr = obj.geometry.attributes.color;
  if (!obj.geometry.userData.origColors) obj.geometry.userData.origColors = colorAttr.array.slice();
  return obj.geometry.userData.origColors;
}

/**
 * Blend the given tracks' region colours into a cloud's own colours, one after another.
 *
 * A track whose `vertex_seg` does not fit the cloud is skipped.
 */
export function applySegBlend(group, segByTrack, tracks) {
  eachPoints(group, useFullGeometry);
  eachPoints(group, (obj) => {
    const colorAttr = obj.geometry.attributes.color;
    if (!colorAttr) return;
    colorAttr.array.set(originalColors(obj));
    for (const track of tracks) {
      const seg = segByTrack?.[track];
      if (!seg || !isLabelArray(seg.vertex_seg) || seg.vertex_seg.length !== colorAttr.count) continue;
      const palette = {};
      for (const cls of seg.classes || []) if (Array.isArray(cls.color)) palette[cls.id] = cls.color.map(pastel01);
      for (let i = 0; i < colorAttr.count; i++) {
        const c = palette[seg.vertex_seg[i]];
        if (!c) continue;
        colorAttr.setXYZ(i,
          colorAttr.getX(i) * (1 - SEG_ALPHA) + c[0] * SEG_ALPHA,
          colorAttr.getY(i) * (1 - SEG_ALPHA) + c[1] * SEG_ALPHA,
          colorAttr.getZ(i) * (1 - SEG_ALPHA) + c[2] * SEG_ALPHA);
      }
    }
    colorAttr.needsUpdate = true;
  });
  applyIsolation(group, segByTrack);
  invalidate();
}

/**
 * Paint the given regions in their highlight colours over the cloud's own colours, and whiten the rest by `dim`.
 *
 * @param {Map<string, number[]>} keyColors  'track:id' → 0–1 RGB.
 * @returns {boolean} whether anything was painted; false when this frame has
 *   no labels for any of the regions, which a region absent from a frame is.
 */
export function applyRegionHighlight(group, segByTrack, keyColors, { mix = 0.65, dim = 0.12 } = {}) {
  const sel = [];
  for (const [key, color] of keyColors) {
    const [track, idStr] = key.split(':');
    const vs = segByTrack?.[track]?.vertex_seg;
    if (isLabelArray(vs)) sel.push({ vs, id: Number(idStr), color });
  }
  if (!sel.length) return false;
  eachPoints(group, useFullGeometry);
  eachPoints(group, (obj) => {
    const colorAttr = obj.geometry.attributes.color;
    if (!colorAttr) return;
    // The backup is the attribute's raw array: possibly normalised bytes,
    // possibly interleaved, and RGBA rather than RGB.
    const orig = originalColors(obj);
    const scale = orig.BYTES_PER_ELEMENT === 1 ? 1 / 255 : 1;
    const stride = colorAttr.isInterleavedBufferAttribute ? colorAttr.data.stride : colorAttr.itemSize;
    const base = colorAttr.isInterleavedBufferAttribute ? colorAttr.offset : 0;
    const o = (i, k) => orig[i * stride + base + k] * scale;
    for (let i = 0; i < colorAttr.count; i++) {
      let hl = null;
      for (const s of sel) if (s.vs.length === colorAttr.count && s.vs[i] === s.id) { hl = s.color; break; }
      if (hl) {
        colorAttr.setXYZ(i, o(i, 0) * (1 - mix) + hl[0] * mix, o(i, 1) * (1 - mix) + hl[1] * mix, o(i, 2) * (1 - mix) + hl[2] * mix);
      } else {
        colorAttr.setXYZ(i, o(i, 0) * (1 - dim) + dim, o(i, 1) * (1 - dim) + dim, o(i, 2) * (1 - dim) + dim);
      }
    }
    colorAttr.needsUpdate = true;
  });
  applyIsolation(group, segByTrack);
  invalidate();
  return true;
}

/** Draw a seg frame over its image into `canvasEl`, emphasising `focusIds`; clear it without one. */
export function segToCanvas(seg, canvasEl, focusIds = null) {
  const px = segImagePixels(seg, focusIds);
  const ctx = canvasEl.getContext('2d');
  if (!px) {
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    return;
  }
  canvasEl.width = seg.width;
  canvasEl.height = seg.height;
  ctx.putImageData(new ImageData(px, seg.width, seg.height), 0, 0);
}
