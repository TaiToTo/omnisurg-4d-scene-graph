// Paint a segmentation frame over its camera image, as RGBA pixels.
//
// Region colours are mixed toward white before they are painted, so masks
// read soft on the light page. Matching a pixel to its class always uses the
// exact palette colour, never the softened one.

/** The share of white mixed into a region colour before it is painted. */
export const PASTEL = 0.35;

/** Mix a 0–1 channel toward white by `PASTEL`. */
export const pastel01 = (v) => v * (1 - PASTEL) + PASTEL;

// How much of the region colour covers the camera image.
const SEG_OVER_RGB = 0.6;

/**
 * Return 255 when `src_colors` holds 0–1 floats and 1 when it holds 0–255 bytes.
 *
 * Decided once for the whole image: deciding per value reads a 0–255 channel
 * that happens to be 0 or 1 as a float, and sprinkles a dark frame with bright dots.
 */
export function srcColorScale(src) {
  if (!src) return 1;
  for (const c of src) if (c[0] > 1 || c[1] > 1 || c[2] > 1) return 1;
  return 255;
}

/**
 * Return the RGBA pixels of a seg frame painted over its image.
 *
 * With `focusIds`, those regions are painted in their full colour and every
 * other pixel shows the plain image, lightened.
 *
 * @param {{width: number, height: number, seg_colors: number[][], src_colors?: number[][], classes?: Array}} seg
 * @param {?(number|Set<number>)} focusIds
 * @returns {?Uint8ClampedArray} null without a raster.
 */
export function segImagePixels(seg, focusIds = null) {
  if (!seg || !Array.isArray(seg.seg_colors) || !seg.width || !seg.height) return null;
  const { width: w, height: h, seg_colors, src_colors } = seg;
  const out = new Uint8ClampedArray(w * h * 4);
  const k = src_colors ? srcColorScale(src_colors) : 255;
  const wanted = focusIds == null ? null : (focusIds instanceof Set ? focusIds : new Set([focusIds]));
  const key = (c) => `${Math.round(c[0] * 255)},${Math.round(c[1] * 255)},${Math.round(c[2] * 255)}`;
  const focusColors = wanted?.size
    ? new Map((seg.classes || []).filter((c) => wanted.has(c.id) && Array.isArray(c.color))
      .map((c) => [key(c.color), c.color.map((v) => Math.round(v * 255))]))
    : null;
  const A = SEG_OVER_RGB;
  const n = Math.min(w * h, seg_colors.length);
  for (let i = 0; i < n; i++) {
    const s = seg_colors[i];
    const r = src_colors ? src_colors[i] : s;
    let R = r[0] * k * (1 - A) + pastel01(s[0]) * 255 * A;
    let G = r[1] * k * (1 - A) + pastel01(s[1]) * 255 * A;
    let B = r[2] * k * (1 - A) + pastel01(s[2]) * 255 * A;
    if (focusColors) {
      const fc = focusColors.get(key(s));
      if (fc) {
        R = R * 0.25 + fc[0] * 0.75; G = G * 0.25 + fc[1] * 0.75; B = B * 0.25 + fc[2] * 0.75;
      } else {
        R = r[0] * k * 0.88 + 31; G = r[1] * k * 0.88 + 31; B = r[2] * k * 0.88 + 31;
      }
    }
    out[i * 4] = R;
    out[i * 4 + 1] = G;
    out[i * 4 + 2] = B;
    out[i * 4 + 3] = 255;
  }
  return out;
}
