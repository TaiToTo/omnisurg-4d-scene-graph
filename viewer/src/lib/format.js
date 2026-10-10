// Escape text for HTML and SVG, and turn colours into the strings they take.
//
// Every string that reaches `innerHTML` from the data (labels, class names,
// track names) goes through `esc`, so data cannot inject markup into the page.

const ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

/** Escape `s` for HTML or SVG text and for a quoted attribute value. */
export function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ESCAPES[c]);
}

/** Return `0xRRGGBB` as `#rrggbb`. */
export function hexColor(c) {
  return `#${((c >>> 0) & 0xffffff).toString(16).padStart(6, '0')}`;
}

/**
 * Return `[r, g, b]` as `0xRRGGBB`, or `fallback` when it is not a colour.
 *
 * A channel at most 1 is read as a float in 0–1, a larger one as 0–255: the
 * overlays store floats, and a few older ones store bytes.
 */
export function rgbToHex(color, fallback = 0xffffff) {
  if (!Array.isArray(color) || color.length < 3) return fallback;
  const ch = (v) => (typeof v === 'number' && v <= 1 ? Math.round(v * 255) : Math.round(v));
  return ((ch(color[0]) << 16) | (ch(color[1]) << 8) | ch(color[2])) >>> 0;
}

/** Return `[r, g, b]` as a CSS colour, or `fallback` when it is not a colour. */
export function cssColor(color, fallback = '#888888') {
  return Array.isArray(color) && color.length >= 3 ? hexColor(rgbToHex(color)) : fallback;
}
