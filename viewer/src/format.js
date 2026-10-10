// Colours, words and escaping shared by the panels and the 3D view.

/** Edge colour per spatial relation, in the 3D view, the 2D graph and the focus band alike. */
export const RELATION_COLORS = {
  right: 0xff4444,
  left: 0x44aaff,
  above: 0x44ff44,
  below: 0xffaa00,
  front: 0xff44ff,
  behind: 0xffff44,
};

/** The relation read from the other node: `a left of b` is `b right of a`. */
export const MIRROR_REL = {
  left: 'right', right: 'left', above: 'below', below: 'above', front: 'behind', behind: 'front',
};

/** The accent colours: the followed region and the neighbour whose edge is read out. */
export const FOCUS_COLOR = 0x0899b4;
export const PARTNER_COLOR = 0xe08a00;

/** Escape text for HTML or SVG, quotes included, so it is safe in an attribute too. */
export function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

/** 0xRRGGBB as "#rrggbb". */
export function hexColor(c) {
  return `#${((c >>> 0) & 0xffffff).toString(16).padStart(6, '0')}`;
}

/** An [r, g, b] colour of 0-1 floats as 0xRRGGBB; `fallback` when it is not one. */
export function rgbToHex(color, fallback = 0x888888) {
  if (!Array.isArray(color) || color.length < 3 || !color.slice(0, 3).every(Number.isFinite)) return fallback;
  const ch = (v) => Math.max(0, Math.min(255, Math.round(v * 255)));
  return ((ch(color[0]) << 16) | (ch(color[1]) << 8) | ch(color[2])) >>> 0;
}

/** An [r, g, b] colour of 0-1 floats as a CSS colour, moved toward white by `k`. */
export function lighten(color, k = 0) {
  if (!Array.isArray(color)) return '#dddddd';
  return `rgb(${color.slice(0, 3).map((c) => Math.round(c * 255 * (1 - k) + 255 * k)).join(',')})`;
}

/** The key that names one node of one track: "track:id". */
export function nodeKey(track, id) {
  return `${track}:${id}`;
}

/** Split a node key into its track and its numeric id. */
export function parseNodeKey(key) {
  const at = String(key).lastIndexOf(':');
  return { track: key.slice(0, at), id: Number(key.slice(at + 1)) };
}
