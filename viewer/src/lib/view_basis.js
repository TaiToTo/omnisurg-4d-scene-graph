// Turn an exported camera pose into an orthonormal view basis.
//
// The manifest's forward and up come from a model's prediction: they can be
// unnormalised, not quite orthogonal, or for a frame the model gave up on,
// zero or parallel. Any of these reaches the camera as NaN, and a NaN camera
// draws an empty view without an error.

const len2 = (v) => v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
const DEGENERATE = 1e-12;

/**
 * Return a right-handed orthonormal basis for looking the way a camera looked.
 *
 * `up` is made orthogonal to `fwd` (Gram–Schmidt): a sheared basis would
 * mismeasure every extent taken in it.
 *
 * @param {number[]} fwd  `camera_forward_glb`.
 * @param {number[]} up   `camera_up_glb`.
 * @returns {?{fwd: number[], up: number[], right: number[]}} null when the pair
 *   cannot define a basis; the caller then keeps a conventional view.
 */
export function viewBasisFrom(fwd, up) {
  if (!Array.isArray(fwd) || !Array.isArray(up) || fwd.length !== 3 || up.length !== 3) return null;
  if (![...fwd, ...up].every(Number.isFinite)) return null;
  const fl = Math.sqrt(len2(fwd));
  if (fl < DEGENERATE) return null;
  const f = fwd.map((v) => v / fl);
  const d = up[0] * f[0] + up[1] * f[1] + up[2] * f[2];
  const u0 = [up[0] - d * f[0], up[1] - d * f[1], up[2] - d * f[2]];
  const ul = Math.sqrt(len2(u0));
  if (ul < DEGENERATE) return null;
  const u = u0.map((v) => v / ul);
  const r = [f[1] * u[2] - f[2] * u[1], f[2] * u[0] - f[0] * u[2], f[0] * u[1] - f[1] * u[0]];
  return { fwd: f, up: u, right: r };
}
