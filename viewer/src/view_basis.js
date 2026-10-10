// Turn a recorded camera pose into an orthonormal view basis. The forward and up a depth model predicts
// can be unnormalised, not quite orthogonal, parallel or zero, and any of those reaches the 3D camera as
// NaN, which draws an empty view without an error.

/** Squared length of a 3-vector. */
const len2 = (v) => v[0] * v[0] + v[1] * v[1] + v[2] * v[2];

const DEGENERATE = 1e-12;

/**
 * Orthonormal right-handed basis for looking the way a camera looked.
 *
 * `up` is orthogonalised against `fwd` (Gram-Schmidt) rather than trusted:
 * exported pairs are close to orthogonal but nothing guarantees it, and a
 * basis that is not orthogonal shears everything measured in it.
 *
 * @param {number[]} fwd  where the camera looked.
 * @param {number[]} up   which way was up in the image.
 * @returns {?{fwd: number[], up: number[], right: number[]}} null when the
 *          pair cannot define a basis: either vector missing, wrong length,
 *          non-finite, zero, or the two parallel. Callers keep their own
 *          framing in that case — better a conventional view than no view.
 */
export function viewBasisFrom(fwd, up) {
  if (!Array.isArray(fwd) || !Array.isArray(up)) return null;
  if (fwd.length !== 3 || up.length !== 3) return null;
  if (![...fwd, ...up].every(Number.isFinite)) return null;

  const fl = Math.sqrt(len2(fwd));
  if (fl < DEGENERATE) return null;
  const f = fwd.map(v => v / fl);

  // Remove the component of `up` along `fwd`, then normalise. When the two are
  // parallel nothing is left — there is no "up" to speak of for that view.
  const d = up[0] * f[0] + up[1] * f[1] + up[2] * f[2];
  const u0 = [up[0] - d * f[0], up[1] - d * f[1], up[2] - d * f[2]];
  const ul = Math.sqrt(len2(u0));
  if (ul < DEGENERATE) return null;
  const u = u0.map(v => v / ul);

  // right = fwd x up, already unit since fwd ⟂ up and both are unit.
  const r = [
    f[1] * u[2] - f[2] * u[1],
    f[2] * u[0] - f[0] * u[2],
    f[0] * u[1] - f[1] * u[0],
  ];
  return { fwd: f, up: u, right: r };
}
