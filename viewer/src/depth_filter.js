// Find the points of a cloud that hang between two surfaces: flying pixels at a depth step, surfaces
// seen edge-on, and the band of mixed pixels round an instrument. The cloud is a dense raster, one point
// per pixel in row-major order, so each rule reads a point's neighbours on the pixel grid.

/** The relative depth step to a neighbour that marks a boundary pixel. */
export const MP_REL_THRESH = 0.04;
/** How many pixels the marked band grows by: the smear across a boundary is a ramp, not one pixel. */
export const MP_DILATE = 1;
/** Surfaces seen more obliquely than this, in degrees from the viewing ray, are dropped. */
export const MP_GRAZING_DEG = 78;
/**
 * How far round an instrument its smear reaches, in pixels. The depth step cannot find it: a depth model
 * smooths across a thin instrument, so the step at its edge is below MP_REL_THRESH.
 */
export const INSTRUMENT_HALO_PX = 2;

const NEIGHBOURS = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];

/**
 * Each point's depth along the camera's optical axis, (p - eye) . forward; 0 behind the camera.
 *
 * @param {ArrayLike<number>} pos  xyz per point, length 3n.
 * @param {number[]} eye  the camera, in the cloud's coordinates.
 * @param {number[]} fwd  the camera's unit forward.
 * @returns {Float32Array}
 */
export function axialDepth(pos, eye, fwd) {
  const n = pos.length / 3;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const z = (pos[3 * i] - eye[0]) * fwd[0] + (pos[3 * i + 1] - eye[1]) * fwd[1] + (pos[3 * i + 2] - eye[2]) * fwd[2];
    out[i] = Number.isFinite(z) && z > 0 ? z : 0;
  }
  return out;
}

/**
 * The camera of a frame in its cloud's own coordinates. A cloud is centred on its centroid, and the
 * record gives the camera in the coordinates before the centring.
 *
 * @param {{centroid:number[], camera_position:number[], camera_forward:number[]}} frame  a frame of clip.json.
 * @returns {?{pos:number[], fwd:number[]}} null when the frame lacks one of the three.
 */
export function cameraInCloudFrame(frame) {
  const pos = frame?.camera_position, fwd = frame?.camera_forward, c = frame?.centroid;
  if (!pos || !fwd || !c) return null;
  return { pos: pos.map((v, k) => v - c[k]), fwd };
}

/**
 * The points on a steep depth step: a neighbour's depth differs by more than `relThresh` of the point's.
 *
 * @param {ArrayLike<number>} depth  per point, row-major w x h; 0 is no depth.
 * @returns {?Uint8Array} 1 to drop; null when the grid does not describe this many points, since a wrong
 *          width would mark a diagonal stripe that looks plausible.
 */
export function flyingPixelMask(depth, width, height, { relThresh = MP_REL_THRESH, dilate = MP_DILATE } = {}) {
  const n = depth.length;
  if (!Number.isInteger(width) || !Number.isInteger(height) || width * height !== n || n === 0) return null;
  if (!Number.isFinite(relThresh)) return new Uint8Array(n);
  const disc = new Uint8Array(n);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const z = depth[i];
      if (!(z > 0)) continue;
      for (const [dy, dx] of NEIGHBOURS) {
        const ny = y + dy, nx = x + dx;
        if (ny < 0 || ny >= height || nx < 0 || nx >= width) continue;
        const zn = depth[ny * width + nx];
        if (!(zn > 0)) continue;
        if (Math.abs(z - zn) / z > relThresh) { disc[i] = 1; break; }
      }
    }
  }
  return dilate > 0 ? dilateMask(disc, width, height, dilate) : disc;
}

/**
 * The points whose surface is seen more edge-on than `maxDeg`. A depth model that smooths across a
 * boundary turns the step into a ramp along the line of sight; the step rule passes it, this one does not.
 *
 * @param {ArrayLike<number>} pos  xyz per pixel, row-major w x h.
 * @param {number[]} eye  the camera, in the coordinates of `pos`.
 * @param {ArrayLike<number>} valid  per pixel depth; 0 is no depth.
 * @returns {?Uint8Array} 1 to drop; null when the grid does not describe `pos`.
 */
export function grazingMask(pos, eye, valid, width, height, { maxDeg = MP_GRAZING_DEG, dilate = MP_DILATE } = {}) {
  const n = width * height;
  if (!Number.isInteger(width) || !Number.isInteger(height) || pos.length !== 3 * n || valid.length !== n) return null;
  const out = new Uint8Array(n);
  const cosMax = Math.cos((maxDeg * Math.PI) / 180);
  const P = (i, k) => pos[3 * i + k];
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const l = i - 1, r = i + 1, u = i - width, d = i + width;
      if (!(valid[i] > 0 && valid[l] > 0 && valid[r] > 0 && valid[u] > 0 && valid[d] > 0)) continue;
      const ax = P(r, 0) - P(l, 0), ay = P(r, 1) - P(l, 1), az = P(r, 2) - P(l, 2);
      const bx = P(d, 0) - P(u, 0), by = P(d, 1) - P(u, 1), bz = P(d, 2) - P(u, 2);
      const nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
      const vx = P(i, 0) - eye[0], vy = P(i, 1) - eye[1], vz = P(i, 2) - eye[2];
      const nn = Math.hypot(nx, ny, nz), vv = Math.hypot(vx, vy, vz);
      if (!(nn > 0 && vv > 0)) continue;
      if (Math.abs(nx * vx + ny * vy + nz * vz) / (nn * vv) < cosMax) out[i] = 1;
    }
  }
  return dilate > 0 ? dilateMask(out, width, height, dilate) : out;
}

/** Grow a mask by `iters` pixels, 8-connected. */
export function dilateMask(mask, width, height, iters = 1) {
  let cur = mask;
  for (let it = 0; it < iters; it++) {
    const next = new Uint8Array(cur);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (cur[y * width + x]) continue;
        for (const [dy, dx] of NEIGHBOURS) {
          const ny = y + dy, nx = x + dx;
          if (ny < 0 || ny >= height || nx < 0 || nx >= width) continue;
          if (cur[ny * width + nx]) { next[y * width + x] = 1; break; }
        }
      }
    }
    cur = next;
  }
  return cur;
}

/**
 * The pixels within `px` of a region of the given classes but outside it: the mixed pixels round an
 * instrument's silhouette, found from the labels.
 *
 * @param {ArrayLike<number>} labels  per pixel, row-major w x h.
 * @param {Set<number>|number[]} ids  the classes that cast the silhouette.
 * @returns {?Uint8Array} 1 to drop; null when the grid does not fit the labels or no such class is there.
 */
export function silhouetteHalo(labels, ids, width, height, px = INSTRUMENT_HALO_PX) {
  const n = labels.length;
  if (!Number.isInteger(width) || !Number.isInteger(height) || width * height !== n) return null;
  const want = ids instanceof Set ? ids : new Set(ids);
  const core = new Uint8Array(n);
  let any = 0;
  for (let i = 0; i < n; i++) if (want.has(labels[i])) { core[i] = 1; any++; }
  if (!any || px <= 0) return null;
  const grown = dilateMask(core, width, height, px);
  const halo = new Uint8Array(n);
  let m = 0;
  for (let i = 0; i < n; i++) if (grown[i] && !core[i]) { halo[i] = 1; m++; }
  return m ? halo : null;
}
