// The cloud filters, each against a case worked out without them. The flying-pixel fixture's expected
// mask came from an independent Python implementation of the same rule, so a change of threshold or of
// neighbourhood here fails.
import { describe, it, expect } from 'vitest';
import { flyingPixelMask, dilateMask, axialDepth, silhouetteHalo, cameraInCloudFrame, grazingMask,
         MP_REL_THRESH, INSTRUMENT_HALO_PX } from './depth_filter.js';

// 9x12: a smooth surface at ~1.0, a step to ~1.3 from column 7, two pixels without depth.
const W = 12, H = 9;
const DEPTH = [
  1.0001, 1.0027, 1.0024, 0.9990, 0.9994, 0.9989, 1.0011, 1.2999, 1.3015, 1.2963, 1.3031, 1.2998,
  1.0014, 0.9997, 0.9992, 1.0009, 1.0016, 0.9996, 0.9997, 1.3014, 1.2983, 1.2970, 1.3008, 1.2987,
  0.9962, 0.0000, 0.0000, 0.9976, 0.9970, 1.0001, 1.0018, 1.2995, 1.2985, 1.3008, 1.3014, 1.2994,
  1.0011, 0.0000, 0.0000, 0.9984, 1.0007, 1.0005, 1.0022, 1.2974, 1.2987, 1.2983, 1.2965, 1.3003,
  1.0011, 0.9985, 1.0028, 1.0016, 1.0013, 1.0008, 1.0019, 1.2973, 1.3012, 1.3012, 1.2965, 1.3007,
  0.9995, 1.0016, 0.9991, 1.0000, 1.0007, 0.9982, 1.0012, 1.2998, 1.3010, 1.2990, 1.3022, 1.3012,
  0.9996, 1.0013, 1.0025, 1.0036, 0.9969, 1.0018, 1.0009, 1.2998, 1.2980, 1.3025, 1.2975, 1.3011,
  1.0026, 0.9968, 0.9994, 0.9974, 1.0005, 1.0030, 1.0040, 1.2964, 1.2989, 1.3014, 1.3032, 1.3008,
  0.9985, 1.0006, 1.0000, 0.9996, 0.9985, 1.0008, 1.0006, 1.2998, 1.2996, 1.2974, 1.2990, 1.3024,
];
// The points to keep: with depth and not on the step.
const PY_KEEP = [
  1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1,
  1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1,
  1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1,
  1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1,
  1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1,
  1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1,
  1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1,
  1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1,
  1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1,
];

describe('flyingPixelMask', () => {
  it('agrees with the Python rule, pixel for pixel', () => {
    const drop = flyingPixelMask(Float32Array.from(DEPTH), W, H);
    const jsKeep = [...drop].map((d, i) => (DEPTH[i] > 0 && !d ? 1 : 0));
    expect(jsKeep).toEqual(PY_KEEP);
  });

  it('leaves a smooth surface entirely alone', () => {
    // Surface noise here is ~0.2 %, well under the 4 % threshold — a filter
    // that nibbled at it would be eating reconstruction, not artefact.
    const flat = Float32Array.from({ length: W * H }, (_, i) => 1 + 0.002 * Math.sin(i));
    expect([...flyingPixelMask(flat, W, H)].some(Boolean)).toBe(false);
  });

  it('refuses a grid that does not describe the data', () => {
    // A wrong width would flag a diagonal stripe and look entirely plausible,
    // so this has to fail closed rather than mask the wrong pixels.
    expect(flyingPixelMask(Float32Array.from(DEPTH), W + 1, H)).toBeNull();
    expect(flyingPixelMask(Float32Array.from(DEPTH), 0, H)).toBeNull();
  });

  it('keeps everything when the threshold is infinite', () => {
    const drop = flyingPixelMask(Float32Array.from(DEPTH), W, H, { relThresh: Infinity });
    expect([...drop].some(Boolean)).toBe(false);
  });

  it('grows the band by exactly the dilation asked for', () => {
    const none = flyingPixelMask(Float32Array.from(DEPTH), W, H, { dilate: 0 });
    const one = flyingPixelMask(Float32Array.from(DEPTH), W, H, { dilate: 1 });
    const nNone = [...none].filter(Boolean).length;
    const nOne = [...one].filter(Boolean).length;
    expect(nOne).toBeGreaterThan(nNone);
    // And dilating the undilated one by hand gets the same answer.
    expect([...dilateMask(none, W, H, 1)]).toEqual([...one]);
  });

  it('uses the documented threshold', () => {
    expect(MP_REL_THRESH).toBe(0.04);
  });
});

describe('axialDepth', () => {
  it('measures along the optical axis, not distance to the eye', () => {
    // Two points the same distance from the eye but at different angles have
    // DIFFERENT axial depth; using range instead would tilt the threshold
    // across a wide frame.
    const eye = [0, 0, 0], fwd = [0, 0, 1];
    const pos = Float32Array.from([0, 0, 2, 1.2, 0, 1.6]);   // both |p| = 2
    const z = axialDepth(pos, eye, fwd);
    expect(z[0]).toBeCloseTo(2, 5);
    expect(z[1]).toBeCloseTo(1.6, 5);
  });

  it('treats anything behind the camera as invalid', () => {
    const z = axialDepth(Float32Array.from([0, 0, -1, 0, 0, NaN]), [0, 0, 0], [0, 0, 1]);
    expect(z[0]).toBe(0);
    expect(z[1]).toBe(0);
  });
});

describe('cameraInCloudFrame', () => {
  // A frame as the bundle records it: a camera at an absolute glTF position, a surface 0.8-1.2 in front
  // of it, and the cloud's points stored relative to their own centroid.
  const eyeAbs = [0.3, -0.2, 0.5], fwd = [0, 0, -1];
  const trueDepth = [0.8, 1.0, 1.2, 0.9];
  const abs = trueDepth.flatMap((d, i) => [eyeAbs[0] + 0.01 * i, eyeAbs[1], eyeAbs[2] - d]);
  const centroid = [0, 1, 2].map(k => abs.filter((_, j) => j % 3 === k).reduce((a, b) => a + b) / 4);
  const centred = Float32Array.from(abs.map((v, j) => v - centroid[j % 3]));
  const entry = { camera_position: eyeAbs, camera_forward: fwd, centroid };

  it('recovers the true depth of a centred cloud', () => {
    const cam = cameraInCloudFrame(entry);
    const z = axialDepth(centred, cam.pos, cam.fwd);
    trueDepth.forEach((d, i) => expect(z[i]).toBeCloseTo(d, 5));
  });

  it('shows why: the absolute eye gets the depths of a centred cloud wrong', () => {
    const z = axialDepth(centred, eyeAbs, fwd);
    expect(Math.abs(z[1] - trueDepth[1])).toBeGreaterThan(0.1);
  });

  it('refuses a frame without its centroid', () => {
    expect(cameraInCloudFrame({ camera_position: eyeAbs, camera_forward: fwd })).toBeNull();
    expect(cameraInCloudFrame(null)).toBeNull();
  });
});

describe('silhouetteHalo', () => {
  // A 7x7 field of tissue (class 2) with a 3x3 instrument (class 5) at the
  // centre. The halo is what hugs the tool, not the tool.
  const W = 7, H = 7;
  const labels = new Int32Array(W * H).fill(2);
  for (let y = 2; y <= 4; y++) for (let x = 2; x <= 4; x++) labels[y * W + x] = 5;

  it('takes the band around the class, never the class itself', () => {
    const halo = silhouetteHalo(labels, [5], W, H, 1);
    for (let y = 2; y <= 4; y++) {
      for (let x = 2; x <= 4; x++) expect(halo[y * W + x], `${x},${y}`).toBe(0);
    }
    // One ring out: the 5x5 square minus the 3x3 core = 16 pixels.
    expect([...halo].filter(Boolean).length).toBe(16);
  });

  it('widens with the distance asked for', () => {
    // Two rings: 7x7 minus 3x3 = 40.
    expect([...silhouetteHalo(labels, [5], W, H, 2)].filter(Boolean).length).toBe(40);
  });

  it('is nothing when the class is not in the frame', () => {
    expect(silhouetteHalo(labels, [9], W, H, 2)).toBeNull();
  });

  it('refuses a grid that does not fit the labels', () => {
    expect(silhouetteHalo(labels, [5], W + 1, H, 2)).toBeNull();
  });

  it('uses the measured width by default', () => {
    expect(INSTRUMENT_HALO_PX).toBe(2);
    expect([...silhouetteHalo(labels, [5], W, H)].filter(Boolean).length).toBe(40);
  });
});

describe('grazingMask', () => {
  // A 9 x 9 grid of points on a plane, camera at the origin looking down +z.
  // `tilt` rotates the plane about the x axis: 0 faces the camera, near 90 runs
  // along the line of sight — the sheet a smoothed boundary leaves.
  const W = 9, H = 9;
  const plane = (tiltDeg) => {
    const t = (tiltDeg * Math.PI) / 180;
    const pos = new Float32Array(W * H * 3);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x, u = (x - 4) * 0.01, v = (y - 4) * 0.01;
      pos.set([u, v * Math.cos(t), 1 + v * Math.sin(t)], 3 * i);
    }
    return pos;
  };
  const inner = (m) => { let n = 0; for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) n += m[y * W + x]; return n; };
  const valid = new Float32Array(W * H).fill(1);

  it('keeps a surface that faces the camera, and a moderately oblique one', () => {
    expect(inner(grazingMask(plane(0), [0, 0, 0], valid, W, H, { dilate: 0 }))).toBe(0);
    expect(inner(grazingMask(plane(60), [0, 0, 0], valid, W, H, { dilate: 0 }))).toBe(0);
  });

  it('drops a surface seen edge-on', () => {
    expect(inner(grazingMask(plane(88), [0, 0, 0], valid, W, H, { dilate: 0 }))).toBe((W - 2) * (H - 2));
  });

  it('refuses a grid that does not describe the points, and skips invalid pixels', () => {
    expect(grazingMask(plane(88), [0, 0, 0], valid, W + 1, H)).toBeNull();
    expect(inner(grazingMask(plane(88), [0, 0, 0], new Float32Array(W * H), W, H, { dilate: 0 }))).toBe(0);
  });
});
