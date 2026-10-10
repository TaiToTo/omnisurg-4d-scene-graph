import { describe, it, expect } from 'vitest';
import { viewBasisFrom } from './view_basis.js';

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a) => Math.hypot(...a);

describe('viewBasisFrom', () => {
  it('normalises and keeps the looking direction', () => {
    const b = viewBasisFrom([0, 0, -4], [0, 7, 0]);
    expect(b.fwd).toEqual([0, 0, -1]);
    expect(b.up).toEqual([0, 1, 0]);
    expect(len(b.right)).toBeCloseTo(1, 12);
  });

  it('orthogonalises an up that leans into fwd', () => {
    const b = viewBasisFrom([0, 0, -1], [0, 1, -0.4]);
    expect(dot(b.fwd, b.up)).toBeCloseTo(0, 12);
    expect(len(b.up)).toBeCloseTo(1, 12);
    expect(b.up[1]).toBeGreaterThan(0);          // still points up, not flipped
  });

  it('is right-handed: right = fwd x up', () => {
    const b = viewBasisFrom([1, 2, 3], [-3, 1, 0.5]);
    for (const [x, y] of [[b.fwd, b.up], [b.up, b.right], [b.right, b.fwd]]) {
      expect(dot(x, y)).toBeCloseTo(0, 12);
    }
    expect(len(b.right)).toBeCloseTo(1, 12);
    expect(dot(b.right, [
      b.fwd[1] * b.up[2] - b.fwd[2] * b.up[1],
      b.fwd[2] * b.up[0] - b.fwd[0] * b.up[2],
      b.fwd[0] * b.up[1] - b.fwd[1] * b.up[0],
    ])).toBeCloseTo(1, 12);
  });

  it('refuses a degenerate pair instead of returning NaN', () => {
    // A NaN camera draws an empty view and throws nothing.
    expect(viewBasisFrom([0, 0, -1], [0, 0, -1])).toBeNull();   // up ∥ fwd
    expect(viewBasisFrom([0, 0, -1], [0, 0, 5])).toBeNull();    // up ∥ -fwd
    expect(viewBasisFrom([0, 0, 0], [0, 1, 0])).toBeNull();     // fwd zero
    expect(viewBasisFrom([0, 0, -1], [0, 0, 0])).toBeNull();    // up zero
  });

  it('refuses malformed input instead of trusting the manifest', () => {
    expect(viewBasisFrom(null, [0, 1, 0])).toBeNull();
    expect(viewBasisFrom([0, 0, -1], undefined)).toBeNull();
    expect(viewBasisFrom([0, -1], [0, 1, 0])).toBeNull();
    expect(viewBasisFrom([0, 0, NaN], [0, 1, 0])).toBeNull();
    expect(viewBasisFrom([0, 0, -1], [0, Infinity, 0])).toBeNull();
  });

  it('handles a Pi3X pose, whose up is far from world +Y', () => {
    const b = viewBasisFrom([0.06, -0.99, 0.09], [-0.24, 0.07, 0.97]);
    expect(dot(b.fwd, b.up)).toBeCloseTo(0, 12);
    expect(len(b.fwd)).toBeCloseTo(1, 12);
    expect(len(b.up)).toBeCloseTo(1, 12);
  });
});
