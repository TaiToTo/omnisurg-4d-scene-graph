import { describe, expect, it } from 'vitest';
import { convexHull2D, fitEllipse, regionEllipses, regionHulls, regionOutlines, simplifyRDP, traceOutline } from './regions.js';

// A frame's regions from rows of digits, '0' being background.
function regions(rows) {
  const labels = Int32Array.from(rows.join('').split('').map(Number));
  const ids = [...new Set(labels)].filter(Boolean);
  return { grid: [rows[0].length, rows.length], labels, classes: ids.map((id) => ({ id, name: `r${id}`, color: [1, 0, 0] })) };
}

const square = regions(['00000000', '01111110', '01111110', '01111110', '01111110', '00000000']);

describe('regionEllipses', () => {
  it('centres each region on its centroid, background left out', () => {
    const e = regionEllipses(regions(['1122', '1122', '0000']));
    expect([...e.keys()].sort()).toEqual([1, 2]);
    expect([e.get(1).cx, e.get(1).cy]).toEqual([0.5, 0.5]);
    expect([e.get(2).cx, e.get(2).cy]).toEqual([2.5, 0.5]);
  });

  it('gives a region of equal area, and a dot under twelve pixels', () => {
    const big = regionEllipses(square).get(1);
    expect(Math.PI * big.a * big.b).toBeCloseTo(24, 6);
    expect(fitEllipse({ n: 3, sx: 3, sy: 3, sxx: 3, syy: 3, sxy: 3 })).toMatchObject({ a: 4, b: 4 });
    expect(fitEllipse({ n: 0 })).toBeNull();
  });
});

describe('hulls', () => {
  it('keeps only the corners of a filled square', () => {
    expect(convexHull2D([[0, 0], [1, 1], [2, 0], [2, 2], [0, 2], [1, 0]]).length).toBe(4);
    const hull = regionHulls(square).get(1);
    expect(hull.map(([x]) => x).sort((a, b) => a - b)).toEqual([1, 1, 6, 6]);
  });

  it('returns a point set under three points as it is', () => {
    expect(convexHull2D([[0, 0], [1, 1]])).toEqual([[0, 0], [1, 1]]);
  });
});

describe('outlines', () => {
  it('traces a closed boundary on the region, spanning its extent', () => {
    const { grid: [w, h], labels } = square;
    const poly = traceOutline(labels, w, h, 1, 1);
    for (const [x, y] of poly) expect(labels[y * w + x]).toBe(1);
    expect(Math.min(...poly.map((p) => p[0]))).toBe(1);
    expect(Math.max(...poly.map((p) => p[0]))).toBe(6);
    expect(Math.max(...poly.map((p) => p[1]))).toBe(4);
  });

  it('follows a notch that a hull would close', () => {
    const notched = regions(['0000000', '0111110', '0110110', '0110110', '0111110', '0000000']);
    const poly = traceOutline(notched.labels, 7, 6, 1, 1);
    expect(poly.some(([x, y]) => x === 3 && y === 1)).toBe(true);
  });

  it('simplifies a straight run of points to its ends', () => {
    const line = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [4, 4]];
    expect(simplifyRDP(line, 0.5)).toEqual([[0, 0], [4, 0], [4, 4]]);
  });

  it('outlines every listed region and nothing for background', () => {
    expect([...regionOutlines(regions(['1100', '1100', '0022', '0022']), 1, 0.5).keys()].sort()).toEqual([1, 2]);
  });
});
