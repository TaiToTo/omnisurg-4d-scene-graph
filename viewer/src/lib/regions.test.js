import { describe, it, expect } from 'vitest';
import { classIdRaster, fitEllipse, regionEllipses, convexHull2D, regionHulls } from './regions.js';

const BG = [0.5, 0.5, 0.5];   // no class has this colour
const RED = [1, 0, 0];

// 6×6 frame; class 1 fills rows 2 and 3.
function band() {
  const seg_colors = [];
  for (let y = 0; y < 6; y++) for (let x = 0; x < 6; x++) seg_colors.push(y === 2 || y === 3 ? RED : BG);
  return { width: 6, height: 6, seg_colors, classes: [{ id: 1, color: RED, name: 'R' }] };
}

describe('classIdRaster', () => {
  it('maps palette colours to ids and anything else to background', () => {
    const r = classIdRaster({ width: 2, height: 1, seg_colors: [RED, BG], classes: [{ id: 1, color: RED }] });
    expect([...r.ids]).toEqual([1, 0]);
  });

  it('builds the raster once per frame object', () => {
    const seg = band();
    expect(classIdRaster(seg)).toBe(classIdRaster(seg));
  });

  it('returns null without a raster', () => {
    expect(classIdRaster(null)).toBeNull();
    expect(classIdRaster({ width: 2 })).toBeNull();
  });
});

describe('fitEllipse', () => {
  it('returns null for empty moments and a small dot below 12 pixels', () => {
    expect(fitEllipse(null)).toBe(null);
    expect(fitEllipse({ n: 0, sx: 0, sy: 0, sxx: 0, syy: 0, sxy: 0 })).toBe(null);
    const dot = fitEllipse({ n: 4, sx: 6, sy: 6, sxx: 10, syy: 10, sxy: 9 });
    expect(dot).toMatchObject({ a: 4, b: 4, theta: 0 });
    expect(dot.cx).toBeCloseTo(1.5);
    expect(dot.cy).toBeCloseTo(1.5);
  });
});

describe('regionEllipses', () => {
  it('fits one ellipse per class at its centroid, wide for a wide band', () => {
    const e = regionEllipses(band()).get(1);
    expect(e.n).toBe(12);
    expect(e.cx).toBeCloseTo(2.5);
    expect(e.cy).toBeCloseTo(2.5);
    expect(e.a).toBeGreaterThan(e.b);
    expect(Math.abs(e.theta)).toBeLessThan(0.2);
  });

  it('ignores background and unknown colours', () => {
    const seg = { width: 2, height: 2, seg_colors: [BG, BG, BG, BG], classes: [{ id: 1, color: RED }] };
    expect(regionEllipses(seg).size).toBe(0);
  });

  it('is empty for malformed input', () => {
    expect(regionEllipses(null).size).toBe(0);
    expect(regionEllipses({}).size).toBe(0);
  });
});

describe('convexHull2D', () => {
  it('returns fewer than 3 points as they are', () => {
    expect(convexHull2D([])).toEqual([]);
    expect(convexHull2D([[1, 1], [2, 2]])).toEqual([[1, 1], [2, 2]]);
  });

  it('keeps only the corners of a filled square', () => {
    const pts = [];
    for (let x = 0; x <= 3; x++) for (let y = 0; y <= 3; y++) pts.push([x, y]);
    const hull = convexHull2D(pts);
    expect(hull).toHaveLength(4);
    expect(new Set(hull.map((p) => p.join(',')))).toEqual(new Set(['0,0', '3,0', '3,3', '0,3']));
  });
});

describe('regionHulls', () => {
  it('spans the region and nothing else', () => {
    const hull = regionHulls(band()).get(1);
    const xs = hull.map((p) => p[0]), ys = hull.map((p) => p[1]);
    expect([Math.min(...xs), Math.max(...xs)]).toEqual([0, 5]);
    expect([Math.min(...ys), Math.max(...ys)]).toEqual([2, 3]);
  });

  it('is empty for malformed input or a frame of background', () => {
    expect(regionHulls(null).size).toBe(0);
    expect(regionHulls({}).size).toBe(0);
    expect(regionHulls({ width: 2, height: 2, seg_colors: [BG, BG, BG, BG], classes: [{ id: 1, color: RED }] }).size).toBe(0);
  });
});
