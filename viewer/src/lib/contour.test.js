import { describe, it, expect } from 'vitest';
import { traceOutline, simplifyRDP, regionOutlines } from './contour.js';
import { classIdRaster } from './regions.js';

const PAL = {
  a: { id: 1, name: 'a', color: [1, 0, 0] },
  b: { id: 2, name: 'b', color: [0, 1, 0] },
};

// A seg_frame from an ASCII mask: '.' is background, a letter is a class of PAL.
function segFromAscii(rows) {
  const seg_colors = [];
  for (const row of rows) for (const ch of row) seg_colors.push(ch === '.' ? [0, 0, 0] : PAL[ch].color);
  return {
    width: rows[0].length, height: rows.length, seg_colors,
    classes: [{ id: 0, name: 'bg', color: [0, 0, 0] }, ...Object.values(PAL)],
  };
}

describe('traceOutline', () => {
  const square = segFromAscii(['........', '.aaaaaa.', '.aaaaaa.', '.aaaaaa.', '.aaaaaa.', '........']);

  it('traces a closed boundary on the region, over its whole extent', () => {
    const { ids, w, h } = classIdRaster(square);
    const poly = traceOutline(ids, w, h, 1, 1);
    expect(poly.length).toBeGreaterThanOrEqual(4);
    for (const [x, y] of poly) expect(ids[y * w + x]).toBe(1);
    const xs = poly.map((p) => p[0]), ys = poly.map((p) => p[1]);
    expect([Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)]).toEqual([1, 6, 1, 4]);
  });

  it('follows a concave notch, unlike a convex hull', () => {
    const notched = segFromAscii(['..........', '.aaaaaaaa.', '.aaa..aaa.', '.aaa..aaa.', '.aaaaaaaa.', '..........']);
    const { ids, w, h } = classIdRaster(notched);
    const rim = traceOutline(ids, w, h, 1, 1).filter(([x, y]) => y >= 2 && x >= 3 && x <= 6);
    expect(rim.length).toBeGreaterThan(0);
  });

  it('returns [] for an absent class', () => {
    const { ids, w, h } = classIdRaster(square);
    expect(traceOutline(ids, w, h, 9, 1)).toEqual([]);
  });
});

describe('simplifyRDP', () => {
  it('drops collinear points', () => {
    expect(simplifyRDP([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [4, 4]], 0.5)).toEqual([[0, 0], [4, 0], [4, 4]]);
  });

  it('keeps corners beyond the tolerance', () => {
    const l = [[0, 0], [5, 4], [10, 0]];
    expect(simplifyRDP(l, 1)).toEqual(l);
  });
});

describe('regionOutlines', () => {
  it('returns one outline per class present', () => {
    const outlines = regionOutlines(segFromAscii(['..........', '.aaaa.bbb.', '.aaaa.bbb.', '.aaaa.bbb.', '..........']), 1, 0.5);
    expect([...outlines.keys()].sort()).toEqual([1, 2]);
    for (const poly of outlines.values()) expect(poly.length).toBeGreaterThanOrEqual(3);
  });

  it('is empty without a raster', () => {
    expect(regionOutlines(null).size).toBe(0);
  });
});
