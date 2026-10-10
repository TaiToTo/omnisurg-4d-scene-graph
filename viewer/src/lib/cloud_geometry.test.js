import { describe, it, expect } from 'vitest';
import {
  gridRemapIndex, remapLabels, eigenSym3, regionNodeGeometry, regraphFromCloud,
  isLabelArray,
} from './cloud_geometry.js';

/** Duck-typed Three.js position attribute over a flat [x,y,z,...] array. */
function attr(xyz) {
  return {
    count: xyz.length / 3,
    getX: (i) => xyz[i * 3],
    getY: (i) => xyz[i * 3 + 1],
    getZ: (i) => xyz[i * 3 + 2],
  };
}

describe('isLabelArray', () => {
  it('accepts both a JSON raster and a resampled typed one', () => {
    // `Array.isArray` is false for the Int32Array that remapLabels returns.
    expect(isLabelArray([1, 2, 3])).toBe(true);
    expect(isLabelArray(new Int32Array([1, 2, 3]))).toBe(true);
    expect(isLabelArray(null)).toBe(false);
    expect(isLabelArray('123')).toBe(false);
    expect(isLabelArray({ length: 3 })).toBe(false);
  });
});

describe('gridRemapIndex', () => {
  it('is the identity when both grids match', () => {
    const map = gridRemapIndex(3, 2, 3, 2);
    expect([...map]).toEqual([0, 1, 2, 3, 4, 5]);
  });

  it('upsamples by repeating source pixels, row-major', () => {
    // 2x2 -> 4x4: each source pixel covers a 2x2 block.
    const map = gridRemapIndex(2, 2, 4, 4);
    expect([...map]).toEqual([
      0, 0, 1, 1,
      0, 0, 1, 1,
      2, 2, 3, 3,
      2, 2, 3, 3,
    ]);
  });

  it('downsamples by picking the pixel under each destination centre', () => {
    // Destination pixel 0 spans source columns 0-1; its centre falls on
    // source column 1, so 1 is what it samples (not the block's corner).
    const map = gridRemapIndex(4, 4, 2, 2);
    expect([...map]).toEqual([5, 7, 13, 15]);
  });

  it('stays inside the source on a non-integer ratio', () => {
    const map = gridRemapIndex(3, 3, 7, 7);
    expect(Math.min(...map)).toBe(0);
    expect(Math.max(...map)).toBe(8);
  });

  it('rejects a degenerate grid', () => {
    expect(gridRemapIndex(0, 2, 2, 2)).toBeNull();
    expect(gridRemapIndex(2, 2, 2, 0)).toBeNull();
  });
});

describe('remapLabels', () => {
  it('returns the input untouched when the grids coincide', () => {
    const labels = [1, 2, 3, 4];
    expect(remapLabels(labels, 2, 2, 2, 2)).toBe(labels);
  });

  it('carries labels onto a denser grid', () => {
    expect([...remapLabels([1, 2, 3, 4], 2, 2, 4, 4)]).toEqual([
      1, 1, 2, 2,
      1, 1, 2, 2,
      3, 3, 4, 4,
      3, 3, 4, 4,
    ]);
  });

  it('accepts an already-resampled (typed) array as input', () => {
    // Chained remaps must not fail on their own output.
    const once = remapLabels([1, 2, 3, 4], 2, 2, 4, 4);
    expect([...remapLabels(once, 4, 4, 2, 2)]).toEqual([1, 2, 3, 4]);
  });

  it('fails closed when the label array is not the full source grid', () => {
    // A cloud that dropped vertices: which ones cannot be recovered.
    expect(remapLabels([1, 2, 3], 2, 2, 4, 4)).toBeNull();
  });
});

describe('eigenSym3', () => {
  it('diagonalizes a diagonal matrix, largest eigenvalue first', () => {
    const { values, vectors } = eigenSym3([[1, 0, 0], [0, 9, 0], [0, 0, 4]]);
    expect(values[0]).toBeCloseTo(9, 9);
    expect(values[1]).toBeCloseTo(4, 9);
    expect(values[2]).toBeCloseTo(1, 9);
    expect(vectors[0].map(Math.abs)).toEqual([0, 1, 0]);
  });

  it('recovers A = V^T diag(l) V for a general symmetric matrix', () => {
    const A = [[4, 1, -2], [1, 3, 0.5], [-2, 0.5, 6]];
    const { values, vectors } = eigenSym3(A);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        let s = 0;
        for (let k = 0; k < 3; k++) s += values[k] * vectors[k][i] * vectors[k][j];
        expect(s).toBeCloseTo(A[i][j], 8);
      }
    }
  });

  it('returns unit, mutually orthogonal eigenvectors', () => {
    const { vectors } = eigenSym3([[2, 0.7, 0.1], [0.7, 5, -0.3], [0.1, -0.3, 1]]);
    for (const v of vectors) expect(Math.hypot(...v)).toBeCloseTo(1, 9);
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    expect(dot(vectors[0], vectors[1])).toBeCloseTo(0, 8);
    expect(dot(vectors[0], vectors[2])).toBeCloseTo(0, 8);
    expect(dot(vectors[1], vectors[2])).toBeCloseTo(0, 8);
  });

  it('keeps a usable basis when the smallest eigenvalue collapses (planar region)', () => {
    const { values, vectors } = eigenSym3([[5, 0, 0], [0, 3, 0], [0, 0, 0]]);
    expect(values[2]).toBeCloseTo(0, 12);
    for (const v of vectors) expect(Math.hypot(...v)).toBeCloseTo(1, 9);
  });
});

describe('regionNodeGeometry', () => {
  it('returns the centroid of each region, background excluded', () => {
    const pos = attr([
      0, 0, 0, 2, 0, 0, 0, 2, 0,      // region 1
      10, 10, 10, 12, 10, 10,         // region 2
      99, 99, 99,                     // background
    ]);
    const geom = regionNodeGeometry(pos, [1, 1, 1, 2, 2, 0]);
    expect([...geom.keys()].sort()).toEqual([1, 2]);
    expect(geom.get(1).pos[0]).toBeCloseTo(2 / 3, 12);
    expect(geom.get(1).pos[1]).toBeCloseTo(2 / 3, 12);
    expect(geom.get(1).pos[2]).toBeCloseTo(0, 12);
    expect(geom.get(1).vertex_count).toBe(3);
    expect(geom.get(2).pos).toEqual([11, 10, 10]);
  });

  it('orients the first axis along the region’s longest spread', () => {
    // A rod along +x, thin in y, thinner in z.
    const xyz = [];
    for (let i = 0; i < 40; i++) xyz.push(i, (i % 2) * 0.2, 0);
    const geom = regionNodeGeometry(attr(xyz), new Array(40).fill(1));
    const g = geom.get(1);
    expect(Math.abs(g.axes[0][0])).toBeCloseTo(1, 3);
    expect(g.axes_radius[0]).toBeGreaterThan(g.axes_radius[1]);
    expect(g.axes_radius[1]).toBeGreaterThanOrEqual(g.axes_radius[2]);
  });

  it('reads a typed label array, not just a plain one', () => {
    const geom = regionNodeGeometry(attr([0, 0, 0, 2, 0, 0]), new Int32Array([1, 1]));
    expect(geom.get(1).pos).toEqual([1, 0, 0]);
  });

  it('is empty when the label array does not match the cloud', () => {
    expect(regionNodeGeometry(attr([0, 0, 0, 1, 1, 1]), [1]).size).toBe(0);
  });
});

describe('regraphFromCloud', () => {
  const geom = new Map([
    [1, { pos: [1, 2, 3], axes: [[1, 0, 0], [0, 1, 0], [0, 0, 1]], axes_radius: [3, 2, 1], vertex_count: 9 }],
  ]);

  it('replaces node geometry and keeps everything the cloud cannot speak to', () => {
    const g = regraphFromCloud({
      nodes: [{ id: 1, label: 'Liver', color: [1, 0, 0], area_frac: 0.25, pos: [9, 9, 9] }],
      edges: [],
      track: 'sam3d',
    }, geom);
    expect(g.nodes[0].pos).toEqual([1, 2, 3]);
    expect(g.nodes[0].axes_radius).toEqual([3, 2, 1]);
    expect(g.nodes[0].label).toBe('Liver');
    expect(g.nodes[0].area_frac).toBe(0.25);
    expect(g.track).toBe('sam3d');
  });

  it('drops nodes absent from this cloud, and the edges that touch them', () => {
    const g = regraphFromCloud({
      nodes: [{ id: 1 }, { id: 2 }],
      edges: [{ src: 1, dst: 2, relation: 'right' }],
    }, geom);
    expect(g.nodes.map(n => n.id)).toEqual([1]);
    expect(g.edges).toEqual([]);
  });

  // An empty map means nothing could be read off this cloud; passing the graph
  // through would draw every node at the other reconstruction's coordinates.
  it('empties the graph when nothing could be read off this cloud', () => {
    const graph = { nodes: [{ id: 1 }], edges: [{ src: 1, dst: 1 }] };
    const out = regraphFromCloud(graph, new Map());
    expect(out.nodes).toEqual([]);
    expect(out.edges).toEqual([]);
    expect(graph.nodes).toHaveLength(1);     // the input is not mutated
  });

  it('has nothing to say about a graph that is not there', () => {
    expect(regraphFromCloud(null, geom)).toBeNull();
    expect(regraphFromCloud(undefined, new Map())).toBeUndefined();
  });
});
