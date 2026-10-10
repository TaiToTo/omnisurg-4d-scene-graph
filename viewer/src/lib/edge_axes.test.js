import { describe, it, expect } from 'vitest';
import { cameraAxes, offsetIn, keptWord, edgeAxes } from './edge_axes.js';

// A camera looking down -Z with +Y up, glTF's default: right is +X.
const cam = cameraAxes([0, 0, -1], [0, 1, 0]);

describe('edge axes', () => {
  it('reads an offset as side, height and depth toward the camera', () => {
    expect(offsetIn([1, 2, 3], [0, 0, 0], cam)).toEqual([1, 2, 3]);
    expect(keptWord([0.3, -0.1, 0.2])).toBe('right');
    expect(keptWord([0.1, -0.4, 0.2])).toBe('below');
    expect(keptWord([0.1, 0.2, -0.3])).toBe('behind');
  });

  it('turns with the camera: looking down +X, +Z is on the right', () => {
    const side = cameraAxes([1, 0, 0], [0, 1, 0]);
    side.right.forEach((v, k) => expect(v).toBeCloseTo([0, 0, 1][k]));
    expect(keptWord(offsetIn([0, 0, 1], [0, 0, 0], side))).toBe('right');
    expect(keptWord(offsetIn([-1, 0, 0], [0, 0, 0], side))).toBe('front');
  });

  it('counts a stored word the axes do not give as wrong', () => {
    const graph = {
      nodes: [{ id: 1, pos: [0.3, 0.1, 0] }, { id: 2, pos: [0, 0, 0] }],
      edges: [
        { src: 1, dst: 2, relation: 'right', edge_type: 'spatial' },
        { src: 2, dst: 1, relation: 'above', edge_type: 'spatial' },   // the axes say 'left'
      ],
    };
    const r = edgeAxes([{ f: 0, graph, cam }], 1, 2);
    expect(r.checked).toBe(2);
    expect(r.wrong).toBe(1);
    expect(r.byFrame.get(0).word).toBe('right');
    expect(r.byFrame.get(0).c[0]).toBeCloseTo(0.3);
  });

  it('checks nothing on a frame without a camera', () => {
    expect(cameraAxes(null, [0, 1, 0])).toBeNull();
    expect(edgeAxes([{ f: 0, graph: { nodes: [], edges: [] }, cam: null }], 1, 2).checked).toBe(0);
  });
});
