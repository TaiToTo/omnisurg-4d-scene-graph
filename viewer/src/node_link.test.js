import { describe, expect, it } from 'vitest';
import { fitLinear, projectNodes, projectToCamera } from './node_link.js';

describe('projectToCamera', () => {
  const frame = { camera_position: [0, 0, 5], camera_forward: [0, 0, -1], camera_up: [0, 1, 0] };

  it('gives a point to the camera\'s right and above it as positive rays, divided by depth', () => {
    const [r] = projectToCamera([[1, 2, 3]], frame, [0, 0, 0]);
    expect(r).toEqual({ X: 0.5, Y: 1, behind: false });
  });

  it('moves the camera by the world origin, and marks a point behind it', () => {
    const [r, back] = projectToCamera([[1, 2, 3], [0, 0, 10]], frame, [0, 0, -1]);
    expect(r.X).toBeCloseTo(1 / 3);
    expect(back.behind).toBe(true);
  });
});

describe('fitLinear', () => {
  it('fits a line and refuses x without spread', () => {
    expect(fitLinear([0, 1, 2], [1, 3, 5])).toEqual({ a: 2, b: 1 });
    expect(fitLinear([1, 1], [0, 5])).toBeNull();
    expect(fitLinear([1], [0])).toBeNull();
  });
});

describe('projectNodes', () => {
  const regions = { grid: [4, 2], labels: Int32Array.from([1, 1, 0, 2, 1, 1, 0, 2]), classes: [] };

  it('places each node at its region\'s centroid, keeping the image\'s aspect', () => {
    const p = projectNodes([{ id: 1 }, { id: 2 }, { id: 9 }], regions, 436, 236);
    expect(p.scale).toBe(100);
    expect(p.positions.get(1)).toEqual([18 + 50, 18 + 50]);
    expect(p.positions.has(9)).toBe(false);
  });

  it('is null when no node has a region on the frame', () => {
    expect(projectNodes([{ id: 9 }], regions, 400, 200)).toBeNull();
  });
});
