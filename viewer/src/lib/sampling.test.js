import { describe, it, expect } from 'vitest';
import { worldIndices, nearestStacked } from './sampling.js';

describe('worldIndices', () => {
  it('takes frame 0, every stride-th frame and the last', () => {
    expect(worldIndices(29, 12)).toEqual([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 29]);
  });

  it('takes every frame of a clip shorter than the target', () => {
    expect(worldIndices(4, 12)).toEqual([0, 1, 2, 3, 4]);
  });

  it('takes a one-frame clip whole', () => {
    expect(worldIndices(0, 12)).toEqual([0]);
  });
});

describe('nearestStacked', () => {
  it('snaps to the nearest stacked frame, the earlier on a tie', () => {
    expect(nearestStacked(4, [0, 3, 6])).toBe(3);
    expect(nearestStacked(5, [0, 3, 6])).toBe(6);
    expect(nearestStacked(1.5, [1, 2])).toBe(1);
  });
});
