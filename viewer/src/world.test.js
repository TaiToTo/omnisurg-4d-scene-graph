import { describe, expect, it } from 'vitest';
import { nearestStacked, worldIndices } from './world.js';

describe('the world stack', () => {
  it('samples a dozen frames, the first and the last always among them', () => {
    expect(worldIndices(14)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]);
    const long = worldIndices(119);
    expect(long[0]).toBe(0);
    expect(long.at(-1)).toBe(119);
    expect(long.length).toBeLessThanOrEqual(13);
  });

  it('puts the nearest stacked frame on stage, the earlier one on a tie', () => {
    expect(nearestStacked(7, [0, 5, 10])).toBe(5);
    expect(nearestStacked(8, [0, 5, 10])).toBe(10);
  });
});
