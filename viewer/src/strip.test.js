import { describe, expect, it } from 'vitest';
import { slabCaption, slabMap } from './strip.js';

describe('the plates', () => {
  it('map the image\'s corners onto the plate\'s corners', () => {
    expect(slabMap(0, 0)).toEqual([0, 10.2]);
    expect(slabMap(1, 1)).toEqual([229.6, 85.9]);
  });

  it('caption a plate with its frame, its time and whether it is the seed', () => {
    expect(slabCaption(3, 12.34, false)).toBe('f3 · 12.3 s');
    expect(slabCaption(0, null, true)).toBe('f0 · seed');
  });
});
