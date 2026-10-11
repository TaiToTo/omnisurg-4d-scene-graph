import { describe, expect, it, vi } from 'vitest';
import { makeLabelSprite } from './stage.js';

describe('a label in the 3D view', () => {
  it('is as wide as its text, so a long name is drawn whole', () => {
    // A canvas whose text is 20 px a character.
    const drawn = [];
    const ctx = {
      measureText: (t) => ({ width: 20 * t.length }),
      strokeText() {},
      fillText: (t, x) => drawn.push(x),
    };
    vi.stubGlobal('document', { createElement: () => ({ width: 300, height: 150, getContext: () => ctx }) });
    const name = 'Gastrointestinal Tract (in Fat)';
    const spr = makeLabelSprite(name, '#000', 0.05);
    const cv = spr.material.map.image;
    expect(cv.width).toBeGreaterThanOrEqual(20 * name.length);
    expect(drawn).toEqual([cv.width / 2]);
    expect(spr.scale.x / spr.scale.y).toBeCloseTo(cv.width / cv.height);
    vi.unstubAllGlobals();
  });
});
