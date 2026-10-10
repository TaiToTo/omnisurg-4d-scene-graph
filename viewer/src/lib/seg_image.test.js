import { describe, it, expect } from 'vitest';
import { srcColorScale, segImagePixels } from './seg_image.js';

describe('srcColorScale', () => {
  it('reads a dark 0–255 frame whose channels sit at 0 and 1 as bytes, not as floats', () => {
    expect(srcColorScale([[23, 0, 1], [64, 1, 0], [62, 1, 1]])).toBe(1);
  });

  it('scales a 0–1 image, and leaves a missing one alone', () => {
    expect(srcColorScale([[0.2, 0.5, 1], [0, 0, 0.9]])).toBe(255);
    expect(srcColorScale(null)).toBe(1);
  });
});

describe('segImagePixels', () => {
  const RED = [1, 0, 0], BG = [0, 0, 0];
  const seg = { width: 2, height: 1, seg_colors: [RED, BG], src_colors: [[100, 100, 100], [100, 100, 100]], classes: [{ id: 1, color: RED }] };

  it('paints the region colour over the image, opaque', () => {
    const px = segImagePixels(seg);
    expect(px).toHaveLength(8);
    expect(px[0]).toBeGreaterThan(px[1]);   // the red region is redder than it is green
    expect(px[3]).toBe(255);
  });

  it('paints a focused region in its colour and lightens the plain image around it', () => {
    const px = segImagePixels(seg, 1);
    expect(px[0]).toBeGreaterThan(200);
    expect([px[4], px[5], px[6]]).toEqual([119, 119, 119]);   // 100 * 0.88 + 31
  });

  it('returns null without a raster', () => {
    expect(segImagePixels(null)).toBeNull();
    expect(segImagePixels({ width: 2, height: 1 })).toBeNull();
  });
});
