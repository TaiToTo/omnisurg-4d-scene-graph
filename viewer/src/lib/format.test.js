import { describe, it, expect } from 'vitest';
import { esc, hexColor, rgbToHex, cssColor } from './format.js';

describe('esc', () => {
  it('escapes markup and both quotes, so text cannot leave a quoted attribute', () => {
    expect(esc(`<img src=x onerror="a('b')">&`)).toBe('&lt;img src=x onerror=&quot;a(&#39;b&#39;)&quot;&gt;&amp;');
  });

  it('turns a number into its text', () => {
    expect(esc(3)).toBe('3');
  });
});

describe('colours', () => {
  it('reads 0–1 floats and 0–255 bytes', () => {
    expect(rgbToHex([1, 0, 0.5])).toBe(0xff0080);
    expect(rgbToHex([12, 200, 30])).toBe(0x0cc81e);
    expect(hexColor(0x0cc81e)).toBe('#0cc81e');
  });

  it('falls back for something that is not a colour', () => {
    expect(rgbToHex(null, 7)).toBe(7);
    expect(cssColor([0, 1], '#123456')).toBe('#123456');
  });
});
