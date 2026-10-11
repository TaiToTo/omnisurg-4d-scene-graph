import { describe, expect, it } from 'vitest';
import { esc, hexColor, lighten, nodeKey, parseNodeKey, rgbToHex } from './format.js';

describe('esc', () => {
  it('escapes the five characters that end text or an attribute', () => {
    expect(esc(`a&b<c>d"e'f`)).toBe('a&amp;b&lt;c&gt;d&quot;e&#39;f');
    expect(esc(7)).toBe('7');
  });
});

describe('colours', () => {
  it('writes a colour as six hex digits, dropping the bits above them', () => {
    expect(hexColor(0x0000ff)).toBe('#0000ff');
    expect(hexColor(0x1ff8800)).toBe('#ff8800');
    expect(hexColor(-1)).toBe('#ffffff');
  });

  it('turns 0-1 floats into a colour, clamping each channel', () => {
    expect(rgbToHex([1, 0.5, 0])).toBe(0xff8000);
    expect(rgbToHex([2, -1, 0.5])).toBe(0xff0080);
  });

  it('gives the fallback for anything but three finite numbers', () => {
    for (const bad of [null, '#fff', [1, 0], [1, NaN, 0]]) expect(rgbToHex(bad, 7)).toBe(7);
    expect(rgbToHex(undefined)).toBe(0x888888);
  });

  it('moves a colour toward white, and draws a missing one pale grey', () => {
    expect(lighten([1, 0, 0], 0.5)).toBe('rgb(255,128,128)');
    expect(lighten(null, 0.5)).toBe('#dddddd');
  });
});

describe('node keys', () => {
  it('join a track and an id, and split at the last colon, so a track may hold one', () => {
    expect(nodeKey('zs', 7)).toBe('zs:7');
    expect(parseNodeKey('zs:a:7')).toEqual({ track: 'zs:a', id: 7 });
    expect(parseNodeKey(nodeKey('gt:v2', 12))).toEqual({ track: 'gt:v2', id: 12 });
  });

  it('refuses what is not a track and an id', () => {
    for (const bad of ['zs', ':7', 'zs:', 'zs:x', 'zs:-1', null, 7]) expect(() => parseNodeKey(bad)).toThrow(/not a node key/);
  });
});
