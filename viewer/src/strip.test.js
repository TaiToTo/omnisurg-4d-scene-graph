import { afterEach, describe, expect, it, vi } from 'vitest';
import { renderTrackStrip, renderWorldStrip, slabCaption, slabMap } from './strip.js';

describe('the plates', () => {
  it('map the image\'s corners onto the plate\'s corners', () => {
    expect(slabMap(0, 0)).toEqual([0, 10.2]);
    expect(slabMap(1, 1)).toEqual([229.6, 85.9]);
  });

  it('caption a plate with its frame, its time and how the regions were made on it', () => {
    expect(slabCaption(3, 12.34)).toBe('f3 · 12.3 s');
    expect(slabCaption(0, null, ['seed'])).toBe('f0 · seed');
    expect(slabCaption(0, 0, ['annotated', 'seed'])).toBe('f0 · 0.0 s · annotated · seed');
  });
});

// Text that opens a tag, or ends an attribute, wherever it is not escaped.
const HOSTILE = '<img src=x onerror=alert(1)>" onmouseover="x';
const INJECTED = /<img|\sonmouseover="/;

// A frame's regions from rows of digits, '0' being background, as data.js hands them over.
function regions(rows, names, stage = 'tracked') {
  const labels = Int32Array.from(rows.join('').split('').map(Number));
  const classes = names.map((name, k) => ({ id: k + 1, name, color: [1, 0, 0] }));
  return { grid: [rows[0].length, rows.length], labels, classes, classById: new Map(classes.map((c) => [c.id, c])), stage };
}

describe('renderWorldStrip', () => {
  it('escapes the class names, the track in every node key and the image URL', () => {
    // Two regions wide enough to keep an outline on the grid the tracing samples.
    const r = regions(['0000000000', '0111112220', '0111112220', '0111112220', '0111112220', '0111112220', '0000000000'], [HOSTILE, HOSTILE]);
    const world = { frames: [{ i: 0, regionsByTrack: { [HOSTILE]: r } }, { i: 1, regionsByTrack: { [HOSTILE]: r } }] };
    const el = { innerHTML: '', classList: { remove() {} }, firstElementChild: { addEventListener() {} } };
    renderWorldStrip(el, world, [{ id: HOSTILE, semantic: false }], 0, [{}, {}], () => {}, { imageURL: () => HOSTILE });
    for (const part of ['class="slab-photo"', 'class="slab-chip"', 'stroke-dasharray', '<title>']) expect(el.innerHTML).toContain(part);
    expect(el.innerHTML).toContain('&lt;img');
    expect(el.innerHTML).not.toMatch(INJECTED);
  });
});

/** An element that keeps what renderTrackStrip sets on it: its class, its markup, its text and its children. */
function fakeElement() {
  const el = {
    className: '', innerHTML: '', textContent: '', style: {}, dataset: {}, children: [],
    classList: { add() {} },
    addEventListener() {},
    append: (...c) => { el.children.push(...c); },
    appendChild: (c) => { el.children.push(c); return c; },
    getContext: () => ({ createImageData: (w, h) => ({ data: new Uint8ClampedArray(4 * w * h) }), putImageData() {} }),
  };
  return el;
}

describe('renderTrackStrip', () => {
  afterEach(() => vi.unstubAllGlobals());

  const tiny = (stage) => regions(['10'], ['liver'], stage);
  const draw = (tracks) => {
    vi.stubGlobal('document', { createElement: fakeElement });
    const frames = [
      { i: 0, regionsByTrack: { gt: tiny('anchor'), zs: tiny('anchor') } },
      { i: 4, regionsByTrack: { gt: tiny('tracked'), zs: tiny('tracked') } },
      { i: 8, regionsByTrack: {} },
    ];
    const el = fakeElement();
    renderTrackStrip(el, { frames, tracks, painted: [], stageFrame: 4, imageURL: (i) => `${i}.jpg`, onFrame() {}, onPickTrack() {} });
    const [labels, scroll] = el.children;
    return { labels: labels.children, rows: scroll.children };
  };

  it('escapes the track\'s name and badges', () => {
    const { labels } = draw([{ id: 'zs', name: HOSTILE, semantic: false, anchor_badge: HOSTILE, tracked_badge: HOSTILE }]);
    expect(labels[0].innerHTML).toContain('&lt;img');
    expect(labels[0].innerHTML).not.toMatch(INJECTED);
  });

  it('marks the cell a track\'s regions were made on, with the track\'s badge', () => {
    const { rows } = draw([
      { id: 'gt', name: 'Annotation', semantic: true, anchor_badge: 'GT', tracked_badge: 'GT tracked' },
      { id: 'zs', name: 'Zero-shot', semantic: false, anchor_badge: 'Seed', tracked_badge: 'Tracked' },
    ]);
    expect(rows.map((row) => row.children.map((cell) => cell.className))).toEqual([
      ['strip-cell anchor-gt', 'strip-cell focus', 'strip-cell strip-gap'],
      ['strip-cell anchor-sam', 'strip-cell focus', 'strip-cell strip-gap'],
    ]);
    const badges = rows.map((row) => row.children.slice(0, 2).map((cell) => {
      const badge = cell.children[0].children[2];
      return [badge.className, badge.textContent];
    }));
    expect(badges).toEqual([
      [['cell-badge badge-gt', 'GT'], ['cell-badge badge-tracked', 'GT tracked']],
      [['cell-badge badge-sam', 'Seed'], ['cell-badge badge-tracked', 'Tracked']],
    ]);
  });
});
