import { describe, expect, it } from 'vitest';
import { axisX, frameAtX, nodeEvents, renderFocusBand, runsOf, topPartners } from './focus_band.js';

const tg = {
  nodes: [{ id: 1, present_frames: [0, 1, 2, 3, 4] }, { id: 2, present_frames: [0, 1, 2, 3, 4] }, { id: 3, present_frames: [0, 1] }],
  relations: [
    { src: 1, dst: 2, relation: 'left', edge_type: 'spatial', frames: [0, 1, 2] },
    { src: 2, dst: 1, relation: 'left', edge_type: 'spatial', frames: [3, 4] },
    { src: 1, dst: 3, relation: 'above', edge_type: 'spatial', frames: [0, 1] },
    { src: 1, dst: 3, relation: 'grasp', edge_type: 'action', frames: [0] },
  ],
};

describe('topPartners', () => {
  it('reads every relation from the followed node, and puts the one that changes first', () => {
    const [[other, e]] = topPartners(tg, 1);
    expect(other).toBe(2);
    // 2 left of 1 is 1 right of 2.
    expect([...e.relByFrame]).toEqual([[0, 'left'], [1, 'left'], [2, 'left'], [3, 'right'], [4, 'right']]);
    expect(e.changes).toEqual([{ f: 3, from: 'left', to: 'right' }]);
  });

  it('counts no change across frames where the two are not both in view', () => {
    const gap = { relations: [
      { src: 1, dst: 2, relation: 'left', edge_type: 'spatial', frames: [0, 1] },
      { src: 1, dst: 2, relation: 'right', edge_type: 'spatial', frames: [3, 4] },
    ] };
    expect(topPartners(gap, 1)[0][1].changes).toEqual([]);
  });

  it('leaves out relations that are not spatial', () => {
    const third = topPartners(tg, 1, 2)[1][1];
    expect([...third.relByFrame.values()]).toEqual(['above', 'above']);
  });
});

describe('values that are not frames', () => {
  it('are left out, so nothing but integers reaches the band', () => {
    const bad = { relations: [{ src: 1, dst: 2, relation: 'left', edge_type: 'spatial', frames: ['<img src=x>', 2] }] };
    expect([...topPartners(bad, 1)[0][1].relByFrame.keys()]).toEqual([2]);
    expect(runsOf(['<img src=x>', 3, 4])).toEqual([[3, 4]]);
  });
});

describe('events', () => {
  it('groups consecutive frames into runs', () => {
    expect(runsOf([5, 1, 2, 3, 7, 8])).toEqual([[1, 3], [5, 5], [7, 8]]);
  });

  it('lists entries, exits and relation changes in time order', () => {
    const ev = nodeEvents(tg.nodes[2], [], [0, 4], String);
    expect(ev).toEqual([{ f: 2, kind: 'exit', label: 'leaves the view' }]);
    const ev1 = nodeEvents(tg.nodes[0], topPartners(tg, 1), [0, 4], (id) => `obj ${id}`);
    expect(ev1).toEqual([{ f: 3, kind: 'rel', partner: 2, label: 'to obj 2: left → right' }]);
  });

  it('lets a node enter or leave only within the frames the graph covers', () => {
    // A track whose regions run from frame 2 to frame 6 of a longer clip.
    const node = { present_frames: [2, 3, 4, 5, 6] };
    expect(nodeEvents(node, [], [2, 6], String)).toEqual([]);
    expect(nodeEvents(node, [], [0, 9], String).map((e) => [e.kind, e.f])).toEqual([['enter', 2], ['exit', 7]]);
  });
});

/** An element just big enough for renderFocusBand: it keeps the markup and hands back a stand-in SVG. */
function bandElement(width) {
  return {
    innerHTML: '',
    getBoundingClientRect: () => ({ width: width + 2 }),
    firstElementChild: { addEventListener() {}, setPointerCapture() {} },
    querySelectorAll: () => [],
  };
}

const attrs = (tag, html) => [...html.matchAll(new RegExp(`<${tag} ([^>]*)>`, 'g'))]
  .map(([, a]) => Object.fromEntries([...a.matchAll(/([\w-]+)="([^"]*)"/g)].map(([, k, v]) => [k, v])));

describe('the band', () => {
  const frames = [...Array(15).keys()];
  const band = {
    frames,
    nodes: [{ id: 1, present_frames: frames }, { id: 2, present_frames: frames }],
    relations: [{ src: 1, dst: 2, relation: 'right', edge_type: 'spatial', frames }],
  };
  const axes = { byFrame: new Map(frames.map((f) => [f, { c: [1, 0.2, -0.3], word: 'right' }])) };

  it('draws every frame\'s column on the axis, clear of the row labels and of the right edge', () => {
    const W = 1576;
    const el = bandElement(W);
    renderFocusBand(el, { tg: band, nodeId: 1, lastFrame: 14, frame: 0, onFrame() {}, labelOf: String, axes });
    const cols = attrs('rect', el.innerHTML).filter((r) => r.width);
    expect(cols.length).toBeGreaterThan(0);
    const labelsEnd = Math.max(...attrs('text', el.innerHTML).filter((t) => t['text-anchor'] === 'end' && t.class === 'nf-axis')
      .map((t) => Number(t.x)));
    expect(Math.min(...cols.map((r) => Number(r.x)))).toBeGreaterThan(labelsEnd);
    expect(Math.max(...cols.map((r) => Number(r.x) + Number(r.width)))).toBeLessThanOrEqual(W);
  });

  it('turns an x anywhere on a frame\'s column back into that frame', () => {
    for (const n of [2, 15, 300]) {
      // A column spans 0.72 of a step, centred on its frame.
      const edge = 0.3 * (axisX(1, 900, n) - axisX(0, 900, n));
      for (const f of [0, 1, Math.floor(n / 2), n - 1]) {
        for (const dx of [-edge, 0, edge]) expect(frameAtX(axisX(f, 900, n) + dx, 900, n)).toBe(f);
      }
    }
    expect(frameAtX(0, 900, 15)).toBe(0);
    expect(frameAtX(900, 900, 15)).toBe(14);
  });
});
