import { describe, expect, it } from 'vitest';
import { nodeEvents, runsOf, topPartners } from './focus_band.js';

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
    const ev = nodeEvents(tg.nodes[2], [], 4, String);
    expect(ev).toEqual([{ f: 2, kind: 'exit', label: 'leaves the view' }]);
    const ev1 = nodeEvents(tg.nodes[0], topPartners(tg, 1), 4, (id) => `obj ${id}`);
    expect(ev1).toEqual([{ f: 3, kind: 'rel', partner: 2, label: 'to obj 2: left → right' }]);
  });
});
