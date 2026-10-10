import { describe, it, expect } from 'vitest';
import { partnerRelations, topPartners, relationRuns, frameRuns, nodeEvents, containmentPartners } from './relations.js';

const spatial = (src, dst, relation, frames) => ({ src, dst, relation, edge_type: 'spatial', frames });
const action = (src, dst, relation, frames) => ({ src, dst, relation, edge_type: 'action', frames });

describe('partnerRelations', () => {
  it('ignores action edges: the band shows spatial relations', () => {
    const e = partnerRelations({ relations: [spatial(1, 2, 'left', [0, 1]), action(1, 2, 'grasp', [0, 1])] }, 1).get(2);
    expect([...e.relByFrame.values()]).toEqual(['left', 'left']);
    expect(e.weight).toBe(2);
  });

  it('mirrors the word when the node is the edge’s dst', () => {
    expect(partnerRelations({ relations: [spatial(2, 1, 'left', [0])] }, 1).get(2).relByFrame.get(0)).toBe('right');
  });

  it('picks one word per frame by depth, height, side, whatever the order of relations', () => {
    const a = { relations: [spatial(1, 2, 'left', [0, 1]), spatial(1, 2, 'above', [0, 1])] };
    const b = { relations: [spatial(1, 2, 'above', [0, 1]), spatial(1, 2, 'left', [0, 1])] };
    expect([...partnerRelations(a, 1).get(2).relByFrame.values()]).toEqual(['above', 'above']);
    expect([...partnerRelations(b, 1).get(2).relByFrame.values()]).toEqual(['above', 'above']);
  });

  it('does not call a gap in presence a change', () => {
    const tg = { relations: [spatial(1, 2, 'left', [0]), spatial(1, 2, 'above', [2])] };
    expect(partnerRelations(tg, 1).get(2).changes).toEqual([]);
  });

  it('reports a change between adjacent frames', () => {
    const tg = { relations: [spatial(1, 2, 'left', [0]), spatial(1, 2, 'above', [1])] };
    expect(partnerRelations(tg, 1).get(2).changes).toEqual([{ f: 1, from: 'left', to: 'above' }]);
  });

  it('is empty for a missing graph or a node in no relation', () => {
    expect(partnerRelations(null, 1).size).toBe(0);
    expect(partnerRelations({ relations: [spatial(3, 4, 'left', [0])] }, 1).size).toBe(0);
  });
});

describe('topPartners', () => {
  it('ranks by changes, then by frames together, then by id', () => {
    const tg = {
      relations: [
        spatial(1, 2, 'left', [0, 1, 2, 3]),
        spatial(1, 3, 'left', [0]), spatial(1, 3, 'above', [1]),
        spatial(1, 4, 'left', [0, 1, 2, 3]),
      ],
    };
    expect(topPartners(tg, 1, 3).map(([id]) => id)).toEqual([3, 2, 4]);
  });
});

describe('relationRuns and frameRuns', () => {
  it('merge consecutive frames and break on a gap', () => {
    expect(relationRuns(new Map([[0, 'left'], [1, 'left'], [2, 'above'], [5, 'above']])))
      .toEqual([{ rel: 'left', a: 0, b: 1 }, { rel: 'above', a: 2, b: 2 }, { rel: 'above', a: 5, b: 5 }]);
    expect(frameRuns([3, 0, 1, 5])).toEqual([[0, 1], [3, 3], [5, 5]]);
  });
});

describe('nodeEvents', () => {
  it('lists entering, leaving and each edge change, in frame order', () => {
    const tg = { relations: [spatial(1, 2, 'left', [2]), spatial(1, 2, 'above', [3])] };
    const events = nodeEvents({ present_frames: [2, 3, 4] }, topPartners(tg, 1), 9, (id) => `obj ${id}`);
    expect(events.map((e) => [e.f, e.kind])).toEqual([[2, 'enter'], [3, 'rel'], [5, 'exit']]);
    expect(events[1]).toMatchObject({ partner: 2, label: 'vs obj 2: left → above' });
  });

  it('lists no entering at frame 0 and no leaving at the last frame', () => {
    expect(nodeEvents({ present_frames: [0, 1, 2] }, [], 2, String)).toEqual([]);
  });
});

describe('containmentPartners', () => {
  it('returns both sides of the contains edges a key is on, and nothing else', () => {
    const h = { edges: [
      { src: 'gt:2', dst: 'auto:1', relation: 'contains' },
      { src: 'gt:2', dst: 'auto:3', relation: 'contains' },
      { src: 'gt:4', dst: 'auto:1', relation: 'overlaps' },
    ] };
    expect(containmentPartners('gt:2', h)).toEqual(['auto:1', 'auto:3']);
    expect(containmentPartners('auto:1', h)).toEqual(['gt:2']);
    expect(containmentPartners('gt:9', null)).toEqual([]);
  });
});
