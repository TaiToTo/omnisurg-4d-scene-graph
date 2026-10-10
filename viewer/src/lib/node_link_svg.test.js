import { describe, it, expect } from 'vitest';
import { renderNodeLink } from './node_link_svg.js';

const positions = new Map([[1, [10, 10]], [2, [100, 100]], [3, [200, 50]]]);
const nodes = [
  { id: 1, label: 'Liver', color: [1, 0, 0] },
  { id: 2, label: 'Grasper' },
  { id: 3, label: 'Fat', color: [12, 200, 30] },
];
const edges = [{ src: 1, dst: 2, relation: 'right' }, { src: 2, dst: 3, relation: 'contains' }];

describe('renderNodeLink', () => {
  it('draws one keyed node per positioned node', () => {
    const svg = renderNodeLink(nodes, edges, positions, { track: 'gt' });
    for (const id of [1, 2, 3]) expect(svg).toContain(`data-node-key="gt:${id}"`);
    expect(svg.match(/<ellipse/g)).toHaveLength(3);
  });

  it('draws one line per edge with both ends positioned, containment dashed', () => {
    const svg = renderNodeLink(nodes, edges, positions, {});
    expect(svg.match(/<line/g)).toHaveLength(2);
    expect(svg.match(/stroke-dasharray/g)).toHaveLength(1);
  });

  it('colours a node by its own colour, else by its track’s tint', () => {
    const svg = renderNodeLink(nodes, edges, positions, { track: 'gt', tint: () => 0x12c2e9 });
    expect(svg).toContain('rgb(255,0,0)');
    expect(svg).toContain('rgb(12,200,30)');
    expect(svg).toContain('#12c2e9');
  });

  it('dims nodes outside the brush and keeps the brushed one bright', () => {
    const svg = renderNodeLink(nodes, edges, positions, { track: 'gt', brushKeys: new Set(['gt:2']) });
    expect(svg).toMatch(/data-node-key="gt:2"[^>]*opacity="1"/);
    expect(svg).toMatch(/data-node-key="gt:1"[^>]*opacity="0\.18"/);
  });

  it('draws a hull polygon where the projection carries one', () => {
    const projById = new Map([[1, { poly: [[-5, -5], [5, -5], [0, 6]], rx: 5, ry: 6 }], [2, { rx: 8, ry: 4 }]]);
    const svg = renderNodeLink(nodes, edges, positions, { track: 'gt', projById });
    expect(svg).toMatch(/<polygon class="nl-node" data-node-key="gt:1" points="5.0,5.0 15.0,5.0 10.0,16.0"/);
    expect(svg).toMatch(/<ellipse class="nl-node" data-node-key="gt:2"/);
  });

  it('marks the hovered node only', () => {
    const svg = renderNodeLink(nodes, edges, positions, { track: 'gt', hoverKey: 'gt:2' });
    expect(svg).toMatch(/class="nl-node nl-hover" data-node-key="gt:2"/);
    expect(svg).toMatch(/class="nl-node" data-node-key="gt:1"/);
  });

  it('says so when no node is positioned', () => {
    expect(renderNodeLink([], [], new Map())).toContain('No graph');
    expect(renderNodeLink(nodes, edges, new Map())).toContain('No graph');
  });

  it('escapes labels and keys taken from the data', () => {
    const svg = renderNodeLink([{ id: 1, label: 'A<b>&"c', key: 'x"y:1' }], [], new Map([[1, [5, 5]]]), { showLabels: true });
    expect(svg).toContain('<title>A&lt;b&gt;&amp;&quot;c</title>');
    expect(svg).toContain('data-node-key="x&quot;y:1"');
    expect(svg).not.toContain('<b>');
  });
});
