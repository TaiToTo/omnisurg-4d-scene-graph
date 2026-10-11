import { describe, expect, it } from 'vitest';
import {
  containmentPartners, fitLinear, labelSVG, projectNodes, projectToCamera, renderFrameNodeLink, renderWorldNodeLink,
} from './node_link.js';

describe('projectToCamera', () => {
  const frame = { camera_position: [0, 0, 5], camera_forward: [0, 0, -1], camera_up: [0, 1, 0] };

  it('gives a point to the camera\'s right and above it as positive rays, divided by depth', () => {
    const [r] = projectToCamera([[1, 2, 3]], frame, [0, 0, 0]);
    expect(r).toEqual({ X: 0.5, Y: 1, behind: false });
  });

  it('moves the camera by the world origin, and marks a point behind it', () => {
    const [r, back] = projectToCamera([[1, 2, 3], [0, 0, 10]], frame, [0, 0, -1]);
    expect(r.X).toBeCloseTo(1 / 3);
    expect(back.behind).toBe(true);
  });
});

describe('fitLinear', () => {
  it('fits a line and refuses x without spread', () => {
    expect(fitLinear([0, 1, 2], [1, 3, 5])).toEqual({ a: 2, b: 1 });
    expect(fitLinear([1, 1], [0, 5])).toBeNull();
    expect(fitLinear([1], [0])).toBeNull();
  });
});

describe('projectNodes', () => {
  const regions = { grid: [4, 2], labels: Int32Array.from([1, 1, 0, 2, 1, 1, 0, 2]), classes: [] };

  it('places each node at its region\'s centroid, keeping the image\'s aspect', () => {
    const p = projectNodes([{ id: 1 }, { id: 2 }, { id: 9 }], regions, 436, 236);
    expect(p.scale).toBe(100);
    expect(p.positions.get(1)).toEqual([18 + 50, 18 + 50]);
    expect(p.positions.has(9)).toBe(false);
  });

  it('is null when no node has a region on the frame', () => {
    expect(projectNodes([{ id: 9 }], regions, 400, 200)).toBeNull();
  });
});

describe('labelSVG', () => {
  it('puts a label to the right of its node, or to its left where it would leave the panel', () => {
    expect(labelSVG('liver', [10, 20], 5, 200, 1)).toBe('<text class="nl-label" x="17.0" y="23.0" opacity="1">liver</text>');
    expect(labelSVG('liver', [190, 20], 5, 200, 1)).toContain('x="183.0" y="23.0" text-anchor="end"');
  });

  it('escapes the label', () => {
    expect(labelSVG('<b>', [0, 0], 1, 100, 1)).toContain('&lt;b&gt;');
  });
});

describe('containmentPartners', () => {
  const hierarchy = { edges: [
    { src: 'gt:2', dst: 'auto:4', relation: 'contains' },
    { src: 'gt:2', dst: 'auto:7', relation: 'contains' },
    { src: 'gt:3', dst: 'auto:4', relation: 'overlaps' },
  ] };

  it('gives the keys a contains edge joins to a key, either way', () => {
    expect(containmentPartners('gt:2', hierarchy)).toEqual(['auto:4', 'auto:7']);
    expect(containmentPartners('auto:4', hierarchy)).toEqual(['gt:2']);
    expect(containmentPartners('gt:3', hierarchy)).toEqual([]);
    expect(containmentPartners('gt:2', null)).toEqual([]);
  });
});

describe('renderFrameNodeLink', () => {
  // Annotated class 2 covers the left half; automatic regions 4 and 7 lie inside it, 5 outside.
  const gt = { grid: [4, 2], labels: Int32Array.from([2, 2, 0, 0, 2, 2, 0, 0]), classes: [] };
  const auto = { grid: [4, 2], labels: Int32Array.from([4, 7, 5, 5, 4, 7, 5, 5]), classes: [] };
  const color = [0.5, 0.5, 0.5];
  const p = {
    graphByTrack: {
      gt: { nodes: [{ id: 2, label: 'liver', color }], edges: [] },
      auto: { nodes: [{ id: 4, color }, { id: 5, color }, { id: 7, color }],
        edges: [{ src: 4, dst: 5, relation: 'left' }] },
    },
    regionsByTrack: { gt, auto },
    hierarchy: { edges: [
      { src: 'gt:2', dst: 'auto:4', relation: 'contains' },
      { src: 'gt:2', dst: 'auto:7', relation: 'contains' },
      { src: 'gt:2', dst: 'auto:5', relation: 'overlaps' },
    ] },
  };
  const draw = (opts) => { const el = { clientWidth: 436, clientHeight: 236, innerHTML: '' }; renderFrameNodeLink(el, { ...p, ...opts }); return el.innerHTML; };

  it('draws one track with its relations', () => {
    const svg = draw({ tracks: ['auto'] });
    expect(svg.match(/data-node-key="auto:/g)).toHaveLength(3);
    expect(svg).not.toContain('data-node-key="gt:');
    expect(svg.match(/<line /g)).toHaveLength(1);
  });

  it('draws two tracks with the contains edges between them, dashed', () => {
    const svg = draw({ tracks: ['gt', 'auto'] });
    expect(svg.match(/data-node-key="/g)).toHaveLength(4);
    expect(svg.match(/stroke-dasharray="3 2"/g)).toHaveLength(2);
    // The annotation is drawn first, under the automatic regions.
    expect(svg.indexOf('gt:2')).toBeLessThan(svg.indexOf('auto:4'));
  });

  it('keeps a followed node and the regions it contains bright, and dims the rest', () => {
    const svg = draw({ tracks: ['gt', 'auto'], focusKey: 'gt:2' });
    const opacity = (key) => svg.match(new RegExp(`data-node-key="${key}"[^>]* opacity="([0-9.]+)"`))[1];
    expect([opacity('gt:2'), opacity('auto:4'), opacity('auto:7'), opacity('auto:5')]).toEqual(['1', '1', '1', '0.18']);
  });
});

// Text that opens a tag, or ends an attribute, wherever it is not escaped.
const HOSTILE = '<img src=x onerror=alert(1)>" onmouseover="x';
const INJECTED = /<img|\sonmouseover="/;

const attrs = (tag, html) => [...html.matchAll(new RegExp(`<${tag} ([^>]*)>`, 'g'))]
  .map(([, a]) => Object.fromEntries([...a.matchAll(/([\w-]+)="([^"]*)"/g)].map(([, k, v]) => [k, v])));

// Region 1 is drawn as its hull, regions 2 and 3 are too thin for one and drawn as ellipses.
const three = { grid: [4, 2], labels: Int32Array.from([1, 1, 3, 2, 1, 1, 0, 2]), classes: [] };
const oneTrack = (track, graph) => ({ tracks: [track], graphByTrack: { [track]: graph }, regionsByTrack: { [track]: three } });

describe('renderFrameNodeLink', () => {
  const el = { clientWidth: 320, clientHeight: 260, innerHTML: '' };
  const edges = (k) => Array.from({ length: k }, () => ({ src: 1, dst: 2, relation: 'left' }));

  it('escapes the node labels and the track in every node key', () => {
    const nodes = [{ id: 1, label: HOSTILE }, { id: 2, label: HOSTILE }];
    renderFrameNodeLink(el, oneTrack(HOSTILE, { nodes, edges: [] }));
    expect(el.innerHTML).toContain('<polygon');
    expect(el.innerHTML).toContain('<ellipse');
    expect(el.innerHTML).toContain('&lt;img');
    expect(el.innerHTML).not.toMatch(INJECTED);
  });

  it('draws a frame\'s edges up to the cap, and none past it', () => {
    const nodes = [{ id: 1 }, { id: 2 }];
    renderFrameNodeLink(el, oneTrack('zs', { nodes, edges: edges(30) }));
    expect(attrs('line', el.innerHTML)).toHaveLength(30);
    renderFrameNodeLink(el, oneTrack('zs', { nodes, edges: edges(31) }));
    expect(attrs('line', el.innerHTML)).toHaveLength(0);
  });

  it('fades an edge only when both its ends are dimmed', () => {
    const graph = { nodes: [{ id: 1 }, { id: 2 }, { id: 3 }], edges: [{ src: 1, dst: 2, relation: 'left' }, { src: 2, dst: 3, relation: 'left' }] };
    renderFrameNodeLink(el, { ...oneTrack('zs', graph), focusKey: 'zs:1' });
    expect(attrs('line', el.innerHTML).map((l) => l.opacity)).toEqual(['0.7', '0.18']);
    renderFrameNodeLink(el, oneTrack('zs', graph));
    expect(attrs('line', el.innerHTML).map((l) => l.opacity)).toEqual(['0.7', '0.7']);
  });
});

/**
 * Two stacked frames, seen by a camera at z = 5 looking down -z. On the frame on stage nodes 1 and 2 sit on
 * their regions. On the other frame node 2 stays put, node 1 and nodes 3 to 5 lie far off the left, right,
 * top and bottom of the view, and node 6 is behind the camera.
 */
function stack(track, label) {
  const node = (id, pos) => ({ id, label, pos });
  const regions = { grid: [4, 2], labels: Int32Array.from([1, 1, 0, 2, 1, 1, 0, 2]), classes: [] };
  const camera = { camera_position: [0, 0, 5], camera_forward: [0, 0, -1], camera_up: [0, 1, 0] };
  const onStage = [node(1, [-1, 0, 0]), node(2, [1, 0, 0])];
  const other = [node(1, [-50, 0, 0]), node(2, [1, 0, 0]), node(3, [50, 0, 0]), node(4, [0, 50, 0]), node(5, [0, -50, 0]), node(6, [0, 0, 10])];
  const world = {
    origin: [0, 0, 0],
    indices: [0, 1],
    frames: [
      { i: 0, offset: [0, 0, 0], regionsByTrack: { [track]: regions }, graphByTrack: { [track]: { nodes: onStage } } },
      { i: 1, offset: [0, 0, 0], regionsByTrack: {}, graphByTrack: { [track]: { nodes: other } } },
    ],
  };
  return { world, refFrame: 0, tracks: [track], frames: [camera, camera] };
}

describe('renderWorldNodeLink', () => {
  const el = { clientWidth: 320, clientHeight: 260, innerHTML: '' };

  it('escapes the node labels and the track in every node key', () => {
    renderWorldNodeLink(el, stack(HOSTILE, HOSTILE));
    for (const tag of ['<polygon', '<ellipse', '<circle', '<polyline']) expect(el.innerHTML).toContain(tag);
    expect(el.innerHTML).toContain('&lt;img');
    expect(el.innerHTML).not.toMatch(INJECTED);
  });

  it('draws a node outside the view small at the edge, and counts those behind the camera', () => {
    renderWorldNodeLink(el, stack('zs', 'n'));
    const dots = attrs('circle', el.innerHTML);
    expect(dots).toHaveLength(5);
    for (const d of dots) {
      expect(Number(d.cx)).toBeGreaterThanOrEqual(0);
      expect(Number(d.cx)).toBeLessThanOrEqual(el.clientWidth);
      expect(Number(d.cy)).toBeGreaterThanOrEqual(0);
      expect(Number(d.cy)).toBeLessThanOrEqual(el.clientHeight);
    }
    expect(dots.filter((d) => d.r === '2.4').map((d) => d['data-node-key'])).toEqual(['zs:1', 'zs:3', 'zs:4', 'zs:5']);
    expect(el.innerHTML).toContain('4 at the edge · 1 behind the camera');
  });

  it('draws a trail only through the frames in view', () => {
    renderWorldNodeLink(el, stack('zs', 'n'));
    expect(attrs('polyline', el.innerHTML).map((p) => p['data-node-key'])).toEqual(['zs:2']);
  });
});
