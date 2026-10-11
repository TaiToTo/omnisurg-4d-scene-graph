import * as THREE from 'three';
import { ConvexGeometry } from 'three/addons/geometries/ConvexGeometry.js';
import { describe, expect, it } from 'vitest';
import { FOCUS_COLOR, RELATION_COLORS } from './format.js';
import { buildGraph3D } from './graph3d.js';
import { groups } from './stage.js';

// A frame of track 't' moved by +10 in x. Region 1 fills a cube of 27 points, region 2 has 5 points,
// region 3 has none, and region 4 lies on a line; the last point is background.
function frame() {
  const pts = [], labels = [];
  const put = (id, xyz) => { pts.push(new THREE.Vector3(...xyz)); labels.push(id); };
  for (let k = 0; k < 27; k++) put(1, [k % 3, Math.floor(k / 3) % 3, Math.floor(k / 9)]);
  for (const xyz of [[5, 0, 0], [6, 0, 0], [5, 1, 0], [5, 0, 1], [6, 1, 1]]) put(2, xyz);
  for (let k = 0; k < 10; k++) put(4, [0, 6, k]);
  put(0, [9, 9, 9]);
  const group = new THREE.Group().add(new THREE.Points(new THREE.BufferGeometry().setFromPoints(pts)));
  const nodes = [
    { id: 1, pos: [1, 1, 1], color: [1, 0, 0] },
    { id: 2, pos: [5.4, 0.4, 0.4], color: [0, 1, 0], area_frac: 0.01, axes_radius: [1, 2, 4], axes: [[0, 1, 0], [-1, 0, 0], [0, 0, 1]] },
    { id: 3, pos: [0, 4, 0], color: [0, 0, 1] },
    { id: 4, pos: [0, 6, 4.5], color: [1, 1, 0] },
  ];
  const edges = [{ src: 1, dst: 2, relation: 'left' }, { src: 3, dst: 1, relation: 'above' }];
  return {
    i: 0, group, offset: [10, 0, 0],
    regionsByTrack: { t: { labels: Int32Array.from(labels) } }, graphByTrack: { t: { nodes, edges } },
  };
}

const drawn = () => groups.graph3d.children;
const nodeMesh = (id) => drawn().find((m) => m.userData.nodeKey === `t:${id}`);
const isSphere = (m) => m.geometry.type === 'SphereGeometry';
const tubes = () => drawn().filter((m) => m.geometry.type === 'TubeGeometry');

describe('the graph in the 3D view', () => {
  it('draws nothing with the graph off', () => {
    buildGraph3D({ on: true, tracks: ['t'], stage: frame() });
    expect(drawn().length).toBeGreaterThan(0);
    buildGraph3D({ on: false, tracks: ['t'], stage: frame() });
    expect(drawn()).toEqual([]);
  });

  it('draws a node as its region\'s hull, or as an ellipsoid when the region is small, empty or a line', () => {
    buildGraph3D({ on: true, tracks: ['t'], stage: frame() });
    const hull = nodeMesh(1).geometry;
    expect(hull).toBeInstanceOf(ConvexGeometry);
    hull.computeBoundingBox();
    expect([hull.boundingBox.min.toArray(), hull.boundingBox.max.toArray()]).toEqual([[10, 0, 0], [12, 2, 2]]);
    expect([2, 3, 4].map((id) => isSphere(nodeMesh(id)))).toEqual([true, true, true]);
    expect(nodeMesh(3).position.toArray()).toEqual([10, 4, 0]);
  });

  it('draws ellipsoids when the labels are missing or do not fit the cloud', () => {
    for (const cut of [() => undefined, (l) => l.slice(1), (l) => Int32Array.of(...l, 0)]) {
      const stage = frame();
      stage.regionsByTrack.t.labels = cut(stage.regionsByTrack.t.labels);
      buildGraph3D({ on: true, tracks: ['t'], stage });
      expect(isSphere(nodeMesh(1))).toBe(true);
    }
  });

  it('reads the regions on the full cloud when the view draws a filtered copy', () => {
    const stage = frame();
    const cloud = stage.group.children[0];
    cloud.userData.fullGeometry = cloud.geometry;
    cloud.geometry = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3()]);
    buildGraph3D({ on: true, tracks: ['t'], stage });
    expect(nodeMesh(1).geometry).toBeInstanceOf(ConvexGeometry);
  });

  it('fades every node and edge but the followed node\'s', () => {
    const opacities = () => [1, 2, 3].map((id) => nodeMesh(id).material.opacity);
    const edgeOpacities = () => tubes().map((m) => m.material.opacity);
    buildGraph3D({ on: true, tracks: ['t'], stage: frame() });
    const [nodesBefore, edgesBefore] = [opacities(), edgeOpacities()];
    buildGraph3D({ on: true, tracks: ['t'], stage: frame(), focusKey: 't:2' });
    const [n1, n2, n3] = opacities();
    expect(n2).toBe(nodesBefore[1]);
    expect(n1).toBeLessThan(nodesBefore[0]);
    expect(n3).toBeLessThan(nodesBefore[2]);
    const [touching, other] = edgeOpacities();
    expect(touching).toBe(edgesBefore[0]);
    expect(other).toBeLessThan(edgesBefore[1]);
  });

  it('outlines the followed node on its own shape, and no other', () => {
    buildGraph3D({ on: true, tracks: ['t'], stage: frame() });
    expect(drawn().filter((m) => m.material.wireframe)).toEqual([]);
    buildGraph3D({ on: true, tracks: ['t'], stage: frame(), focusKey: 't:2' });
    const outlines = drawn().filter((m) => m.material.wireframe);
    expect(outlines).toHaveLength(1);
    const [o] = outlines, node = nodeMesh(2);
    expect(o.geometry).toBe(node.geometry);
    expect(o.userData.sharedGeometry).toBe(true);
    expect(o.material.color.getHex()).toBe(FOCUS_COLOR);
    for (const k of ['position', 'quaternion', 'scale']) expect(o[k].toArray()).toEqual(node[k].toArray());
  });

  it('colours each edge, its arc and its arrowhead, by its relation', () => {
    buildGraph3D({ on: true, tracks: ['t'], stage: frame() });
    const arcs = drawn().filter((m) => ['TubeGeometry', 'ConeGeometry'].includes(m.geometry.type));
    expect(arcs.map((m) => m.material.color.getHex())).toEqual(
      [RELATION_COLORS.left, RELATION_COLORS.left, RELATION_COLORS.above, RELATION_COLORS.above]);
  });
});

describe('the graph in the 3D view with two tracks', () => {
  // Both tracks label the frame alike; the annotation's region 1 contains the automatic region 2.
  function pair() {
    const f = frame();
    const { labels } = f.regionsByTrack.t;
    return { ...f, regionsByTrack: { gt: { labels }, auto: { labels } }, graphByTrack: { gt: f.graphByTrack.t, auto: f.graphByTrack.t },
      hierarchy: { edges: [{ src: 'gt:1', dst: 'auto:2', relation: 'contains' }] } };
  }
  const opacity = (key) => drawn().find((m) => m.userData.nodeKey === key).material.opacity;

  it('draws the contains links instead of each track\'s relations', () => {
    buildGraph3D({ on: true, tracks: ['gt', 'auto'], stage: pair() });
    expect(tubes()).toHaveLength(0);
    expect(drawn().filter((o) => o.material?.isLineDashedMaterial)).toHaveLength(1);
  });

  it('keeps the region the followed node contains bright', () => {
    buildGraph3D({ on: true, tracks: ['gt', 'auto'], stage: pair(), focusKey: 'gt:1' });
    expect(opacity('auto:2')).toBeGreaterThan(opacity('auto:3'));
  });
});
