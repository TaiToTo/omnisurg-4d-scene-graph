// Draw the scene graph on the point cloud: each node as the convex hull of
// its region's points, relation edges as arcs, containment as dashed links,
// and in World mode every stacked frame's nodes threaded through time.
//
// Also draws the followed region's path through a World stack, and finds the
// region under the pointer.
import * as THREE from 'three';
import { ConvexGeometry } from 'three/addons/geometries/ConvexGeometry.js';
import { state, onStage, focusFamily } from './state.js';
import { camera, canvas, clearGroup, invalidate, onSceneScale, scene, sz } from './scene.js';
import { RELATION_COLORS } from './lib/relations.js';
import { isLabelArray } from './lib/cloud_geometry.js';
import { fullGeometryOf, srcVertexIndex } from './lib/vertex_filter.js';
import { MAX_RELATION_EDGES } from './nodelink.js';

const HULL_MIN = 8, HULL_MAX = 3000;
// Ellipsoid radii in reference units, sized from `area_frac`, aspect clamped.
const R_BASE = 0.15, R_MIN = 0.012, R_MAX = 0.11, R_FALLBACK = 0.022;
const RATIO_MIN = 0.4, RATIO_MAX = 2.8;
const ACCENT = 0x0899b4, PARTNER = 0xe08a00, PATH = 0x0a8fb0, LIT = 0xe07f00;

const colorOf = (n) => (Array.isArray(n.color) ? new THREE.Color(...n.color) : new THREE.Color(0xffff00));
const basic = (color, opacity, extra = {}) => new THREE.MeshBasicMaterial({ color, transparent: true, opacity, ...extra });

// A node's PCA ellipsoid, for a region with too few points for a hull.
function ellipsoid(n, color, opacity) {
  const a = n.area_frac;
  const base = sz(Number.isFinite(a) && a > 0 ? Math.min(R_MAX, Math.max(R_MIN, R_BASE * Math.sqrt(a))) : R_FALLBACK);
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 12), basic(color, opacity));
  const ar = n.axes_radius, ax = n.axes;
  if (Array.isArray(ar) && ar.length === 3 && Array.isArray(ax) && ax.length === 3 && ar.every((v) => v > 0)) {
    const g = Math.cbrt(Math.max(ar[0] * ar[1] * ar[2], 1e-12));
    const f = ar.map((r) => Math.min(RATIO_MAX, Math.max(RATIO_MIN, r / g)));
    mesh.scale.set(base * f[0], base * f[1], base * f[2]);
    mesh.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(...ax.map((v) => new THREE.Vector3(...v))));
  } else {
    mesh.scale.setScalar(base);
  }
  return mesh;
}

// Every region's points in one pass over the labels, moved by `offset`.
function regionPoints(seg, posAttr, offset) {
  const vs = seg?.vertex_seg;
  if (!isLabelArray(vs) || vs.length !== posAttr.count) return null;
  const byId = new Map();
  for (let i = 0; i < vs.length; i++) {
    const id = vs[i];
    if (!id) continue;
    if (!byId.has(id)) byId.set(id, []);
    byId.get(id).push(new THREE.Vector3(posAttr.getX(i) + offset[0], posAttr.getY(i) + offset[1], posAttr.getZ(i) + offset[2]));
  }
  return byId;
}

// An arc from a to b with an arrowhead three quarters along.
function edgeArc(a, b, color, opacity) {
  const src = new THREE.Vector3(...a), dst = new THREE.Vector3(...b);
  const dir = new THREE.Vector3().subVectors(dst, src);
  const len = dir.length();
  if (len < 1e-6) return [];
  let perp = new THREE.Vector3().crossVectors(dir, new THREE.Vector3(0, 1, 0));
  if (perp.length() < 1e-4) perp = new THREE.Vector3().crossVectors(dir, new THREE.Vector3(1, 0, 0));
  perp.normalize().multiplyScalar(len * 0.08);
  const curve = new THREE.QuadraticBezierCurve3(src, src.clone().lerp(dst, 0.5).add(perp), dst);
  const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, 16, sz(0.002), 8, false), basic(color, opacity));
  const cone = new THREE.Mesh(new THREE.ConeGeometry(sz(0.008), sz(0.02), 8), basic(color, opacity));
  cone.position.copy(curve.getPointAt(0.75));
  cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), curve.getTangentAt(0.75).normalize());
  return [tube, cone];
}

/**
 * Draw the followed region's path through the World stack: solid up to the frame in focus, faint after it.
 *
 * Each piece is drawn twice, white then coloured, so the path reads over dark tissue and pale scene alike.
 */
function drawFocusPath() {
  if (state.mode !== 'world' || !state.world || !state.focusNode) return;
  const [track, idStr] = state.focusNode.split(':');
  const id = Number(idStr);
  const pts = [];
  let here = -1;
  for (const wf of state.world.frames) {
    const n = wf.graphByTrack?.[track]?.nodes?.find((m) => m.id === id);
    if (!Array.isArray(n?.pos)) continue;
    pts.push(new THREE.Vector3(n.pos[0] + wf.offset[0], n.pos[1] + wf.offset[1], n.pos[2] + wf.offset[2]));
    if (wf.i <= state.worldFocus) here = pts.length - 1;
  }
  const mat = (color, opacity) => basic(color, opacity, { depthTest: false });
  const tube = (part, radius, opacity) => {
    if (part.length < 2) return;
    const curve = new THREE.CatmullRomCurve3(part);
    for (const [r, color, order] of [[radius * 1.7, 0xffffff, 95], [radius, PATH, 96]]) {
      const m = new THREE.Mesh(new THREE.TubeGeometry(curve, Math.max(8, part.length * 8), sz(r), 8, false), mat(color, opacity));
      m.renderOrder = order;
      state.graph3dGroup.add(m);
    }
  };
  tube(pts.slice(0, here + 1), 0.0028, 1.0);
  tube(pts.slice(Math.max(0, here)), 0.0014, 0.35);
  pts.forEach((p, k) => {
    const r = k === here ? 0.009 : 0.0045;
    for (const [rr, color, order] of [[r * 1.45, 0xffffff, 97], [r, k === here ? LIT : PATH, 98]]) {
      const dot = new THREE.Mesh(new THREE.SphereGeometry(sz(rr), 14, 14), mat(color, k <= here ? 1.0 : 0.35));
      dot.position.copy(p);
      dot.renderOrder = order;
      state.graph3dGroup.add(dot);
    }
  });
}

/**
 * Rebuild the graph on the cloud for the frame on stage.
 *
 * Without the graph turned on, only the followed region's path is drawn. The
 * followed node's family stays bright and the rest fades.
 */
export function buildGraph3D() {
  clearGroup(state.graph3dGroup);
  if (!state.graph3d) { drawFocusPath(); invalidate(); return; }
  const stage = onStage();
  if (!stage) return;
  let posAttr = null;
  // The full geometry: a compacted copy would not line up with `vertex_seg`.
  stage.group.traverse((o) => { if (!posAttr && o.isPoints) posAttr = fullGeometryOf(o)?.attributes?.position ?? null; });
  if (!posAttr) return;
  const fam = focusFamily();
  const bright = (track, id) => (state.focusNode ? fam[track]?.has(id) : true);
  const posOf = new Map();

  // Nodes and relation edges, track by track.
  for (const track of state.tracks) {
    const seg = stage.segByTrack[track], graph = stage.graphByTrack[track];
    if (!seg || !graph?.nodes) continue;
    const points = regionPoints(seg, posAttr, stage.offset);
    for (const n of graph.nodes) {
      const key = `${track}:${n.id}`;
      const on = bright(track, n.id);
      const color = colorOf(n);
      if (Array.isArray(n.pos)) posOf.set(key, n.pos.map((v, k) => v + stage.offset[k]));
      let mesh = null;
      const pts = points?.get(n.id);
      if (pts && pts.length >= HULL_MIN) {
        const sample = pts.length > HULL_MAX ? pts.filter((_, i) => i % Math.ceil(pts.length / HULL_MAX) === 0) : pts;
        try {
          mesh = new THREE.Mesh(new ConvexGeometry(sample), basic(color, on ? 0.3 : 0.05, { side: THREE.DoubleSide, depthWrite: false }));
        } catch {
          mesh = null;   // coplanar points have no hull; the ellipsoid stands in
        }
      }
      if (!mesh && posOf.has(key)) {
        mesh = ellipsoid(n, color, on ? 0.6 : 0.08);
        mesh.position.set(...posOf.get(key));
      }
      if (!mesh) continue;
      mesh.renderOrder = 94;
      mesh.userData.nodeKey = key;
      state.graph3dGroup.add(mesh);
      if (state.focusNode && on) {
        const outline = new THREE.Mesh(mesh.geometry, basic(key === state.focusNode ? ACCENT : PARTNER, 0.75, { wireframe: true, depthTest: false }));
        outline.userData.sharedGeometry = true;
        outline.position.copy(mesh.position);
        outline.quaternion.copy(mesh.quaternion);
        outline.scale.copy(mesh.scale);
        outline.renderOrder = 96;
        state.graph3dGroup.add(outline);
      }
    }
    const edges = graph.edges || [];
    if (edges.length > MAX_RELATION_EDGES) continue;
    for (const e of edges) {
      const a = posOf.get(`${track}:${e.src}`), b = posOf.get(`${track}:${e.dst}`);
      if (!a || !b) continue;
      const touched = !state.focusNode || fam[track]?.has(e.src) || fam[track]?.has(e.dst);
      for (const obj of edgeArc(a, b, RELATION_COLORS[e.relation] ?? 0x888888, touched ? 0.75 : 0.05)) state.graph3dGroup.add(obj);
    }
  }

  // Containment, between two tracks on stage.
  if (state.tracks.length > 1) {
    for (const e of stage.hierarchy?.edges || []) {
      if (e.relation !== 'contains') continue;
      const a = posOf.get(e.src), b = posOf.get(e.dst);
      if (!a || !b) continue;
      const op = state.focusNode && e.src !== state.focusNode && e.dst !== state.focusNode ? 0.06 : 0.5;
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...a), new THREE.Vector3(...b)]),
        new THREE.LineDashedMaterial({ color: 0x39434f, transparent: true, opacity: op, depthTest: false, dashSize: sz(0.012), gapSize: sz(0.009) }));
      line.computeLineDistances();
      line.renderOrder = 94;
      state.graph3dGroup.add(line);
    }
  }

  // World mode: every stacked frame's node centroids, threaded per node through time.
  if (state.mode === 'world' && state.world) {
    const byKey = new Map();
    for (const wf of state.world.frames) {
      for (const track of state.tracks) {
        for (const n of wf.graphByTrack[track]?.nodes || []) {
          if (!Array.isArray(n.pos)) continue;
          const key = `${track}:${n.id}`;
          if (!byKey.has(key)) byKey.set(key, { track, id: n.id, color: colorOf(n), pts: [] });
          byKey.get(key).pts.push(new THREE.Vector3(n.pos[0] + wf.offset[0], n.pos[1] + wf.offset[1], n.pos[2] + wf.offset[2]));
        }
      }
    }
    for (const [key, e] of byKey) {
      const op = bright(e.track, e.id) ? 0.85 : 0.08;
      if (e.pts.length >= 2) {
        const thread = new THREE.Line(new THREE.BufferGeometry().setFromPoints(e.pts),
          new THREE.LineBasicMaterial({ color: e.color, transparent: true, opacity: op * 0.7, depthTest: false }));
        thread.renderOrder = 93;
        thread.userData.nodeKey = key;
        state.graph3dGroup.add(thread);
      }
      for (const p of e.pts) {
        const dot = new THREE.Mesh(new THREE.SphereGeometry(sz(0.0045), 8, 8), basic(e.color, op, { depthTest: false }));
        dot.position.copy(p);
        dot.renderOrder = 93;
        state.graph3dGroup.add(dot);
      }
    }
  }
  invalidate();
}

// The region under the pointer: an outline over its glyph, and its key.
const hoverGroup = new THREE.Group();
scene.add(hoverGroup);
const raycaster = new THREE.Raycaster();
const PICK_RADIUS = 0.006;
raycaster.params.Points.threshold = PICK_RADIUS;
onSceneScale((scale) => { raycaster.params.Points.threshold = PICK_RADIUS * scale; });

/** Outline the glyph of `key` on the cloud, or clear the outline. */
export function setHover3D(key) {
  clearGroup(hoverGroup);
  if (!key || !state.graph3d) return;
  for (const m of state.graph3dGroup.children) {
    if (!m.isMesh || m.userData.nodeKey !== key) continue;
    const o = new THREE.Mesh(m.geometry, basic(0x1d2530, 0.7, { wireframe: true, depthTest: false }));
    o.userData.sharedGeometry = true;
    o.position.copy(m.position);
    o.quaternion.copy(m.quaternion);
    o.scale.copy(m.scale);
    o.renderOrder = 99;
    hoverGroup.add(o);
  }
}

/**
 * Return the key of the region under a pointer event: a glyph of the graph, else a point of the cloud on stage.
 *
 * A point maps to its region through the primary track's labels first.
 */
export function pickKey(ev) {
  const r = canvas.getBoundingClientRect();
  raycaster.setFromCamera(new THREE.Vector2(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1), camera);
  const stage = onStage();
  // Glyph meshes only: a thread line is hit from far off, since a Line's pick radius is a world unit.
  const targets = state.graph3d ? state.graph3dGroup.children.filter((o) => o.isMesh && o.userData.nodeKey) : [];
  stage?.group.traverse((o) => { if (o.isPoints) targets.push(o); });
  for (const hit of raycaster.intersectObjects(targets, false)) {
    if (hit.object.userData.nodeKey) return hit.object.userData.nodeKey;
    const idx = srcVertexIndex(hit.object, hit.index);
    if (idx < 0) continue;
    for (const track of state.tracks) {
      const id = stage?.segByTrack?.[track]?.vertex_seg?.[idx];
      if (id) return `${track}:${id}`;
    }
  }
  return null;
}
