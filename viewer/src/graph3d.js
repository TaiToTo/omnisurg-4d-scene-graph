// Draw the scene graph in the 3D view: each node as the convex hull of its region's points, or as an
// ellipsoid of its axes when the region is too small or flat for a hull; the edges as arcs coloured by
// relation; in world mode, a dot per stacked frame joined by a line per node. Also the followed region's
// path, the connector of a selected event, and the outline of a hovered node.
import * as THREE from 'three';
import { ConvexGeometry } from 'three/addons/geometries/ConvexGeometry.js';
import { clearGroup, groups, invalidate, makeLabelSprite, sz } from './stage.js';
import { fullGeometryOf } from './vertex_filter.js';
import { FOCUS_COLOR, PARTNER_COLOR, RELATION_COLORS, hexColor, nodeKey } from './format.js';

const HULL_MIN = 8, HULL_MAX = 3000;
// Ellipsoid radii in reference units (stage.sz), from the region's share of the image.
const R_BASE = 0.15, R_MIN = 0.012, R_MAX = 0.11, R_FALLBACK = 0.022;
const RATIO_MIN = 0.4, RATIO_MAX = 2.8;
// A frame's spatial edges hold every ordered pair of nodes; past this many arcs the cloud is buried.
const MAX_EDGES = 30;
const PATH_COLOR = 0x0a8fb0;

const colorOf = (n) => (Array.isArray(n.color) ? new THREE.Color(...n.color) : new THREE.Color(0xffff00));

function ellipsoid(n, color, opacity) {
  const a = n.area_frac;
  const base = sz(Number.isFinite(a) && a > 0 ? Math.min(R_MAX, Math.max(R_MIN, R_BASE * Math.sqrt(a))) : R_FALLBACK);
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 12, 12), new THREE.MeshBasicMaterial({ color, transparent: true, opacity }));
  const ar = n.axes_radius, ax = n.axes;
  if (Array.isArray(ar) && ar.length === 3 && Array.isArray(ax) && ax.length === 3 && ar.every((v) => v > 0)) {
    const g = Math.cbrt(Math.max(ar[0] * ar[1] * ar[2], 1e-12));
    const f = ar.map((r) => Math.min(RATIO_MAX, Math.max(RATIO_MIN, r / g)));
    mesh.scale.set(base * f[0], base * f[1], base * f[2]);
    mesh.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(
      new THREE.Vector3(...ax[0]), new THREE.Vector3(...ax[1]), new THREE.Vector3(...ax[2])));
  } else {
    mesh.scale.setScalar(base);
  }
  return mesh;
}

/** Each region's points, moved by `offset`, in one pass over the labels. */
function regionPoints(labels, posAttr, offset) {
  if (!labels || labels.length !== posAttr.count) return null;
  const byId = new Map();
  for (let i = 0; i < labels.length; i++) {
    const id = labels[i];
    if (!id) continue;
    if (!byId.has(id)) byId.set(id, []);
    byId.get(id).push(new THREE.Vector3(posAttr.getX(i) + offset[0], posAttr.getY(i) + offset[1], posAttr.getZ(i) + offset[2]));
  }
  return byId;
}

function edgeArc(a, b, color, opacity) {
  const src = new THREE.Vector3(...a), dst = new THREE.Vector3(...b);
  const dir = new THREE.Vector3().subVectors(dst, src);
  const len = dir.length();
  if (len < 1e-6) return [];
  let perp = new THREE.Vector3().crossVectors(dir, new THREE.Vector3(0, 1, 0));
  if (perp.length() < 1e-4) perp = new THREE.Vector3().crossVectors(dir, new THREE.Vector3(1, 0, 0));
  perp.normalize().multiplyScalar(len * 0.08);
  const curve = new THREE.QuadraticBezierCurve3(src, new THREE.Vector3().lerpVectors(src, dst, 0.5).add(perp), dst);
  const mat = () => new THREE.MeshBasicMaterial({ color, transparent: true, opacity });
  const tube = new THREE.Mesh(new THREE.TubeGeometry(curve, 16, sz(0.002), 8, false), mat());
  const cone = new THREE.Mesh(new THREE.ConeGeometry(sz(0.008), sz(0.02), 8), mat());
  cone.position.copy(curve.getPointAt(0.75));
  cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), curve.getTangentAt(0.75).normalize());
  return [tube, cone];
}

/**
 * Draw the graph of the frame on stage on its cloud, and in world mode every stacked frame's nodes.
 *
 * @param {object} p
 * @param {boolean} p.on  draw the graph; off, only the followed region's path is drawn.
 * @param {string} p.track
 * @param {object} p.stage  the frame on stage: {group, regionsByTrack, graphByTrack, offset}.
 * @param {?object} p.world  the world stack, in world mode.
 * @param {?string} p.focusKey  the followed node; the others fade.
 */
export function buildGraph3D({ on, track, stage, world = null, focusKey = null }) {
  clearGroup(groups.graph3d);
  if (world && focusKey) drawFocusPath(world, focusKey, stage?.i);
  if (!on || !stage?.group) { invalidate(); return; }
  let posAttr = null;
  // The full geometry, since the labels index the full cloud and a view may draw a filtered copy.
  stage.group.traverse((o) => { if (!posAttr && o.isPoints) posAttr = fullGeometryOf(o)?.attributes?.position ?? null; });
  const offset = stage.offset ?? [0, 0, 0];
  const graph = stage.graphByTrack?.[track];
  const bright = (id) => !focusKey || nodeKey(track, id) === focusKey;
  const posOf = new Map();
  if (graph?.nodes && posAttr) {
    const pts = regionPoints(stage.regionsByTrack?.[track]?.labels, posAttr, offset);
    for (const n of graph.nodes) {
      if (!Array.isArray(n.pos)) continue;
      const key = nodeKey(track, n.id);
      const p = n.pos.map((v, k) => v + offset[k]);
      posOf.set(n.id, p);
      const color = colorOf(n);
      let mesh = null;
      const region = pts?.get(n.id);
      if (region && region.length >= HULL_MIN) {
        const sample = region.length > HULL_MAX ? region.filter((_, i) => i % Math.ceil(region.length / HULL_MAX) === 0) : region;
        try {
          mesh = new THREE.Mesh(new ConvexGeometry(sample), new THREE.MeshBasicMaterial({
            color, transparent: true, opacity: bright(n.id) ? 0.3 : 0.05, side: THREE.DoubleSide, depthWrite: false,
          }));
        } catch {
          mesh = null;   // coplanar points have no hull; the ellipsoid stands in
        }
      }
      if (!mesh) {
        mesh = ellipsoid(n, color, bright(n.id) ? 0.6 : 0.08);
        mesh.position.set(...p);
      }
      mesh.renderOrder = 94;
      mesh.userData.nodeKey = key;
      groups.graph3d.add(mesh);
      if (focusKey && key === focusKey) {
        const outline = new THREE.Mesh(mesh.geometry, new THREE.MeshBasicMaterial({ color: FOCUS_COLOR, wireframe: true, transparent: true, opacity: 0.75, depthTest: false }));
        outline.userData.sharedGeometry = true;
        outline.position.copy(mesh.position);
        outline.quaternion.copy(mesh.quaternion);
        outline.scale.copy(mesh.scale);
        outline.renderOrder = 96;
        groups.graph3d.add(outline);
      }
    }
    const edges = graph.edges || [];
    if (edges.length <= MAX_EDGES) {
      for (const e of edges) {
        const a = posOf.get(e.src), b = posOf.get(e.dst);
        if (!a || !b) continue;
        const touched = bright(e.src) || bright(e.dst);
        for (const obj of edgeArc(a, b, RELATION_COLORS[e.relation] ?? 0x00ff88, touched ? 0.75 : 0.05)) groups.graph3d.add(obj);
      }
    }
  }
  if (world) drawThreads(world, track, bright);
  invalidate();
}

/** World mode: each node's centroid in every stacked frame, as dots joined in time order. */
function drawThreads(world, track, bright) {
  const byId = new Map();
  for (const wf of world.frames) {
    for (const n of wf.graphByTrack[track]?.nodes || []) {
      if (!Array.isArray(n.pos)) continue;
      if (!byId.has(n.id)) byId.set(n.id, { color: colorOf(n), pts: [] });
      byId.get(n.id).pts.push(new THREE.Vector3(n.pos[0] + wf.offset[0], n.pos[1] + wf.offset[1], n.pos[2] + wf.offset[2]));
    }
  }
  for (const [id, e] of byId) {
    const op = bright(id) ? 0.85 : 0.08;
    if (e.pts.length >= 2) {
      const thread = new THREE.Line(new THREE.BufferGeometry().setFromPoints(e.pts),
        new THREE.LineBasicMaterial({ color: e.color, transparent: true, opacity: op * 0.7, depthTest: false }));
      thread.renderOrder = 93;
      thread.userData.nodeKey = nodeKey(track, id);
      groups.graph3d.add(thread);
    }
    for (const p of e.pts) {
      const dot = new THREE.Mesh(new THREE.SphereGeometry(sz(0.0045), 8, 8),
        new THREE.MeshBasicMaterial({ color: e.color, transparent: true, opacity: op, depthTest: false }));
      dot.position.copy(p);
      dot.renderOrder = 93;
      groups.graph3d.add(dot);
    }
  }
}

/**
 * The followed region's path through the stack: its centroid in each stacked frame, solid up to the
 * frame on stage and faint after it, each piece over a white halo so it reads on any tissue.
 */
function drawFocusPath(world, focusKey, stageFrame) {
  const at = focusKey.lastIndexOf(':');
  const track = focusKey.slice(0, at), id = Number(focusKey.slice(at + 1));
  const pts = [];
  let here = -1;
  for (const wf of world.frames) {
    const n = wf.graphByTrack?.[track]?.nodes?.find((m) => m.id === id);
    if (!Array.isArray(n?.pos)) continue;
    pts.push(new THREE.Vector3(n.pos[0] + wf.offset[0], n.pos[1] + wf.offset[1], n.pos[2] + wf.offset[2]));
    if (wf.i <= stageFrame) here = pts.length - 1;
  }
  const mat = (color, opacity) => new THREE.MeshBasicMaterial({ color, transparent: true, opacity, depthTest: false });
  const tube = (part, radius, opacity) => {
    if (part.length < 2) return;
    const curve = new THREE.CatmullRomCurve3(part);
    for (const [r, color, order] of [[radius * 1.7, 0xffffff, 95], [radius, PATH_COLOR, 96]]) {
      const m = new THREE.Mesh(new THREE.TubeGeometry(curve, Math.max(8, part.length * 8), sz(r), 8, false), mat(color, opacity));
      m.renderOrder = order;
      groups.graph3d.add(m);
    }
  };
  tube(pts.slice(0, here + 1), 0.0028, 1.0);
  tube(pts.slice(Math.max(0, here)), 0.0014, 0.35);
  pts.forEach((p, k) => {
    const now = k === here;
    const r = now ? 0.009 : 0.0045;
    for (const [rr, color, order] of [[r * 1.45, 0xffffff, 97], [r, now ? 0xe07f00 : PATH_COLOR, 98]]) {
      const dot = new THREE.Mesh(new THREE.SphereGeometry(sz(rr), 14, 14), mat(color, k <= here ? 1.0 : 0.35));
      dot.position.copy(p);
      dot.renderOrder = order;
      groups.graph3d.add(dot);
    }
  });
}

/** Join two node centroids with a shaft, a marker at each end and the nodes' names. */
export function drawEventConnector(a, b, labelA, labelB) {
  clearGroup(groups.highlight);
  const from = new THREE.Vector3(...a), to = new THREE.Vector3(...b);
  const dir = new THREE.Vector3().subVectors(to, from);
  const len = dir.length();
  if (len < 1e-6) return;
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(sz(0.0035), sz(0.0035), len, 8),
    new THREE.MeshBasicMaterial({ color: 0x334050, transparent: true, opacity: 0.85, depthTest: false }));
  shaft.position.copy(from).addScaledVector(dir.clone().normalize(), len / 2);
  shaft.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  shaft.renderOrder = 90;
  groups.highlight.add(shaft);
  for (const [p, c, text] of [[from, FOCUS_COLOR, labelA], [to, PARTNER_COLOR, labelB]]) {
    const marker = new THREE.Mesh(new THREE.SphereGeometry(sz(0.009), 12, 12), new THREE.MeshBasicMaterial({ color: c, depthTest: false }));
    marker.position.copy(p);
    marker.renderOrder = 91;
    groups.highlight.add(marker);
    const label = makeLabelSprite(text, hexColor(c), sz(0.045));
    label.position.copy(p).add(new THREE.Vector3(0, sz(0.045), 0));
    groups.highlight.add(label);
  }
  invalidate();
}

/** Outline the drawn node `key` in dark wire; nothing when the graph is not on the cloud. */
export function setHover3D(key) {
  clearGroup(groups.hover);
  if (!key) return;
  for (const m of groups.graph3d.children) {
    if (!m.isMesh || m.userData.nodeKey !== key) continue;
    const o = new THREE.Mesh(m.geometry, new THREE.MeshBasicMaterial({ color: 0x1d2530, wireframe: true, transparent: true, opacity: 0.7, depthTest: false }));
    o.userData.sharedGeometry = true;
    o.position.copy(m.position);
    o.quaternion.copy(m.quaternion);
    o.scale.copy(m.scale);
    o.renderOrder = 99;
    groups.hover.add(o);
  }
}
