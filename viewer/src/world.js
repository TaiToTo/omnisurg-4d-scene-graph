// Stack a sample of a clip's frames in one space: each frame's cloud moved by its centroid's offset from
// the first stacked frame's, with the camera's path through them and the world axes. One frame is on
// stage: drawn full, its two neighbours faint either side, the rest hidden; or every frame at once.
import * as THREE from 'three';
import { clearGroup, groups, invalidate, makeLabelSprite, measureSceneScale, sz } from './stage.js';
import { fetchGraph, fetchHierarchy, fetchRegions, loadCloud } from './data.js';
import { cameraInCloudFrame } from './depth_filter.js';

/** How many frames world mode stacks: each is a whole cloud, so a longer clip is sampled more sparsely. */
export const TARGET_FRAMES = 12;
// Opacity and point size of the stage frame and its neighbours, by distance in the stack. Past the list a
// frame is hidden: a dozen faint layers composite into a curtain over the stage frame.
const NEIGHBOUR_OPACITY = [1.0, 0.10, 0.05];
const NEIGHBOUR_SIZE = [1.35, 0.8, 0.7];
const CAMERA_COLOR = 0x0899b4;
const STAGE_COLOR = 0xe07f00;
const AXES = [
  { dir: [1, 0, 0], color: 0xe05c5c, name: 'x' },
  { dir: [0, 1, 0], color: 0x58c96b, name: 'y' },
  { dir: [0, 0, 1], color: 0x5b8ce0, name: 'z' },
];

/** The frames to stack: frame 0, every `stride`-th after it, and the last. */
export function worldIndices(lastFrame, target = TARGET_FRAMES) {
  const stride = Math.max(1, Math.round((lastFrame + 1) / target));
  const out = [0];
  for (let i = stride; i <= lastFrame; i += stride) out.push(i);
  if (out[out.length - 1] !== lastFrame) out.push(lastFrame);
  return out;
}

/** The stacked frame nearest frame `i`. */
export function nearestStacked(i, indices) {
  return indices.reduce((best, k) => (Math.abs(k - i) < Math.abs(best - i) ? k : best), indices[0]);
}

/**
 * Load the stacked frames' clouds, regions and graphs, and put the clouds in the scene.
 *
 * @param {object} clip  the clip's record.
 * @param {(group:THREE.Group, regionsByTrack:object) => void} paint  colours a frame's cloud.
 * @param {() => boolean} stale  true once the page has moved on; the stack is then dropped.
 * @returns {Promise<?object>} the stack: {origin, indices, frames: [{i, offset, group, regionsByTrack,
 *          graphByTrack, hierarchy}]}, or null when stale.
 */
export async function enterWorld(clip, paint, stale) {
  const indices = worldIndices(clip.n_frames - 1);
  const tracks = clip.tracks.map((t) => t.id);
  const loaded = await Promise.all(indices.map(async (i) => {
    const [gltf, regions, graphs, hierarchy] = await Promise.all([
      loadCloud(clip, i),
      Promise.all(tracks.map((t) => fetchRegions(clip, t, i))),
      Promise.all(tracks.map((t) => fetchGraph(clip, t, i))),
      fetchHierarchy(clip, i),
    ]);
    return {
      i, gltf, hierarchy,
      regionsByTrack: Object.fromEntries(tracks.map((t, k) => [t, regions[k]])),
      graphByTrack: Object.fromEntries(tracks.map((t, k) => [t, graphs[k]])),
    };
  }));
  if (stale()) return null;

  clearGroup(groups.points);
  clearGroup(groups.cameraViz);
  const origin = clip.frames[indices[0]].centroid;
  const world = { origin, indices, frames: [] };
  for (const fr of loaded) {
    const c = clip.frames[fr.i].centroid;
    const offset = [c[0] - origin[0], c[1] - origin[1], c[2] - origin[2]];
    const group = new THREE.Group();
    fr.gltf.scene.traverse((obj) => {
      if (!obj.isPoints) return;
      const clone = obj.clone();
      clone.userData.sharedGeometry = true;
      clone.material = obj.material.clone();
      clone.position.set(...offset);
      group.add(clone);
    });
    group.userData.cam = cameraInCloudFrame(clip.frames[fr.i]);
    group.userData.grid = clip.geometry.grid;
    groups.points.add(group);
    paint(group, fr.regionsByTrack);
    world.frames.push({ i: fr.i, offset, group, regionsByTrack: fr.regionsByTrack, graphByTrack: fr.graphByTrack,
      hierarchy: fr.hierarchy });
  }
  measureSceneScale();
  buildCameraPath(world, clip.frames);
  buildWorldAxes();
  invalidate();
  return world;
}

/** Take the stack out of the scene. */
export function exitWorld() {
  clearGroup(groups.points);
  clearGroup(groups.cameraViz);
  clearGroup(groups.graph3d);
}

/**
 * Put the stacked frame nearest frame `i` on stage and fade the others.
 *
 * @param {'spotlight'|'overlay'} view  one frame on stage, or every frame alike.
 * @param {boolean} isolated  a region is followed: only the stage frame's cloud is drawn, since stacking a
 *        region's surface from every frame shows the depth estimates disagreeing, not the region moving.
 * @returns {number} the frame on stage.
 */
export function focusWorldFrame(world, i, view, isolated) {
  const stage = nearestStacked(i, world.indices);
  const overlay = view === 'overlay';
  const rank = world.indices.indexOf(stage);
  for (const fr of world.frames) {
    const away = Math.abs(world.indices.indexOf(fr.i) - rank);
    const isStage = fr.i === stage;
    fr.group.visible = overlay || (isolated ? isStage : away < NEIGHBOUR_OPACITY.length);
    fr.group.traverse((o) => {
      if (!o.isPoints) return;
      if (o.userData.baseSize === undefined) o.userData.baseSize = o.material.size;
      o.material.transparent = true;
      o.material.opacity = overlay ? 0.9 : (NEIGHBOUR_OPACITY[away] ?? 0);
      o.material.size = o.userData.baseSize * (overlay ? 1 : isolated ? NEIGHBOUR_SIZE[0] : (NEIGHBOUR_SIZE[away] ?? 1));
      o.material.depthWrite = overlay || isStage;
      o.material.needsUpdate = true;
    });
  }
  for (const obj of groups.cameraViz.children) {
    const fid = obj.userData.cameraFrame;
    if (fid === undefined) continue;
    const lit = overlay || fid === stage;
    obj.material.color.set(lit ? STAGE_COLOR : 0x7c8798);
    obj.material.opacity = lit ? (fid === stage ? 1.0 : 0.85) : 0.35;
    obj.scale.setScalar(!overlay && fid === stage ? 1.7 : 1.0);
  }
  invalidate();
  return stage;
}

/** A small pyramid at each stacked frame's camera, joined by a tube in time order. */
function buildCameraPath(world, frames) {
  const APEX = sz(0.03), HALF = sz(0.02);
  const P = [[0, 0, -APEX], [-HALF, -HALF, 0], [HALF, -HALF, 0], [HALF, HALF, 0], [-HALF, HALF, 0]];
  const EDGES = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]];
  const geo = new THREE.BufferGeometry().setFromPoints(EDGES.flatMap(([a, b]) => [new THREE.Vector3(...P[a]), new THREE.Vector3(...P[b])]));
  const path = [];
  for (const fr of world.frames) {
    const f = frames[fr.i];
    const p = f.camera_position.map((v, k) => v - world.origin[k]);
    path.push(new THREE.Vector3(...p));
    const pyr = new THREE.LineSegments(geo.clone(), new THREE.LineBasicMaterial({ color: 0x7c8798, transparent: true, opacity: 0.6, depthTest: false }));
    pyr.position.set(...p);
    pyr.lookAt(p[0] + f.camera_forward[0], p[1] + f.camera_forward[1], p[2] + f.camera_forward[2]);
    pyr.renderOrder = 100;
    pyr.userData.cameraFrame = fr.i;
    groups.cameraViz.add(pyr);
  }
  geo.dispose();
  if (path.length < 2) return;
  const mat = () => new THREE.MeshBasicMaterial({ color: CAMERA_COLOR, transparent: true, opacity: 0.9, depthTest: false });
  const tube = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(path), Math.max(32, path.length * 8), sz(0.0018), 8, false), mat());
  tube.renderOrder = 100;
  groups.cameraViz.add(tube);
  for (let k = 0; k + 1 < path.length; k++) {
    const dir = new THREE.Vector3().subVectors(path[k + 1], path[k]);
    if (dir.length() < 1e-5) continue;
    const cone = new THREE.Mesh(new THREE.ConeGeometry(sz(0.006), sz(0.016), 8), mat());
    cone.position.copy(path[k]).addScaledVector(dir, 0.5);
    cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
    cone.renderOrder = 100;
    groups.cameraViz.add(cone);
  }
  const label = makeLabelSprite('camera path', `#${CAMERA_COLOR.toString(16).padStart(6, '0')}`, sz(0.045));
  label.position.set(path[0].x, path[0].y + sz(0.055), path[0].z);
  groups.cameraViz.add(label);
}

/** The world's x, y and z axes at its origin, the first stacked frame's centroid. */
function buildWorldAxes() {
  const len = sz(0.16);
  for (const { dir, color, name } of AXES) {
    const arm = new THREE.Mesh(new THREE.CylinderGeometry(sz(0.003), sz(0.003), len, 6),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, depthTest: false }));
    arm.position.set(...dir.map((v) => v * len / 2));
    arm.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(...dir));
    arm.renderOrder = 99;
    groups.cameraViz.add(arm);
    const label = makeLabelSprite(name, `#${color.toString(16).padStart(6, '0')}`, sz(0.04));
    label.position.set(...dir.map((v) => v * len * 1.18));
    groups.cameraViz.add(label);
  }
  const origin = makeLabelSprite('world origin', '#334050', sz(0.038));
  origin.position.set(0, sz(-0.05), 0);
  groups.cameraViz.add(origin);
}
