// Stack a sample of the clip's frames in one space (World mode), spotlight
// one of them, and draw the camera's path and the world axes.
//
// A frame's cloud is stored centred on its own `glb_centroid`; stacking moves
// it by its centroid minus the first stacked frame's, the stack's origin.
import * as THREE from 'three';
import { state } from './state.js';
import { clearGroup, invalidate, makeLabelSprite, measureSceneScale, setStatus, sz } from './scene.js';
import { fetchJSON, glbURL, loadGLB, overlayURL } from './data.js';
import { adaptOverlays } from './lib/overlays.js';
import { nearestStacked, worldIndices } from './lib/sampling.js';
import { applySegBlend } from './cloud.js';

// Spotlight draws the frame in focus and two neighbours each side, fading:
// more faint layers composite into a curtain over the frame in focus.
const SPOT_OPACITY = [1.0, 0.10, 0.05];
const SPOT_SIZE = [1.35, 0.8, 0.7];
const POSE_COLOR = 0x0899b4;
const LIT_COLOR = 0xe07f00;

/**
 * Load and stack the sampled frames: cloud, segmentation and graph of every track, and the hierarchy.
 *
 * @returns {Promise<?object>} the stack, or null when the load went stale or the manifest has no poses.
 */
export async function enterWorld() {
  const seq = state.loadSeq;
  const stale = () => state.loadSeq !== seq || state.mode !== 'world';
  const clip = state.clip;
  const indices = worldIndices(state.maxFrame, state.worldFrames);
  setStatus(`Stacking ${indices.length} frames in world coordinates…`);

  // Fetch every sampled frame's files at once.
  const frames = (await Promise.all(indices.map(async (i) => {
    let gltf;
    try {
      gltf = await loadGLB(glbURL(clip.path, i, clip.geometry));
    } catch {
      return null;
    }
    const [segs, graphs, hierarchy] = await Promise.all([
      Promise.all(clip.tracks.map((t) => fetchJSON(overlayURL(clip.path, 'seg_frame', i, t)))),
      Promise.all(clip.tracks.map((t) => fetchJSON(overlayURL(clip.path, 'graph_frame', i, t)))),
      clip.hierarchy ? fetchJSON(overlayURL(clip.path, 'hierarchy_frame', i)) : null,
    ]);
    return {
      i, gltf, hierarchy,
      segByTrack: Object.fromEntries(clip.tracks.map((t, k) => [t, segs[k]])),
      graphByTrack: Object.fromEntries(clip.tracks.map((t, k) => [t, graphs[k]])),
    };
  }))).filter(Boolean);
  if (stale()) return null;
  const meta = state.geoManifest?.frames;
  if (!frames.length || !meta?.[frames[0].i]?.glb_centroid) {
    setStatus('World mode is unavailable: no stacked frame loaded, or the manifest has no poses.');
    return null;
  }

  // Place each cloud, fit its overlays and paint it.
  clearGroup(state.pointsGroup);
  clearGroup(state.cameraVizGroup);
  const origin = meta[frames[0].i].glb_centroid;
  const world = { indices: frames.map((f) => f.i), origin, frames: [] };
  let unaligned = 0;
  for (const fr of frames) {
    const c = meta[fr.i].glb_centroid;
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
    state.pointsGroup.add(group);
    const fit = adaptOverlays({
      segByTrack: fr.segByTrack, graphByTrack: fr.graphByTrack, group, manifest: state.manifest, source: clip.geometry,
    });
    if (!fit.aligned) unaligned++;
    applySegBlend(group, fit.segByTrack, state.segVisible ? state.tracks : []);
    world.frames.push({ i: fr.i, offset, group, hierarchy: fr.hierarchy, segByTrack: fit.segByTrack, graphByTrack: fit.graphByTrack });
  }

  // Size and draw the camera path and the axes from the stacked clouds.
  measureSceneScale();
  buildCameraTrail(world, meta);
  buildWorldAxes();
  state.segAligned = unaligned === 0;
  setStatus(`World mode: ${world.indices.length} of ${state.maxFrame + 1} frames stacked`
    + (unaligned ? ` — the overlays of ${unaligned} could not be fitted to their cloud` : ''));
  invalidate();
  return world;
}

/** Remove the stack. */
export function exitWorld() {
  clearGroup(state.pointsGroup);
  clearGroup(state.cameraVizGroup);
  clearGroup(state.graph3dGroup);
  state.world = null;
}

/**
 * Emphasise the stacked frame nearest `frameIdx`, and return it.
 *
 * Spotlight fades its neighbours and hides the rest; Overlay shows every frame
 * alike. While a region is followed alone, only the frame in focus is drawn:
 * the region's path says where it has been, and earlier frames would show each
 * frame's own depth estimate rather than motion.
 */
export function focusWorldFrame(world, frameIdx, view = 'focus') {
  const focus = nearestStacked(frameIdx, world.indices);
  const overlay = view === 'overlay';
  const rank = world.indices.indexOf(focus);
  const iso = state.isolateFocus && state.focusNode;
  for (const fr of world.frames) {
    const isCur = fr.i === focus;
    const away = Math.abs(world.indices.indexOf(fr.i) - rank);
    fr.group.visible = overlay || (iso ? isCur : away < SPOT_OPACITY.length);
    fr.group.traverse((o) => {
      if (!o.isPoints || !o.material) return;
      if (o.userData.baseSize === undefined) o.userData.baseSize = o.material.size;
      o.material.transparent = true;
      o.material.opacity = overlay ? 0.9 : (isCur ? 1.0 : SPOT_OPACITY[away] ?? 0);
      o.material.size = o.userData.baseSize * (overlay ? 1 : (iso ? SPOT_SIZE[0] : SPOT_SIZE[away] ?? 1));
      o.material.depthWrite = overlay || isCur;
      o.material.needsUpdate = true;
    });
  }
  // The camera marks follow: the frame in focus is lit and larger; Overlay lights them all.
  for (const obj of state.cameraVizGroup.children) {
    const fid = obj.userData.cameraFrameId;
    if (fid === undefined) continue;
    const isCur = fid === focus;
    obj.material.color.set(overlay || isCur ? LIT_COLOR : 0x7c8798);
    obj.material.opacity = overlay || isCur ? (isCur ? 1.0 : 0.85) : 0.35;
    obj.scale.setScalar(!overlay && isCur ? 1.7 : 1.0);
  }
  invalidate();
  return focus;
}

// A pose pyramid per stacked frame, joined by a tube with direction cones.
function buildCameraTrail(world, meta) {
  const APEX = sz(0.03), HALF = sz(0.02);
  const P = [[0, 0, -APEX], [-HALF, -HALF, 0], [HALF, -HALF, 0], [HALF, HALF, 0], [-HALF, HALF, 0]];
  const EDGES = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]];
  const pyramid = new THREE.BufferGeometry().setFromPoints(EDGES.flatMap(([a, b]) => [new THREE.Vector3(...P[a]), new THREE.Vector3(...P[b])]));
  const o = world.origin;
  const trail = [];
  for (const fr of world.frames) {
    const m = meta[fr.i];
    if (!m?.camera_pos_glb || !m?.camera_forward_glb) continue;
    const p = new THREE.Vector3(m.camera_pos_glb[0] - o[0], m.camera_pos_glb[1] - o[1], m.camera_pos_glb[2] - o[2]);
    trail.push(p);
    const pyr = new THREE.LineSegments(pyramid.clone(), new THREE.LineBasicMaterial({
      color: 0x7c8798, transparent: true, opacity: 0.6, depthTest: false,
    }));
    pyr.position.copy(p);
    pyr.lookAt(p.x + m.camera_forward_glb[0], p.y + m.camera_forward_glb[1], p.z + m.camera_forward_glb[2]);
    pyr.renderOrder = 100;
    pyr.userData.cameraFrameId = fr.i;
    state.cameraVizGroup.add(pyr);
  }
  pyramid.dispose();
  if (trail.length < 2) return;
  const mat = () => new THREE.MeshBasicMaterial({ color: POSE_COLOR, transparent: true, opacity: 0.9, depthTest: false });
  // A tube rather than a line: WebGL draws every line one pixel wide.
  const tube = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(trail), Math.max(32, trail.length * 8), sz(0.0018), 8, false), mat());
  tube.renderOrder = 100;
  state.cameraVizGroup.add(tube);
  for (let k = 0; k + 1 < trail.length; k++) {
    const dir = new THREE.Vector3().subVectors(trail[k + 1], trail[k]);
    if (dir.length() < 1e-5) continue;
    const cone = new THREE.Mesh(new THREE.ConeGeometry(sz(0.006), sz(0.016), 8), mat());
    cone.position.copy(trail[k]).addScaledVector(dir, 0.5);
    cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
    cone.renderOrder = 100;
    state.cameraVizGroup.add(cone);
  }
  const label = makeLabelSprite('camera path', `#${POSE_COLOR.toString(16).padStart(6, '0')}`, sz(0.045));
  label.position.set(trail[0].x, trail[0].y + sz(0.055), trail[0].z);
  state.cameraVizGroup.add(label);
}

// The world's x, y and z at the stack's origin.
function buildWorldAxes() {
  const len = sz(0.16);
  for (const [dir, color, name] of [[[1, 0, 0], 0xe05c5c, 'x'], [[0, 1, 0], 0x58c96b, 'y'], [[0, 0, 1], 0x5b8ce0, 'z']]) {
    const tip = dir.map((v) => v * len);
    const arm = new THREE.Mesh(new THREE.CylinderGeometry(sz(0.003), sz(0.003), len, 6),
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, depthTest: false }));
    arm.position.set(tip[0] / 2, tip[1] / 2, tip[2] / 2);
    arm.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(...dir));
    arm.renderOrder = 99;
    state.cameraVizGroup.add(arm);
    const label = makeLabelSprite(name, `#${color.toString(16).padStart(6, '0')}`, sz(0.04));
    label.position.set(tip[0] * 1.18, tip[1] * 1.18, tip[2] * 1.18);
    state.cameraVizGroup.add(label);
  }
  const origin = makeLabelSprite('world origin', '#334050', sz(0.038));
  origin.position.set(0, -sz(0.05), 0);
  state.cameraVizGroup.add(origin);
}
