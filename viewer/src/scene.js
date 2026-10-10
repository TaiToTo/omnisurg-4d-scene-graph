// Run the 3D view: renderer, camera, orbit controls and a render loop that
// draws only when something changed, and frame the clouds on screen.
//
// Annotation sizes (camera marks, node glyphs, connectors) follow the cloud's
// size: depth models are scale-invariant, so two models put one scene at
// different scales, and a fixed size vanishes on the larger one.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { state } from './state.js';
import { viewBasisFrom } from './lib/view_basis.js';

export const canvas = document.getElementById('gl');
export const viewport = document.getElementById('viewport');
const statusEl = document.getElementById('status');

export const scene = new THREE.Scene();
scene.background = new THREE.Color('#e9edf2');
scene.add(new THREE.AmbientLight(0xffffff, 1));
for (const g of [state.pointsGroup, state.cameraVizGroup, state.highlightGroup, state.graph3dGroup]) scene.add(g);

export const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 100);
camera.position.set(0, 0, 1.5);

export const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
// Capped at 2: a higher ratio multiplies the fill cost for no visible gain.
renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));

export const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
// Right-drag pans in screen space; the default pans along a ground plane.
controls.screenSpacePanning = true;

let needsRender = true;

/** Ask for one more frame to be drawn. */
export function invalidate() { needsRender = true; }

controls.addEventListener('change', invalidate);
// A restored WebGL context draws nothing until asked.
canvas.addEventListener('webglcontextlost', (ev) => ev.preventDefault());
canvas.addEventListener('webglcontextrestored', invalidate);

function resizeToViewport() {
  const w = viewport.clientWidth, h = viewport.clientHeight;
  if (!w || !h) return;
  const cur = renderer.getSize(new THREE.Vector2());
  if (cur.x === w && cur.y === h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

/** Start the loop that draws whenever the view was invalidated or the camera is still easing. */
export function startRenderLoop() {
  const tick = () => {
    requestAnimationFrame(tick);
    if (controls.update()) needsRender = true;
    if (!needsRender) return;
    needsRender = false;
    resizeToViewport();
    renderer.render(scene, camera);
  };
  window.addEventListener('resize', invalidate);
  tick();
}

// The cloud radius at which annotations have their reference size, and the
// range the scale is held in so one odd reconstruction cannot blow it up.
const ANNOTATION_REF_RADIUS = 0.55;
const SCALE_CLAMP = [0.5, 4];
let scaleHook = null;

/**
 * Measure `state.sceneScale` from the clouds on screen.
 *
 * Call after the clouds are added and before anything sized by `sz` is built.
 */
export function measureSceneScale() {
  const r = robustSpan(new THREE.Vector3(), null)?.radius ?? 0;
  state.sceneScale = r > 0 ? Math.min(SCALE_CLAMP[1], Math.max(SCALE_CLAMP[0], r / ANNOTATION_REF_RADIUS)) : 1;
  scaleHook?.(state.sceneScale);
  return state.sceneScale;
}

/** Register a function that follows the scene scale (the picking radius). */
export function onSceneScale(fn) { scaleHook = fn; }

/** Return annotation size `v`, given in reference units, at the current scale. */
export function sz(v) { return v * state.sceneScale; }

// The share of a cloud's points the reset view must contain. A monocular
// reconstruction throws a long tail of points off the frame border, and
// framing all of them shrinks the tissue to a fraction of the view.
const FIT_QUANTILE = 0.97;
const FIT_SAMPLES = 4000;
// How far beyond the tissue the frame reaches to keep the camera path in view,
// in cloud radii, so one bad pose cannot shrink the tissue to a dot.
const CAMERA_FIT_CAP = 2.2;
const Y_UP = new THREE.Vector3(0, 1, 0);

function viewBasis() {
  const frames = state.geoManifest?.frames ?? state.manifest?.frames;
  const m = frames?.[state.frame] ?? frames?.[0];
  const b = m && viewBasisFrom(m.camera_forward_glb, m.camera_up_glb);
  return b && { fwd: new THREE.Vector3(...b.fwd), up: new THREE.Vector3(...b.up), right: new THREE.Vector3(...b.right) };
}

/**
 * Frame the clouds on screen from behind the endoscope, looking the way it looked.
 *
 * Without a usable pose the view looks down -Z.
 */
export function resetView() {
  const centre = new THREE.Vector3();
  const basis = viewBasis();
  const axes = basis ? [basis.right, basis.up, basis.fwd]
    : [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
  const span = robustSpan(centre, basis);
  let dist = 1.5;
  if (span) {
    const { lo, hi, radius } = span;
    // Include the camera marks, clamped around the tissue, unless one region is followed alone.
    if (state.cameraVizGroup.visible && state.cameraVizGroup.children.length && !state.isolateFocus) {
      const viz = new THREE.Box3().setFromObject(state.cameraVizGroup);
      if (!viz.isEmpty()) {
        const reach = radius * CAMERA_FIT_CAP;
        viz.min.max(centre.clone().subScalar(reach));
        viz.max.min(centre.clone().addScalar(reach));
        const c = viz.getCenter(new THREE.Vector3());
        const h = viz.getSize(new THREE.Vector3()).multiplyScalar(0.5);
        const v = new THREE.Vector3();
        for (let i = 0; i < 8; i++) {
          v.set(c.x + (i & 1 ? h.x : -h.x), c.y + (i & 2 ? h.y : -h.y), c.z + (i & 4 ? h.z : -h.z)).sub(centre);
          for (let a = 0; a < 3; a++) {
            const t = v.dot(axes[a]);
            lo[a] = Math.min(lo[a], t);
            hi[a] = Math.max(hi[a], t);
          }
        }
      }
    }
    for (let a = 0; a < 3; a++) centre.addScaledVector(axes[a], (lo[a] + hi[a]) / 2);
    // Fit the box's width and height to the frustum separately: a bounding
    // sphere of a diagonal scene is far larger than the frame needs.
    const w = hi[0] - lo[0], h = hi[1] - lo[1], d = hi[2] - lo[2];
    const aspect = (viewport.clientWidth || 1) / (viewport.clientHeight || 1);
    const halfFovY = (camera.fov / 2) * Math.PI / 180;
    const halfFovX = Math.atan(Math.tan(halfFovY) * aspect);
    dist = Math.max(0.05, Math.max(h / 2 / Math.tan(halfFovY), w / 2 / Math.tan(halfFovX)) * 1.06 + d / 2);
  }
  if (basis && setCameraUp(basis.up)) {
    camera.position.copy(centre).addScaledVector(basis.fwd, -dist);
  } else {
    setCameraUp(Y_UP);
    camera.position.set(centre.x, centre.y, centre.z + dist);
  }
  controls.target.copy(centre);
  controls.update();
  invalidate();
}

/**
 * Point the camera's up and the orbit axis the same way.
 *
 * OrbitControls caches its up rotation in `_quat` and `_quatInverse` when it is
 * built and offers no setter; moving only `camera.up` makes a drag turn the
 * scene about the old axis. Returns false, leaving the view unrolled, when
 * those fields are not where this version of three keeps them.
 */
function setCameraUp(up) {
  if (!controls._quat?.isQuaternion || !controls._quatInverse?.isQuaternion) {
    camera.up.copy(Y_UP);
    return false;
  }
  camera.up.copy(up);
  controls._quat.setFromUnitVectors(up, Y_UP);
  controls._quatInverse.copy(controls._quat).invert();
  return true;
}

// A strided sample of every cloud in the scene, hidden stacked frames
// included, so World mode frames the whole stack.
function sampleClouds() {
  scene.updateMatrixWorld(true);
  const meshes = [];
  state.pointsGroup.traverse((o) => { if (o.isPoints && o.geometry?.attributes?.position) meshes.push(o); });
  if (!meshes.length) return null;
  const total = meshes.reduce((n, m) => n + m.geometry.attributes.position.count, 0);
  const step = Math.max(1, Math.floor(total / FIT_SAMPLES));
  const pts = [];
  const v = new THREE.Vector3();
  for (const m of meshes) {
    const pos = m.geometry.attributes.position;
    for (let i = 0; i < pos.count; i += step) {
      v.fromBufferAttribute(pos, i).applyMatrix4(m.matrixWorld);
      pts.push(v.x, v.y, v.z);
    }
  }
  return pts.length ? { pts, n: pts.length / 3 } : null;
}

/**
 * Measure where the clouds are and how far they reach along the view's own axes.
 *
 * Each axis is trimmed at both ends, so an outlier on one side does not pad the other.
 *
 * @returns {?{lo: number[], hi: number[], radius: number}} offsets from `outCentre`, which is written.
 */
function robustSpan(outCentre, basis) {
  const s = sampleClouds();
  if (!s) return null;
  const { pts, n } = s;
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < n; i++) { cx += pts[i * 3]; cy += pts[i * 3 + 1]; cz += pts[i * 3 + 2]; }
  outCentre.set(cx / n, cy / n, cz / n);
  const axes = basis ? [basis.right, basis.up, basis.fwd]
    : [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
  const tail = (1 - FIT_QUANTILE) / 2;
  const loIdx = Math.min(n - 1, Math.floor(n * tail));
  const hiIdx = Math.min(n - 1, Math.floor(n * (1 - tail)));
  const lo = [0, 0, 0], hi = [0, 0, 0];
  const proj = new Float64Array(n);
  const v = new THREE.Vector3();
  for (let a = 0; a < 3; a++) {
    for (let i = 0; i < n; i++) {
      v.set(pts[i * 3] - outCentre.x, pts[i * 3 + 1] - outCentre.y, pts[i * 3 + 2] - outCentre.z);
      proj[i] = v.dot(axes[a]);
    }
    proj.sort();
    lo[a] = proj[loIdx];
    hi[a] = proj[hiIdx];
  }
  const d = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    d[i] = Math.hypot(pts[i * 3] - outCentre.x, pts[i * 3 + 1] - outCentre.y, pts[i * 3 + 2] - outCentre.z);
  }
  d.sort();
  return { lo, hi, radius: d[Math.min(n - 1, Math.floor(n * FIT_QUANTILE))] };
}

/** Show `msg` on the status line. */
export function setStatus(msg) {
  if (statusEl) statusEl.textContent = msg;
}

/** Return a text label that always faces the camera, `height` world units tall. */
export function makeLabelSprite(text, color, height = 0.05) {
  const cv = document.createElement('canvas');
  cv.width = 320;
  cv.height = 64;
  const ctx = cv.getContext('2d');
  ctx.font = '700 38px Inter, Arial, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.lineWidth = 7;
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.92)';
  ctx.strokeText(text, 160, 34);
  ctx.fillStyle = color;
  ctx.fillText(text, 160, 34);
  const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cv), transparent: true, depthTest: false }));
  spr.scale.set(height * 5, height, 1);
  spr.renderOrder = 101;
  return spr;
}

/**
 * Remove and free everything under `group`.
 *
 * A mesh flagged `userData.sharedGeometry` keeps its geometry: it belongs to
 * the GLB cache or to another mesh.
 */
export function clearGroup(group) {
  while (group.children.length) {
    const child = group.children[0];
    group.remove(child);
    child.traverse?.((obj) => {
      if (obj.geometry && !obj.userData.sharedGeometry) obj.geometry.dispose();
      for (const m of [obj.material].flat()) {
        m?.map?.dispose();
        m?.dispose();
      }
    });
  }
  invalidate();
}
