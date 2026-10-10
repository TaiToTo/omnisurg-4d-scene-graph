// Hold the 3D view: the scene, camera, orbit controls and renderer, drawn only when something changed.
// Sizes of what the page adds to the scene (camera marks, node shapes, labels) scale with the cloud on
// stage, since a depth model's units are arbitrary.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { viewBasisFrom } from './view_basis.js';

export const scene = new THREE.Scene();
scene.background = new THREE.Color('#e9edf2');
scene.add(new THREE.AmbientLight(0xffffff, 1));

export const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 100);
camera.position.set(0, 0, 1.5);

/** The groups the page draws into, from back to front. */
export const groups = {
  points: new THREE.Group(),
  cameraViz: new THREE.Group(),
  highlight: new THREE.Group(),
  graph3d: new THREE.Group(),
  hover: new THREE.Group(),
};
for (const g of Object.values(groups)) scene.add(g);

let renderer = null;
let controls = null;
let viewport = null;
let needsRender = true;
let sceneScale = 1;

/** Ask for one redraw. */
export function invalidate() { needsRender = true; }

/**
 * Create the renderer and the orbit controls on `canvas`, sized to `container`.
 *
 * @returns {boolean} false when the browser cannot draw WebGL; the caller says so on the page.
 */
export function initStage(canvas, container) {
  viewport = container;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  } catch {
    return false;
  }
  // Two device pixels per CSS pixel at most: a high-density screen otherwise quadruples the fill cost.
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.screenSpacePanning = true;
  controls.addEventListener('change', invalidate);
  // A context the browser drops and restores needs a redraw nobody else will ask for.
  canvas.addEventListener('webglcontextlost', (ev) => ev.preventDefault());
  canvas.addEventListener('webglcontextrestored', invalidate);
  window.addEventListener('resize', invalidate);
  return true;
}

function resizeToViewport() {
  const w = viewport.clientWidth, h = viewport.clientHeight;
  if (!w || !h) return;
  const cur = renderer.getSize(new THREE.Vector2());
  if (cur.x === w && cur.y === h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

/** Start the loop that draws a frame whenever the scene or the camera has changed. */
export function startRenderLoop() {
  let last = performance.now();
  const tick = () => {
    requestAnimationFrame(tick);
    const now = performance.now();
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (controls.update(dt)) needsRender = true;
    if (!needsRender) return;
    needsRender = false;
    resizeToViewport();
    renderer.render(scene, camera);
  };
  tick();
}

// Annotation sizes are given for a cloud of this radius; the clamp keeps one odd cloud from blowing them up.
const ANNOTATION_REF_RADIUS = 0.55;
const SCALE_CLAMP = [0.5, 4];
let onScale = () => {};

/** Measure the clouds on stage and set the annotation scale from them. Call before building annotations. */
export function measureSceneScale() {
  const r = robustSpan(new THREE.Vector3(), null)?.radius ?? 0;
  sceneScale = r > 0 ? Math.min(SCALE_CLAMP[1], Math.max(SCALE_CLAMP[0], r / ANNOTATION_REF_RADIUS)) : 1;
  onScale(sceneScale);
  return sceneScale;
}

/** Call `fn(scale)` whenever the annotation scale changes. */
export function onSceneScale(fn) { onScale = fn; }

/** An annotation size, given for the reference cloud, at the scale of the cloud on stage. */
export function sz(v) { return v * sceneScale; }

// The view frames this share of the points; a monocular reconstruction throws a long tail off the
// frame's border, and fitting all of it shrinks the tissue to a fraction of the view.
const FIT_QUANTILE = 0.97;
const FIT_SAMPLES = 4000;
// How far beyond the tissue the view reaches to keep the camera marks in it, in cloud radii.
const CAMERA_FIT_CAP = 2.2;
const Y_UP = new THREE.Vector3(0, 1, 0);

/**
 * Frame the clouds on stage from behind the recorded camera, looking where it looked.
 *
 * @param {?object} frame  the clip.json frame whose camera gives the direction; null looks down -Z.
 * @param {{withCameraMarks?: boolean}} [opts]  also keep the camera marks in view.
 */
export function resetView(frame, { withCameraMarks = true } = {}) {
  const b = frame && viewBasisFrom(frame.camera_forward, frame.camera_up);
  const basis = b && { fwd: new THREE.Vector3(...b.fwd), up: new THREE.Vector3(...b.up), right: new THREE.Vector3(...b.right) };
  const axes = basis ? [basis.right, basis.up, basis.fwd]
    : [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
  const centre = new THREE.Vector3();
  const span = robustSpan(centre, axes);
  let dist = 1.5;
  if (span) {
    const { lo, hi, radius } = span;
    // Widen the box to hold the camera marks, each clamped to a few radii of the tissue.
    if (withCameraMarks && groups.cameraViz.children.length) {
      const viz = new THREE.Box3().setFromObject(groups.cameraViz);
      if (!viz.isEmpty()) {
        viz.min.max(centre.clone().subScalar(radius * CAMERA_FIT_CAP));
        viz.max.min(centre.clone().addScalar(radius * CAMERA_FIT_CAP));
        const v = new THREE.Vector3();
        for (let i = 0; i < 8; i++) {
          v.set(i & 1 ? viz.max.x : viz.min.x, i & 2 ? viz.max.y : viz.min.y, i & 4 ? viz.max.z : viz.min.z).sub(centre);
          for (let a = 0; a < 3; a++) {
            const t = v.dot(axes[a]);
            lo[a] = Math.min(lo[a], t);
            hi[a] = Math.max(hi[a], t);
          }
        }
      }
    }
    // Centre on the box and fit its width and height to the view separately.
    for (let a = 0; a < 3; a++) centre.addScaledVector(axes[a], (lo[a] + hi[a]) / 2);
    const w = hi[0] - lo[0], h = hi[1] - lo[1], d = hi[2] - lo[2];
    const aspect = (viewport?.clientWidth || 1) / (viewport?.clientHeight || 1);
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
 * Point the camera's up and the orbit axis the same way. OrbitControls keeps the rotation from its up to
 * +Y in two private fields set once; changing `camera.up` alone rolls the view while the drag still turns
 * about the old axis.
 *
 * @returns {boolean} false when those fields are not where this version of three keeps them; the view
 *          then stays unrolled, so the mouse still turns it the way it looks.
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

/** A strided sample of every cloud on stage, in world coordinates. */
function sampleClouds() {
  scene.updateMatrixWorld(true);
  const meshes = [];
  groups.points.traverse((o) => { if (o.isPoints && o.geometry?.attributes?.position) meshes.push(o); });
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
 * Where the clouds are and how far they reach along each of three axes, each trimmed at both ends.
 *
 * @param {THREE.Vector3} outCentre  set to the sample's mean.
 * @param {?THREE.Vector3[]} axes  the view's axes; world axes when null.
 * @returns {?{lo:number[], hi:number[], radius:number}}
 */
function robustSpan(outCentre, axes) {
  const s = sampleClouds();
  if (!s) return null;
  const { pts, n } = s;
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < n; i++) { cx += pts[i * 3]; cy += pts[i * 3 + 1]; cz += pts[i * 3 + 2]; }
  outCentre.set(cx / n, cy / n, cz / n);
  const ax = axes ?? [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
  const tail = (1 - FIT_QUANTILE) / 2;
  const loIdx = Math.min(n - 1, Math.floor(n * tail));
  const hiIdx = Math.min(n - 1, Math.floor(n * (1 - tail)));
  const lo = [0, 0, 0], hi = [0, 0, 0];
  const proj = new Float64Array(n);
  for (let a = 0; a < 3; a++) {
    for (let i = 0; i < n; i++) {
      proj[i] = (pts[i * 3] - outCentre.x) * ax[a].x + (pts[i * 3 + 1] - outCentre.y) * ax[a].y
        + (pts[i * 3 + 2] - outCentre.z) * ax[a].z;
    }
    proj.sort();
    lo[a] = proj[loIdx];
    hi[a] = proj[hiIdx];
  }
  const d = new Float64Array(n);
  for (let i = 0; i < n; i++) d[i] = Math.hypot(pts[i * 3] - outCentre.x, pts[i * 3 + 1] - outCentre.y, pts[i * 3 + 2] - outCentre.z);
  d.sort();
  return { lo, hi, radius: d[Math.min(n - 1, Math.floor(n * FIT_QUANTILE))] };
}

/** A text label that always faces the camera, `height` scene units tall. */
export function makeLabelSprite(text, color, height = 0.05) {
  const cv = document.createElement('canvas');
  cv.width = 320; cv.height = 64;
  const ctx = cv.getContext('2d');
  ctx.font = '700 38px system-ui, Arial, sans-serif';
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
 * Remove everything under a group and free it. A mesh flagged `userData.sharedGeometry` keeps its
 * geometry, which the cloud cache owns.
 */
export function clearGroup(group) {
  while (group.children.length) {
    const child = group.children[0];
    group.remove(child);
    child.traverse?.((obj) => {
      if (obj.geometry && !obj.userData.sharedGeometry) obj.geometry.dispose();
      if (obj.material) [].concat(obj.material).forEach((m) => { m.map?.dispose(); m.dispose(); });
    });
  }
  invalidate();
}
