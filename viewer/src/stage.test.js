import * as THREE from 'three';
import { describe, expect, it, vi } from 'vitest';
import { camera, clearGroup, groups, initStage, makeLabelSprite, resetView } from './stage.js';

// A renderer that needs no WebGL: resetView reads only the controls and the viewport that initStage sets.
vi.mock('three', async (importOriginal) => ({ ...(await importOriginal()), WebGLRenderer: class { setPixelRatio() {} } }));
const made = vi.hoisted(() => []);
vi.mock('three/addons/controls/OrbitControls.js', async (importOriginal) => {
  const { OrbitControls } = await importOriginal();
  return { OrbitControls: class extends OrbitControls { constructor(...args) { super(...args); made.push(this); } } };
});

describe('a label in the 3D view', () => {
  it('is as wide as its text, so a long name is drawn whole', () => {
    // A canvas whose text is 20 px a character.
    const drawn = [];
    const ctx = {
      measureText: (t) => ({ width: 20 * t.length }),
      strokeText() {},
      fillText: (t, x) => drawn.push(x),
    };
    vi.stubGlobal('document', { createElement: () => ({ width: 300, height: 150, getContext: () => ctx }) });
    const name = 'Gastrointestinal Tract (in Fat)';
    const spr = makeLabelSprite(name, '#000', 0.05);
    const cv = spr.material.map.image;
    expect(cv.width).toBeGreaterThanOrEqual(20 * name.length);
    expect(drawn).toEqual([cv.width / 2]);
    expect(spr.scale.x / spr.scale.y).toBeCloseTo(cv.width / cv.height);
    vi.unstubAllGlobals();
  });
});

describe('framing the view', () => {
  // A square of tissue facing +X, centred on (2, 1, -3).
  const CENTRE = new THREE.Vector3(2, 1, -3);
  function putCloud() {
    clearGroup(groups.points);
    const pts = [];
    for (let k = 0; k < 121; k++) pts.push(new THREE.Vector3(2, (k % 11) / 5, -4 + Math.floor(k / 11) / 5));
    groups.points.add(new THREE.Points(new THREE.BufferGeometry().setFromPoints(pts)));
  }
  // Start the stage on a canvas and a page that only take listeners, with the camera's up at +Y as on load.
  function start() {
    const target = { addEventListener() {} };
    vi.stubGlobal('window', { devicePixelRatio: 1, addEventListener() {} });
    camera.up.set(0, 1, 0);
    initStage({ ...target, getRootNode: () => target, style: {} }, { clientWidth: 400, clientHeight: 300 });
    vi.unstubAllGlobals();
    return made.at(-1);
  }
  const expectNear = (v, xyz) => v.toArray().forEach((x, k) => expect(x).toBeCloseTo(xyz[k], 6));
  const facingX = { camera_forward: [1, 0, 0], camera_up: [0, 0, 1] };

  it('puts the camera behind the scene along the frame\'s forward, looking along it with the frame\'s up', () => {
    putCloud();
    const controls = start();
    resetView(facingX);
    expectNear(camera.position.clone().sub(CENTRE).normalize(), [-1, 0, 0]);
    expectNear(camera.getWorldDirection(new THREE.Vector3()), [1, 0, 0]);
    expectNear(camera.up, [0, 0, 1]);
    // The orbit turns about that up: the controls rotate it to their +Y and back.
    expectNear(new THREE.Vector3(0, 0, 1).applyQuaternion(controls._quat), [0, 1, 0]);
    expectNear(new THREE.Vector3(0, 1, 0).applyQuaternion(controls._quatInverse), [0, 0, 1]);
  });

  it('keeps +Y up and looks down -Z when the controls\' rotation is not where this three keeps it', () => {
    putCloud();
    for (const field of ['_quat', '_quatInverse']) {
      const controls = start();
      // Not a Quaternion, as in a three that keeps the rotation elsewhere; update reads it as no turn.
      controls[field] = { x: 0, y: 0, z: 0, w: 1 };
      camera.up.set(0, 0, 1);
      resetView(facingX);
      expectNear(camera.up, [0, 1, 0]);
      expectNear(camera.position.clone().sub(CENTRE).normalize(), [0, 0, 1]);
    }
  });
});

describe('clearGroup', () => {
  it('frees everything it removes but a geometry flagged as shared', () => {
    const freed = [];
    const watch = (o) => { o.addEventListener('dispose', () => freed.push(o)); return o; };
    const cached = watch(new THREE.BufferGeometry());
    const outline = new THREE.Mesh(cached, watch(new THREE.MeshBasicMaterial()));
    outline.userData.sharedGeometry = true;
    const own = new THREE.Mesh(watch(new THREE.BoxGeometry()), watch(new THREE.MeshBasicMaterial()));
    const group = new THREE.Group().add(outline, new THREE.Group().add(own));
    clearGroup(group);
    expect(group.children).toEqual([]);
    expect(new Set(freed)).toEqual(new Set([outline.material, own.geometry, own.material]));
  });
});
