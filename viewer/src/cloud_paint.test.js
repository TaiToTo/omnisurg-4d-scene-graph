import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import { applyCloudFilter, applyRegionHighlight, applySegBlend, cloudDropMask, regionsToCanvas } from './cloud_paint.js';

// Six points: the annotation calls 1 and 2 an instrument (class 5), the rest liver (class 2); the
// zero-shot track has region 7 on the first three points.
const regionsByTrack = {
  gt: { labels: Int32Array.from([5, 5, 2, 2, 2, 2]) },
  zs: { labels: Int32Array.from([7, 7, 7, 3, 3, 3]) },
};
const instruments = { track: 'gt', ids: [5] };

describe('cloudDropMask', () => {
  it('hides nothing when every point is shown', () => {
    expect(cloudDropMask(regionsByTrack, 6, { show: 'all', instruments, isolate: null })).toBeNull();
  });

  it('hides the instruments for the tissue, and the tissue for the instruments', () => {
    const st = {};
    expect([...cloudDropMask(regionsByTrack, 6, { show: 'tissue', instruments, isolate: null }, { stats: st })]).toEqual([1, 1, 0, 0, 0, 0]);
    expect(st).toMatchObject({ total: 6, instruments: 2 });
    expect([...cloudDropMask(regionsByTrack, 6, { show: 'tools', instruments, isolate: null })]).toEqual([0, 0, 1, 1, 1, 1]);
  });

  it('shows no instrument on a frame without their labels, and says so', () => {
    const st = {};
    const mask = cloudDropMask({ zs: regionsByTrack.zs }, 6, { show: 'tools', instruments, isolate: null }, { stats: st });
    expect([...mask]).toEqual([1, 1, 1, 1, 1, 1]);
    expect(st.unlabeled).toBe(1);
    expect([...cloudDropMask(regionsByTrack, 6, { show: 'tools', instruments: null, isolate: null })]).toEqual([1, 1, 1, 1, 1, 1]);
  });

  it('keeps only the isolated region, except on the frame on stage', () => {
    const v = { show: 'all', instruments, isolate: 'zs:7' };
    expect([...cloudDropMask(regionsByTrack, 6, v)]).toEqual([0, 0, 0, 1, 1, 1]);
    expect(cloudDropMask(regionsByTrack, 6, v, { context: true })).toBeNull();
  });

  it('leaves a frame whole when its labels do not fit the cloud', () => {
    expect(cloudDropMask(regionsByTrack, 5, { show: 'all', instruments, isolate: 'zs:7' })).toBeNull();
  });
});

// A cloud of one point per pixel on a w x h grid, the point of pixel (x, y) at `at(x, y)`, seen by a camera at
// the origin looking down +z.
function raster(w, h, at) {
  const pos = new Float32Array(w * h * 3);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) pos.set(at(x, y), 3 * (y * w + x));
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  return geom;
}
const cam = { pos: [0, 0, 0], fwd: [0, 0, 1] };

function drops(geom, [w, h], { labels = new Int32Array(w * h).fill(2), show = 'tissue' } = {}) {
  const stats = {};
  const mask = cloudDropMask({ gt: { labels } }, w * h, { show, instruments, isolate: null }, { geom, cam, grid: [w, h], stats });
  return { mask: mask && [...mask], stats };
}

describe('the tissue view', () => {
  it('drops the flying pixels on a depth step', () => {
    // Depth 1 left of column 4 and 1.5 from it: the step marks columns 3 and 4, grown by one to 2-5.
    const { mask, stats } = drops(raster(8, 4, (x, y) => [x * 0.01, y * 0.01, x < 4 ? 1 : 1.5]), [8, 4]);
    expect(mask.slice(0, 8)).toEqual([0, 0, 1, 1, 1, 1, 0, 0]);
    expect(stats).toEqual({ total: 32, depthStep: 16 });
  });

  it('drops a surface seen edge-on', () => {
    // A sheet running away from the camera, 1 % deeper per row: no step, but nearly along the line of sight.
    const { stats } = drops(raster(6, 6, (x, y) => [x * 0.01, y * 0.0002, 1 + y * 0.01]), [6, 6]);
    expect(stats).toEqual({ total: 36, edgeOn: 36 });
  });

  it('drops the halo round an instrument, two pixels wide', () => {
    const labels = new Int32Array(45).fill(2);
    labels[2 * 9 + 2] = 5;
    const { mask, stats } = drops(raster(9, 5, (x, y) => [x * 0.01, y * 0.01, 1]), [9, 5], { labels });
    expect(mask.slice(0, 9)).toEqual([1, 1, 1, 1, 1, 0, 0, 0, 0]);
    expect(stats).toEqual({ total: 45, instruments: 1, toolHalo: 24 });
  });

  it('drops nothing when every point is shown', () => {
    const geom = raster(6, 6, (x, y) => [x * 0.01, y * 0.0002, 1 + y * 0.01]);
    expect(drops(geom, [6, 6], { show: 'all' }).mask).toBeNull();
  });
});

// A group holding one cloud of three points, black unless `colors` says otherwise.
function cloudGroup(colors = new Float32Array(9), normalized = false) {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(9), 3));
  geom.setAttribute('color', new THREE.BufferAttribute(colors, 3, normalized));
  const group = new THREE.Group();
  group.add(new THREE.Points(geom, new THREE.PointsMaterial()));
  return group;
}
const reds = (group) => {
  const c = group.children[0].geometry.attributes.color;
  return [0, 1, 2].map((i) => Number(c.getX(i).toFixed(2)));
};
const showAll = { show: 'all', instruments: null, isolate: null };

describe('painting the regions', () => {
  const classes = [{ id: 1, color: [1, 0, 0] }];

  it('paints a track whose labels fit the cloud, and skips one whose labels do not', () => {
    const fits = cloudGroup();
    applySegBlend(fits, { zs: { labels: [1, 1, 0], classes } }, ['zs'], showAll);
    expect(reds(fits)).toEqual([0.45, 0.45, 0]);
    const short = cloudGroup();
    applySegBlend(short, { zs: { labels: [1, 1], classes } }, ['zs'], showAll);
    expect(reds(short)).toEqual([0, 0, 0]);
  });

  it('lights a region and pales the rest, and lights nothing from labels that do not fit', () => {
    const key = new Map([['zs:1', [1, 0, 0]]]);
    const fits = cloudGroup();
    expect(applyRegionHighlight(fits, { zs: { labels: [1, 1, 0] } }, key, showAll)).toBe(true);
    expect(reds(fits)).toEqual([0.65, 0.65, 0.12]);
    const short = cloudGroup();
    applyRegionHighlight(short, { zs: { labels: [1, 1] } }, key, showAll);
    expect(reds(short)).toEqual([0.12, 0.12, 0.12]);
  });

  it('reads a normalized byte colour as a fraction of 255 when it lights', () => {
    const g = cloudGroup(Uint8Array.from([100, 0, 0, 100, 0, 0, 100, 0, 0]), true);
    applyRegionHighlight(g, { zs: { labels: [0, 0, 0] } }, new Map([['zs:1', [1, 0, 0]]]), showAll);
    expect(g.children[0].geometry.attributes.color.getX(0)).toBeCloseTo((100 / 255) * 0.88 + 0.12, 2);
  });

  it('frees the filtered copy when the view shows every point again', () => {
    const g = cloudGroup();
    const obj = g.children[0];
    const full = obj.geometry;
    const regions = { gt: { labels: Int32Array.from([5, 2, 2]) } };
    applyCloudFilter(g, regions, { show: 'tools', instruments, isolate: null });
    const copy = obj.geometry;
    expect(copy.attributes.position.count).toBe(1);
    let freed = false;
    copy.addEventListener('dispose', () => { freed = true; });
    applyCloudFilter(g, regions, showAll);
    expect(obj.geometry).toBe(full);
    expect(freed).toBe(true);
  });
});

describe('the regions over the image', () => {
  // A canvas that keeps the image put on it.
  const canvas = () => {
    const cv = { width: 0, height: 0, image: null };
    cv.getContext = () => ({
      createImageData: (w, h) => ({ data: new Uint8ClampedArray(w * h * 4) }),
      putImageData: (img) => { cv.image = img; },
    });
    return cv;
  };
  const regions = { grid: [3, 1], labels: [1, 2, 0], classes: [{ id: 1, color: [1, 0, 0] }, { id: 2, color: [0, 0, 1] }] };
  const alphas = (cv) => [3, 7, 11].map((k) => cv.image.data[k]);

  it('draws each region in its colour over a clear background', () => {
    const cv = canvas();
    regionsToCanvas(regions, cv);
    expect(alphas(cv)).toEqual([153, 153, 0]);
  });

  it('draws the focused regions strong and a white veil over the rest', () => {
    const cv = canvas();
    regionsToCanvas(regions, cv, new Set([1]));
    expect(alphas(cv)).toEqual([191, 31, 31]);
    expect([...cv.image.data.slice(4, 7)]).toEqual([255, 255, 255]);
  });
});
