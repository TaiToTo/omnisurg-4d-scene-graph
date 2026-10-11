import * as THREE from 'three';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  checkClipId, checkCloud, checkFormat, clearCaches, decodeRuns, fetchGraph, fetchHierarchy, fetchRegions, fetchTemporal,
  FORMAT, frameName, loadCatalog, loadClip, loadCloud,
} from './data.js';

const gltfLoad = vi.hoisted(() => ({ fn: null }));
vi.mock('three/addons/loaders/GLTFLoader.js', () => ({ GLTFLoader: class { loadAsync(url) { return gltfLoad.fn(url); } } }));

/** A parsed glTF holding one point list of each of `counts` points. */
function cloudOf(...counts) {
  const scene = new THREE.Group();
  for (const n of counts) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(3 * n), 3));
    scene.add(new THREE.Points(g, new THREE.PointsMaterial()));
  }
  return { scene };
}

describe('decodeRuns', () => {
  it('expands value/count pairs into one label per point', () => {
    expect([...decodeRuns([0, 2, 3, 3, 7, 1], 6)]).toEqual([0, 0, 3, 3, 3, 7]);
  });

  it('refuses runs that cover more or fewer points than the cloud has', () => {
    expect(() => decodeRuns([0, 2, 3, 3], 6)).toThrow(/cover 5 of 6/);
    expect(() => decodeRuns([0, 7], 6)).toThrow(/run past/);
  });

  it('keeps labels in 16 bits when they fit, and in 32 when they do not', () => {
    expect(decodeRuns([3, 2], 2)).toBeInstanceOf(Uint16Array);
    expect([...decodeRuns([70000, 2], 2)]).toEqual([70000, 70000]);
    expect(decodeRuns([70000, 2], 2)).toBeInstanceOf(Int32Array);
  });

  it('refuses a label that is no region id, or that 32 bits would wrap round', () => {
    for (const bad of [-1, 1.5, '3', 2 ** 31, 2 ** 32 + 5]) expect(() => decodeRuns([bad, 2], 2)).toThrow(/is not a region id/);
    expect([...decodeRuns([2 ** 31 - 1, 1], 1)]).toEqual([2 ** 31 - 1]);
  });

  it('refuses what is not value/count pairs', () => {
    expect(() => decodeRuns([0, 2, 3], 5)).toThrow(/pairs/);
    expect(() => decodeRuns([0, -1, 3, 7], 6)).toThrow(/run past/);
    expect(() => decodeRuns([0, 1.5, 3, 4.5], 6)).toThrow(/run past/);
  });
});

describe('the bundle records', () => {
  it('refuses a record of another format', () => {
    expect(checkFormat({ format: FORMAT }, 'x')).toEqual({ format: FORMAT });
    expect(() => checkFormat({ format: FORMAT + 1 }, 'x')).toThrow(/not a bundle of format/);
    expect(() => checkFormat(null, 'x')).toThrow(/format missing/);
  });

  it('refuses a cloud without one point per pixel of the grid', () => {
    const clip = { id: 'c', geometry: { grid: [2, 3] } };
    expect(checkCloud(cloudOf(6), clip, 0).scene.children).toHaveLength(1);
    expect(() => checkCloud(cloudOf(5), clip, 4)).toThrow(/c frame 4: the cloud holds 5 points, not one per pixel of the 2x3 grid/);
    expect(() => checkCloud(cloudOf(6, 6), clip, 0)).toThrow(/holds 6 \+ 6 points/);
    expect(() => checkCloud(cloudOf(), clip, 0)).toThrow(/holds no points/);
  });

  it('refuses a clip id that would leave the bundle directory', () => {
    expect(checkClipId('lar___7H7G-4sevQ__gt_0001')).toBe('lar___7H7G-4sevQ__gt_0001');
    for (const bad of ['../x', 'a/b', '', '.hidden', 'a b', 'https://x', null]) expect(() => checkClipId(bad)).toThrow(/not a clip id/);
  });

  it('names frames with four digits', () => {
    expect(frameName(7)).toBe('0007');
  });
});

describe('fetching a clip', () => {
  const clipJSON = {
    format: FORMAT, id: 'c', n_frames: 2, geometry: { grid: [2, 2] },
    tracks: [{ id: 't', region_frames: [0], graph_frames: [], temporal: false }],
  };
  const respond = (body, ok = true) => Promise.resolve({ ok, status: ok ? 200 : 404, json: () => Promise.resolve(body) });
  afterEach(() => { vi.unstubAllGlobals(); clearCaches(); });

  it('asks only for the frames a track lists, and decodes the labels', async () => {
    const fetch = vi.fn((url) => respond(url.endsWith('clip.json') ? clipJSON
      : { grid: [2, 2], runs: [0, 1, 4, 3], classes: [{ id: 4, name: 'liver', color: [1, 0, 0] }], stage: 'anchor', anchor_frame: null }));
    vi.stubGlobal('fetch', fetch);
    const clip = await loadClip('c');
    expect(await fetchRegions(clip, 't', 1)).toBeNull();
    expect(fetch).toHaveBeenCalledTimes(1);
    const r = await fetchRegions(clip, 't', 0);
    expect([...r.labels]).toEqual([0, 4, 4, 4]);
    expect(r.classById.get(4).name).toBe('liver');
    expect(await fetchRegions(clip, 't', 0)).toBe(r);
  });

  it('refuses regions on another grid than the cloud', async () => {
    for (const grid of [[4, 1], [2, 1]]) {
      clearCaches();
      vi.stubGlobal('fetch', vi.fn((url) => respond(url.endsWith('clip.json') ? clipJSON
        : { grid, runs: [0, grid[0] * grid[1]], classes: [], stage: 'anchor', anchor_frame: null })));
      const clip = await loadClip('c');
      await expect(fetchRegions(clip, 't', 0)).rejects.toThrow(`${grid[0]}x${grid[1]} grid`);
    }
  });

  it('takes a seed frame of the clip, and refuses one that is not', async () => {
    const regions = (anchor) => ({ grid: [2, 2], runs: [0, 4], classes: [], stage: 'propagated', anchor_frame: anchor });
    for (const [anchor, want] of [[null, null], [undefined, null], [0, 0], [1, 1]]) {
      clearCaches();
      vi.stubGlobal('fetch', vi.fn((url) => respond(url.endsWith('clip.json') ? clipJSON : regions(anchor))));
      expect((await fetchRegions(await loadClip('c'), 't', 0)).anchorFrame).toBe(want);
    }
    for (const anchor of [-1, 2, 1.5, '0']) {
      clearCaches();
      vi.stubGlobal('fetch', vi.fn((url) => respond(url.endsWith('clip.json') ? clipJSON : regions(anchor))));
      await expect(fetchRegions(await loadClip('c'), 't', 0)).rejects.toThrow(/as their seed frame, no frame of the clip/);
    }
  });

  it('asks for no graph, graph through time or hierarchy the record does not list', async () => {
    const fetch = vi.fn((url) => respond(clipJSON));
    vi.stubGlobal('fetch', fetch);
    const clip = await loadClip('c');
    expect(await fetchGraph(clip, 't', 0)).toBeNull();
    expect(await fetchTemporal(clip, 't')).toBeNull();
    expect(await fetchHierarchy({ ...clip, hierarchy: { tracks: ['t', 'u'], frames: [1] } }, 0)).toBeNull();
    expect(await fetchHierarchy(clip, 0)).toBeNull();
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it('refuses a record of another format, an unsafe track id, or no list of tracks', async () => {
    for (const [rec, msg] of [[{ ...clipJSON, format: FORMAT + 1 }, /not a bundle of format/],
      [{ ...clipJSON, tracks: [{ ...clipJSON.tracks[0], id: '../t' }] }, /not a track id: "\.\.\/t"/],
      [{ ...clipJSON, tracks: null }, /lists no tracks/], [{ ...clipJSON, tracks: { t: {} } }, /lists no tracks/]]) {
      clearCaches();
      vi.stubGlobal('fetch', vi.fn(() => respond(rec)));
      await expect(loadClip('c')).rejects.toThrow(msg);
    }
    await expect(loadClip('../c')).rejects.toThrow(/not a clip id/);
  });

  it('names the file that is not JSON', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve({ ok: true, status: 200, json: () => Promise.reject(new SyntaxError('Unexpected token <')) })));
    await expect(loadClip('c')).rejects.toThrow(/c\/clip\.json is not JSON \(Unexpected token </);
  });

  it('refuses a record of another clip, and does not keep a failed fetch', async () => {
    let fail = true;
    vi.stubGlobal('fetch', vi.fn(() => (fail ? respond(null, false) : respond({ ...clipJSON, id: 'd' }))));
    await expect(loadClip('c')).rejects.toThrow(/HTTP 404/);
    fail = false;
    await expect(loadClip('c')).rejects.toThrow(/describes d/);
  });
});

describe('the catalog', () => {
  const respond = (body) => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(body) });
  afterEach(() => { vi.unstubAllGlobals(); clearCaches(); });

  it('lists the clips of a bundle of this format, and refuses any other', async () => {
    const good = { format: FORMAT, clips: [{ id: 'a' }, { id: 'b' }] };
    for (const [cat, msg] of [[{ ...good, format: 0 }, /not a bundle of format/], [{ format: FORMAT }, /lists no clips/],
      [{ format: FORMAT, clips: [{ id: 'a/../../x' }] }, /not a clip id/]]) {
      clearCaches();
      vi.stubGlobal('fetch', vi.fn(() => respond(cat)));
      await expect(loadCatalog()).rejects.toThrow(msg);
    }
    clearCaches();
    vi.stubGlobal('fetch', vi.fn(() => respond(good)));
    expect((await loadCatalog()).clips.map((c) => c.id)).toEqual(['a', 'b']);
  });
});

describe('a cloud', () => {
  const clip = { id: 'c', n_frames: 2, geometry: { grid: [2, 2] } };
  afterEach(() => { clearCaches(); });

  it('is refused with too few points, and asked for again on the next call', async () => {
    gltfLoad.fn = vi.fn().mockResolvedValueOnce(cloudOf(3)).mockResolvedValueOnce(cloudOf(4));
    await expect(loadCloud(clip, 1)).rejects.toThrow(/holds 3 points/);
    expect((await loadCloud(clip, 1)).scene.children).toHaveLength(1);
    expect(gltfLoad.fn).toHaveBeenCalledTimes(2);
  });
});
