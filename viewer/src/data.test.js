import * as THREE from 'three';
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  checkClipId, checkCloud, checkFormat, clearCaches, decodeRuns, fetchGraph, fetchHierarchy, fetchRegions, fetchTemporal,
  FORMAT, frameName, loadCatalog, loadClip, loadCloud, temporalProblem,
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

  it('refuses a malformed graph through time, and reads a sound one', async () => {
    // The track has a scene graph on frame 1 only, of the clip's two frames.
    const withTemporal = { ...clipJSON, tracks: [{ ...clipJSON.tracks[0], graph_frames: [1], temporal: true }] };
    const sound = { frames: [1], nodes: [{ id: 1, label: 'obj 1', present_frames: [1] }], relations: [{ src: 1, dst: 2, relation: 'left', frames: [1] }] };
    let body = sound;
    vi.stubGlobal('fetch', vi.fn((url) => respond(url.endsWith('clip.json') ? withTemporal : body)));
    expect(await fetchTemporal(await loadClip('c'), 't')).toEqual(sound);
    for (const [bad, msg] of [
      [{ ...sound, relations: [{ src: 1, dst: 2, relation: 'left', frames: ['<img src=x>'] }] }, /graph through time of c is malformed/],
      [{ nodes: [], relations: [] }, /are not lists/],
      [{ ...sound, frames: [0, 1] }, /not the 1 frames the track has scene graphs on/],
    ]) {
      clearCaches();
      body = bad;
      await expect(fetchTemporal(await loadClip('c'), 't')).rejects.toThrow(msg);
    }
  });
});

describe('temporalProblem', () => {
  const node = { id: 1, label: 'obj 1', present_frames: [0] };
  const rel = { src: 1, dst: 2, relation: 'left', frames: [0] };
  const sound = { frames: [0, 1], nodes: [node], relations: [rel] };
  const problem = (tg) => temporalProblem(tg, [0, 1]);
  const withNode = (x) => problem({ ...sound, nodes: [{ ...node, ...x }] });
  const withRelation = (x) => problem({ ...sound, relations: [{ ...rel, ...x }] });

  it('passes a sound graph, a node without label or frames included', () => {
    expect(problem(sound)).toBeNull();
    expect(problem({ frames: [0, 1], nodes: [{ id: 1 }], relations: [{ src: 1, dst: 2, relation: 'left' }] })).toBeNull();
    expect(temporalProblem({ frames: [3, 5], nodes: [{ id: 1, present_frames: [5] }], relations: [] }, [3, 5])).toBeNull();
  });

  it('names what is wrong with each part', () => {
    expect(problem(null)).toMatch(/not an object/);
    expect(problem([])).toMatch(/not an object/);
    expect(problem(7)).toMatch(/not an object/);
    for (const k of ['frames', 'nodes', 'relations']) {
      expect(problem({ ...sound, [k]: undefined })).toMatch(/are not lists/);
      expect(problem({ ...sound, [k]: {} })).toMatch(/are not lists/);
    }
    for (const frames of [[0], [1, 0], [0, 1, 1], [0, 500], [0, true]]) {
      expect(problem({ ...sound, frames })).toMatch(/its frames are not the 2 frames the track has scene graphs on/);
    }
    expect(withNode({ id: '1' })).toMatch(/node "1"/);
    expect(withNode({ present_frames: [0.5] })).toMatch(/node 1/);
    expect(withNode({ present_frames: [2] })).toMatch(/node 1 .* frames the graph lists/);
    expect(withNode({ present_frames: [-1] })).toMatch(/node 1/);
    expect(withNode({ label: 7 })).toMatch(/a string label/);
    expect(problem({ ...sound, nodes: [7] })).toMatch(/node undefined/);
    expect(withRelation({ dst: null })).toMatch(/relation of 1 and null/);
    expect(withRelation({ src: 1.5 })).toMatch(/relation of 1.5/);
    expect(withRelation({ relation: 3 })).toMatch(/a word/);
    expect(withRelation({ frames: null })).toMatch(/frames the graph lists/);
    expect(withRelation({ frames: [2] })).toMatch(/frames the graph lists/);
    expect(problem({ ...sound, relations: ['left'] })).toMatch(/relation of undefined/);
  });

  it('refuses a frame of the clip the graph does not list', () => {
    expect(temporalProblem({ frames: [3, 5], nodes: [{ id: 1, present_frames: [4] }], relations: [] }, [3, 5])).toMatch(/node 1/);
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
