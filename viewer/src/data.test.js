import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  checkClipId, checkFormat, clearCaches, decodeRuns, fetchRegions, fetchTemporal, FORMAT, frameName, loadClip, temporalProblem,
} from './data.js';

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
    expect(decodeRuns([-1, 2], 2)).toBeInstanceOf(Int32Array);
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
    vi.stubGlobal('fetch', vi.fn((url) => respond(url.endsWith('clip.json') ? clipJSON
      : { grid: [4, 1], runs: [0, 4], classes: [], stage: 'anchor', anchor_frame: null })));
    const clip = await loadClip('c');
    await expect(fetchRegions(clip, 't', 0)).rejects.toThrow(/4x1 grid/);
  });

  it('refuses a record of another clip, and does not keep a failed fetch', async () => {
    let fail = true;
    vi.stubGlobal('fetch', vi.fn(() => (fail ? respond(null, false) : respond({ ...clipJSON, id: 'd' }))));
    await expect(loadClip('c')).rejects.toThrow(/HTTP 404/);
    fail = false;
    await expect(loadClip('c')).rejects.toThrow(/describes d/);
  });

  it('refuses a malformed graph through time, and reads a sound one', async () => {
    const withTemporal = { ...clipJSON, tracks: [{ ...clipJSON.tracks[0], temporal: true }] };
    const sound = { nodes: [{ id: 1, label: 'obj 1', present_frames: [0, 1] }], relations: [{ src: 1, dst: 2, relation: 'left', frames: [0] }] };
    let body = sound;
    vi.stubGlobal('fetch', vi.fn((url) => respond(url.endsWith('clip.json') ? withTemporal : body)));
    expect(await fetchTemporal(await loadClip('c'), 't')).toEqual(sound);
    clearCaches();
    body = { nodes: [] };
    expect((await fetchTemporal(await loadClip('c'), 't')).relations).toEqual([]);
    clearCaches();
    body = { ...sound, relations: [{ src: 1, dst: 2, relation: 'left', frames: ['<img src=x>'] }] };
    await expect(fetchTemporal(await loadClip('c'), 't')).rejects.toThrow(/graph through time of c is malformed/);
    clearCaches();
    // The clip has two frames, so frame 2 is past its end.
    body = { ...sound, relations: [{ src: 1, dst: 2, relation: 'left', frames: [2] }] };
    await expect(fetchTemporal(await loadClip('c'), 't')).rejects.toThrow(/frames 0 to 1/);
  });
});

describe('temporalProblem', () => {
  const node = { id: 1, label: 'obj 1', present_frames: [0] };
  const rel = { src: 1, dst: 2, relation: 'left', frames: [0] };
  const problem = (tg) => temporalProblem(tg, 2);

  it('passes what the bundle stage publishes, lists left out included', () => {
    expect(problem({ nodes: [node], relations: [rel] })).toBeNull();
    expect(problem({ nodes: [{ id: 1 }], relations: [{ src: 1, dst: 2, relation: 'left' }] })).toBeNull();
    expect(problem({})).toBeNull();
  });

  it('names what is wrong with each part', () => {
    expect(problem(null)).toMatch(/not an object/);
    expect(problem({ relations: {} })).toMatch(/not a list/);
    expect(problem({ nodes: [{ ...node, id: '1' }] })).toMatch(/node "1"/);
    expect(problem({ nodes: [{ ...node, present_frames: [0.5] }] })).toMatch(/node 1/);
    expect(problem({ nodes: [{ ...node, present_frames: [2] }] })).toMatch(/frames 0 to 1/);
    expect(problem({ nodes: [{ ...node, present_frames: [-1] }] })).toMatch(/node 1/);
    expect(problem({ nodes: [{ ...node, label: 7 }] })).toMatch(/and a label/);
    expect(problem({ relations: [{ ...rel, dst: null }] })).toMatch(/relation of 1 and null/);
    expect(problem({ relations: [{ ...rel, src: 1.5 }] })).toMatch(/relation of 1.5/);
    expect(problem({ relations: [{ ...rel, relation: 3 }] })).toMatch(/a word/);
    expect(problem({ relations: [{ ...rel, frames: null }] })).toMatch(/frames 0 to 1/);
    expect(problem({ relations: [{ ...rel, frames: [2] }] })).toMatch(/frames 0 to 1/);
  });
});
