// Drive app.js with the modules it calls replaced, to check what the page does when loads end out of order.
import { afterEach, describe, expect, it, vi } from 'vitest';

function fakeEl() {
  const parts = new Map();
  return {
    textContent: '', value: '', max: '', hidden: false, innerHTML: '', src: '', className: '', title: '', disabled: false,
    style: {}, dataset: {},
    classList: { toggle() {}, add() {}, remove() {}, contains() { return false; } },
    setAttribute() {}, getAttribute() { return null; }, appendChild() {}, append() {}, addEventListener() {},
    querySelectorAll() { return []; }, firstElementChild: null,
    querySelector(sel) { if (!parts.has(sel)) parts.set(sel, fakeEl()); return parts.get(sel); },
    getContext() { return { clearRect() {} }; }, getBoundingClientRect() { return { width: 800 }; },
  };
}
const els = new Map();
globalThis.document = {
  getElementById: (id) => { if (!els.has(id)) els.set(id, fakeEl()); return els.get(id); },
  querySelectorAll: () => [],
  body: { classList: { toggle() {}, add() {}, remove() {} } },
  createElement: () => fakeEl(),
};
globalThis.window = { innerHeight: 1000 };

vi.mock('./data.js', () => ({
  clearCaches: vi.fn(), fetchGraph: vi.fn(), fetchHierarchy: vi.fn(), fetchRegions: vi.fn(), fetchTemporal: vi.fn(),
  frameImageURL: (clip, i) => `${clip.id}/${i}.jpg`, loadClip: vi.fn(), loadCloud: vi.fn(), prefetchCloud: vi.fn(),
}));
vi.mock('./stage.js', async () => {
  const THREE = await import('three');
  const groups = {
    points: new THREE.Group(), cameraViz: new THREE.Group(), highlight: new THREE.Group(),
    graph3d: new THREE.Group(), hover: new THREE.Group(),
  };
  const clearGroup = (g) => { while (g.children.length) g.remove(g.children[0]); };
  return {
    camera: new THREE.PerspectiveCamera(), groups, clearGroup, invalidate: () => {}, measureSceneScale: () => 1,
    onSceneScale: () => {},
    // Each call returns how many clouds were on stage when the view was framed.
    resetView: vi.fn(() => groups.points.children.length),
  };
});
vi.mock('./world.js', async () => {
  const actual = await vi.importActual('./world.js');
  const stage = await import('./stage.js');
  return {
    ...actual,
    enterWorld: vi.fn(),
    exitWorld: () => { stage.clearGroup(stage.groups.points); stage.clearGroup(stage.groups.cameraViz); },
    focusWorldFrame: (world, i) => actual.nearestStacked(i, world.indices),
  };
});
vi.mock('./graph3d.js', () => ({ buildGraph3D: vi.fn(), drawEventConnector: vi.fn(), setHover3D: vi.fn() }));
vi.mock('./node_link.js', () => ({ renderFrameNodeLink: vi.fn(), renderWorldNodeLink: vi.fn(), setHoverKey: vi.fn() }));
vi.mock('./strip.js', () => ({ renderTrackStrip: vi.fn(), renderWorldStrip: vi.fn(), updateStripStage: vi.fn() }));
vi.mock('./cloud_paint.js', () => ({
  applySegBlend: vi.fn(), applyRegionHighlight: vi.fn(() => true), regionsToCanvas: vi.fn(),
}));
vi.mock('./focus_band.js', () => ({
  renderFocusBand: vi.fn(() => []), topPartners: vi.fn(() => [[7, {}]]), updateFocusCursor: vi.fn(),
}));

const flush = async () => { for (let k = 0; k < 20; k++) await Promise.resolve(); };
function deferred() {
  let resolve, reject;
  const promise = new Promise((a, b) => { resolve = a; reject = b; });
  return { promise, resolve, reject };
}
const gltf = { scene: { traverse() {} } };

function makeClip(id, n) {
  const all = [...Array(n).keys()];
  const tracks = [{ id: 't', name: 'T', short: 'T', semantic: false, instrument_ids: [], region_frames: all,
    graph_frames: all, temporal: true, anchor_badge: 'A', tracked_badge: 'B' }];
  return {
    id, n_frames: n, tracks, trackById: new Map(tracks.map((t) => [t.id, t])), hierarchy: null,
    frames: all.map((i) => ({ time_s: i, centroid: [0, 0, 0], camera_position: [0, 0, 1], camera_forward: [0, 0, -1], camera_up: [0, 1, 0] })),
    geometry: { grid: [2, 2], name: 'DA3' }, dataset_name: 'X', procedure: 'P', video: 'v',
  };
}

function makeWorld(THREE, indices) {
  return {
    origin: [0, 0, 0], indices,
    frames: indices.map((i) => ({
      i, offset: [0, 0, 0], group: new THREE.Group(), hierarchy: null,
      regionsByTrack: { t: { labels: new Uint16Array([1, 1, 7, 7]), classes: [] } },
      graphByTrack: { t: { nodes: [{ id: 1, pos: [0, 0, 0] }, { id: 7, pos: [1, 0, 0] }] } },
    })),
  };
}

const STACK = [0, 10, 20, 30, 39];

async function fresh() {
  vi.resetModules();
  els.clear();
  const app = await import('./app.js');
  const data = await import('./data.js');
  const stage = await import('./stage.js');
  const world = await import('./world.js');
  const band = await import('./focus_band.js');
  const paint = await import('./cloud_paint.js');
  const THREE = await import('three');
  data.loadCloud.mockResolvedValue(gltf);
  data.fetchRegions.mockResolvedValue(null);
  data.fetchGraph.mockResolvedValue(null);
  data.fetchHierarchy.mockResolvedValue(null);
  const all = [...Array(40).keys()];
  data.fetchTemporal.mockResolvedValue({ frames: all, nodes: [{ id: 1, present_frames: all }, { id: 7, present_frames: all }], relations: [] });
  const reports = [];
  app.onError((m) => reports.push(m));
  return { app, data, stage, world, band, paint, THREE, reports };
}

afterEach(() => { vi.useRealTimers(); });

describe('framing the view', () => {
  it('frames the view on a cloud when playback steps on during the switch', async () => {
    vi.useFakeTimers();
    const { app, data, stage, world, THREE } = await fresh();
    data.loadClip.mockResolvedValue(makeClip('a', 40));
    await app.openClip('a');
    world.enterWorld.mockResolvedValue(makeWorld(THREE, STACK));
    await app.switchMode('world');
    app.setPlaying(true);
    await vi.advanceTimersByTimeAsync(700 * 3);
    const d1 = deferred(), d2 = deferred();
    data.loadCloud.mockReturnValueOnce(d1.promise).mockReturnValueOnce(d2.promise);
    stage.resetView.mockClear();
    const sw = app.switchMode('frame');
    await vi.advanceTimersByTimeAsync(700);
    d1.resolve(gltf);
    await sw;
    d2.resolve(gltf);
    await flush();
    app.setPlaying(false);
    expect(stage.resetView.mock.results.map((r) => r.value)).toEqual([1]);
  });

  it('frames a clip opened while the switch to per-frame mode loads', async () => {
    const { app, data, stage, world, THREE } = await fresh();
    data.loadClip.mockResolvedValueOnce(makeClip('a', 40)).mockResolvedValueOnce(makeClip('b', 40));
    await app.openClip('a');
    world.enterWorld.mockResolvedValue(makeWorld(THREE, STACK));
    await app.switchMode('world');
    const d1 = deferred(), dB = deferred();
    data.loadCloud.mockReturnValueOnce(d1.promise).mockReturnValueOnce(dB.promise);
    stage.resetView.mockClear();
    const sw = app.switchMode('frame');
    const op = app.openClip('b');
    await flush();
    d1.resolve(gltf);
    await sw;
    dB.resolve(gltf);
    await op;
    expect(app.state.clip.id).toBe('b');
    expect(stage.resetView.mock.results.map((r) => r.value)).toEqual([1]);
  });
});

describe('the frame panel', () => {
  it('pairs the image with the regions while the next frame loads', async () => {
    const { app, data } = await fresh();
    data.loadClip.mockResolvedValue(makeClip('a', 40));
    await app.openClip('a');
    const d = deferred();
    data.loadCloud.mockReturnValueOnce(d.promise);
    const seek = app.seekFrame(5);
    await app.setFocusNode('t:1');
    expect(document.getElementById('thumb-rgb').src).toBe('a/0.jpg');
    d.resolve(gltf);
    await seek;
    expect(document.getElementById('thumb-rgb').src).toBe('a/5.jpg');
  });
});

describe('the per-frame strip', () => {
  it('says that its regions did not load, rather than loading for ever', async () => {
    const { app, data, reports } = await fresh();
    data.loadClip.mockResolvedValue(makeClip('a', 40));
    data.fetchRegions.mockImplementation((clip, t, i) => (i === 0 ? Promise.resolve(null)
      : Promise.reject(new Error(`could not load a/regions/t/${i}.json (HTTP 404)`))));
    await app.openClip('a');
    await flush();
    expect(reports).toEqual([expect.stringMatching(/^The regions through time did not load: could not load/)]);
    expect(document.getElementById('strip').innerHTML).toMatch(/did not load/);
  });
});

describe('following a region', () => {
  it('marks the region again on going back from an event to the first step', async () => {
    const { app, data, world, band, paint, THREE } = await fresh();
    data.loadClip.mockResolvedValue(makeClip('a', 40));
    await app.openClip('a');
    world.enterWorld.mockResolvedValue(makeWorld(THREE, STACK));
    band.renderFocusBand.mockImplementation(() => [{ f: 0, kind: 'rel', partner: 7, label: '' }]);
    await app.setFocusNode('t:1');
    await app.setFocusStage(3);
    await flush();
    await band.renderFocusBand.mock.calls.at(-1)[1].onEvent(0);
    expect(app.state.selectedEvent).toBe(0);
    paint.applyRegionHighlight.mockClear();
    await app.setFocusStage(1);
    expect(app.state.selectedEvent).toBe(null);
    expect(paint.applyRegionHighlight).toHaveBeenCalled();
  });

  it('shows the event picked last', async () => {
    const { app, data, world, band, THREE } = await fresh();
    data.loadClip.mockResolvedValue(makeClip('a', 40));
    await app.openClip('a');
    world.enterWorld.mockResolvedValue(makeWorld(THREE, STACK));
    band.renderFocusBand.mockImplementation(() => [{ f: 5, kind: 'enter', label: '' }, { f: 15, kind: 'enter', label: '' }]);
    await app.setFocusNode('t:1');
    await app.setFocusStage(3);
    await flush();
    const onEvent = band.renderFocusBand.mock.calls.at(-1)[1].onEvent;
    const d1 = deferred(), d2 = deferred(), d3 = deferred();
    data.loadCloud.mockReturnValueOnce(d1.promise).mockReturnValueOnce(d2.promise).mockReturnValueOnce(d3.promise);
    const first = onEvent(0);
    await flush();
    const second = onEvent(1);
    await flush();
    d1.resolve(gltf); await flush();
    d2.resolve(gltf); await flush();
    d3.resolve(gltf);
    await Promise.all([first, second]);
    expect(app.state.frame).toBe(15);
    expect(app.state.selectedEvent).toBe(1);
  });

  it('drops an edge whose graphs arrive after another clip opened', async () => {
    const { app, data, world, THREE, reports } = await fresh();
    data.loadClip.mockResolvedValueOnce(makeClip('a', 40)).mockResolvedValueOnce(makeClip('b', 10));
    await app.openClip('a');
    world.enterWorld.mockResolvedValue(makeWorld(THREE, STACK));
    const pending = [];
    data.fetchGraph.mockImplementation(() => { const d = deferred(); pending.push(d); return d.promise; });
    await app.setFocusNode('t:1');
    await app.setFocusStage(3);
    await flush();
    data.fetchGraph.mockResolvedValue(null);
    await app.openClip('b');
    const status = document.getElementById('status').textContent;
    for (const d of pending) d.resolve({ nodes: [], edges: [] });
    await flush();
    expect(document.getElementById('status').textContent).toBe(status);
    expect(reports).toEqual([]);
  });

  it('reports an edge whose graph files do not load', async () => {
    const { app, data, world, THREE, reports } = await fresh();
    data.loadClip.mockResolvedValue(makeClip('a', 40));
    await app.openClip('a');
    world.enterWorld.mockResolvedValue(makeWorld(THREE, STACK));
    data.fetchGraph.mockRejectedValue(new Error('could not load a/graphs/t/0003.json (HTTP 404)'));
    await app.setFocusNode('t:1');
    await app.setFocusStage(3);
    await flush();
    expect(reports).toEqual([expect.stringMatching(/edge's axes did not load \(could not load a\/graphs\/t\/0003\.json/)]);
  });

  it('keeps the graph through time of the clip on screen', async () => {
    const { app, data } = await fresh();
    data.loadClip.mockResolvedValueOnce(makeClip('a', 40)).mockResolvedValueOnce(makeClip('b', 40));
    await app.openClip('a');
    const tgA = deferred();
    const tgB = { frames: [...Array(40).keys()], nodes: [{ id: 3, present_frames: [0] }], relations: [] };
    data.fetchTemporal.mockReturnValueOnce(tgA.promise).mockResolvedValueOnce(tgB);
    const followA = app.setFocusNode('t:3');
    await app.openClip('b');
    await app.setFocusNode('t:3');
    tgA.resolve({ frames: [...Array(40).keys()], nodes: [], relations: [] });
    await followA;
    expect(app.state.temporal).toBe(tgB);
  });

  it('keeps the reader\'s own playback after the second step\'s clip is closed', async () => {
    vi.useFakeTimers();
    const { app, data, world, THREE } = await fresh();
    data.loadClip.mockResolvedValueOnce(makeClip('a', 40)).mockResolvedValueOnce(makeClip('b', 40));
    await app.openClip('a');
    world.enterWorld.mockResolvedValue(makeWorld(THREE, STACK));
    await app.setFocusNode('t:1');
    await app.setFocusStage(2);
    expect(app.state.playing).toBe(true);
    await app.openClip('b');
    app.setPlaying(true);
    await app.setFocusNode('t:1');
    await app.setFocusNode(null);
    expect(app.state.playing).toBe(true);
    app.setPlaying(false);
  });

  it('repaints only the frames leaving and taking the stage while isolated', async () => {
    const { app, data, world, paint, THREE } = await fresh();
    data.loadClip.mockResolvedValue(makeClip('a', 40));
    await app.openClip('a');
    const stack = makeWorld(THREE, STACK);
    world.enterWorld.mockResolvedValue(stack);
    await app.setFocusNode('t:1');
    await app.setFocusStage(2);
    app.setPlaying(false);
    app.seekFrame(0);
    paint.applySegBlend.mockClear();
    app.seekFrame(10);
    const painted = new Set(paint.applySegBlend.mock.calls.map(([g]) => stack.frames.find((fr) => fr.group === g).i));
    expect([...painted].sort((a, b) => a - b)).toEqual([0, 10]);
  });
});

describe('world mode', () => {
  it('keeps the stack asked for last, and does not report one no longer asked for', async () => {
    const { app, data, world, THREE, reports } = await fresh();
    data.loadClip.mockResolvedValue(makeClip('a', 40));
    await app.openClip('a');
    const w1 = deferred(), w2 = deferred(), c1 = deferred();
    world.enterWorld
      .mockImplementationOnce(async (clip, paint, stale) => { const w = await w1.promise; return stale() ? null : w; })
      .mockImplementationOnce(async (clip, paint, stale) => { const w = await w2.promise; return stale() ? null : w; });
    const a = app.switchMode('world');
    data.loadCloud.mockReturnValueOnce(c1.promise);
    const b = app.switchMode('frame');
    const c = app.switchMode('world');
    w1.reject(new Error('could not load a/clouds/0010.glb (HTTP 404)'));
    await a;
    c1.resolve(gltf);
    await b;
    w2.resolve(makeWorld(THREE, STACK));
    await c;
    expect(reports).toEqual([]);
    expect(app.state.mode).toBe('world');
    expect(app.state.world).not.toBe(null);
  });

  it('goes back to the first step when the stack does not load', async () => {
    const { app, data, world, reports } = await fresh();
    data.loadClip.mockResolvedValue(makeClip('a', 40));
    await app.openClip('a');
    world.enterWorld.mockRejectedValue(new Error('could not load a/clouds/0010.glb (HTTP 404)'));
    await app.setFocusNode('t:1');
    await app.setFocusStage(2);
    expect(app.state.mode).toBe('frame');
    expect(app.state.focusStage).toBe(1);
    expect(reports).toHaveLength(1);
  });
});

/** A clip with an annotation `gt` and an automatic track `auto`, which its hierarchy relates. */
function makePairClip(id, n, { gtSemantic = true } = {}) {
  const clip = makeClip(id, n);
  const all = [...Array(n).keys()];
  const track = (tid, semantic) => ({ ...clip.tracks[0], id: tid, name: tid, short: tid, semantic });
  clip.tracks = [track('auto', false), track('gt', gtSemantic)];
  clip.trackById = new Map(clip.tracks.map((t) => [t.id, t]));
  clip.hierarchy = { tracks: ['auto', 'gt'], frames: all };
  return clip;
}

describe('both tracks of the hierarchy', () => {
  const painted = (paint) => paint.applySegBlend.mock.calls.at(-1)[2];

  it('paints the annotation and the automatic regions, the annotation first', async () => {
    const { app, data, paint } = await fresh();
    data.loadClip.mockResolvedValue(makePairClip('a', 4));
    await app.openClip('a');
    app.toggleRegions(app.BOTH);
    expect(painted(paint)).toEqual(['gt', 'auto']);
    app.setTrack('auto');
    expect(painted(paint)).toEqual(['auto']);
    app.setTrack(app.BOTH);
    expect(painted(paint)).toEqual(['gt', 'auto']);
  });

  it('offers no Both for two tracks of one kind', async () => {
    const { app, data, paint } = await fresh();
    data.loadClip.mockResolvedValue(makePairClip('a', 4, { gtSemantic: false }));
    await app.openClip('a');
    app.toggleRegions(app.BOTH);
    expect(app.state.both).toBe(false);
    expect(painted(paint)).toEqual([]);
  });

  it('takes one track off the cloud from its strip row, and leaves the other painted', async () => {
    const { app, data, paint } = await fresh();
    const strip = await import('./strip.js');
    data.loadClip.mockResolvedValue(makePairClip('a', 4));
    await app.openClip('a');
    await flush();
    app.toggleRegions(app.BOTH);
    strip.renderTrackStrip.mock.calls.at(-1)[1].onPickTrack('gt');
    expect(painted(paint)).toEqual(['auto']);
  });

  it('stops following a region whose track is no longer shown', async () => {
    const { app, data } = await fresh();
    data.loadClip.mockResolvedValue(makePairClip('a', 4));
    await app.openClip('a');
    app.setTrack(app.BOTH);
    await app.setFocusNode('gt:1');
    app.setTrack('auto');
    expect(app.state.focusKey).toBe(null);
  });

  it('drops the graph through time of the other track while the followed region\'s loads', async () => {
    const { app, data } = await fresh();
    data.loadClip.mockResolvedValue(makePairClip('a', 4));
    await app.openClip('a');
    app.setTrack(app.BOTH);
    await app.setFocusNode('gt:1');
    expect(app.state.temporal).not.toBe(null);
    const d = deferred();
    data.fetchTemporal.mockReturnValueOnce(d.promise);
    const follow = app.setFocusNode('auto:1');
    expect(app.state.temporal).toBe(null);
    d.resolve({ nodes: [], relations: [] });
    await follow;
  });

  it('picks the automatic region on the cloud before the annotated region that contains it', async () => {
    const { app, data, THREE } = await fresh();
    data.loadClip.mockResolvedValue(makePairClip('a', 4));
    const cloud = new THREE.Group().add(new THREE.Points(new THREE.BufferGeometry().setFromPoints(
      [[0, 0, -1], [5, 5, -1], [-5, 5, -1], [5, -5, -1]].map((p) => new THREE.Vector3(...p)))));
    data.loadCloud.mockResolvedValue({ scene: cloud });
    data.fetchRegions.mockImplementation((clip, track) => Promise.resolve({ grid: [2, 2], classes: [], classById: new Map(),
      stage: 'anchor', labels: Uint16Array.from(track === 'gt' ? [3, 3, 3, 3] : [7, 0, 0, 0]) }));
    await app.openClip('a');
    // A click in the middle of the canvas casts the camera's ray down -z, through the first point.
    const click = () => app.pickNode({ getBoundingClientRect: () => ({ left: 0, top: 0, width: 100, height: 100 }) }, { clientX: 50, clientY: 50 });
    app.setTrack(app.BOTH);
    expect(click()).toBe('auto:7');
    app.setTrack('gt');
    expect(click()).toBe('gt:3');
  });
});
