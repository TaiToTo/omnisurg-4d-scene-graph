// Drive main.js against a stand-in app.js, to check the picker and the address.
import { beforeEach, describe, expect, it, vi } from 'vitest';

const pending = {};
const state = {};

vi.mock('./stage.js', () => ({ initStage: () => true, invalidate() {}, resetView() {}, startRenderLoop() {} }));
vi.mock('./data.js', () => ({
  loadCatalog: async () => ({ format: 1, clips: ['X', 'A', 'B'].map((id) => ({ id, dataset_name: 'd', procedure: 'p', video: 'v', n_frames: 2 })) }),
}));
vi.mock('./app.js', () => {
  const noop = () => {};
  return {
    state,
    BOTH: 'both',
    // As app.js's openClip: a sequence token, the record, then the first frame; each waits for the test.
    openClip: async (id) => {
      const seq = ++state.clipSeq;
      await new Promise((r) => { pending[`${id}:record`] = r; });
      if (state.clipSeq !== seq) return false;
      state.clip = { id, n_frames: 2, trackById: new Map([['t', {}]]), frames: [], hierarchy: state.hierarchy };
      await new Promise((r) => { pending[`${id}:frame`] = r; });
      return state.clipSeq === seq;
    },
    cycleCloudShow: noop, onChange: noop, onError: noop, pickNode: noop, refitPanels: noop, setCloudShow: noop,
    setFocusNode: vi.fn(), setFocusStage: noop, setGraph3D: noop, setHover: noop, setIsolate: noop, setMapBig: noop,
    setPlaying: noop, setRegionsMenu: noop, setStatus: noop, setStripCellHeight: noop, setStripPhoto: noop,
    setWorldView: noop, seekFrame: vi.fn(), setTrack: vi.fn(), stepFrame: noop, stripCellHeight: () => 60,
    switchMode: noop, syncControls: noop, toggleRegions: noop,
  };
});

function el() {
  const listeners = {};
  return {
    listeners, hidden: true, value: '', innerHTML: '', textContent: '', style: {}, classList: { add() {}, remove() {} },
    addEventListener(t, f) { (listeners[t] ??= []).push(f); }, appendChild() {}, blur() {}, focus() {},
    querySelector: () => null, contains: () => false, setPointerCapture() {}, removeEventListener() {},
  };
}
let els = {};
const tick = () => new Promise((r) => setTimeout(r, 0));

/** Start the page on address `search`, and let the clip it opens finish loading. */
async function start(search, { hierarchy = null } = {}) {
  vi.resetModules();
  vi.clearAllMocks();
  els = {};
  Object.assign(state, { clip: null, clipSeq: 0, frame: 0, mode: 'frame', focusKey: null, track: null, hierarchy });
  globalThis.document = {
    getElementById: (id) => (els[id] ??= el()), querySelectorAll: () => [], addEventListener() {},
    createElement: () => el(), documentElement: { style: { setProperty() {} } },
  };
  globalThis.location = { search, href: `http://localhost/${search}` };
  globalThis.ResizeObserver = class { observe() {} };
  globalThis.requestAnimationFrame = () => 0;
  await import('./main.js');
  await tick();
  const id = new URLSearchParams(search).get('clip') ?? 'X';
  pending[`${id}:record`](); await tick(); pending[`${id}:frame`](); await tick();
}

beforeEach(() => { vi.spyOn(console, 'error').mockImplementation(() => {}); });

describe('the clip picker', () => {
  it('names the clip on screen after two quick picks', async () => {
    await start('');
    const sel = els['clip-select'];
    sel.value = 'A'; sel.listeners.change[0]();
    await tick(); pending['A:record'](); await tick();
    // B is picked while A's first frame loads, and A's frame arrives before B's record.
    sel.value = 'B'; sel.listeners.change[0]();
    await tick(); pending['A:frame'](); await tick();
    pending['B:record'](); await tick(); pending['B:frame'](); await tick();
    expect(state.clip.id).toBe('B');
    expect(sel.value).toBe('B');
  });

  it('names the clip still on screen when the one picked does not load', async () => {
    await start('');
    const sel = els['clip-select'];
    sel.value = 'A'; sel.listeners.change[0]();
    // openClip ends false with X still on screen, as when A's record does not load.
    state.clipSeq++;
    await tick(); pending['A:record'](); await tick();
    expect(sel.value).toBe('X');
  });
});

describe('the address', () => {
  it('opens a link\'s frame, track and region without a notice', async () => {
    await start('?clip=A&frame=1&track=t&focus=t:3');
    const app = await import('./app.js');
    expect(app.setTrack).toHaveBeenLastCalledWith('t');
    expect(app.seekFrame).toHaveBeenLastCalledWith(1);
    expect(app.setFocusNode).toHaveBeenLastCalledWith('t:3');
    expect(els['toast-text']?.textContent ?? '').toBe('');
  });

  it('names a link\'s settings the clip lacks', async () => {
    // The clip has frames 0 and 1, and one track, t.
    await start('?clip=A&frame=2&track=nope&focus=other:3');
    const app = await import('./app.js');
    expect(els.toast.hidden).toBe(false);
    expect(els['toast-text'].textContent).toBe('This clip has no track nope, frame 2, region other:3; the link opens without it.');
    expect(app.setFocusNode).not.toHaveBeenCalled();
  });

  it('tells the reader of a frame that is not a number', async () => {
    await start('?clip=A&frame=two');
    expect(els['toast-text'].textContent).toBe('This clip has no frame two; the link opens without it.');
  });

  it('names a frame that is not a whole number once, and does not seek it', async () => {
    await start('?clip=A&frame=1.5');
    const app = await import('./app.js');
    expect(app.seekFrame).not.toHaveBeenCalled();
    expect(console.error).toHaveBeenCalledTimes(1);
    expect(els['toast-text'].textContent).toBe('This clip has no frame 1.5; the link opens without it.');
  });

  it('shows both tracks on a clip with a hierarchy, and names track=both on one without', async () => {
    await start('?clip=A&track=both', { hierarchy: { tracks: ['t', 'u'], frames: [0] } });
    const app = await import('./app.js');
    expect(app.setTrack).toHaveBeenLastCalledWith('both');
    expect(els['toast-text']?.textContent ?? '').toBe('');
    await start('?clip=A&track=both');
    expect(els['toast-text'].textContent).toBe('This clip has no track both; the link opens without it.');
  });
});
