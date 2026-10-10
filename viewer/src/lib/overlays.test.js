import { describe, it, expect } from 'vitest';
import { adaptOverlays, alignSeg, manifestForSource, sourceGrid } from './overlays.js';

// A cloud of `n` vertices, in the shape adaptOverlays walks a group.
function fakeGroup(n) {
  const attr = { count: n, getX: (i) => i * 0.01, getY: (i) => i * 0.02, getZ: (i) => 1 + i * 0.03 };
  const points = { isPoints: true, geometry: { attributes: { position: attr } } };
  return { traverse: (fn) => fn(points) };
}

// Labels on a 2×3 grid; the Pi3X grid is 4×3, so they are resampled to 12.
const seg = { vertex_seg: [1, 1, 1, 1, 1, 1], width: 2, height: 3 };
const graph = { nodes: [{ id: 1, pos: [9, 9, 9], label: 'liver' }], edges: [] };
const manifest = { geometry_sources: { pi3x: { resolution: [4, 3] } }, depth_info: { depth_shape: [5, 3, 2] } };
const fit = (group, overlays = {}) => adaptOverlays({
  segByTrack: { gt: seg }, graphByTrack: { gt: graph }, ...overlays, group, manifest, source: 'pi3x',
});

describe('sourceGrid', () => {
  it('reads a source’s resolution as [W, H], and the default source’s from the depth shape', () => {
    expect(sourceGrid(manifest, 'pi3x')).toEqual({ w: 4, h: 3 });
    expect(sourceGrid(manifest, 'da3')).toEqual({ w: 2, h: 3 });
    expect(sourceGrid(manifest, 'other')).toBeNull();
  });
});

describe('alignSeg', () => {
  it('refuses labels that are not the whole source grid', () => {
    expect(alignSeg({ ...seg, vertex_seg: [1, 1, 1] }, { w: 4, h: 3 })).toBeNull();
  });

  it('refuses an unknown grid', () => {
    expect(alignSeg(seg, null)).toBeNull();
  });
});

describe('adaptOverlays on another geometry source', () => {
  it('reads node geometry off the cloud when the counts agree', () => {
    const out = fit(fakeGroup(12));
    expect(out.aligned).toBe(true);
    expect(out.graphByTrack.gt.nodes[0].pos).not.toEqual([9, 9, 9]);
    expect(out.graphByTrack.gt.nodes[0].label).toBe('liver');
  });

  it('empties the graph and says so when the cloud and the labels disagree', () => {
    const out = fit(fakeGroup(10));
    expect(out.graphByTrack.gt.nodes).toEqual([]);
    expect(out.aligned).toBe(false);
  });

  it('empties a graph that has no labels on this frame, rather than drawing it at the default cloud’s coordinates', () => {
    const out = fit(fakeGroup(12), { segByTrack: { gt: null } });
    expect(out.graphByTrack.gt.nodes).toEqual([]);
    expect(out.aligned).toBe(false);
  });

  it('leaves the default source alone', () => {
    const out = adaptOverlays({ segByTrack: { gt: seg }, graphByTrack: { gt: graph }, group: fakeGroup(6), manifest, source: 'da3' });
    expect(out.graphByTrack.gt).toBe(graph);
    expect(out.segByTrack.gt).toBe(seg);
    expect(out.aligned).toBe(true);
  });
});

describe('manifestForSource', () => {
  const m = {
    n_frames: 2,
    frames: [
      { glb_centroid: [0, 0, 0], camera_pos_glb: [1, 1, 1], geometry_sources: { pi3x: { glb_centroid: [5, 5, 5], camera_pos_glb: [6, 6, 6] } } },
      { glb_centroid: [0, 0, 1], camera_pos_glb: [1, 1, 2], geometry_sources: { pi3x: { glb_centroid: [5, 5, 6], camera_pos_glb: [6, 6, 7] } } },
    ],
  };

  it('puts the source’s pose in each frame’s own fields', () => {
    const out = manifestForSource(m, 'pi3x');
    expect(out.frames[1].glb_centroid).toEqual([5, 5, 6]);
    expect(m.frames[1].glb_centroid).toEqual([0, 0, 1]);
  });

  it('returns the manifest itself for the default source', () => {
    expect(manifestForSource(m, 'da3')).toBe(m);
  });

  it('refuses a frame without the source’s pose', () => {
    const broken = { ...m, frames: [m.frames[0], { glb_centroid: [0, 0, 1] }] };
    expect(() => manifestForSource(broken, 'pi3x')).toThrow(/frame 1 has no pi3x pose/);
  });
});
