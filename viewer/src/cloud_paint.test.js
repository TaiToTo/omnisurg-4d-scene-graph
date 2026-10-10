import { describe, expect, it } from 'vitest';
import { cloudDropMask } from './cloud_paint.js';

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
