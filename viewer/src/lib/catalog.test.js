import { describe, it, expect } from 'vitest';
import { CATALOG_FORMAT, parseCatalog, groupClips } from './catalog.js';

function catalog() {
  return {
    format: CATALOG_FORMAT,
    tracks: {
      gt: { label: 'Manual annotation', short: 'Manual', semantic: true, seeded: false, anchor_badge: 'ANNOTATED FRAME', tracked_badge: 'annotated region' },
      auto: { label: 'Automatic', short: 'Auto', semantic: false, seeded: true, anchor_badge: 'SEED', tracked_badge: 'tracked region' },
    },
    clips: [
      { path: 'cholec/VID01_a', dataset: 'CholecSeg8k', group: 'VID01', label: 'VID01_a', frames: 30, geometry: 'da3', hierarchy: true, tracks: ['gt', 'auto'], coverage: { gt: 22, auto: 30 } },
      { path: 'cholec/VID01_b', dataset: 'CholecSeg8k', group: 'VID01', label: 'VID01_b', frames: 30, geometry: 'da3', hierarchy: false, tracks: ['auto'], coverage: { auto: 30 } },
      { path: 'atlas/x__y__gt_0001', dataset: 'ATLAS-120k', group: 'x / y', label: 'gt_0001', frames: 21, geometry: 'pi3x', hierarchy: false, tracks: ['auto'], coverage: { auto: 21 } },
    ],
  };
}

describe('parseCatalog', () => {
  it('reads tracks and clips into the viewer’s shape', () => {
    const c = parseCatalog(catalog());
    expect(c.tracks.get('gt')).toMatchObject({ semantic: true, anchorBadge: 'ANNOTATED FRAME' });
    expect(c.byPath.get('atlas/x__y__gt_0001').geometry).toBe('pi3x');
    expect(c.clips[0].coverage).toEqual({ gt: 22, auto: 30 });
  });

  // Each case plants one fault the viewer would otherwise turn into a bad URL,
  // an unknown track or half a clip, and names the refusal it must meet.
  const faults = {
    'another format': [(j) => { j.format = 'something/2'; }, /format is not/],
    'no clip': [(j) => { j.clips = []; }, /has no clip/],
    'a path that climbs out': [(j) => { j.clips[0].path = 'cholec/../../etc'; }, /relative path of plain names/],
    'an absolute path': [(j) => { j.clips[0].path = '/cholec/VID01_a'; }, /relative path of plain names/],
    'a path with a query': [(j) => { j.clips[0].path = 'cholec/VID01_a?x=1'; }, /relative path of plain names/],
    'a clip listed twice': [(j) => { j.clips[1].path = j.clips[0].path; }, /listed twice/],
    'a track not in the table': [(j) => { j.clips[0].tracks.push('other'); j.clips[0].coverage.other = 1; }, /not in the track table/],
    'a clip without a track': [(j) => { j.clips[1].tracks = []; }, /has no track/],
    'a track id that is not plain': [(j) => { j.tracks['a b'] = j.tracks.gt; }, /is not lower-case letters/],
    'a track without a label': [(j) => { delete j.tracks.gt.label; }, /label is not a non-empty string/],
    'a track whose semantic is not a boolean': [(j) => { j.tracks.gt.semantic = 'yes'; }, /semantic is not a boolean/],
    'frames that are not a positive integer': [(j) => { j.clips[0].frames = 0; }, /frames is not a positive integer/],
    'a geometry that is not an id': [(j) => { j.clips[0].geometry = 'pi3x/../x'; }, /geometry is not an id/],
    'coverage above the frame count': [(j) => { j.clips[0].coverage.gt = 31; }, /coverage of gt/],
    'coverage missing for a track': [(j) => { delete j.clips[0].coverage.auto; }, /coverage of auto/],
    'hierarchy that is not a boolean': [(j) => { j.clips[0].hierarchy = 1; }, /hierarchy is not a boolean/],
    'an empty dataset name': [(j) => { j.clips[0].dataset = ' '; }, /dataset is not a non-empty string/],
  };
  for (const [name, [plant, message]] of Object.entries(faults)) {
    it(`refuses ${name}`, () => {
      const j = catalog();
      plant(j);
      expect(() => parseCatalog(j)).toThrow(message);
    });
  }
});

describe('groupClips', () => {
  it('groups by dataset, then group, in catalog order', () => {
    const g = groupClips(parseCatalog(catalog()));
    expect(g.map((d) => d.dataset)).toEqual(['CholecSeg8k', 'ATLAS-120k']);
    expect(g[0].groups[0].clips.map((c) => c.label)).toEqual(['VID01_a', 'VID01_b']);
  });
});
