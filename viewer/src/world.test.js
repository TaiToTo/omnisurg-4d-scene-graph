import * as THREE from 'three';
import { describe, expect, it } from 'vitest';
import { focusWorldFrame, nearestStacked, worldIndices } from './world.js';

describe('the world stack', () => {
  it('samples a dozen frames, the first and the last always among them', () => {
    expect(worldIndices(14)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]);
    const long = worldIndices(119);
    expect(long[0]).toBe(0);
    expect(long.at(-1)).toBe(119);
    expect(long.length).toBeLessThanOrEqual(13);
  });

  it('puts the nearest stacked frame on stage, the earlier one on a tie', () => {
    expect(nearestStacked(7, [0, 5, 10])).toBe(5);
    expect(nearestStacked(8, [0, 5, 10])).toBe(10);
    expect(nearestStacked(5, [0, 4, 6])).toBe(4);
  });
});

describe('the frames on view', () => {
  const stack = () => {
    const indices = [0, 5, 10, 15];
    return {
      indices,
      frames: indices.map((i) => {
        const group = new THREE.Group();
        group.add(new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ size: 2 })));
        return { i, group };
      }),
    };
  };
  const visible = (world) => world.frames.filter((fr) => fr.group.visible).map((fr) => fr.i);

  it('shows every frame in overlay, and the stage with its neighbours in spotlight', () => {
    const world = stack();
    focusWorldFrame(world, 5, 'overlay', false);
    expect(visible(world)).toEqual([0, 5, 10, 15]);
    focusWorldFrame(world, 0, 'spotlight', false);
    expect(visible(world)).toEqual([0, 5, 10]);
  });

  it('shows the stage alone while a region is isolated, in either view', () => {
    const world = stack();
    for (const view of ['spotlight', 'overlay']) {
      focusWorldFrame(world, 10, view, true);
      expect(visible(world)).toEqual([10]);
    }
  });
});
