import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { filteredGeometry, srcVertexIndex, fullGeometryOf } from './vertex_filter.js';

/** n vertices whose x coordinate IS the vertex index, so a mix-up is visible. */
function cloud(n, { interleaved = false } = {}) {
  const g = new THREE.BufferGeometry();
  if (interleaved) {
    // Position (3) and colour (4) sharing one buffer.
    const stride = 7;
    const arr = new Float32Array(n * stride);
    for (let i = 0; i < n; i++) {
      arr[i * stride] = i; arr[i * stride + 1] = 0; arr[i * stride + 2] = 0;
      arr[i * stride + 3] = i; arr[i * stride + 4] = 0; arr[i * stride + 5] = 0; arr[i * stride + 6] = 1;
    }
    const ib = new THREE.InterleavedBuffer(arr, stride);
    g.setAttribute('position', new THREE.InterleavedBufferAttribute(ib, 3, 0));
    g.setAttribute('color', new THREE.InterleavedBufferAttribute(ib, 4, 3));
  } else {
    const pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) pos[i * 3] = i;
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const col = new Uint8Array(n * 4);
    for (let i = 0; i < n; i++) col[i * 4] = i;
    g.setAttribute('color', new THREE.BufferAttribute(col, 4, true));
  }
  return g;
}

describe('filteredGeometry', () => {
  it('keeps exactly the unmasked vertices, in order', () => {
    const out = filteredGeometry(cloud(6), Uint8Array.from([0, 1, 1, 0, 0, 1]));
    const pos = out.attributes.position;
    expect(pos.count).toBe(3);
    expect([pos.getX(0), pos.getX(1), pos.getX(2)]).toEqual([0, 3, 4]);
  });

  it('records srcIndex so the compaction stays reversible', () => {
    const out = filteredGeometry(cloud(6), Uint8Array.from([0, 1, 1, 0, 0, 1]));
    expect([...out.userData.srcIndex]).toEqual([0, 3, 4]);
  });

  it('preserves a normalized Uint8 colour attribute raw, not rescaled', () => {
    const out = filteredGeometry(cloud(6), Uint8Array.from([1, 0, 0, 0, 0, 0]));
    const col = out.attributes.color;
    expect(col.normalized).toBe(true);
    expect(col.array[0]).toBe(1);            // vertex 1's raw 0–255 value survives
  });

  it('de-interleaves without losing the per-vertex correspondence', () => {
    const out = filteredGeometry(cloud(6, { interleaved: true }), Uint8Array.from([1, 1, 0, 0, 0, 0]));
    expect(out.attributes.position.getX(0)).toBe(2);
    expect(out.attributes.color.getX(0)).toBe(2);   // same vertex on both attributes
    expect([...out.userData.srcIndex]).toEqual([2, 3, 4, 5]);
  });
});

describe('srcVertexIndex', () => {
  // With point 1 dropped, a click on filtered index 1 is the original point 2.
  it('maps a filtered hit index back to the original vertex', () => {
    const obj = { geometry: filteredGeometry(cloud(5), Uint8Array.from([0, 1, 0, 0, 0])) };
    expect(srcVertexIndex(obj, 0)).toBe(0);
    expect(srcVertexIndex(obj, 1)).toBe(2);
    expect(srcVertexIndex(obj, 2)).toBe(3);
    expect(srcVertexIndex(obj, 3)).toBe(4);
  });

  it('is the identity on an unfiltered geometry', () => {
    const obj = { geometry: cloud(5) };
    expect(srcVertexIndex(obj, 3)).toBe(3);
  });

  it('returns -1 rather than a wrong vertex for an unusable index', () => {
    const obj = { geometry: filteredGeometry(cloud(5), Uint8Array.from([0, 1, 0, 0, 0])) };
    expect(srcVertexIndex(obj, 4)).toBe(-1);        // past the filtered count
    expect(srcVertexIndex(obj, -1)).toBe(-1);
    expect(srcVertexIndex(obj, undefined)).toBe(-1);
    expect(srcVertexIndex(null, 0)).toBe(-1);
  });

  it('survives the drop-everything mask', () => {
    const obj = { geometry: filteredGeometry(cloud(3), Uint8Array.from([1, 1, 1])) };
    expect(obj.geometry.attributes.position.count).toBe(0);
    expect(srcVertexIndex(obj, 0)).toBe(-1);
  });
});

describe('fullGeometryOf', () => {
  it('prefers the pristine geometry the filter stashed', () => {
    const full = cloud(5);
    const obj = { geometry: filteredGeometry(full, Uint8Array.from([0, 1, 0, 0, 0])), userData: { fullGeometry: full } };
    expect(fullGeometryOf(obj).attributes.position.count).toBe(5);
  });

  it('falls back to the live geometry when the filter never ran', () => {
    const obj = { geometry: cloud(5), userData: {} };
    expect(fullGeometryOf(obj).attributes.position.count).toBe(5);
  });

  it('is null-safe', () => {
    expect(fullGeometryOf(null)).toBe(null);
    expect(fullGeometryOf({})).toBe(null);
  });
});
