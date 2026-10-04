"""`edge_reliable_mask` removes a ring exactly `EDGE_RING_PX` wide, for any width.

The width is 2 and is not meant to move. The test exists because the border
slices are written as `r[-px:]`, and `r[-0:]` is the whole array: a width of
0 used to mask every pixel instead of none, and nothing said so.
"""

import numpy as np

from surgical_core.geometry import normals as geo


def _mask():
    m = np.ones((20, 30), dtype=bool)
    m[8:12, 10:15] = False
    return m


def test_width_zero_removes_nothing(monkeypatch):
    monkeypatch.setattr(geo, "EDGE_RING_PX", 0)
    m = _mask()
    assert np.array_equal(geo.edge_reliable_mask(m), m)


def test_width_two_removes_the_ring_and_the_border(monkeypatch):
    monkeypatch.setattr(geo, "EDGE_RING_PX", 2)
    m = _mask()
    r = geo.edge_reliable_mask(m)
    assert not r[:2].any() and not r[-2:].any() and not r[:, :2].any() and not r[:, -2:].any()
    assert not r[6:14, 8:17].any()            # the hole, grown by 2 on every side
    assert r[2:6, 2:-2].all()                  # untouched pixels stay
    assert r[14:-2, 2:-2].all()
