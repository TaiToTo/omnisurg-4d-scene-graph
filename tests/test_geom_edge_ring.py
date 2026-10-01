"""The ring removal in `geom_edge_map` removes the ring and nothing else.

The normals are central differences, so on the outer pixel of the image and
next to an invalid pixel they are 0 or NaN; the 1 - cos to the neighbour is
then 1, the strongest possible crease, and the depth gradient looks like a
step at the rim of a hole. A contour standing there is not geometry.
`EDGE_MASK_RING` zeroes that ring, and must not change a single pixel
inside it: if it did, regenerating the edge conditions would measure some
other change under the name of ring removal, and the images would look the
same.
"""

import numpy as np
import pytest

from surgical_core.geometry import normals as geo


def _smooth_depth_with_hole():
    """A smooth slope (no real geometric boundary) with a depth hole: only
    the ring should stand."""
    h, w = 60, 80
    yy, xx = np.mgrid[0:h, 0:w]
    d = 0.30 + 0.0005 * xx + 0.0003 * yy
    d[25:35, 30:45] = 0.0                      # invalid: a depth hole
    k = np.array([[150.0, 0, w / 2], [0, 150.0, h / 2], [0, 0, 1]])
    return d, k


def test_ring_is_the_only_difference():
    """With the ring removed, the trustworthy pixels are bitwise the same."""
    d, k = _smooth_depth_with_hole()
    m = np.isfinite(d) & (d > 1e-6)
    inside = geo.edge_reliable_mask(m)
    for parts in ("both", "normal", "depth"):
        old = geo.geom_edge_map(d, k, parts=parts, mask_ring=False)
        new = geo.geom_edge_map(d, k, parts=parts, mask_ring=True)
        assert np.array_equal(old[inside], new[inside]), parts
        assert not new[~inside].any(), parts


def test_smooth_surface_has_no_edge_left():
    """A smooth surface has no real crease, so after the ring goes no strong
    edge remains."""
    d, k = _smooth_depth_with_hole()
    old = geo.geom_edge_map(d, k, mask_ring=False)
    new = geo.geom_edge_map(d, k, mask_ring=True)
    assert (old > 0.5).sum() > 0                # the ring did stand before
    assert (new > 0.5).sum() == 0


def test_ring_covers_image_border_and_hole_rim():
    """The ring is 2 pixels, matched to the crease term: 2 at the border and
    2 around invalid pixels."""
    d, k = _smooth_depth_with_hole()
    m = np.isfinite(d) & (d > 1e-6)
    r = geo.edge_reliable_mask(m)
    assert not r[:2, :].any() and not r[-2:, :].any()
    assert not r[:, :2].any() and not r[:, -2:].any()
    assert not r[25 - 2:35 + 2, 30 - 2:45 + 2].any()      # the hole plus 2
    assert r[25 - 3, 30 - 3] and r[2, 2]                  # one further out is trusted


def test_dense_depth_masks_only_the_border():
    """With no invalid pixel at all, only the outer 2 pixels go."""
    h, w = 40, 50
    yy, xx = np.mgrid[0:h, 0:w]
    d = 0.30 + 0.0005 * xx
    r = geo.edge_reliable_mask(np.isfinite(d) & (d > 1e-6))
    assert r[2:-2, 2:-2].all()
    assert r.sum() == (h - 4) * (w - 4)


@pytest.mark.parametrize("parts", ["both", "normal", "depth"])
def test_default_follows_module_flag(monkeypatch, parts):
    """`mask_ring=None` follows the module flag, which is how old labels are
    reproduced."""
    d, k = _smooth_depth_with_hole()
    monkeypatch.setattr(geo, "EDGE_MASK_RING", False)
    assert np.array_equal(geo.geom_edge_map(d, k, parts=parts),
                          geo.geom_edge_map(d, k, parts=parts, mask_ring=False))
    monkeypatch.setattr(geo, "EDGE_MASK_RING", True)
    assert np.array_equal(geo.geom_edge_map(d, k, parts=parts),
                          geo.geom_edge_map(d, k, parts=parts, mask_ring=True))
