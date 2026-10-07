"""The ring removal in `geom_edge_map` removes the ring and nothing else.

The normals are central differences, so on the outer pixel of the image and
next to an invalid pixel they are 0 or NaN; the 1 - cos to the neighbour is
then 1, the strongest possible crease, and the depth gradient looks like a
step at the rim of a hole. A contour standing there is not geometry.
`mask_ring` zeroes that ring, and must not change a single pixel inside
it: if it did, regenerating the edge conditions would measure some other
change under the name of ring removal, and the images would look the same.
The setting is an argument all the way from the segmenter input down.
"""

import numpy as np
import pytest

import surgical_core.geometry as geometry
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
    m = geometry.valid_depth_mask(d)
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
    m = geometry.valid_depth_mask(d)
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
    r = geo.edge_reliable_mask(geometry.valid_depth_mask(d))
    assert r[2:-2, 2:-2].all()
    assert r.sum() == (h - 4) * (w - 4)


def test_no_module_flag_decides_the_ring():
    """No module global sets the ring for a whole process: one process could
    then make two settings' inputs under one record."""
    assert not hasattr(geo, "EDGE_MASK_RING")
    assert not hasattr(geometry, "EDGE_MASK_RING")
    assert not hasattr(geometry, "EDGE_RING_PX")


@pytest.mark.parametrize("parts", ["both", "normal", "depth"])
def test_the_ring_is_masked_by_default(parts):
    d, k = _smooth_depth_with_hole()
    assert np.array_equal(geo.geom_edge_map(d, k, parts=parts),
                          geo.geom_edge_map(d, k, parts=parts, mask_ring=True))


@pytest.mark.parametrize("bad", [None, 0, 1, "yes"])
def test_a_ring_setting_that_is_not_a_bool_is_refused(bad):
    d, k = _smooth_depth_with_hole()
    with pytest.raises(ValueError, match="mask_ring is True or False"):
        geo.geom_edge_map(d, k, mask_ring=bad)


@pytest.mark.parametrize("mode", ["normal_edge", "rgb_edge"])
def test_the_segmenter_input_passes_the_ring_setting_down(mode):
    """An edge input made with the ring left in differs from the default, and
    equals the edges burnt with the ring left in."""
    render = pytest.importorskip("surgical_core.geometry.render")
    d, k = _smooth_depth_with_hole()
    rgb = np.full((*d.shape, 3), 200, np.uint8)
    kept = render.sam_input_image(mode, d, k, None, rgb, mask_ring=False)
    assert not np.array_equal(kept, render.sam_input_image(mode, d, k, None, rgb))
    want = (geo.normal_edge_map(d, k, mask_ring=False) if mode == "normal_edge"
            else geo.burn_geom_edge(rgb, d, k, mask_ring=False))
    assert np.array_equal(kept, want)
