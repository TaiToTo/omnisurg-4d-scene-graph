"""Boundary pixels and their agreement, pinned on frames small enough to count by hand.

The frames are 60 x 100 px: class 1 fills the left half and class 2 the right
half, and a predicted map puts its edge `k` px to the right of the true one.
"""
import numpy as np
import pytest

from evalkit.boundary import BOUNDARY_TOL_PX, BoundaryScore, boundary_pixels, boundary_score, within_tolerance

H, W = 60, 100


# These are the `exact` and `shifted(k)` frames of `tests/scenes.py`, built
# here because that file is on its way to main on another branch and this one
# depends on nothing under review. Once both are on main they are built from
# it, so that every metric is tested on one set of scenes.
def halves(col: int = 50, ids=(1, 2)) -> np.ndarray:
    labels = np.empty((H, W), dtype=np.int32)
    labels[:, :col] = ids[0]
    labels[:, col:] = ids[1]
    return labels


def test_both_sides_of_an_edge_are_marked_so_a_boundary_is_two_pixels_wide():
    b = boundary_pixels(halves())
    cols = np.flatnonzero(b.any(axis=0))
    assert cols.tolist() == [49, 50]
    assert b[:, 49].all() and b[:, 50].all()
    assert int(b.sum()) == 2 * H


def test_a_map_of_one_label_has_no_boundary():
    assert not boundary_pixels(np.ones((H, W), dtype=np.int32)).any()


def test_an_edge_against_a_pixel_that_is_not_scored_is_not_a_boundary():
    # The top ten rows hold a third label and are removed (unannotated
    # background, say). Without the mask the row 9 / row 10 edge is a boundary
    # across the whole width; with it, row 10 keeps only the class 1 / class 2
    # edge, and the vertical edge keeps its fifty scored rows.
    labels = halves()
    labels[:10] = 0
    scored = np.ones((H, W), dtype=bool)
    scored[:10] = False
    assert boundary_pixels(labels)[10].all()
    b = boundary_pixels(labels, scored)
    assert not b[:10].any()
    assert np.flatnonzero(b[10]).tolist() == [49, 50]
    assert int(b.sum()) == 2 * (H - 10)


def test_a_horizontal_edge_is_marked_like_a_vertical_one():
    # The same frame turned on its side: the `down` pass has to find what the
    # `across` pass found.
    labels = np.ascontiguousarray(halves().T)
    b = boundary_pixels(labels)
    assert (b == boundary_pixels(halves()).T).all()
    assert np.flatnonzero(b.any(axis=1)).tolist() == [49, 50]


def test_without_a_scored_mask_every_pixel_is_scored_which_is_the_pilot_rule():
    # Pilot mode marks boundaries on the whole map and masks afterwards, so an
    # edge against background is a boundary there.
    labels = halves()
    labels[:10] = 0
    valid = labels != 0
    pilot = boundary_pixels(labels) & valid
    normal = boundary_pixels(labels, valid)
    assert pilot[10, 0] and not normal[10, 0]


def test_an_objects_bool_mask_is_a_two_label_map():
    # `inst_BF` takes the boundary of one object's mask, which arrives as bool.
    labels = halves()
    assert (boundary_pixels(labels == 1) == boundary_pixels(labels)).all()


def test_a_label_map_must_be_a_2d_integer_array_with_a_matching_bool_mask():
    with pytest.raises(ValueError, match=r"\(H, W\)"):
        boundary_pixels(np.zeros((H, W, 3), dtype=np.int32))
    with pytest.raises(ValueError, match="integer or bool"):
        boundary_pixels(np.zeros((H, W), dtype=np.float32))
    with pytest.raises(ValueError, match="bool"):
        boundary_pixels(halves(), np.ones((H, W), dtype=np.uint8))
    with pytest.raises(ValueError, match="shape"):
        boundary_pixels(halves(), np.ones((H, W - 1), dtype=bool))


def test_the_tolerance_is_a_square_not_a_disk():
    b = np.zeros((H, W), dtype=bool)
    b[10, 10] = True
    near = within_tolerance(b, 2)
    assert near[12, 12] and near[8, 8] and near[12, 8]   # the corners of the square
    assert not near[13, 10] and not near[10, 13]
    assert int(near.sum()) == 25
    assert (within_tolerance(b, 0) == b).all()
    assert (within_tolerance(b, np.int64(2)) == near).all()   # a tolerance read from an array
    for bad in (-1, 1.5, True, np.bool_(True)):
        with pytest.raises(ValueError, match="non-negative integer"):
            within_tolerance(b, bad)


def test_an_exact_boundary_scores_one_at_every_tolerance():
    gt = boundary_pixels(halves())
    for tol in (0, 1, BOUNDARY_TOL_PX, 5):
        assert boundary_score(gt, gt, tol) == BoundaryScore(1.0, 1.0, 1.0)


@pytest.mark.parametrize("shift, expected", [(1, 1.0), (2, 1.0), (3, 0.5), (4, 0.0)])
def test_how_far_an_edge_may_be_off_at_the_default_tolerance(shift, expected):
    # The GT boundary is columns 49 and 50, the predicted one 49 + k and
    # 50 + k. With a square tolerance of 2, a predicted pixel counts when a
    # GT pixel lies within 2 columns of it, so an edge 2 px off still scores
    # in full, one 3 px off has one of its two columns within reach (half the
    # pixels, each way), and one 4 px off has none.
    gt = boundary_pixels(halves())
    pred = boundary_pixels(halves(50 + shift))
    s = boundary_score(pred, gt, BOUNDARY_TOL_PX)
    assert (s.precision, s.recall, s.f) == (expected, expected, expected)


def test_at_tolerance_zero_an_edge_one_pixel_off_scores_half():
    gt = boundary_pixels(halves())
    pred = boundary_pixels(halves(51))
    assert boundary_score(pred, gt, 0) == BoundaryScore(0.5, 0.5, 0.5)


def test_precision_and_recall_are_taken_from_their_own_sides():
    # Two predicted edges for one true edge: every GT pixel is recovered, but
    # half of the predicted pixels are far from any GT pixel.
    gt = boundary_pixels(halves())
    pred = boundary_pixels(halves()) | boundary_pixels(halves(80))
    s = boundary_score(pred, gt, BOUNDARY_TOL_PX)
    assert (s.precision, s.recall) == (0.5, 1.0)
    assert s.f == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_no_gt_boundary_leaves_the_score_undefined_and_no_predicted_boundary_scores_zero():
    none = np.zeros((H, W), dtype=bool)
    some = boundary_pixels(halves())
    assert boundary_score(some, none) is None
    assert boundary_score(none, none) is None
    assert boundary_score(none, some) == BoundaryScore(0.0, 0.0, 0.0)


def test_boundaries_must_be_bool_arrays_of_one_shape():
    some = boundary_pixels(halves())
    with pytest.raises(ValueError, match="bool"):
        boundary_score(some.astype(np.uint8), some)
    with pytest.raises(ValueError, match="one shape"):
        boundary_score(some, some[:, :-1])
