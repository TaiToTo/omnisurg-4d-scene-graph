"""`inst_BF`, derived by hand on the 60 x 100 scenes of `tests/scenes.py`.

GT class 1 fills the left half and class 2 the right half (3,000 px each).
Each object's boundary is the two columns on either side of the true edge,
so a hit's score follows the shifted-edge table of `tests/test_boundary.py`.
"""
import numpy as np
import pytest

import scenes as S
from evalkit.boundary import BOUNDARY_TOL_PX, BoundaryScore
from evalkit.inst_bf import InstanceBoundary, instance_boundary_f
from evalkit.objects import gt_objects, instance_scores, predicted_objects

ALL = np.ones((S.H, S.W), dtype=bool)


def score(scene: S.Scene, scored: np.ndarray = ALL, **kw) -> InstanceBoundary:
    gt = gt_objects(scene.gt, scored)
    pred = predicted_objects(scene.lab, scored)
    return instance_boundary_f(gt, pred, instance_scores(gt, pred), **kw)


def test_an_exact_prediction_scores_one_on_every_hit():
    r = score(S.exact())
    assert r == InstanceBoundary(
        inst_bf=1.0, scores=(BoundaryScore(1.0, 1.0, 1.0),) * 2, n_hits=2, n_entered=2,
    )


@pytest.mark.parametrize("shift, expected", [(1, 1.0), (2, 1.0), (3, 0.5), (4, 0.0)])
def test_a_shifted_edge_costs_each_hit_what_it_costs_the_frame(shift, expected):
    # Both objects stay hits (IoU 50/54 and 46/50 at the largest shift), and
    # each one's contour is the shifted edge, so each hit scores what the
    # class map's boundary scores on the same frame.
    r = score(S.shifted(shift))
    assert r.n_hits == 2 and r.n_entered == 2
    assert [s.f for s in r.scores] == [pytest.approx(expected)] * 2
    assert r.inst_bf == pytest.approx(expected)


def test_a_miss_is_not_scored():
    # The edge 30 px off: class 1 is still found (IoU 50/80), class 2 is not
    # (20/50). The one hit's contour is 30 px off, so it scores 0, and the miss
    # enters nothing.
    r = score(S.shifted(30))
    assert r == InstanceBoundary(inst_bf=0.0, scores=(BoundaryScore(0.0, 0.0, 0.0),), n_hits=1, n_entered=1)


def test_a_frame_with_no_hit_has_no_value():
    # Five regions of 20 columns each: the best IoU with a class is 20/50.
    lab = np.repeat(np.arange(S.W, dtype=np.int32) // 20, S.H).reshape(S.W, S.H).T
    r = score(S.Scene("thirds", "", S._two_classes(), lab))
    assert r == InstanceBoundary(inst_bf=None, scores=(), n_hits=0, n_entered=0)


def test_a_predicted_object_without_a_boundary_scores_zero():
    # One region over both classes: it is paired with class 2 (IoU exactly
    # 0.5, a hit, the higher GT index first). The predicted object fills the
    # scored pixels and has no boundary; the GT object has one, so the hit
    # scores 0 rather than being left out.
    r = score(S.merged())
    assert r == InstanceBoundary(inst_bf=0.0, scores=(BoundaryScore(0.0, 0.0, 0.0),), n_hits=1, n_entered=1)


def test_a_gt_object_that_fills_the_scored_pixels_has_no_contour_to_recover():
    # The frame shows one class, exactly predicted: a hit, but its GT object
    # has no boundary within the scored pixels, so the hit enters no mean and
    # the frame has no value, as the frame-level boundary metrics have none.
    gt = np.ones((S.H, S.W), dtype=np.int32)
    r = score(S.Scene("one", "", gt, np.zeros((S.H, S.W), dtype=np.int32)))
    assert r == InstanceBoundary(inst_bf=None, scores=(None,), n_hits=1, n_entered=0)


def test_an_object_cut_off_by_removed_pixels_is_left_out_and_counted():
    # Classes 1 and 2 share an edge at column 30; class 4 sits beyond a band
    # of removed pixels (a tool, say) in columns 60-69. The prediction is
    # exact. The two adjacent objects score 1; the cut-off one has no
    # boundary, enters nothing, and the mean is over the two.
    gt = np.full((S.H, S.W), 4, dtype=np.int32)
    gt[:, :30] = 1
    gt[:, 30:60] = 2
    gt[:, 60:70] = 9
    scored = gt != 9
    lab = np.full((S.H, S.W), 2, dtype=np.int32)
    lab[:, :30] = 0
    lab[:, 30:60] = 1
    r = score(S.Scene("cut_off", "", gt, lab), scored)
    assert r.inst_bf == 1.0
    # The hits come in the order the pairing took them: at equal IoU the
    # higher GT index first, so the cut-off class 4 is listed first.
    assert r.scores == (None, BoundaryScore(1.0, 1.0, 1.0), BoundaryScore(1.0, 1.0, 1.0))
    assert (r.n_hits, r.n_entered) == (3, 2)


def test_an_edge_against_removed_pixels_is_not_part_of_an_objects_contour():
    # The top ten rows are removed. Without that, the shifted prediction would
    # also be scored on its edge against them; with it, each object's contour
    # is the vertical edge alone, and the shift costs what it costs on a
    # whole frame.
    scored = ALL.copy()
    scored[:10] = False
    assert score(S.shifted(3), scored).inst_bf == pytest.approx(0.5)
    assert score(S.shifted(2), scored).inst_bf == 1.0


def test_the_tolerance_is_passed_through():
    # At tolerance 0 the shifted boundary (columns 50 and 51) meets the true
    # one (49 and 50) in one of its two columns, each way.
    r = score(S.shifted(1), tol=0)
    assert r.scores[0] == BoundaryScore(0.5, 0.5, 0.5)
    assert score(S.shifted(1), tol=BOUNDARY_TOL_PX).inst_bf == 1.0


def test_objects_over_different_masks_or_a_pairing_of_other_objects_are_refused():
    scene = S.exact()
    gt = gt_objects(scene.gt, ALL)
    pred = predicted_objects(scene.lab, ALL)
    pairing = instance_scores(gt, pred)
    other = ALL.copy()
    other[:10] = False
    with pytest.raises(ValueError, match="different scored masks"):
        instance_boundary_f(gt, predicted_objects(scene.lab, other), pairing)
    # A pairing computed on a frame with a third class, applied to this one.
    gt3 = scene.gt.copy()
    gt3[:10, :10] = 7
    pairing3 = instance_scores(gt_objects(gt3, ALL), pred)
    with pytest.raises(ValueError, match="computed from other objects"):
        instance_boundary_f(gt, pred, pairing3)


def test_pilot_mode_marks_an_objects_edge_against_removed_pixels():
    # The pilot evaluator marked an object's boundary on the whole map and
    # masked it afterwards, so the edge against a removed pixel is a boundary
    # on the object's side. One class filling the scored rows, exactly
    # predicted: under the evaluator's rule the GT object has no contour and
    # the hit is left out; under the pilot's its top edge (row 10, 100 px)
    # is the contour, recovered exactly.
    scored = ALL.copy()
    scored[:10] = False
    gt = np.ones((S.H, S.W), dtype=np.int32)
    scene = S.Scene("one", "", gt, np.zeros((S.H, S.W), dtype=np.int32))
    assert score(scene, scored) == InstanceBoundary(inst_bf=None, scores=(None,), n_hits=1, n_entered=0)
    assert score(scene, scored, pilot=True) == InstanceBoundary(
        inst_bf=1.0, scores=(BoundaryScore(1.0, 1.0, 1.0),), n_hits=1, n_entered=1,
    )
    # With the edge 3 px off, the top edge enters both contours. GT object 1
    # (columns 0-49, rows 10-59): columns 49 and 50 and its top edge, 149 px.
    # Its region (columns 0-52): columns 52 and 53 and its top edge, 152 px.
    # Within 2 px of the other: column 52 and the top edge of the region
    # (102 px), and columns 49-50 in rows 10-12 and the top edge of the
    # GT object (102 px). The evaluator's rule sees the vertical edge alone.
    first = score(S.shifted(3), scored, pilot=True).scores[0]
    p, r = 102 / 152, 102 / 149
    assert (first.precision, first.recall, first.f) == (pytest.approx(p), pytest.approx(r), pytest.approx(2 * p * r / (p + r)))
    assert score(S.shifted(3), scored).inst_bf == pytest.approx(0.5)
