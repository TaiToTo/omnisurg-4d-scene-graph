"""Pilot mode's objects and domains, derived by hand on the 60 x 100 scenes of `tests/scenes.py`.

GT class 1 fills the left half and class 2 the right half (3,000 px each).
Where the pilot evaluator's rule differs from the evaluator's, the test says
what the evaluator's objects would give on the same frame.
"""
import numpy as np
import pytest

import scenes as S
from evalkit.objects import gt_objects, instance_scores, predicted_objects
from evalkit.pilot import (
    PILOT_DOMAINS, PILOT_INSTRUMENT_IDS, PILOT_MIN_CC_PX, pilot_domains, pilot_gt_objects,
    pilot_predicted_objects)

ALL = np.ones((S.H, S.W), dtype=bool)


def test_an_exact_prediction_gives_the_same_objects_as_the_evaluator():
    scene = S.exact()
    g, p = pilot_gt_objects(scene.gt, ALL), pilot_predicted_objects(scene.lab, ALL)
    assert (g.ids, g.areas) == ((1, 2), (3000, 3000))
    assert (p.ids, p.areas) == ((0, 1), (3000, 3000))
    assert (g.index == gt_objects(scene.gt, ALL).index).all()
    assert (p.index == predicted_objects(scene.lab, ALL).index).all()
    assert instance_scores(g, p).f1_50 == 1.0


def test_a_class_cut_in_two_is_two_objects_for_the_pilot_and_one_for_the_evaluator():
    # A band of class 2 across class 1: class 1 falls into two components
    # (1,200 px each) and class 2 into two as well (the band, 600 px, and
    # its half). Objects are numbered class by class, then in the order
    # `cv2.connectedComponents` labels them. That order is OpenCV's own, not
    # raster order in general; on this scene it puts the left piece first.
    gt = S._two_classes()
    gt[:, 20:30] = 2
    g = pilot_gt_objects(gt, ALL)
    assert g.ids == (1, 1, 2, 2)
    assert g.areas == (1200, 1200, 600, 3000)
    assert g.index[0, 0] == 1 and g.index[0, 30] == 2 and g.index[0, 20] == 3 and g.index[0, 99] == 4
    assert gt_objects(gt, ALL).ids == (1, 2)


def test_a_component_under_the_cut_is_no_object_and_the_evaluator_keeps_it():
    gt = S._two_classes()
    gt[:10, :10] = 7                          # 100 px of class 7
    g = pilot_gt_objects(gt, ALL)
    assert g.ids == (1, 2) and g.areas == (2900, 3000)
    assert not (g.index == 3).any() and g.index[0, 0] == 0
    assert gt_objects(gt, ALL).ids == (1, 2, 7)
    gt[:30, :10] = 7                          # 300 px: exactly the cut, kept
    assert pilot_gt_objects(gt, ALL).ids == (1, 2, 7)


@pytest.mark.parametrize("px, counted", [(PILOT_MIN_CC_PX - 1, False), (PILOT_MIN_CC_PX, True)])
def test_a_predicted_sliver_is_an_object_at_the_cut_and_not_below_it(px, counted):
    # The pilot evaluator's numbers on the sliver scene: a third region of
    # 300 px is a false positive (F1 0.8), one of 299 px is dropped (F1 1).
    scene = S.sliver(px)
    p = pilot_predicted_objects(scene.lab, ALL)
    assert len(p) == (3 if counted else 2)
    s = instance_scores(pilot_gt_objects(scene.gt, ALL), p)
    assert s.f1_50 == pytest.approx(0.8 if counted else 1.0)
    assert len(predicted_objects(scene.lab, ALL)) == 3


def test_a_region_is_measured_by_its_pixels_in_the_domain():
    # Region 3 has 600 px in the frame but 200 in the domain: dropped. The
    # other two are the objects of their pixels in the domain.
    lab = S._split_at(S.HALF)
    lab[:10, :60] = 3
    domain = ALL.copy()
    domain[:, :40] = False
    p = pilot_predicted_objects(lab, domain)
    assert p.ids == (0, 1) and p.areas == (500, 2900)
    domain[:, 20:40] = True                  # now 400 px in the domain
    assert pilot_predicted_objects(lab, domain).ids == (0, 1, 3)


def test_the_split_scene_scores_as_the_pilot_evaluator_scored_it():
    # One class in two regions of IoU exactly 0.5: the pilot found class 2
    # and one half of class 1 (F1 2·2/(2·2 + 1), SQ (1 + 0.5)/2).
    scene = S.split()
    s = instance_scores(pilot_gt_objects(scene.gt, ALL), pilot_predicted_objects(scene.lab, ALL))
    assert s.f1_50 == pytest.approx(0.8) and s.sq == pytest.approx(0.75)


def test_the_four_domains():
    # The top ten rows are background; class 5 (a grasper, under the
    # CholecSeg8k ids) fills columns 40-59; the depth is invalid in the
    # last ten rows.
    gt = S.on_background(10).gt
    gt[:, 40:60] = 5
    valid = ALL.copy()
    valid[50:] = False
    d = pilot_domains(gt, valid, PILOT_INSTRUMENT_IDS["cholecseg8k"])
    assert list(d) == list(PILOT_DOMAINS) == ["full", "labeled", "tissue", "labeled_tissue"]
    assert (d["full"] == valid).all()
    # Background: ten rows less the grasper's twenty columns, 800 px.
    assert (d["labeled"] == (valid & (gt != 0))).all() and int(d["labeled"].sum()) == 4200
    assert (d["tissue"] == (valid & (gt != 5))).all() and int(d["tissue"].sum()) == 4000
    assert int(d["labeled_tissue"].sum()) == 3200
    assert not d["labeled_tissue"][5, 0] and not d["labeled_tissue"][20, 50] and d["labeled_tissue"][20, 0]


def test_the_tissue_domains_need_the_instrument_ids():
    with pytest.raises(ValueError, match="instrument ids"):
        pilot_domains(S.exact().gt, ALL, frozenset())
    assert PILOT_INSTRUMENT_IDS == {"cholecseg8k": {5, 9}, "atlas120k": {1}}


def test_a_region_on_background_is_an_object_in_full_and_none_in_labeled():
    # The pilot's `labeled` domain is why the evaluator removes background.
    scene = S.on_background(10)
    d = pilot_domains(scene.gt, ALL, PILOT_INSTRUMENT_IDS["atlas120k"])
    full = instance_scores(pilot_gt_objects(scene.gt, d["full"]), pilot_predicted_objects(scene.lab, d["full"]))
    assert (full.n_gt, full.n_pred, full.f1_50) == (2, 3, pytest.approx(0.8))
    labeled = instance_scores(
        pilot_gt_objects(scene.gt, d["labeled"]), pilot_predicted_objects(scene.lab, d["labeled"]))
    assert (labeled.n_gt, labeled.n_pred, labeled.f1_50) == (2, 2, 1.0)


def test_the_inputs_are_checked():
    gt = S.exact().gt
    with pytest.raises(ValueError, match="negative"):
        pilot_gt_objects(np.full_like(gt, -1), ALL)
    with pytest.raises(ValueError, match="-1 for no region"):
        pilot_predicted_objects(np.full_like(gt, -2), ALL)
    with pytest.raises(ValueError, match="bool"):
        pilot_gt_objects(gt, ALL.astype(np.uint8))
    with pytest.raises(ValueError, match="integer"):
        pilot_domains(gt.astype(np.float32), ALL, frozenset({1}))
