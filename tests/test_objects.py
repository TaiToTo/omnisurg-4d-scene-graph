"""Objects, pairing, `F1_50` and `SQ`, derived by hand on 60 x 100 frames.

GT class 1 fills the left half and class 2 the right half (3,000 px each);
each prediction has one controlled fault.
"""
import numpy as np
import pytest

from evalkit.objects import (
    MATCH_IOU, Pair, gt_objects, instance_scores, intersections, pair, predicted_objects)

H, W = 60, 100
HALF = W // 2
ALL = np.ones((H, W), dtype=bool)


def two_classes() -> np.ndarray:
    gt = np.zeros((H, W), dtype=np.int32)
    gt[:, :HALF] = 1
    gt[:, HALF:] = 2
    return gt


def split_at(col: int, ids=(0, 1)) -> np.ndarray:
    lab = np.empty((H, W), dtype=np.int32)
    lab[:, :col] = ids[0]
    lab[:, col:] = ids[1]
    return lab


def test_one_gt_object_per_class_indexed_by_class_id_ascending():
    gt = two_classes()
    gt[:10, :10] = 7
    objs = gt_objects(gt, ALL)
    assert objs.ids == (1, 2, 7)
    assert objs.areas == (3000 - 100, 3000, 100)
    assert objs.mask(3).sum() == 100 and objs.index[0, 0] == 3
    with pytest.raises(IndexError):
        objs.mask(0)


def test_a_class_cut_in_two_is_still_one_object():
    gt = two_classes()
    gt[:, 20:30] = 2          # an instrument's worth of class 2 across class 1
    objs = gt_objects(gt, ALL)
    assert objs.ids == (1, 2)
    assert objs.areas == (3000 - 600, 3000 + 600)


def test_objects_are_taken_over_the_scored_pixels_only():
    scored = ALL.copy()
    scored[:10] = False
    objs = gt_objects(two_classes(), scored)
    assert objs.areas == (2500, 2500)
    assert not objs.index[:10].any()


def test_a_region_with_no_scored_pixel_is_no_object():
    lab = split_at(HALF)
    lab[:10] = 2
    scored = ALL.copy()
    scored[:10] = False
    objs = predicted_objects(lab, scored)
    assert objs.ids == (0, 1)
    assert objs.areas == (2500, 2500)


def test_no_region_is_no_object():
    lab = split_at(HALF)
    lab[:, 45:55] = -1
    objs = predicted_objects(lab, ALL)
    assert objs.ids == (0, 1)
    assert objs.areas == (2700, 2700)
    assert not objs.index[:, 45:55].any()


def test_the_maps_are_checked_before_anything_is_counted():
    with pytest.raises(ValueError, match="bool"):
        gt_objects(two_classes(), ALL.astype(np.uint8))
    with pytest.raises(ValueError, match="shape"):
        predicted_objects(split_at(HALF), ALL[:, :-1])
    with pytest.raises(ValueError, match="integer"):
        gt_objects(two_classes().astype(np.float32), ALL)
    with pytest.raises(ValueError, match="negative"):
        gt_objects(split_at(HALF, (-1, 1)), ALL)
    with pytest.raises(ValueError, match=">= 0"):
        predicted_objects(split_at(HALF, (-2, 1)), ALL)


def test_intersections_is_the_overlap_table():
    gt = gt_objects(two_classes(), ALL)
    pred = predicted_objects(split_at(HALF + 10), ALL)
    assert intersections(gt, pred).tolist() == [[3000, 0], [600, 2400]]


def test_objects_over_different_scored_masks_are_refused():
    # Scoring the prediction without the top ten rows would set region 0's
    # area against the whole class's; nothing would crash and every union
    # would be off. The fault is planted and must be caught before any pairing.
    scored = ALL.copy()
    scored[:10] = False
    gt = gt_objects(two_classes(), ALL)
    pred = predicted_objects(split_at(HALF), scored)
    with pytest.raises(ValueError, match="scored masks"):
        intersections(gt, pred)
    with pytest.raises(ValueError, match="scored masks"):
        instance_scores(gt, pred)
    # Equal masks held in separate arrays are one mask.
    assert instance_scores(gt, predicted_objects(split_at(HALF), ALL.copy())).f1_50 == 1.0


def test_objects_compare_by_identity():
    # The dataclass holds arrays, so a generated `==` would raise; identity is
    # what is left, and a test compares fields.
    gt = gt_objects(two_classes(), ALL)
    assert gt == gt and gt != gt_objects(two_classes(), ALL)


def test_an_exact_prediction_pairs_each_class_with_its_region():
    s = instance_scores(gt_objects(two_classes(), ALL), predicted_objects(split_at(HALF), ALL))
    # Both pairs have IoU 1, so the higher GT index is taken first.
    assert s.pairs == (Pair(1.0, 2, 2), Pair(1.0, 1, 1))
    assert (s.f1_50, s.sq, s.n_gt, s.n_pred, s.n_hits) == (1.0, 1.0, 2, 2, 2)
    assert s.hits == s.pairs


def test_a_split_class_is_found_once_and_the_other_piece_counts_against_f1():
    # Class 1 is cut into regions 0 and 1 of 1,500 px each: both have IoU
    # exactly 0.5 with it, which is a hit (>=). Of the two equal pairs the
    # higher predicted index is taken first, so region 1 is the one found.
    lab = split_at(HALF, (0, 2))
    lab[:, HALF // 2:HALF] = 1
    s = instance_scores(gt_objects(two_classes(), ALL), predicted_objects(lab, ALL))
    assert s.pairs == (Pair(1.0, 2, 3), Pair(0.5, 1, 2))
    assert s.n_hits == 2 and s.pairs[1].iou == MATCH_IOU
    assert s.f1_50 == pytest.approx(2 * 2 / (2 + 3))
    assert s.sq == pytest.approx((1.0 + 0.5) / 2)


def test_a_merged_region_is_paired_with_the_higher_gt_index_on_a_tie():
    # One region covers both classes, IoU 0.5 with each. Of two equal pairs
    # the higher GT index is taken first, so class 2 is found and class 1 is
    # missed; the region cannot be used twice.
    s = instance_scores(gt_objects(two_classes(), ALL), predicted_objects(np.zeros((H, W), np.int32), ALL))
    assert s.pairs == (Pair(0.5, 2, 1),)
    assert s.f1_50 == pytest.approx(2 * 1 / (2 + 1))
    assert s.sq == 0.5


def test_a_four_way_tie_takes_the_pair_with_the_highest_indices_first():
    # GT 1 and 2 are the left and right halves; regions 0 and 1 are the top
    # and bottom halves, so each region meets each class in 1,500 of 4,500 px
    # and every IoU is 1/3. (2, 2) goes first; then (1, 1) is all that is left.
    top_bottom = np.zeros((H, W), dtype=np.int32)
    top_bottom[H // 2:] = 1
    pairs = pair(gt_objects(two_classes(), ALL), predicted_objects(top_bottom, ALL))
    assert pairs == [Pair(1 / 3, 2, 2), Pair(1 / 3, 1, 1)]


def test_the_gt_index_is_the_tie_key_before_the_predicted_index():
    # Region 0 is the right half and region 1 the left, so the two exact
    # pairs are (GT 2, pred 1) and (GT 1, pred 2). Both are taken whichever
    # key comes first; the order of the list tells the GT key from the
    # predicted key, and sorting by the predicted index first would reverse it.
    swapped = np.zeros((H, W), dtype=np.int32)
    swapped[:, :HALF] = 1
    pairs = pair(gt_objects(two_classes(), ALL), predicted_objects(swapped, ALL))
    assert pairs == [Pair(1.0, 2, 1), Pair(1.0, 1, 2)]


def test_a_gap_lowers_sq_but_not_f1():
    lab = split_at(HALF)
    lab[:, 45:55] = -1
    s = instance_scores(gt_objects(two_classes(), ALL), predicted_objects(lab, ALL))
    assert s.f1_50 == 1.0
    assert s.sq == pytest.approx(2700 / 3000)


def test_a_region_over_removed_pixels_is_neither_an_object_nor_a_false_positive():
    # The top ten rows are unannotated background, removed from the scored
    # pixels, and region 2 lies on them alone.
    gt = two_classes()
    gt[:10] = 0
    lab = split_at(HALF)
    lab[:10] = 2
    scored = gt != 0
    s = instance_scores(gt_objects(gt, scored), predicted_objects(lab, scored))
    assert (s.n_gt, s.n_pred, s.f1_50, s.sq) == (2, 2, 1.0, 1.0)


def test_f1_is_undefined_without_a_gt_object_and_sq_without_a_hit():
    # Every scored pixel has a GT class, so a frame has no GT object only when
    # it has no scored pixel, and then no predicted object either.
    none = np.zeros((H, W), dtype=bool)
    s = instance_scores(gt_objects(two_classes(), none), predicted_objects(split_at(HALF), none))
    assert (s.f1_50, s.sq, s.n_gt, s.n_pred, s.pairs) == (None, None, 0, 0, ())
    far = split_at(HALF)
    far[:, :HALF] = -1            # class 1 has no region at all
    far[:, HALF:HALF + 20] = -1   # class 2's region covers 30 of its 50 columns: IoU 0.6, a hit
    s = instance_scores(gt_objects(two_classes(), ALL), predicted_objects(far, ALL))
    assert s.pairs == (Pair(0.6, 2, 1),) and s.f1_50 == pytest.approx(2 / 3)
    far[:, HALF:HALF + 30] = -1   # now 20 of 50 columns: IoU 0.4, no hit
    s = instance_scores(gt_objects(two_classes(), ALL), predicted_objects(far, ALL))
    assert (s.f1_50, s.sq, s.n_hits) == (0.0, None, 0)
    assert s.pairs == (Pair(0.4, 2, 1),) and s.hits == ()
