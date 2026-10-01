"""The class map and `mIoU`, derived by hand on 60 x 100 frames.

GT class 1 fills the left half and class 2 the right half (3,000 px each).
"""
import numpy as np
import pytest

from evalkit.classmap import NO_CLASS, class_map, class_scores

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


def test_an_exact_prediction_names_each_region_after_its_class():
    gt = two_classes()
    cmap = class_map(gt, split_at(HALF), ALL)
    assert cmap.names == {0: 1, 1: 2}
    assert (cmap.classes == gt).all()
    s = class_scores(gt, cmap, ALL)
    assert (s.miou, s.ious) == (1.0, {1: 1.0, 2: 1.0})


def test_a_region_takes_the_class_most_of_its_pixels_have():
    # Region 0 covers class 1 (3,000 px) and spills 10 columns (600 px) into
    # class 2: it is named 1, and class 2 loses the spill.
    gt = two_classes()
    cmap = class_map(gt, split_at(HALF + 10), ALL)
    assert cmap.names == {0: 1, 1: 2}
    s = class_scores(gt, cmap, ALL)
    assert s.ious[1] == pytest.approx(3000 / 3600)
    assert s.ious[2] == pytest.approx(2400 / 3000)
    assert s.miou == pytest.approx((3000 / 3600 + 2400 / 3000) / 2)


def test_splitting_a_class_costs_miou_nothing():
    lab = split_at(HALF, (0, 2))
    lab[:, HALF // 2:HALF] = 1
    cmap = class_map(two_classes(), lab, ALL)
    assert cmap.names == {0: 1, 1: 1, 2: 2}
    assert class_scores(two_classes(), cmap, ALL).miou == 1.0


def test_a_tie_goes_to_the_smaller_class_id():
    # One region over both halves: 3,000 votes each, so it is named 1.
    # Class 1 then has IoU 3000/6000 and class 2 has 0.
    gt = two_classes()
    cmap = class_map(gt, np.zeros((H, W), dtype=np.int32), ALL)
    assert cmap.names == {0: 1}
    s = class_scores(gt, cmap, ALL)
    assert s.ious == {1: 0.5, 2: 0.0}
    assert s.miou == 0.25


def test_the_tie_rule_is_by_id_not_by_order_of_appearance():
    # The same frame with the ids swapped: the region now meets class 2
    # first (left) and class 1 second, and is still named 1.
    gt = two_classes()
    gt[:, :HALF], gt[:, HALF:] = 2, 1
    assert class_map(gt, np.zeros((H, W), dtype=np.int32), ALL).names == {0: 1}


def test_a_pixel_with_no_region_has_no_class():
    lab = split_at(HALF)
    lab[:, 45:55] = -1
    gt = two_classes()
    cmap = class_map(gt, lab, ALL)
    assert (cmap.classes[:, 45:55] == NO_CLASS).all()
    s = class_scores(gt, cmap, ALL)
    assert s.ious == {1: pytest.approx(2700 / 3000), 2: pytest.approx(2700 / 3000)}


def test_only_scored_pixels_vote_and_a_region_without_one_gets_no_name():
    # The top ten rows are removed. Region 2 lies on them alone and gets no
    # name; region 0 also covers them, but only its scored pixels vote.
    gt = two_classes()
    gt[:10] = 0
    lab = split_at(HALF)
    lab[:10, :HALF] = 0
    lab[:10, HALF:] = 2
    scored = gt != 0
    cmap = class_map(gt, lab, scored)
    assert cmap.names == {0: 1, 1: 2}
    assert (cmap.classes[:10] == NO_CLASS).all()
    assert class_scores(gt, cmap, scored).miou == 1.0


def test_miou_is_undefined_when_no_class_is_present():
    none = np.zeros((H, W), dtype=bool)
    cmap = class_map(two_classes(), split_at(HALF), none)
    assert cmap.names == {}
    assert class_scores(two_classes(), cmap, none).miou is None


def test_the_maps_are_checked_before_anything_is_counted():
    with pytest.raises(ValueError, match="one shape"):
        class_map(two_classes(), split_at(HALF)[:, :-1], ALL)
    with pytest.raises(ValueError, match="bool"):
        class_map(two_classes(), split_at(HALF), ALL.astype(np.uint8))
    with pytest.raises(ValueError, match="negative"):
        class_map(split_at(HALF, (-1, 2)), split_at(HALF), ALL)
    with pytest.raises(ValueError, match=">= 0"):
        class_map(two_classes(), split_at(HALF, (-2, 1)), ALL)
    with pytest.raises(ValueError, match="integer"):
        class_map(two_classes(), split_at(HALF).astype(np.float64), ALL)
