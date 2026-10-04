"""`VI_split` and `VI_merge`, derived by hand on 60 x 100 frames.

GT class 1 fills the left half and class 2 the right half. With `h` the
binary entropy in bits, h(1/2) = 1 and h(0.1) = 0.4690.
"""
import numpy as np
import pytest

from evalkit.vi import VIScores, variation_of_information

H, W = 60, 100
HALF = W // 2
ALL = np.ones((H, W), dtype=bool)


def h(p: float) -> float:
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


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


def test_an_exact_prediction_has_no_information_lost_either_way():
    assert variation_of_information(two_classes(), split_at(HALF), ALL) == VIScores(0.0, 0.0)


def test_a_class_halved_costs_one_bit_on_that_class_and_nothing_on_merge():
    # Class 1 is cut into two equal regions: H(X | Y = 1) = 1 bit, weighted
    # by class 1's half of the pixels.
    lab = split_at(HALF, (0, 2))
    lab[:, HALF // 2:HALF] = 1
    vi = variation_of_information(two_classes(), lab, ALL)
    assert vi.split == pytest.approx(0.5)
    assert vi.merge == pytest.approx(0.0)


def test_one_region_over_two_equal_classes_costs_one_bit_of_merge():
    vi = variation_of_information(two_classes(), np.zeros((H, W), np.int32), ALL)
    assert vi.split == pytest.approx(0.0)
    assert vi.merge == pytest.approx(1.0)


def test_pixels_with_no_region_count_together_as_one_region():
    # A 10-column band around the boundary has no region: within each class,
    # a tenth of the pixels fall in that one extra "region", so
    # VI_split = h(0.1); the band is half class 1 and half class 2, so it
    # carries one bit of merge on a tenth of the pixels.
    lab = split_at(HALF)
    lab[:, HALF - 5:HALF + 5] = -1
    vi = variation_of_information(two_classes(), lab, ALL)
    assert vi.split == pytest.approx(h(0.1))
    assert vi.merge == pytest.approx(0.1)
    # Had every such pixel been its own region, the split would be far larger.
    none = np.full((H, W), -1, dtype=np.int32)
    vi = variation_of_information(two_classes(), none, ALL)
    assert vi.split == pytest.approx(0.0) and vi.merge == pytest.approx(1.0)


def test_only_scored_pixels_enter():
    # Region 0 runs over the removed top rows, where the GT is background.
    # Counted, those pixels would put region 0 across two GT labels and
    # cost merge; under the scored mask they are not seen at all.
    gt = two_classes()
    gt[:10] = 0
    lab = split_at(HALF)
    lab[:10] = 0
    assert variation_of_information(gt, lab, gt != 0) == VIScores(0.0, 0.0)
    assert variation_of_information(gt, lab, ALL).merge > 0


def test_a_negative_gt_id_on_a_scored_pixel_is_refused():
    gt = two_classes()
    gt[:10] = -1
    with pytest.raises(ValueError, match="negative"):
        variation_of_information(gt, split_at(HALF), ALL)
    # Off the scored pixels it is not looked at.
    assert variation_of_information(gt, split_at(HALF), gt >= 0) == VIScores(0.0, 0.0)


def test_undefined_without_a_scored_pixel():
    assert variation_of_information(two_classes(), split_at(HALF), np.zeros((H, W), bool)) is None


def test_the_maps_are_checked_before_anything_is_counted():
    with pytest.raises(ValueError, match="one shape"):
        variation_of_information(two_classes(), split_at(HALF)[:, :-1], ALL)
    with pytest.raises(ValueError, match="bool"):
        variation_of_information(two_classes(), split_at(HALF), ALL.astype(np.uint8))
    with pytest.raises(ValueError, match="integer"):
        variation_of_information(two_classes(), split_at(HALF).astype(np.float32), ALL)
