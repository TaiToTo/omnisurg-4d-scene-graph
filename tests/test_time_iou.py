"""`time_IoU`, derived by hand on 60 x 100 frames with two regions."""
import numpy as np
import pytest

from evalkit.time_iou import time_iou

H, W = 60, 100
HALF = W // 2
ALL = np.ones((H, W), dtype=bool)


def split_at(col: int, ids=(0, 1)) -> np.ndarray:
    lab = np.empty((H, W), dtype=np.int32)
    lab[:, :col] = ids[0]
    lab[:, col:] = ids[1]
    return lab


def test_regions_that_stay_put_score_one():
    assert time_iou([split_at(HALF)] * 3, [ALL] * 3) == 1.0


def test_regions_that_move_five_columns_score_their_overlap():
    # Region 0 grows from 50 to 55 columns (IoU 50/55) and region 1 shrinks
    # from 50 to 45 (IoU 45/50); the two are pooled.
    assert time_iou([split_at(HALF), split_at(HALF + 5)], [ALL, ALL]) == pytest.approx((50 / 55 + 45 / 50) / 2)


def test_regions_that_exchange_ids_score_zero():
    assert time_iou([split_at(HALF), split_at(HALF, (1, 0))], [ALL, ALL]) == 0.0


def test_values_are_pooled_over_every_id_and_frame_pair_not_averaged_per_frame():
    # Pair 1: both ids at IoU 1. Pair 2: only id 0 carries over, at 50/55.
    # Pooled, that is three values; a per-frame mean would give two.
    frames = [split_at(HALF), split_at(HALF), split_at(HALF + 5, (0, 2))]
    assert time_iou(frames, [ALL] * 3) == pytest.approx((1 + 1 + 50 / 55) / 3)


def test_an_id_that_disappears_costs_nothing_which_is_a_known_fault():
    # Id 1 is gone from the second frame; only id 0 is compared.
    frames = [split_at(HALF), np.where(split_at(HALF) == 1, -1, 0)]
    assert time_iou(frames, [ALL, ALL]) == 1.0


def test_one_region_over_the_whole_frame_scores_one_whatever_happens_which_is_a_known_fault():
    assert time_iou([np.zeros((H, W), np.int32)] * 2, [ALL, ALL]) == 1.0


def test_no_region_is_not_an_id_and_is_never_pooled():
    # The no-region half moves five columns with region 0; were -1 an id it
    # would add a second value, 45/50, and the mean would not be 50/55.
    frames = [split_at(HALF, (0, -1)), split_at(HALF + 5, (0, -1))]
    assert time_iou(frames, [ALL, ALL]) == pytest.approx(50 / 55)


def test_only_pixels_valid_in_both_frames_are_compared():
    # Region 0 moves five columns, but the valid pixels stop at column 50 in
    # one of the two frames, so within them it does not move at all. Either
    # frame may be the restricted one: the rule reads both masks.
    narrow = ALL.copy()
    narrow[:, HALF:] = False
    assert time_iou([split_at(HALF), split_at(HALF + 5)], [ALL, narrow]) == 1.0
    assert time_iou([split_at(HALF), split_at(HALF + 5)], [narrow, ALL]) == 1.0


def test_an_id_left_only_on_invalid_pixels_still_counts_at_zero():
    # Region 1 shrinks to the last five columns, where the depth of the second
    # frame is not valid. It is "present in both" all the same, and within the
    # shared valid pixels its second mask is empty, so it scores 0; had it
    # vanished from the frame it would add nothing. The pilot evaluator's rule.
    second = split_at(W - 5)
    valid_second = ALL.copy()
    valid_second[:, W - 5:] = False
    gone = split_at(W - 5, (0, -1))
    assert time_iou([split_at(HALF), second], [ALL, valid_second]) == pytest.approx((50 / 95 + 0.0) / 2)
    assert time_iou([split_at(HALF), gone], [ALL, valid_second]) == pytest.approx(50 / 95)


def test_undefined_when_nothing_is_pooled():
    assert time_iou([split_at(HALF)], [ALL]) is None
    assert time_iou([split_at(HALF), split_at(HALF, (2, 3))], [ALL, ALL]) is None
    assert time_iou([split_at(HALF), split_at(HALF)], [ALL, np.zeros((H, W), bool)]) is None


def test_the_inputs_are_checked_first():
    with pytest.raises(ValueError, match="valid masks"):
        time_iou([split_at(HALF), split_at(HALF)], [ALL])
    with pytest.raises(ValueError, match="bool"):
        time_iou([split_at(HALF)], [ALL.astype(np.uint8)])
    with pytest.raises(ValueError, match="frame 1 has shape"):
        time_iou([split_at(HALF), split_at(HALF)[:, :-1]], [ALL, ALL[:, :-1]])
    with pytest.raises(ValueError, match="integer"):
        time_iou([split_at(HALF).astype(np.float32)], [ALL])
