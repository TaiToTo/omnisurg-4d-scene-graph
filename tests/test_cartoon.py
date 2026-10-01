"""What the cartoon frame promises: the four things in it, and the one typical
mistake the prediction makes on each.

Nothing here names a metric. The cartoon is drawn with OpenCV, so the exact
pixel counts are not pinned; the relations between the regions are.
"""
import cv2
import numpy as np

import cartoon as C

GT = C.ground_truth()
LAB = C.prediction(GT)


def _components(mask: np.ndarray) -> int:
    n, _ = cv2.connectedComponents(mask.astype(np.uint8))
    return n - 1


def test_shape_and_ids():
    assert GT.shape == LAB.shape == C.valid().shape == (C.H, C.W)
    assert GT.dtype == np.int32 and LAB.dtype == np.int32
    assert set(np.unique(GT)) == {C.BACKGROUND, C.LIVER, C.GALLBLADDER, C.FAT, C.TOOL}
    assert set(np.unique(LAB)) == {-1, 0, 1, 2, 3, 4}
    assert set(C.CLASS_NAMES) == {C.LIVER, C.GALLBLADDER, C.FAT, C.TOOL}
    assert C.valid().all()


def test_instrument_cuts_the_liver_in_two():
    assert _components(GT == C.LIVER) == 2
    assert _components(GT == C.TOOL) == 1


def test_liver_is_split_into_regions_0_and_1():
    liver = GT == C.LIVER
    for r in (0, 1):
        assert (LAB == r).any() and liver[LAB == r].all()
    # Every liver pixel is one of its two regions, the grown instrument, or the band.
    assert set(np.unique(LAB[liver])) <= {-1, 0, 1, 3}


def test_gallbladder_and_fat_are_merged_into_region_2():
    soft = (GT == C.GALLBLADDER) | (GT == C.FAT)
    assert soft[LAB == 2].all()
    assert (LAB[GT == C.GALLBLADDER] == 2).any() and (LAB[GT == C.FAT] == 2).any()
    assert _components(LAB == 2) == 1


def test_instrument_region_is_the_tool_grown_outward():
    tool = GT == C.TOOL
    assert (LAB[tool] == 3).all()
    assert (LAB == 3).sum() > tool.sum()


def test_region_4_lies_on_unannotated_background():
    assert (LAB == 4).any()
    assert (GT[LAB == 4] == C.BACKGROUND).all()


def test_a_band_between_liver_and_gallbladder_has_no_region():
    unassigned = LAB == -1
    assert (unassigned & (GT != C.BACKGROUND)).any()
    assert (unassigned & (GT == C.LIVER)).any()


def test_tool_offset_moves_the_instrument_left():
    later = C.prediction(GT, tool_offset=20)
    x_now = np.nonzero(LAB == 3)[1].mean()
    x_later = np.nonzero(later == 3)[1].mean()
    assert x_later < x_now
    assert ((later == 2) == (LAB == 2)).sum() > 0.9 * (C.H * C.W)


def test_swap_liver_exchanges_regions_0_and_1_only():
    swapped = C.prediction(GT, swap_liver=True)
    assert ((swapped == 0) == (LAB == 1)).all()
    assert ((swapped == 1) == (LAB == 0)).all()
    others = LAB >= 2
    assert (swapped[others] == LAB[others]).all()
