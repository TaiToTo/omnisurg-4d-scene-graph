"""`unlabelled_share`, pinned on a frame small enough to count by hand.

The frame is 60 x 100 px. The GT paints a liver in columns 20-59; columns
0-19 are ignored (the black surround) and columns 60-99 are unlabelled
tissue. A region that spills past the liver is scored like one that stops at
its edge, and this is the number that tells the two apart.
"""
import numpy as np
import pytest

from evalkit.unlabelled import unlabelled_share

H, W = 60, 100
SCORED = np.zeros((H, W), dtype=bool); SCORED[:, 20:60] = True
BACKGROUND = np.zeros((H, W), dtype=bool); BACKGROUND[:, 60:] = True


def region(c0: int, c1: int, rid: int = 0, base: np.ndarray | None = None) -> np.ndarray:
    r = np.full((H, W), -1, dtype=np.int32) if base is None else base.copy()
    r[:, c0:c1] = rid
    return r


def test_a_region_that_stops_at_the_organ_has_no_spill():
    assert unlabelled_share(region(20, 60), SCORED, BACKGROUND) == 0.0


def test_the_spill_is_the_share_of_the_region_lying_on_unlabelled_tissue():
    # 40 columns on the liver, 20 on unlabelled tissue: a third of the region.
    assert unlabelled_share(region(20, 80), SCORED, BACKGROUND) == pytest.approx(20 / 60)


def test_pixels_the_view_removed_for_another_reason_are_not_counted():
    # The region also covers the ignored columns 0-19; they are in neither
    # the numerator nor the denominator, so the share is the same third.
    assert unlabelled_share(region(0, 80), SCORED, BACKGROUND) == pytest.approx(20 / 60)


def test_a_region_on_unlabelled_tissue_alone_is_not_among_the_scored_regions():
    alone = region(70, 90, rid=1)
    assert unlabelled_share(alone, SCORED, BACKGROUND) is None
    # Beside a scored region it still does not count: the share is the
    # scored region's own, pooled over pixels.
    both = region(20, 60, rid=0, base=alone)
    assert unlabelled_share(both, SCORED, BACKGROUND) == 0.0
    spilled = region(20, 80, rid=0, base=region(85, 95, rid=1))
    assert unlabelled_share(spilled, SCORED, BACKGROUND) == pytest.approx(20 / 60)


def test_the_share_is_pooled_over_pixels_not_averaged_over_regions():
    # Region 0: 30 liver columns, no spill. Region 1: 10 liver columns and 30
    # unlabelled ones. Pooled: 30 of 70 pixels per row, not the mean of 0 and 3/4.
    r = region(20, 50, rid=0); r[:, 50:90] = 1
    assert unlabelled_share(r, SCORED, BACKGROUND) == pytest.approx(30 / 70)


def test_scored_and_background_may_not_overlap():
    bad = BACKGROUND.copy(); bad[:, 59] = True
    with pytest.raises(ValueError, match="60 pixels are both"):
        unlabelled_share(region(20, 60), SCORED, bad)


def test_the_inputs_are_checked():
    with pytest.raises(ValueError, match="integer"):
        unlabelled_share(region(20, 60).astype(np.float32), SCORED, BACKGROUND)
    with pytest.raises(ValueError, match=r"\(H, W\)"):
        unlabelled_share(np.zeros((2, H, W), dtype=np.int32), SCORED, BACKGROUND)
    with pytest.raises(ValueError, match="bool"):
        unlabelled_share(region(20, 60), SCORED.astype(np.uint8), BACKGROUND)
    with pytest.raises(ValueError, match="shape"):
        unlabelled_share(region(20, 60), SCORED, BACKGROUND[:, :-1])
    with pytest.raises(ValueError, match=">= 0"):
        unlabelled_share(region(20, 60, rid=-2), SCORED, BACKGROUND)
