"""The fill of a GT mask's `filled` classes: each pixel takes the nearest region's id, the smaller of two equally near.

The rule is pinned on lines drawn by hand, and against a search that walks
out from each filled pixel on its own, on random masks.
"""
from collections import deque

import numpy as np
import pytest

from evalkit.fill import fill_from_nearest

LINE, LIVER, GALLBLADDER = 13, 2, 10


def nearest_by_search(mask_ids: np.ndarray, filled_ids: set[int]) -> np.ndarray:
    """Fill each pixel on its own: walk out through filled pixels, step by step, to the first other ids met."""
    h, w = mask_ids.shape
    out = mask_ids.copy()
    for y, x in zip(*np.nonzero(np.isin(mask_ids, sorted(filled_ids)))):
        seen, frontier, met = {(y, x)}, deque([(y, x)]), set()
        while frontier and not met:
            nxt = deque()
            for cy, cx in frontier:
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in seen:
                        seen.add((ny, nx))
                        if mask_ids[ny, nx] in filled_ids:
                            nxt.append((ny, nx))
                        else:
                            met.add(int(mask_ids[ny, nx]))
            frontier = nxt
        out[y, x] = min(met)
    return out


def test_a_1_px_line_between_two_regions_takes_the_smaller_id_whole_and_is_marked():
    ids = np.full((5, 7), GALLBLADDER, dtype=np.int32)
    ids[:, :3] = LIVER
    ids[:, 3] = LINE
    filled, was = fill_from_nearest(ids, frozenset({LINE}))
    assert (filled[:, :4] == LIVER).all() and (filled[:, 4:] == GALLBLADDER).all()
    assert (was == (ids == LINE)).all()
    assert filled.dtype == ids.dtype and (ids[:, 3] == LINE).all()   # the input is left as it was


def test_a_wide_gap_is_filled_from_its_nearer_side():
    # Columns 2 to 6 are a gap 5 px wide: the two columns nearer the liver take it, the two nearer the
    # gallbladder take that, and the middle one, equally near both, takes the smaller id.
    ids = np.full((3, 9), GALLBLADDER, dtype=np.int32)
    ids[:, :2] = LIVER
    ids[:, 2:7] = LINE
    filled, _ = fill_from_nearest(ids, frozenset({LINE}))
    assert filled[0].tolist() == [LIVER] * 5 + [GALLBLADDER] * 4


def test_nearness_counts_steps_between_four_neighbours_not_diagonals():
    # The centre pixel touches the liver only diagonally (2 steps) and the gallbladder below (1 step).
    ids = np.full((3, 3), LINE, dtype=np.int32)
    ids[0, 0] = LIVER
    ids[2, 1] = GALLBLADDER
    filled, _ = fill_from_nearest(ids, frozenset({LINE}))
    assert filled[1, 1] == GALLBLADDER
    assert filled[0, 1] == LIVER and filled[1, 0] == LIVER   # one step from the liver, two from the gallbladder


@pytest.mark.parametrize("seed", range(6))
def test_the_fill_equals_a_search_from_each_pixel_on_random_masks(seed):
    rng = np.random.default_rng(seed)
    ids = rng.choice([0, LIVER, 4, GALLBLADDER], size=(13, 17)).astype(np.int32)
    ids[rng.random(ids.shape) < (0.3, 0.6, 0.85)[seed % 3]] = LINE
    filled, was = fill_from_nearest(ids, frozenset({LINE}))
    assert (filled == nearest_by_search(ids, {LINE})).all()
    assert (was == (ids == LINE)).all() and not np.isin(filled, [LINE]).any()


def test_a_mask_without_a_filled_id_is_returned_as_it_is():
    ids = np.arange(12, dtype=np.int32).reshape(3, 4)
    filled, was = fill_from_nearest(ids, frozenset({LINE}))
    assert (filled == ids).all() and not was.any()
    filled, was = fill_from_nearest(ids, frozenset())
    assert (filled == ids).all() and not was.any()


def test_a_mask_of_filled_ids_alone_or_not_an_integer_map_is_refused():
    with pytest.raises(ValueError, match="no region is near"):
        fill_from_nearest(np.full((4, 4), LINE, dtype=np.int32), frozenset({LINE}))
    with pytest.raises(ValueError, match="integer"):
        fill_from_nearest(np.zeros((4, 4), dtype=np.float32), frozenset({LINE}))
    with pytest.raises(ValueError, match=r"\(H, W\)"):
        fill_from_nearest(np.zeros((2, 4, 4), dtype=np.int32), frozenset({LINE}))
