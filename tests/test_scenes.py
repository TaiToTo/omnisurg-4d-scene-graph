"""What the hand-derived scenes promise: their shape, their ids, and the one
fault each is built to contain.

Nothing here names a metric. These tests say what a scene *is*, so that the
tests that derive metric values from the areas stand on frames that are what
their names say. A scene called "split" that did not split would make a
derivation right for the wrong reason.
"""
import numpy as np
import pytest

import scenes as S

ALL = [S.exact(), S.shifted(3), S.split(), S.merged(), S.gap(10), S.gap(10, painted=True),
       S.on_background(), S.sliver(100), S.spill(4), S.moved(5), S.swapped()]


def _regions(lab: np.ndarray) -> set[int]:
    return {int(v) for v in np.unique(lab) if v >= 0}


def _objects_under(lab: np.ndarray, mask: np.ndarray) -> dict[int, int]:
    """Region id -> pixels it has inside `mask`."""
    ids, counts = np.unique(lab[mask], return_counts=True)
    return {int(i): int(c) for i, c in zip(ids, counts)}


@pytest.mark.parametrize("scene", ALL, ids=lambda s: s.key)
def test_shape_dtype_and_ids(scene):
    assert scene.gt.shape == scene.lab.shape == scene.valid.shape == (S.H, S.W)
    assert scene.gt.dtype == np.int32 and scene.lab.dtype == np.int32
    assert scene.valid.dtype == bool and scene.valid.all()
    assert set(np.unique(scene.gt)) <= {0, 1, 2}
    assert scene.lab.min() >= -1
    if scene.next_lab is not None:
        assert scene.next_lab.shape == (S.H, S.W) and scene.next_lab.dtype == np.int32


def test_keys_are_distinct():
    keys = [s.key for s in ALL]
    assert len(keys) == len(set(keys))


def test_two_classes_fill_the_halves():
    gt = S.exact().gt
    assert (gt[:, :S.HALF] == 1).all() and (gt[:, S.HALF:] == 2).all()
    assert (gt == 1).sum() == (gt == 2).sum() == S.H * S.HALF


def test_exact_matches_region_for_region():
    sc = S.exact()
    assert _objects_under(sc.lab, sc.gt == 1) == {0: S.H * S.HALF}
    assert _objects_under(sc.lab, sc.gt == 2) == {1: S.H * S.HALF}


@pytest.mark.parametrize("k", [1, 3, 10])
def test_shifted_moves_the_boundary_right_by_k(k):
    sc = S.shifted(k)
    assert sc.key == f"shift{k}"
    assert (sc.lab[:, :S.HALF + k] == 0).all() and (sc.lab[:, S.HALF + k:] == 1).all()
    assert _objects_under(sc.lab, sc.gt == 1) == {0: S.H * S.HALF}
    assert _objects_under(sc.lab, sc.gt == 2) == {0: S.H * k, 1: S.H * (S.HALF - k)}


def test_split_cuts_class_1_into_two_equal_regions():
    sc = S.split()
    under_1 = _objects_under(sc.lab, sc.gt == 1)
    assert len(under_1) == 2 and len(set(under_1.values())) == 1
    assert _objects_under(sc.lab, sc.gt == 2) == {2: S.H * S.HALF}
    assert _regions(sc.lab) == {0, 1, 2}


def test_merged_covers_both_classes_with_one_region():
    sc = S.merged()
    assert _regions(sc.lab) == {0}
    assert (sc.lab == 0).all()


def test_gap_leaves_a_band_without_a_region():
    sc = S.gap(10)
    band = np.zeros((S.H, S.W), bool)
    band[:, S.HALF - 5:S.HALF + 5] = True
    assert (sc.lab[band] == -1).all() and (sc.lab[~band] >= 0).all()
    assert (sc.lab == -1).sum() == S.H * 10
    assert _regions(sc.lab) == {0, 1}


def test_band_paints_the_gap_as_a_third_region():
    sc = S.gap(10, painted=True)
    assert sc.key == "band"
    plain = S.gap(10)
    assert ((sc.lab == 2) == (plain.lab == -1)).all()
    assert (sc.lab >= 0).all() and _regions(sc.lab) == {0, 1, 2}


def test_on_background_puts_a_region_where_nothing_is_annotated():
    sc = S.on_background(10)
    assert (sc.gt[:10] == 0).all() and (sc.gt[10:] != 0).all()
    assert ((sc.lab == 2) == (sc.gt == 0)).all()
    assert _objects_under(sc.lab, sc.gt == 1) == {0: S.H * S.HALF - 10 * S.HALF}


@pytest.mark.parametrize("px", [100, 299, 300, 400])
def test_sliver_has_exactly_px_pixels(px):
    sc = S.sliver(px)
    assert sc.key == f"sliver{px}"
    assert (sc.lab == 3).sum() == px
    assert (sc.lab.reshape(-1)[:px] == 3).all()


@pytest.mark.parametrize("width", [2, 4, 8])
def test_spill_covers_class_1_and_width_columns_of_class_2(width):
    sc = S.spill(width)
    small = 16
    assert (sc.gt[:, :small] == 1).all() and (sc.gt[:, small:] == 2).all()
    assert _objects_under(sc.lab, sc.gt == 1) == {0: S.H * small}
    assert _objects_under(sc.lab, sc.gt == 2) == {0: S.H * width, 1: S.H * (S.W - small - width)}


def test_moved_keeps_ids_and_shifts_both_regions():
    sc = S.moved(5)
    assert sc.next_lab is not None
    assert _regions(sc.lab) == _regions(sc.next_lab) == {0, 1}
    assert (sc.next_lab[:, :S.HALF + 5] == 0).all() and (sc.next_lab[:, S.HALF + 5:] == 1).all()
    assert (sc.lab != sc.next_lab).sum() == S.H * 5


def test_swapped_exchanges_the_ids_and_nothing_else():
    sc = S.swapped()
    assert sc.next_lab is not None
    assert ((sc.lab == 0) == (sc.next_lab == 1)).all()
    assert ((sc.lab == 1) == (sc.next_lab == 0)).all()
