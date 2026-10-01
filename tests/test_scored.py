"""The scored pixels of a view, and the count of what it removes, on the real tables.

The frame is 60 x 100 px, laid out in columns so that every count is a
multiple of 60: ten columns each of the classes named in the fixtures.
"""
import numpy as np
import pytest

from evalkit.classes import VIEWS, load_table
from evalkit.scored import PixelCounts, frame_is_excluded, scored_pixels

H, W = 60, 100
ALL = np.ones((H, W), dtype=bool)


def columns(ids: list[int]) -> np.ndarray:
    """Ten columns of each id, in order."""
    frame = np.empty((H, W), dtype=np.int32)
    for i, cid in enumerate(ids):
        frame[:, 10 * i:10 * (i + 1)] = cid
    return frame


@pytest.fixture
def cholec():
    return load_table("cholecseg8k")


@pytest.fixture
def atlas():
    return load_table("atlas120k")


# CholecSeg8k: 0 Black Background (ignored), 1 Abdominal Wall (backdrop),
# 2 Liver (tissue), 5 Grasper (tool), 7 Blood (appearance), 8 Cystic Duct
# (expert), 13 the region line (ignored). Each ten columns; 30 of liver.
CHOLEC = [0, 1, 2, 2, 2, 5, 7, 8, 13, 2]


def test_the_all_view_scores_every_class_that_is_not_ignored(cholec):
    s = scored_pixels(columns(CHOLEC), cholec, "all", ALL)
    assert s.view == "all"
    assert s.counts == PixelCounts(invalid_depth=0, ignored=1200, background=0, left_out=0, scored=4800)
    assert s.mask[:, 10:80].all() and s.mask[:, 90:].all()
    assert not s.mask[:, :10].any() and not s.mask[:, 80:90].any()


def test_the_tissue_view_leaves_out_the_tool(cholec):
    s = scored_pixels(columns(CHOLEC), cholec, "tissue", ALL)
    assert s.counts == PixelCounts(invalid_depth=0, ignored=1200, background=0, left_out=600, scored=4200)
    assert not s.mask[:, 50:60].any()


def test_the_geometric_view_keeps_tissue_alone(cholec):
    s = scored_pixels(columns(CHOLEC), cholec, "geometric", ALL)
    # Left out: the abdominal wall, the grasper, blood and the cystic duct.
    assert s.counts == PixelCounts(invalid_depth=0, ignored=1200, background=0, left_out=2400, scored=2400)
    assert s.mask[:, 20:50].all() and s.mask[:, 90:].all()
    assert int(s.mask.sum()) == 2400


def test_invalid_depth_is_counted_first_and_the_counts_sum_to_the_frame(cholec):
    valid = ALL.copy()
    valid[:30] = False           # the top half, across every column
    s = scored_pixels(columns(CHOLEC), cholec, "geometric", valid)
    assert s.counts == PixelCounts(invalid_depth=3000, ignored=600, background=0, left_out=1200, scored=1200)
    assert s.counts.total == H * W
    assert not s.mask[:30].any()


def test_background_is_removed_in_every_view_and_counted(atlas):
    # ATLAS-120k: 0 Background, 1 Tools/camera (tool), 12 Liver (tissue),
    # 7 Abdominal wall (backdrop).
    frame = columns([0, 0, 1, 12, 12, 12, 7, 7, 12, 0])
    for view in VIEWS:
        s = scored_pixels(frame, atlas, view, ALL)
        assert s.counts.background == 1800, view
        assert not s.mask[:, :20].any() and not s.mask[:, 90:].any()
    assert scored_pixels(frame, atlas, "all", ALL).counts.scored == 4200
    assert scored_pixels(frame, atlas, "tissue", ALL).counts == PixelCounts(0, 0, 1800, 600, 3600)
    assert scored_pixels(frame, atlas, "geometric", ALL).counts == PixelCounts(0, 0, 1800, 1800, 2400)


def test_the_benchmark_set_types_its_own_classes():
    # Benchmark class 11, Bile/lymph duct, is tissue although Cystic duct
    # (original id 13) is expert; the view reads the set it was given.
    original, benchmark = load_table("atlas120k"), load_table("atlas120k", "benchmark")
    assert scored_pixels(columns([13] * 10), original, "geometric", ALL).counts.scored == 0
    assert scored_pixels(columns([11] * 10), benchmark, "geometric", ALL).counts.scored == H * W


def test_a_class_the_table_does_not_have_raises(cholec):
    with pytest.raises(KeyError, match="14"):
        scored_pixels(columns([2] * 9 + [14]), cholec, "all", ALL)


def test_an_unknown_view_raises(cholec):
    with pytest.raises(KeyError, match="labeled"):
        scored_pixels(columns(CHOLEC), cholec, "labeled", ALL)


def test_a_frame_with_the_excluded_marker_is_refused_not_scored(atlas):
    frame = columns([12] * 9 + [42])
    with pytest.raises(ValueError, match="excluded"):
        scored_pixels(frame, atlas, "all", ALL)


def test_the_marker_is_seen_on_the_mask_ids_in_both_class_sets():
    original, benchmark = load_table("atlas120k"), load_table("atlas120k", "benchmark")
    frame = columns([12] * 9 + [42])
    assert frame_is_excluded(frame, original) and frame_is_excluded(frame, benchmark)
    assert not frame_is_excluded(columns([12] * 10), benchmark)
    # In the benchmark set the marker has become background by the time the
    # classes are typed, so the check has to run on the mask ids.
    assert scored_pixels(benchmark.classes_of(frame), benchmark, "all", ALL).counts.background == 600


def test_the_marker_check_refuses_an_id_no_mask_may_hold(atlas):
    with pytest.raises(KeyError, match="47"):
        frame_is_excluded(columns([12] * 9 + [47]), atlas)


def test_the_inputs_are_checked_first(cholec):
    with pytest.raises(ValueError, match="bool"):
        scored_pixels(columns(CHOLEC), cholec, "all", ALL.astype(np.uint8))
    with pytest.raises(ValueError, match="integer"):
        scored_pixels(columns(CHOLEC).astype(np.float32), cholec, "all", ALL)
    with pytest.raises(ValueError, match="shape"):
        scored_pixels(columns(CHOLEC), cholec, "all", ALL[:, :-1])
