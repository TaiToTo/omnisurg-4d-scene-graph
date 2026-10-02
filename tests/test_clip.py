"""The clip summary: each key's mean over the frames it is defined on, with the counts.

The frames are the 60 x 100 scenes of `tests/scenes.py` read through the
ATLAS-120k table, as in `tests/test_frame.py`; the means are derived by hand
from the per-frame values those tests pin.
"""
import numpy as np
import pytest

import scenes as S
from evalkit.classes import VIEWS, load_table
from evalkit.clip import ClipScores, ScoredFrame, pq, summarize_clip
from evalkit.frame import KEYS, score_frame
from evalkit.scored import PixelCounts

DEPTH = np.ones((S.H, S.W), dtype=np.float32)
LIVER, GALLBLADDER, BACKGROUND, EXCLUDED = 12, 14, 0, 42
ATLAS = load_table("atlas120k")


def atlas_ids(gt: np.ndarray) -> np.ndarray:
    return np.array([BACKGROUND, LIVER, GALLBLADDER], dtype=np.int32)[gt]


def frame(number: int, gt: np.ndarray, lab: np.ndarray) -> ScoredFrame:
    return ScoredFrame(number, score_frame(gt, lab, DEPTH, ATLAS))


def exact(number: int) -> ScoredFrame:
    return frame(number, atlas_ids(S.exact().gt), S.exact().lab)


def merged(number: int) -> ScoredFrame:
    return frame(number, atlas_ids(S.merged().gt), S.merged().lab)


def one_class(number: int) -> ScoredFrame:
    """One class fills the frame, one region covers it: no GT boundary, so no boundary key."""
    return frame(number, np.full((S.H, S.W), LIVER, np.int32), np.zeros((S.H, S.W), np.int32))


def excluded(number: int) -> ScoredFrame:
    gt = atlas_ids(S.exact().gt)
    gt[:10, :10] = EXCLUDED
    return frame(number, gt, S.exact().lab)


@pytest.fixture(scope="module")
def clip() -> ClipScores:
    # In time order, the numbers not rising with it, and one excluded frame.
    return summarize_clip([exact(10), merged(30), one_class(20), excluded(40)], time_iou=0.75)


def test_each_mean_covers_the_frames_its_key_is_defined_on(clip):
    assert set(clip.views) == set(VIEWS)
    for view, v in clip.views.items():
        assert v.view == view
        assert list(v.means) == list(KEYS) and list(v.n_frames) == list(KEYS)
        assert v.means["F1_50"] == pytest.approx((1 + 2 / 3 + 1) / 3) and v.n_frames["F1_50"] == 3
        assert v.means["SQ"] == pytest.approx((1 + 0.5 + 1) / 3) and v.n_frames["SQ"] == 3
        assert v.means["mIoU"] == pytest.approx((1 + 0.25 + 1) / 3) and v.n_frames["mIoU"] == 3
        assert v.means["VI_merge"] == pytest.approx(1 / 3) and v.n_frames["VI_merge"] == 3
        assert v.means["unlabelled_share"] == 0.0 and v.n_frames["unlabelled_share"] == 3
        # The one-class frame has no GT boundary: the boundary keys cover two frames.
        for key in ("inst_BF", "boundary_F", "boundary_R_raw"):
            assert v.means[key] == 0.5 and v.n_frames[key] == 2, key


def test_the_counts_are_kept_and_the_excluded_frame_is_counted_not_scored(clip):
    assert (clip.n_scored_frames, clip.n_excluded_frames, clip.time_iou) == (3, 1, 0.75)
    assert [f.frame for f in clip.frames] == [10, 30, 20, 40]
    assert clip.frames[3].scores.excluded
    for v in clip.views.values():
        assert v.pixels == PixelCounts(0, 0, 0, 0, 3 * S.H * S.W)
        # exact 2/2/2/2, merged 2/1/1/1, one class 1/1/1 and a hit whose
        # GT object has no boundary pixel, so it did not enter inst_BF.
        assert (v.n_gt_objects, v.n_pred_objects, v.n_hits, v.n_inst_bf_hits) == (5, 4, 4, 3)


def test_a_key_defined_on_no_frame_has_no_mean_and_a_count_of_zero():
    c = summarize_clip([one_class(1)], time_iou=None)
    v = c.views["all"]
    assert v.means["boundary_F"] is None and v.n_frames["boundary_F"] == 0
    assert v.means["F1_50"] == 1.0 and v.n_frames["F1_50"] == 1
    assert c.time_iou is None


def test_the_means_and_counts_are_read_only(clip):
    v = clip.views["all"]
    with pytest.raises(TypeError):
        v.means["F1_50"] = 0.0
    with pytest.raises(TypeError):
        clip.views["all"] = v


def test_pq_is_sq_times_f1_and_zero_on_a_frame_with_objects_but_no_hit():
    assert pq(exact(1).scores.views["all"]) == 1.0
    assert pq(merged(1).scores.views["all"]) == pytest.approx(0.5 * 2 / 3)
    # The regions cover a fifth of each class: IoU 0.2, no hit, SQ undefined.
    far = S.exact().lab.copy()
    far[:, 10:S.HALF] = -1
    far[:, S.HALF + 10:] = -1
    v = frame(1, atlas_ids(S.exact().gt), far).scores.views["all"]
    assert v.sq is None and v.n_gt_objects == 2 and pq(v) == 0.0
    # No GT object: no F1_50, no PQ.
    blank = frame(1, np.zeros((S.H, S.W), np.int32), S.exact().lab).scores.views["all"]
    assert blank.f1_50 is None and pq(blank) is None


def test_a_clip_with_no_frame_or_with_every_frame_excluded_is_refused():
    with pytest.raises(ValueError, match="no GT frame"):
        summarize_clip([], None)
    with pytest.raises(ValueError, match="every one of the 2 GT frames is excluded"):
        summarize_clip([excluded(1), excluded(2)], None)


def test_a_frame_given_twice_is_refused():
    f = exact(5)
    with pytest.raises(ValueError, match="frame 5 is given twice"):
        summarize_clip([f, f], None)


def test_a_frame_scored_in_fewer_views_than_the_specification_is_refused():
    f = exact(1)
    views = {k: v for k, v in f.scores.views.items() if k != "geometric"}
    partial = ScoredFrame(1, type(f.scores)(excluded=False, views=views))
    with pytest.raises(ValueError, match="not in"):
        summarize_clip([partial], None)


def test_time_iou_is_none_or_an_iou():
    f = exact(1)
    for bad in (-0.1, 1.5, True, float("nan")):
        with pytest.raises(ValueError, match="time_iou"):
            summarize_clip([f], bad)
    assert summarize_clip([f], 0).time_iou == 0.0
