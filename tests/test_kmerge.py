"""The merge to K regions, on maps small enough to work out by hand, and the merged condition the evaluator scores.

Each map is drawn row by row; the comment beside it gives the areas and
the boundaries the rule reads.
"""

import json

import numpy as np
import pytest

import clip_dirs as C
import evalkit.tools.kmerge as KM
from evalkit.evaluate import condition_rule, score_condition
from evalkit.inputs import DEPTH_FILE
from evalkit.tools.kmerge import kmerge, merge_condition

BOTH_WAYS = {"seed_source": "sam", "seed_frame": 1, "bidir": True, "frames": [0, 1, 2]}


def test_the_smallest_region_joins_the_neighbour_it_shares_most_boundary_with():
    # Areas 0: 12, 1: 10, 2: 6. Region 2 shares 4 pixels of boundary with 0 and 3 with 1.
    lab = np.array([[0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 0, 1, 1, 1],
                    [0, 0, 2, 2, 2, 1, 1],
                    [0, 0, 2, 2, 2, 1, 1]])
    want = np.where(lab == 2, 0, lab)
    assert np.array_equal(kmerge(lab, 2), want)


def test_of_two_smallest_regions_the_lower_id_goes_first():
    # Areas 5: 8, 3: 2, 4: 2. Region 3 goes first, and shares 2 with 5 and 1 with 4.
    lab = np.array([[5, 5, 5, 5],
                    [5, 5, 5, 5],
                    [3, 3, 4, 4]])
    assert np.array_equal(kmerge(lab, 2), np.where(lab == 3, 5, lab))


def test_of_two_equal_boundaries_the_smaller_neighbour_takes_the_region():
    # Areas 2: 4, 0: 1, 3: 3. Region 0 shares 1 with each neighbour.
    lab = np.array([[2, 2, 2, 2, 0, 3, 3, 3]])
    assert np.array_equal(kmerge(lab, 2), np.where(lab == 0, 3, lab))


def test_regions_parted_by_a_seam_still_merge_and_the_seam_stays():
    # Region 1, of area 1, touches no region until the unlabelled pixels beside it are filled.
    lab = np.array([[0, 0, 0, -1, 1, -1, 2, 2, 2, 2]])
    out = kmerge(lab, 2)
    assert len(np.unique(out[out >= 0])) == 2
    assert out[0, 3] == out[0, 5] == -1


def test_a_region_without_a_neighbour_is_refused(monkeypatch):
    monkeypatch.setattr(KM, "fill_seams", lambda labels: labels)
    with pytest.raises(RuntimeError, match="has no neighbour"):
        kmerge(np.array([[0, 0, 0, -1, 1, -1, 2, 2, 2, 2]]), 2)


def test_a_map_of_k_regions_or_fewer_is_kept():
    lab = np.array([[0, 0, 1, -1, 2]])
    assert np.array_equal(kmerge(lab, 3), lab)


def test_k_below_one_is_refused():
    with pytest.raises(ValueError, match="at least 1"):
        kmerge(np.zeros((2, 2), np.int32), 0)


def _condition(tmp_path):
    """Two clips of three frames, each predicted as four quarter regions."""
    gt = [C.two_organs() for _ in range(3)]
    quarters = np.zeros((20, 30), np.int32)
    quarters[:10, 15:] = 1
    quarters[10:, :15] = 2
    quarters[10:, 15:] = 3
    quarters[:10, :15][:2, :2] = 4
    clips = ["VID01_a", "VID02_b"]
    for clip in clips:
        C.write_clip(tmp_path / "data", clip, gt)
        C.write_labels(tmp_path / "tracks", clip, "t", {i: quarters.copy() for i in range(3)}, scale=2,
                       seed_info=BOTH_WAYS)
    return clips


def test_the_merged_condition_is_written_at_the_depth_shape_and_scored(tmp_path):
    clips = _condition(tmp_path)
    merge_condition("cholecseg8k", clips, tmp_path / "data", tmp_path / "tracks", "t", "t_k2", k=2)
    out = tmp_path / "tracks" / "VID01_a" / "t_k2"
    assert sorted(p.name for p in out.glob("label_*.npy")) == [f"label_{i:04d}.npy" for i in range(3)]
    merged = np.load(out / "label_0001.npy")
    assert merged.shape == (20, 30) and len(np.unique(merged)) == 2
    assert json.loads((out / "seed_info.json").read_text())["kmerge"] == {"from_tag": "t", "k": 2}
    assert json.loads((out / "kmerge.json").read_text()) == {"from_tag": "t", "k": 2}
    assert condition_rule(tmp_path / "tracks", "t_k2", clips) == condition_rule(tmp_path / "tracks", "t", clips)
    summary = score_condition("cholecseg8k", None, clips, tmp_path / "data", tmp_path / "tracks", "t_k2")
    assert summary["propagation"] == "both_ways_from_centre"


def test_merged_predictions_left_by_an_earlier_run_are_refused_unless_replaced(tmp_path):
    clips = _condition(tmp_path)
    merge_condition("cholecseg8k", clips, tmp_path / "data", tmp_path / "tracks", "t", "t_k2", k=2)
    stale = tmp_path / "tracks" / "VID02_b" / "t_k2" / "label_0099.npy"
    np.save(stale, np.zeros((20, 30), np.int32))
    with pytest.raises(ValueError, match="pass --overwrite"):
        merge_condition("cholecseg8k", clips, tmp_path / "data", tmp_path / "tracks", "t", "t_k2", k=2)
    merge_condition("cholecseg8k", clips, tmp_path / "data", tmp_path / "tracks", "t", "t_k2", k=2, overwrite=True)
    assert not stale.exists()


def test_the_merged_condition_needs_a_tag_of_its_own(tmp_path):
    clips = _condition(tmp_path)
    with pytest.raises(ValueError, match="tag of its own"):
        merge_condition("cholecseg8k", clips, tmp_path / "data", tmp_path / "tracks", "t", "t", k=2)


def test_a_merge_of_several_steps_carries_area_and_boundaries_on():
    # Areas 1: 3, 4: 2, 0: 2, 2: 1, 3: 1. Each step below holds only if the step before added the taken
    # region's area and boundaries to its taker, and recorded the taker among its new neighbours' neighbours.
    # 2 shares 1 with each of 0, 1 and 3, and goes to 3, the neighbour of least area.
    # 0 shares 2 with 1 and, now, 1 with 3, and goes to 1.
    # 3, of area 2 like 4 and the lower id, shares 1 with 4 and, through 0, 2 with 1, and goes to 1.
    lab = np.array([[1, 1, 4],
                    [0, 1, 4],
                    [0, 2, 3]])
    assert np.array_equal(kmerge(lab, 2), np.where(lab == 4, 4, 1))


def test_a_refused_clip_leaves_every_clip_as_it_was(tmp_path):
    clips = _condition(tmp_path)
    merge_condition("cholecseg8k", clips, tmp_path / "data", tmp_path / "tracks", "t", "t_k2", k=2)
    earlier = tmp_path / "tracks" / "VID01_a" / "t_k2" / "marker"
    earlier.touch()
    # The second clip loses a GT frame's prediction, which `read_clip` refuses.
    (tmp_path / "tracks" / "VID02_b" / "t" / "label_0001.npy").unlink()
    with pytest.raises(ValueError, match="no prediction"):
        merge_condition("cholecseg8k", clips, tmp_path / "data", tmp_path / "tracks", "t", "t_k2", k=2,
                        overwrite=True)
    assert earlier.exists()


def test_k_below_one_is_refused_before_anything_is_written(tmp_path):
    clips = _condition(tmp_path)
    with pytest.raises(ValueError, match="at least 1"):
        merge_condition("cholecseg8k", clips, tmp_path / "data", tmp_path / "tracks", "t", "t_k0", k=0)
    assert not (tmp_path / "tracks" / "VID01_a" / "t_k0").exists()


def test_a_condition_that_is_not_merged_predictions_is_not_replaced(tmp_path):
    clips = _condition(tmp_path)
    # A tracked condition named as the output: it has no `kmerge.json`.
    for clip in clips:
        C.write_labels(tmp_path / "tracks", clip, "s", {0: np.zeros((20, 30), np.int32)})
    tracked = tmp_path / "tracks" / "VID01_a" / "s" / "label_0000.npy"
    with pytest.raises(ValueError, match="not merged predictions"):
        merge_condition("cholecseg8k", clips, tmp_path / "data", tmp_path / "tracks", "t", "s", k=2,
                        overwrite=True)
    assert tracked.exists()


def test_a_frame_without_valid_depth_is_refused_before_anything_is_written(tmp_path):
    clips = _condition(tmp_path)
    depth_file = tmp_path / "data" / "VID02_b" / DEPTH_FILE
    depth = np.load(depth_file)["depth"]
    depth[1, 3, 4] = 0.0
    np.savez(depth_file, depth=depth)
    with pytest.raises(ValueError, match="VID02_b: frame 1: .*no valid depth"):
        merge_condition("cholecseg8k", clips, tmp_path / "data", tmp_path / "tracks", "t", "t_k2", k=2)
    assert not (tmp_path / "tracks" / "VID01_a" / "t_k2").exists()


def test_a_region_id_below_minus_one_is_refused_before_anything_is_written(tmp_path):
    clips = _condition(tmp_path)
    label = tmp_path / "tracks" / "VID02_b" / "t" / "label_0000.npy"
    bad = np.load(label)
    bad[0, 0] = -2
    np.save(label, bad)
    with pytest.raises(ValueError, match="VID02_b: frame 0 holds a region id below -1"):
        merge_condition("cholecseg8k", clips, tmp_path / "data", tmp_path / "tracks", "t", "t_k2", k=2)
    assert not (tmp_path / "tracks" / "VID01_a" / "t_k2").exists()
