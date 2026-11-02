"""Check the rules of the paper's node scripts that would give a wrong number without a crash.

The scripts under `paper/ipcai2027/nodes/` run by path, so each is imported here by path, with its
directory on `sys.path` for the sibling module it imports.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

import clip_dirs as C
from evalkit.classes import load_table

AREA = Path(__file__).resolve().parent.parent / "paper" / "ipcai2027" / "nodes"
TABLE = load_table("atlas120k")
OMENTUM, LIVER, EXCLUDED = 9, 12, 42


def two_organs() -> np.ndarray:
    """An ATLAS-120k frame of class ids: liver in the three left columns, omentum elsewhere."""
    frame = np.full((6, 8), OMENTUM, dtype=np.int32)
    frame[:, :3] = LIVER
    return frame


@pytest.fixture
def load(monkeypatch):
    """Return a function that imports one script of the area by its name."""
    monkeypatch.syspath_prepend(str(AREA))

    def _load(name: str):
        spec = importlib.util.spec_from_file_location(name, AREA / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return _load


def test_the_seed_map_holds_one_region_per_class_of_frame_zero(load, tmp_path):
    gt_seeds = load("gt_seeds")
    clip_dir = C.write_clip(tmp_path, "c__gt_0001", [two_organs(), two_organs()], dataset="atlas120k")
    labels, record = gt_seeds.seed_regions(*gt_seeds.first_frame_reference(clip_dir, TABLE), TABLE)
    assert (record["class_ids"], record["areas"]) == ([OMENTUM, LIVER], [30, 18])
    assert np.array_equal(labels, np.where(two_organs() == OMENTUM, 0, 1))


@pytest.mark.parametrize("fault, match", [
    ("not a reference frame", "not a reference frame"),
    ("excluded", "excluded from the score"),
    ("background alone", "holds no class"),
])
def test_a_frame_zero_that_cannot_seed_is_refused(load, tmp_path, fault, match):
    gt_seeds = load("gt_seeds")
    first = two_organs()
    if fault == "excluded":
        first[0, 0] = EXCLUDED
    if fault == "background alone":
        first[:] = 0
    clip_dir = C.write_clip(tmp_path, "c__gt_0001", [first, two_organs()], dataset="atlas120k",
                            gt_frames={1} if fault == "not a reference frame" else None)
    with pytest.raises(ValueError, match=match):
        gt_seeds.seed_regions(*gt_seeds.first_frame_reference(clip_dir, TABLE), TABLE)


def test_a_frame_the_excluded_marker_takes_out_has_no_reference_labels(load, tmp_path):
    script = load("reprojection_by_group")
    marked = two_organs()
    marked[0, 0] = EXCLUDED
    clip_dir = C.write_clip(tmp_path, "c__gt_0001", [two_organs(), marked, two_organs()], dataset="atlas120k")
    labels, have = script.reference_labels(clip_dir, TABLE, (6, 8))
    assert have.tolist() == [True, False, True]
    assert np.array_equal(labels[0], two_organs()) and not labels[1].any()


def test_a_camera_moved_right_carries_the_labels_left(load):
    warp_labels = load("reprojection_by_group").warp_labels
    K = np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]])
    E_t = np.hstack([np.eye(3), np.zeros((3, 1))])
    # Frame t + g's camera centre is 0.2 to the right of frame t's; at depth 2 that is 10 pixels.
    E_g = np.hstack([np.eye(3), np.array([[-0.2], [0.0], [0.0]])])
    labels_t = np.zeros((64, 64), dtype=np.int32)
    labels_t[:, 40:45] = 7
    warped = warp_labels(labels_t, np.full((64, 64), 2.0), K, E_g, K, E_t)
    assert np.array_equal(np.flatnonzero(warped[32] == 7), np.arange(30, 35))
    # The pixels whose source falls right of frame t have no label.
    assert not warped[:, 54:].any()


def test_a_class_needs_the_fewest_pairs_to_give_a_row(load):
    script = load("reprojection_by_group")
    short = [0.5] * (script.MIN_PAIRS - 1)
    enough = [0.5] * script.MIN_PAIRS
    rows = script.clip_rows({"reprojection": {1: short, 2: enough}, "fixed": {1: short, 2: [0.25] * 3}})
    assert list(rows) == [2]
    assert rows[2] == {"reprojection": 0.5, "fixed": 0.25}


def test_a_row_without_a_fixed_mask_pair_is_refused(load):
    script = load("reprojection_by_group")
    with pytest.raises(ValueError, match="none under the fixed mask"):
        script.clip_rows({"reprojection": {3: [0.5] * script.MIN_PAIRS}, "fixed": {}})


def test_a_track_no_region_picked_is_left_out_of_the_tracking_iou(load):
    row = load("tracking_iou").row

    def clip(name, video, tracks):
        return {"clip": name, "video": video, "views": {"all": tracks}}

    picked = {"type": "tissue", "picked": True, "tracked": [0.8, 0.6], "fixed": [0.4, 0.2]}
    unpicked = {"type": "tissue", "picked": False, "tracked": [0.0, 0.0], "fixed": [0.5, 0.5]}
    r = row([clip("a__gt_0001", "a", [picked, unpicked]), clip("b__gt_0001", "b", [picked])], "all", None)
    assert (r["tracks"], r["clips"], r["videos"]) == (2, 2, 2)
    assert r["tracked"] == pytest.approx(0.7)
    assert r["fixed"] == pytest.approx(0.3)
