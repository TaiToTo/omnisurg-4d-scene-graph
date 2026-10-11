"""Test the D4D census on a clip, session details and a motion record made up for the test.

The made-up clip has three frames, 32 × 32 pixels, so four blocks of 16 × 16. Each refusal has a test that
plants its fault:

- session details with a flag other than `Y` or `N`, a session named twice, and a line that is not a session;
- a clip whose session the details do not list;
- a motion record that lacks a clip, holds one twice, or holds a clip the dataset does not.
"""

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import surgical_core.d4d as d4d
from pipeline import d4d_census as census

FRAMES = ["100_000000000", "100_500000000", "101_000000000"]
PARAMS = {"fx": 100.0, "fy": 100.0, "cx": 0.0, "cy": 0.0, "width": 32, "height": 32, "baseline": 0.004}


def pixel_points(pixels: dict[tuple[int, int], int]) -> np.ndarray:
    """Return points 1 m in front of the camera that put `n` points in each pixel `(row, column)`."""
    return np.array([[(c + 0.5) / 100, (r + 0.5) / 100, 1.0] for (r, c), n in pixels.items() for _ in range(n)])


# Block (0, 0) is covered, block (0, 1) has too few points, block (1, 0) is an instrument in the first frame,
# and block (1, 1) is covered.
POINTS = pixel_points({(1, 1): 5, (2, 20): 4, (20, 2): 9, (20, 20): 10})


def make_clip(tmp_path, scans: dict | None = None, params: dict | None = PARAMS) -> SimpleNamespace:
    """Write a clip's masks, an instrument over block (1, 0) in the first frame only, and return the clip."""
    (tmp_path / "left_images_rect_masks").mkdir(exist_ok=True)
    for i, f in enumerate(FRAMES):
        m = np.zeros((32, 32), np.uint8)
        if i == 0:
            m[16:, :16] = 255
        cv2.imwrite(str(tmp_path / "left_images_rect_masks" / f"{f}.png"), m)
    if scans is None:
        scans = {"start": str(tmp_path / "95_000000000.ply"), "end": str(tmp_path / "103_000000000.ply")}
    return SimpleNamespace(name="Clip_1", path=tmp_path, pointclouds=scans,
                           poses={s: np.eye(4) for s in d4d.SIDES}, endoscope_params=params,
                           session=SimpleNamespace(name="s_a", specimen=SimpleNamespace(name="specimen_1")),
                           left_img_paths=[str(tmp_path / "left_images_rect" / f"{f}.png") for f in FRAMES])


def write_details(tmp_path, lines: list[str]) -> str:
    (tmp_path / "info").mkdir()
    (tmp_path / "info" / "session_details.txt").write_text("path,moved_camera\n" + "\n".join(lines) + "\n\n")
    return str(tmp_path)


def test_moved_camera_is_read_per_session(tmp_path):
    root = write_details(tmp_path, ["specimen_1/s_a,N", "specimen_1/s_b,Y"])
    assert census.read_moved_camera(root) == {"specimen_1/s_a": False, "specimen_1/s_b": True}


@pytest.mark.parametrize("lines", [
    ["specimen_1/s_a,maybe"],
    ["specimen_1/s_a,N", "specimen_1/s_a,Y"],
    ["s_a,N"],
])
def test_session_details_with_a_line_that_is_not_a_new_session_and_flag_are_refused(tmp_path, lines):
    with pytest.raises(ValueError, match="is not a new `<specimen>/<session>,Y` or `,N`"):
        census.read_moved_camera(write_details(tmp_path, lines))


def test_a_block_counts_when_mostly_free_of_instruments_and_is_covered_by_five_points():
    hits = d4d.pixel_hits(POINTS, np.eye(4), PARAMS)
    free = np.ones((32, 32), bool)
    assert census.block_cover(hits, free) == 0.75
    free[16:, :16] = False
    assert census.block_cover(hits, free) == 2 / 3
    assert census.block_cover(hits, np.zeros((32, 32), bool)) == 0.0


def test_a_side_takes_the_frame_nearest_its_scan_and_leaves_out_the_instrument(tmp_path, monkeypatch):
    monkeypatch.setattr(d4d, "read_scan", lambda path: POINTS)
    rec = census.measure_clip((make_clip(tmp_path), True, {"intervals": [[0.0, 0.5]]}))
    assert rec == {"key": "specimen_1/s_a/Clip_1", "moved_camera": True,
                   # The scan 5 s before the first frame meets the instrument in block (1, 0).
                   "start": {"present": True, "frame_index": 0, "frame_minus_gt_s": 5.0, "gt_blk_frac": 2 / 3,
                             "active": True},
                   "end": {"present": True, "frame_index": 2, "frame_minus_gt_s": -2.0, "gt_blk_frac": 0.75,
                           "active": False}}


@pytest.mark.parametrize("intervals, start, end", [
    ([[0.0, 0.5]], True, False),
    ([[0.5, 1.0]], False, True),
    ([[0.2, 0.8]], False, False),
])
def test_a_side_is_active_when_the_clips_nearest_end_lies_in_a_stretch_of_motion(tmp_path, intervals, start, end):
    rec = census.measure_clip((make_clip(tmp_path, params=None), False, {"intervals": intervals}))
    assert (rec["start"]["active"], rec["end"]["active"]) == (start, end)


@pytest.mark.parametrize("clip_fields", [{"scans": {}}, {"params": None}])
def test_a_side_without_a_scan_or_a_calibration_is_not_present(tmp_path, clip_fields):
    rec = census.measure_clip((make_clip(tmp_path, **clip_fields), False, {"intervals": []}))
    assert rec["start"] == rec["end"] == {"present": False, "active": False}


def test_a_side_without_a_pose_is_not_present(tmp_path, monkeypatch):
    monkeypatch.setattr(d4d, "read_scan", lambda path: POINTS)
    c = make_clip(tmp_path)
    del c.poses["end"]
    rec = census.measure_clip((c, False, {"intervals": []}))
    assert rec["start"]["present"] and rec["end"] == {"present": False, "active": False}


def test_each_clip_is_paired_with_its_session_and_its_motion_record(tmp_path):
    c = make_clip(tmp_path)
    motion = [{"clip": "specimen_1/s_a/Clip_1", "intervals": []}]
    assert census.tasks([c], {"specimen_1/s_a": True}, motion) == [(c, True, motion[0])]


def test_a_clip_whose_session_is_not_listed_is_refused(tmp_path):
    with pytest.raises(ValueError, match="does not list 1 sessions, the first being specimen_1/s_a"):
        census.tasks([make_clip(tmp_path)], {"specimen_1/s_b": True}, [{"clip": "specimen_1/s_a/Clip_1"}])


@pytest.mark.parametrize("motion", [
    [],
    [{"clip": "specimen_1/s_a/Clip_1"}, {"clip": "specimen_1/s_a/Clip_1"}],
    [{"clip": "specimen_1/s_a/Clip_1"}, {"clip": "specimen_1/s_a/Clip_2"}],
])
def test_a_motion_record_that_does_not_hold_each_clip_once_is_refused(tmp_path, motion):
    with pytest.raises(ValueError, match="does not hold each of the 1 clips once"):
        census.tasks([make_clip(tmp_path)], {"specimen_1/s_a": False}, motion)


def test_main_writes_the_census_of_every_clip(tmp_path, monkeypatch):
    trimesh = pytest.importorskip("trimesh")
    c = make_clip(tmp_path)
    for path in c.pointclouds.values():
        trimesh.PointCloud(POINTS).export(path)
    root = write_details(tmp_path, ["specimen_1/s_a,Y"])
    (tmp_path / "motion.json").write_text(json.dumps([{"clip": "specimen_1/s_a/Clip_1", "intervals": [[0.5, 1.0]]}]))
    monkeypatch.setattr(d4d, "clips", lambda root: [c])
    census.main(["--root", root, "--motion", str(tmp_path / "motion.json"), "--out", str(tmp_path / "census.json"),
                 "--jobs", "1"])
    [rec] = json.loads((tmp_path / "census.json").read_text())
    assert rec["moved_camera"] is True
    assert (rec["start"]["gt_blk_frac"], rec["end"]["gt_blk_frac"]) == (2 / 3, 0.75)
    assert (rec["start"]["active"], rec["end"]["active"]) == (False, True)
