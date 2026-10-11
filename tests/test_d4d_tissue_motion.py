"""Test the D4D tissue-motion step on depth maps and masks made up for the test.

A made-up clip has 40 frames at 20 a second, so a sample compares frame `i` with frame `i + 10`, every 5
frames. Its tissue sinks 5 mm at frame 15, and an instrument under the mask moves 50 mm at frame 30. Each
refusal has a test that plants its fault:

- a clip with one frame, which has no frame interval, and a clip too short for `MIN_SAMPLES` samples;
- a clip without per-frame poses, or with poses of another shape.
"""

import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import surgical_core.d4d as d4d
from pipeline import d4d_tissue_motion as tm

H, W = 40, 60
INSTRUMENT_ROWS = 12


def write_clip(tmp_path, n: int = 40, masked: bool = True) -> tuple[list, list]:
    """Write a clip's depth maps and masks named by their times; return their paths."""
    (tmp_path / "depth").mkdir()
    (tmp_path / "masks").mkdir()
    depth, masks = [], []
    for i in range(n):
        name = f"100_{i * 50_000_000:09d}"
        d = np.full((H, W), 0.1)
        if i >= 15:
            d += 0.005
        d[:INSTRUMENT_ROWS] = 0.05 if i < 30 else 0.1
        d[-1, -1] = np.nan
        m = np.zeros((H, W), np.uint8)
        if masked:
            m[:INSTRUMENT_ROWS] = 255
        np.save(tmp_path / "depth" / f"{name}.npy", d)
        cv2.imwrite(str(tmp_path / "masks" / f"{name}.png"), m)
        depth.append(tmp_path / "depth" / f"{name}.npy")
        masks.append(tmp_path / "masks" / f"{name}.png")
    return depth, masks


def test_the_motion_is_the_90th_percentile_of_the_depth_change_over_the_lag(tmp_path):
    t, p90 = tm.series(*write_clip(tmp_path))
    assert t == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0, 1.25])
    # Samples 5 and 10 straddle the tissue's 5 mm sink; the instrument's move, masked, counts in none.
    assert p90 == pytest.approx([0.0, 5.0, 5.0, 0.0, 0.0, 0.0])


def test_without_its_mask_the_instrument_counts_as_motion(tmp_path):
    _, p90 = tm.series(*write_clip(tmp_path, masked=False))
    assert p90 == pytest.approx([0.0, 5.0, 5.0, 0.0, 50.0, 50.0])


def test_a_clip_with_one_frame_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no frame interval"):
        tm.series(*write_clip(tmp_path, n=1))


def test_a_clip_too_short_for_the_samples_is_refused(tmp_path):
    with pytest.raises(ValueError, match="1 samples, fewer than 3"):
        tm.series(*write_clip(tmp_path, n=12))


T = [0.25 * i for i in range(13)]


@pytest.mark.parametrize("hot, want", [
    ([1, 2], [[0.25, 0.75]]),
    # A gap of 0.5 s, up to `MERGE_GAP_S`, joins two stretches.
    ([1, 2, 5], [[0.25, 1.5]]),
    # Beyond it, the second stretch stands alone, and at 0.25 s is shorter than `MIN_INTERVAL_S`.
    ([1, 2, 7], [[0.25, 0.75]]),
    # A stretch at the last sample has no later sample to end at, so it lasts 0 s.
    ([12], []),
    ([], []),
])
def test_a_stretch_runs_from_its_first_sample_above_the_threshold_to_the_sample_after_its_last(hot, want):
    p90 = [2.0 if i in hot else 1.0 for i in range(len(T))]
    assert tm.intervals(T, p90, threshold=1.0) == want


def record(quiet: float, moves: bool) -> dict:
    return {"camera_moves": moves, "quiet_level_mm": quiet, "t": [0.0, 0.25, 0.5, 0.75],
            "p90_mm": [0.0, 3.0, 3.0, 0.0]}


@pytest.mark.parametrize("records, threshold", [
    # The moving camera's quiet level does not count.
    ([record(0.5, False), record(0.7, False), record(0.9, False), record(5.0, True)], 2.1),
    ([record(0.1, False), record(5.0, True)], tm.MIN_THRESHOLD_MM),
    # With no still camera, every clip counts.
    ([record(1.0, True), record(2.0, True)], 4.5),
])
def test_the_threshold_is_a_multiple_of_the_median_quiet_level_of_the_still_clips(records, threshold):
    assert tm.add_intervals(records) == pytest.approx(threshold)
    assert all(r["threshold_mm"] == pytest.approx(threshold) for r in records)
    assert records[0]["intervals"] == ([[0.25, 0.75]] if threshold < 3.0 else [])


def bounds(span_m: float) -> np.ndarray:
    """Return LLFF pose bounds of three frames whose positions span `span_m` along x."""
    b = np.zeros((3, 17))
    b[:, :15].reshape(3, 3, 5)[:, 0, 3] = [0.0, span_m / 2, span_m]
    return b


@pytest.mark.parametrize("span, moves", [(0.006, True), (0.004, False)])
def test_the_camera_moves_when_its_positions_span_more_than_5_mm(span, moves):
    assert tm.camera_moves(SimpleNamespace(load_pose_bounds=lambda: bounds(span))) is moves


@pytest.mark.parametrize("pose_bounds", [None, np.zeros((3, 15))])
def test_a_clip_without_per_frame_poses_of_17_numbers_is_refused(pose_bounds):
    c = SimpleNamespace(load_pose_bounds=lambda: pose_bounds, name="Clip_1",
                        session=SimpleNamespace(name="s_a", specimen=SimpleNamespace(name="specimen_1")))
    with pytest.raises(ValueError, match="specimen_1/s_a/Clip_1: no per-frame poses"):
        tm.camera_moves(c)


def test_main_writes_one_record_per_clip_with_the_shared_threshold(tmp_path, monkeypatch):
    depth, masks = write_clip(tmp_path)
    c = SimpleNamespace(name="Clip_1", path=tmp_path, load_pose_bounds=lambda: bounds(0.0),
                        session=SimpleNamespace(name="s_a", specimen=SimpleNamespace(name="specimen_1")),
                        left_img_paths=[str(p) for p in depth])
    monkeypatch.setattr(d4d, "clips", lambda root: [c])
    monkeypatch.setattr(d4d, "frame_files", lambda clip, folder, suffix: depth if folder == "stereo_depth" else masks)
    tm.main(["--root", str(tmp_path), "--out", str(tmp_path / "motion.json"), "--jobs", "1"])
    [r] = json.loads((tmp_path / "motion.json").read_text())
    assert list(r) == ["clip", "camera_moves", "t", "p90_mm", "quiet_level_mm", "threshold_mm", "intervals"]
    assert r["clip"] == "specimen_1/s_a/Clip_1" and r["camera_moves"] is False
    assert r["threshold_mm"] == tm.MIN_THRESHOLD_MM
    assert r["intervals"] == [[0.25, 0.75]]
