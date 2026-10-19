"""Test the StereoMIS reader on a sequence made up for the test.

The calibration is two identical cameras side by side, so rectifying leaves a
view as it is. Each view of the video holds stripes one pixel wide of two
colours, whose mean differs in red and blue, so a view halved by area takes
that mean. The ground truth moves the camera on circles that pass through the
origin, one turn per clip, so each clip's RMS radius is the circle's radius.
The video tests skip where FFmpeg is not installed. Each refusal has a test
that plants its fault.
"""

import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

import cv2  # noqa: E402

import surgical_core.stereomis as S  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="FFmpeg is not installed")

BASELINE_MM = 4.0
N_VIDEO_FRAMES = 8


@pytest.fixture(autouse=True)
def _fresh_caches():
    """Clear the reader's caches, since every test writes a dataset of its own under the same name."""
    caches = (S.calib, S.video_path, S.video_info, S.frame_times, S.load_gt, S.gt_jump_rows, S.mask_numbers)
    for f in caches:
        f.cache_clear()
    yield
    for f in caches:
        f.cache_clear()


def write_calibration(seq_dir: Path, res: tuple[int, int] = S.VIEW_SIZE, right_cc: tuple[float, float] = (640, 512),
                      kc_0: float = 0.0) -> None:
    """Write a calibration of two cameras, the right one `BASELINE_MM` to the right.

    The cameras are identical unless the right one's principal point is moved to `right_cc`. Both have the
    radial distortion `kc_0`, none by default.
    """
    seq_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for section, tx, (cc_x, cc_y) in (("StereoLeft", 0.0, (640, 512)), ("StereoRight", -BASELINE_MM, right_cc)):
        lines.append(f"[{section}]")
        lines += [f"res_x = {res[0]}", f"res_y = {res[1]}", "fc_x = 1000", "fc_y = 1000", f"cc_x = {cc_x}",
                  f"cc_y = {cc_y}"]
        lines += [f"kc_0 = {kc_0}"] + [f"kc_{i} = 0" for i in range(1, 5)]
        lines += [f"R_{i} = {1 if i in (0, 4, 8) else 0}" for i in range(9)]
        lines += [f"T_0 = {tx}", "T_1 = 0", "T_2 = 0"]
    (seq_dir / "StereoCalibration.ini").write_text("\n".join(lines) + "\n")


def write_gt(seq_dir: Path, t_m: np.ndarray, quat: np.ndarray | None = None) -> None:
    """Write `groundtruth.txt` with translations `t_m` in m and quaternions (x, y, z, w), identity by default."""
    seq_dir.mkdir(parents=True, exist_ok=True)
    if quat is None:
        quat = np.tile([0.0, 0.0, 0.0, 1.0], (len(t_m), 1))
    a = np.column_stack([np.arange(len(t_m)), t_m, quat])
    np.savetxt(seq_dir / "groundtruth.txt", a)


# Each view's stripes lie this far above and below its colour, in every channel.
STRIPE = 10


def colours(i: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the RGB colours of frame `i`'s left and right views, each with red and blue at least 20 apart."""
    return np.array([40 + 20 * i, 128, 220 - 20 * i]), np.array([220 - 20 * i, 64, 40 + 20 * i])


@pytest.fixture(scope="module")
def video_file(tmp_path_factory):
    """Write, once, a video of `N_VIDEO_FRAMES` stacked frames at 60 frames a second, each view striped."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg is not installed")
    path = tmp_path_factory.mktemp("video") / "video.mp4"
    w, h = S.VIDEO_SIZE
    frames = []
    for i in range(N_VIDEO_FRAMES):
        f = np.empty((h, w, 3), np.uint8)
        for top, colour in zip((0, h // 2), colours(i)):
            f[top:top + h // 2, 0::2] = colour + STRIPE
            f[top:top + h // 2, 1::2] = colour - STRIPE
        frames.append(f)
    # Lossless H.264 in RGB, so that every pixel comes back as it was written.
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{w}x{h}", "-r", "60", "-i", "-", "-c:v", "libx264rgb", "-preset", "ultrafast", "-qp", "0",
                    "-pix_fmt", "rgb24", str(path)], input=b"".join(f.tobytes() for f in frames), check=True)
    return path


@pytest.fixture
def video_seq(tmp_path, video_file):
    """A sequence P1 with the calibration of `write_calibration` and the video of `video_file`."""
    seq_dir = tmp_path / "StereoMIS" / "P1"
    write_calibration(seq_dir)
    (seq_dir / "video.mp4").symlink_to(video_file)
    return tmp_path / "StereoMIS"


# --------------------------------------------------------------------------- calibration


def test_two_identical_cameras_give_the_baseline_and_half_the_intrinsics(tmp_path):
    write_calibration(tmp_path / "P1")
    c = S.Calib(tmp_path, "P1")
    assert c.size == S.VIEW_SIZE
    assert c.baseline_mm == pytest.approx(BASELINE_MM, abs=1e-12)
    np.testing.assert_array_equal(c.K_half[:2], c.K_full[:2] / 2.0)
    np.testing.assert_array_equal(c.K_half[2], [0.0, 0.0, 1.0])


def test_a_flat_view_stays_flat_when_rectified(tmp_path):
    write_calibration(tmp_path / "P1")
    c = S.Calib(tmp_path, "P1")
    img = np.full((S.VIEW_SIZE[1], S.VIEW_SIZE[0], 3), 77, np.uint8)
    for side in ("l", "r"):
        out = c.rectify(img, side)
        assert out.shape == img.shape and (out == 77).all()


def test_each_view_is_rectified_by_its_own_camera(tmp_path):
    """The right camera's principal point sits 40 px lower, so the two maps differ and each side takes its own."""
    write_calibration(tmp_path / "P1", right_cc=(640, 552))
    c = S.Calib(tmp_path, "P1")
    assert not np.array_equal(c.map_left[1], c.map_right[1])
    img = np.random.default_rng(0).integers(0, 256, (S.VIEW_SIZE[1], S.VIEW_SIZE[0], 3), dtype=np.uint8)
    for side, (mx, my) in (("l", c.map_left), ("r", c.map_right)):
        np.testing.assert_array_equal(c.rectify(img, side), cv2.remap(img, mx, my, cv2.INTER_LINEAR))


def test_the_rectified_views_keep_no_black_border(tmp_path):
    """Under barrel distortion, a rectification that kept every pixel of the view would leave black corners."""
    write_calibration(tmp_path / "P1", kc_0=-0.3)
    c = S.Calib(tmp_path, "P1")
    white = np.full((S.VIEW_SIZE[1], S.VIEW_SIZE[0], 3), 255, np.uint8)
    for side in ("l", "r"):
        assert (c.rectify(white, side) > 0).all(), f"side {side}: the rectified view has black pixels"


def test_both_rectified_cameras_share_one_principal_point(tmp_path):
    """The right camera's principal point sits 40 px to the left; rectified, both views put it at one column."""
    write_calibration(tmp_path / "P1", right_cc=(600, 512))
    c = S.Calib(tmp_path, "P1")
    assert c.P_left[0, 2] == pytest.approx(c.P_right[0, 2], abs=1e-9)


def test_a_side_other_than_left_or_right_is_refused(tmp_path):
    write_calibration(tmp_path / "P1")
    with pytest.raises(ValueError, match="side"):
        S.Calib(tmp_path, "P1").rectify(np.zeros((4, 4, 3), np.uint8), "left")


def test_a_sequence_without_a_calibration_is_refused(tmp_path):
    (tmp_path / "P1").mkdir()
    with pytest.raises(FileNotFoundError):
        S.Calib(tmp_path, "P1")


def test_a_calibration_for_another_view_size_is_refused(tmp_path):
    write_calibration(tmp_path / "P1", res=(1280, 1000))
    with pytest.raises(ValueError, match="1280x1000, not 1280x1024"):
        S.Calib(tmp_path, "P1")


# --------------------------------------------------------------------------- video


@needs_ffmpeg
def test_each_position_decodes_its_own_frame_in_order_with_the_left_view_on_top(video_seq):
    got = list(S.iter_frame_set(video_seq, "P1", [5, 0, 3, 5, 7]))
    assert [f for f, _, _ in got] == [0, 3, 5, 7]
    for f, left, right in got:
        assert left.shape == right.shape == (S.HALF_SIZE[1], S.HALF_SIZE[0], 3)
        assert left.dtype == right.dtype == np.uint8
        lc, rc = colours(f)
        assert np.abs(left.astype(int) - lc).max() <= 1, f"frame {f}: the left view is not frame {f}'s top half"
        assert np.abs(right.astype(int) - rc).max() <= 1, f"frame {f}: the right view is not frame {f}'s bottom half"


@needs_ffmpeg
def test_the_views_come_back_in_rgb_order(video_seq):
    [(_, left, right)] = S.iter_frame_set(video_seq, "P1", [0])
    for view, colour in zip((left, right), colours(0)):
        np.testing.assert_allclose(view.reshape(-1, 3).mean(0), colour, atol=1)


@needs_ffmpeg
def test_a_view_is_halved_by_area_so_its_stripes_average_out(video_seq):
    """A pixel of the halved view covers two stripes; taking one pixel of the two would be `STRIPE` away."""
    [(_, left, _)] = S.iter_frame_set(video_seq, "P1", [3])
    assert np.abs(left.astype(int) - colours(3)[0]).max() <= 1


@needs_ffmpeg
def test_the_video_is_probed_for_its_size_count_and_rate(video_seq):
    assert S.video_info(video_seq, "P1") == dict(width=1280, height=2048, n_frames=N_VIDEO_FRAMES, fps=60.0)
    assert S.clip_stride(video_seq, "P1") == 12


@needs_ffmpeg
def test_each_frame_s_time_is_its_timestamp(video_seq):
    np.testing.assert_array_equal(S.frame_times(video_seq, "P1"), np.arange(N_VIDEO_FRAMES) / 60.0)


def paused_video(tmp_path: Path, monkeypatch) -> Path:
    """Write a video of 6 frames at 60 a second whose last 3 come 7 frames late, as where a recording paused.

    Its B-frames make the packets' order, the order they are decoded in, differ from the order of their timestamps.
    """
    seq_dir = tmp_path / "P1"
    seq_dir.mkdir()
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=64x48:rate=60", "-frames:v", "6", "-vf", "setpts=N+gte(N\\,3)*7",
                    "-fps_mode", "passthrough", "-pix_fmt", "yuv420p", str(seq_dir / "video.mp4")], check=True)
    monkeypatch.setattr(S, "video_info", lambda root, seq: dict(width=1280, height=2048, n_frames=6, fps=60.0))
    return tmp_path


@needs_ffmpeg
def test_a_pause_in_the_recording_stays_in_the_times_and_the_times_come_in_order(tmp_path, monkeypatch):
    root = paused_video(tmp_path, monkeypatch)
    np.testing.assert_array_equal(S.frame_times(root, "P1"), np.array([0, 1, 2, 10, 11, 12]) / 60.0)
    with pytest.raises(ValueError, match="read-only"):
        S.frame_times(root, "P1")[0] = 1.0


@needs_ffmpeg
def test_timestamps_of_another_number_of_frames_are_refused(tmp_path, monkeypatch):
    root = paused_video(tmp_path, monkeypatch)
    monkeypatch.setattr(S, "video_info", lambda root, seq: dict(width=1280, height=2048, n_frames=7, fps=60.0))
    with pytest.raises(ValueError, match="6 timestamps for 7 frames"):
        S.frame_times(root, "P1")


@pytest.mark.parametrize("packets, refusal", [
    ([{"pts": 0}, {"pts": 256}, {"pts": 256}], "1 frames share a timestamp"),
    ([{"pts": 0}, {}, {"pts": 512}], "1 frames have no timestamp"),
])
def test_frames_without_a_timestamp_of_their_own_are_refused(tmp_path, monkeypatch, packets, refusal):
    """The planted fault: what ffprobe reports is replaced."""
    (tmp_path / "P1").mkdir()
    (tmp_path / "P1" / "video.mp4").write_bytes(b"")
    probe = json.dumps({"packets": packets, "streams": [{"time_base": "1/15360"}]})
    monkeypatch.setattr(S.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=probe))
    monkeypatch.setattr(S, "video_info", lambda root, seq: dict(width=1280, height=2048, n_frames=3, fps=60.0))
    with pytest.raises(ValueError, match=refusal):
        S.frame_times(tmp_path, "P1")


def test_no_frame_wanted_decodes_nothing(tmp_path):
    assert list(S.iter_frame_set(tmp_path, "P1", [])) == []


@needs_ffmpeg
def test_a_position_outside_the_video_is_refused(video_seq):
    with pytest.raises(IndexError):
        list(S.iter_frame_set(video_seq, "P1", [N_VIDEO_FRAMES]))
    with pytest.raises(IndexError):
        list(S.iter_frame_set(video_seq, "P1", [-1, 2]))


@needs_ffmpeg
def test_a_video_that_ends_before_the_frames_wanted_is_refused(video_seq, monkeypatch):
    """The planted fault: the probe claims more frames than the video holds."""
    info = dict(S.video_info(video_seq, "P1"), n_frames=N_VIDEO_FRAMES + 4)
    monkeypatch.setattr(S, "video_info", lambda root, seq: info)
    with pytest.raises(RuntimeError, match="3 frames wanted, 1 decoded"):
        list(S.iter_frame_set(video_seq, "P1", [7, 9, 10]))


@needs_ffmpeg
def test_a_video_that_is_not_two_stacked_views_is_refused(tmp_path):
    seq_dir = tmp_path / "P1"
    seq_dir.mkdir()
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=64x48:rate=10", "-frames:v", "3", "-pix_fmt", "yuv420p", str(seq_dir / "video.mp4")],
                   check=True)
    with pytest.raises(ValueError, match="stacked views"):
        S.video_info(tmp_path, "P1")


@needs_ffmpeg
def test_a_video_that_does_not_say_how_many_frames_it_has_is_refused(tmp_path):
    """The planted fault: a Matroska file under an mp4 name, whose stream carries no frame count."""
    seq_dir = tmp_path / "P1"
    seq_dir.mkdir()
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=64x48:rate=10", "-frames:v", "2", "-pix_fmt", "yuv420p", "-f", "matroska",
                    str(seq_dir / "video.mp4")], check=True)
    with pytest.raises(ValueError, match="how many frames"):
        S.video_info(tmp_path, "P1")


def test_a_sequence_needs_exactly_one_video(tmp_path):
    (tmp_path / "P1").mkdir()
    with pytest.raises(FileNotFoundError):
        S.video_path(tmp_path, "P1")
    (tmp_path / "P1" / "a.mp4").write_bytes(b"")
    (tmp_path / "P1" / "b.mp4").write_bytes(b"")
    S.video_path.cache_clear()
    with pytest.raises(FileNotFoundError, match="a.mp4"):
        S.video_path(tmp_path, "P1")


# --------------------------------------------------------------------------- ground truth


def test_a_quarter_turn_about_z_and_a_translation_in_m_are_read_as_a_matrix_and_mm(tmp_path):
    s = math.sqrt(0.5)
    write_gt(tmp_path / "P1", np.array([[0.001, 0.0, -0.002], [0.0, 0.0, 0.0]]),
             np.array([[0.0, 0.0, s, s], [0.0, 0.0, 0.0, 1.0]]))
    R, t = S.load_gt(tmp_path, "P1")
    np.testing.assert_allclose(R[0], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-15)
    np.testing.assert_array_equal(R[1], np.eye(3))
    np.testing.assert_array_equal(t, [[1.0, 0.0, -2.0], [0.0, 0.0, 0.0]])


def test_every_sequence_numbers_its_files_one_ahead_and_its_rows_three_behind_its_frames():
    # P1's shipped frame N is the video's frame N - 1, as P2's files are; the common row offset follows from it.
    assert S.DEPTH_FILE_OFFSET == {s: -1 for s in S.SEQUENCES}
    assert S.GT_ROW_OFFSET == {s: -3 for s in S.SEQUENCES}


def test_the_poses_of_ground_truth_rows_are_read_in_mm(tmp_path):
    write_gt(tmp_path / "P2_0", np.arange(30.0)[:, None] * [0.001, 0.0, 0.0])
    T = S.gt_pose(tmp_path, "P2_0", [0, 6, 29])
    assert T.shape == (3, 4, 4)
    np.testing.assert_array_equal(T[:, :3, 3], [[0.0, 0, 0], [6.0, 0, 0], [29.0, 0, 0]])
    np.testing.assert_array_equal(T[:, 3], np.tile([0.0, 0, 0, 1], (3, 1)))


def test_a_row_outside_the_ground_truth_is_refused(tmp_path):
    write_gt(tmp_path / "P1", np.zeros((30, 3)))
    with pytest.raises(IndexError):
        S.gt_pose(tmp_path, "P1", [-1])
    with pytest.raises(IndexError):
        S.gt_pose(tmp_path, "P1", [30])
    with pytest.raises(ValueError, match="no row"):
        S.gt_pose(tmp_path, "P1", [])


def test_a_sequence_without_a_measured_offset_is_refused():
    with pytest.raises(KeyError, match="ground truth's rows has not been measured"):
        S.measured_offset(S.GT_ROW_OFFSET, "P3")


def test_a_sequence_without_a_measured_offset_has_no_break_read(tmp_path):
    # The workbench took an offset of 0 here, and read the breaks of rows that are not the clip's.
    write_gt(tmp_path / "P3", np.zeros((30, 3)))
    with pytest.raises(KeyError, match="P3: the offset between its frames and the ground truth's rows has not"):
        S.has_gt_jump(tmp_path, "P3", 11, 20)


def test_a_sequence_without_a_measured_offset_is_not_cut_into_clips(tmp_path):
    with pytest.raises(KeyError, match="P3: the offset between its frames and the ground truth's rows has not"):
        S.clips(tmp_path, tmp_path, "P3")


def test_a_sequence_without_a_measured_file_offset_has_no_mask_found(tmp_path, monkeypatch):
    monkeypatch.setitem(S.GT_ROW_OFFSET, "P3", -3)
    write_mask(tmp_path / "P3", 10)
    with pytest.raises(KeyError, match="P3: the offset between its frames and the file numbers of its masks"):
        S.nearest_mask(tmp_path, "P3", 10)
    with pytest.raises(KeyError, match="the file numbers of its masks and depth maps has not been measured"):
        S.clips(tmp_path, tmp_path, "P3")


def test_the_ground_truth_is_returned_read_only(tmp_path):
    # It is cached: a caller that wrote into it would move every later pose of the sequence.
    write_gt(tmp_path / "P1", np.zeros((6, 3)))
    R, t = S.load_gt(tmp_path, "P1")
    with pytest.raises(ValueError, match="read-only"):
        t -= 1.0
    with pytest.raises(ValueError, match="read-only"):
        R[0, 0, 0] = 2.0
    with pytest.raises(ValueError, match="read-only"):
        S.gt_jump_rows(tmp_path, "P1")[:] = 0


def test_a_ground_truth_without_eight_columns_is_refused(tmp_path):
    (tmp_path / "P1").mkdir()
    np.savetxt(tmp_path / "P1" / "groundtruth.txt", np.zeros((4, 7)))
    with pytest.raises(ValueError, match="8 columns"):
        S.load_gt(tmp_path, "P1")


def test_a_ground_truth_with_a_gap_in_its_rows_is_refused(tmp_path):
    write_gt(tmp_path / "P1", np.zeros((4, 3)))
    path = tmp_path / "P1" / "groundtruth.txt"
    a = np.loadtxt(path)
    a[2:, 0] += 1
    np.savetxt(path, a)
    with pytest.raises(ValueError, match="row number"):
        S.load_gt(tmp_path, "P1")


def test_a_ground_truth_with_a_value_that_is_not_finite_is_refused(tmp_path):
    t = np.zeros((4, 3))
    t[2, 1] = np.nan
    write_gt(tmp_path / "P1", t)
    with pytest.raises(ValueError, match="not finite"):
        S.load_gt(tmp_path, "P1")


def test_a_step_above_five_mm_or_five_degrees_is_a_break_and_five_mm_is_not(tmp_path):
    t = np.zeros((6, 3))
    t[2:] += [0.005, 0.0, 0.0]          # rows 1 to 2: exactly 5 mm
    t[4:] += [0.0, 0.0051, 0.0]         # rows 3 to 4: 5.1 mm
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (6, 1))
    half = math.radians(5.1) / 2        # rows 4 to 5: a turn of 5.1 degrees about x
    quat[5] = [math.sin(half), 0.0, 0.0, math.cos(half)]
    write_gt(tmp_path / "P1", t, quat)
    np.testing.assert_array_equal(S.gt_jump_rows(tmp_path, "P1"), [3, 4])


@pytest.mark.parametrize("jump_row, expected", [(6, False), (7, True), (17, True), (18, False)])
def test_a_break_counts_for_a_clip_from_the_step_into_its_first_row_to_the_step_out_of_its_last(
        tmp_path, jump_row, expected):
    """Positions 11 to 20 take rows 8 to 17; a break is the step from its row to the next."""
    t = np.zeros((30, 3))
    t[jump_row + 1:] += [0.01, 0.0, 0.0]
    write_gt(tmp_path / "P1", t)
    assert S.has_gt_jump(tmp_path, "P1", 11, 20) is expected


# --------------------------------------------------------------------------- masks


def write_mask(seq_dir: Path, number: int, tissue: bool = True, size: tuple[int, int] = S.HALF_SIZE) -> None:
    (seq_dir / "masks").mkdir(parents=True, exist_ok=True)
    a = np.zeros((size[1], size[0], 3), np.uint8)
    a[:, : size[0] // 2] = 255 if tissue else 0
    cv2.imwrite(str(seq_dir / "masks" / ("%06dl.png" % number)), a)


def test_a_mask_is_read_by_its_file_number_with_tissue_true(tmp_path):
    write_mask(tmp_path / "P2_0", 10)
    m = S.load_mask(tmp_path, "P2_0", 10)
    assert m.shape == (S.HALF_SIZE[1], S.HALF_SIZE[0]) and m.dtype == bool
    assert m[:, :320].all() and not m[:, 320:].any()
    with pytest.raises(FileNotFoundError):
        S.load_mask(tmp_path, "P2_0", 9)


def test_a_frame_s_own_mask_is_one_file_number_ahead_of_it(tmp_path):
    write_mask(tmp_path / "P1", 11)
    assert S.nearest_mask(tmp_path, "P1", 10) == 11


def test_the_nearest_mask_within_two_frames_is_taken_the_earlier_on_a_tie(tmp_path):
    """Masks 11 and 13 are those of frames 10 and 12."""
    write_mask(tmp_path / "P1", 11)
    write_mask(tmp_path / "P1", 13)
    assert [S.nearest_mask(tmp_path, "P1", f) for f in (7, 8, 11, 13, 14, 15)] == [None, 11, 11, 13, 13, None]


def test_a_file_that_is_not_a_mask_is_not_found_as_one(tmp_path):
    write_mask(tmp_path / "P1", 11)
    (tmp_path / "P1" / "masks" / "000013r.png").write_bytes(b"")
    (tmp_path / "P1" / "masks" / "13l.png").write_bytes(b"")
    assert S.mask_numbers(tmp_path, "P1") == {11}


def test_a_mask_that_cannot_be_read_is_refused(tmp_path):
    (tmp_path / "P1" / "masks").mkdir(parents=True)
    (tmp_path / "P1" / "masks" / "000010l.png").write_bytes(b"not a png")
    with pytest.raises(ValueError, match="cannot be read"):
        S.load_mask(tmp_path, "P1", 10)


def test_a_mask_of_another_size_is_refused(tmp_path):
    write_mask(tmp_path / "P1", 10, size=(320, 256))
    with pytest.raises(ValueError, match="320x256"):
        S.load_mask(tmp_path, "P1", 10)


# --------------------------------------------------------------------------- clips

FPS = 60.0
STRIDE = 12
SPAN = S.CLIP_FRAMES * STRIDE
# Seven grid places from `CLIP_GRID_START`; the ground truth ends inside the seventh, so six clips are returned.
N_FRAMES = 4 + 7 * SPAN
N_ROWS = 6 * SPAN + 600
RADII_MM = (20.0, 5.0, 0.5, 20.0, 0.5, 0.5)
# The camera steps 6 mm inside clip 3, and turns 6 degrees about x inside clip 5.
JUMP_ROW = 3 * SPAN + 600
TURN_ROW = 5 * SPAN + 600


def circle_gt(n_rows: int = N_ROWS) -> tuple[np.ndarray, np.ndarray]:
    """Move the camera on one circle per clip, of `RADII_MM`, each through the origin; one turn per clip.

    Returns translations in m and quaternions (x, y, z, w).
    """
    rows = np.arange(n_rows)
    clip = np.minimum(rows // SPAN, len(RADII_MM) - 1)
    r = np.asarray(RADII_MM)[clip] / 1000.0
    theta = 2 * np.pi * (rows % SPAN) / SPAN
    t = np.column_stack([r - r * np.cos(theta), r * np.sin(theta), np.zeros(n_rows)])
    t[JUMP_ROW + 1:, 2] += 0.006
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (n_rows, 1))
    quat[TURN_ROW + 1:] = [math.sin(math.radians(3)), 0.0, 0.0, math.cos(math.radians(3))]
    return t, quat


@pytest.fixture
def clip_seq(tmp_path, monkeypatch):
    """A sequence P2_0 at 60 frames a second whose clips are moving, slow, static, broken or without surface.

    P2_0's depth files are numbered one ahead of their positions. A map every 30 positions shows a surface,
    but for the first 15 of clip 1 and every map of clip 4. One more map, on clip 0's last position, shows none.
    Each frame's time is its position over the frame rate, and no frame has a mask.
    """
    root, depth_root = tmp_path / "StereoMIS", tmp_path / "StereoMIS_depth"
    write_gt(root / "P2_0", *circle_gt())
    positions = np.append(np.arange(30, N_FRAMES, 30), 4 + SPAN - 1)
    stats = np.column_stack([positions + 1, np.full(len(positions), 100.0), np.full(len(positions), 0.9)])
    stats[-1, 1] = 10.0
    clip1 = np.flatnonzero((positions >= 4 + SPAN) & (positions < 4 + 2 * SPAN))
    stats[clip1[:15], 1] = 10.0
    clip4 = (positions >= 4 + 4 * SPAN) & (positions < 4 + 5 * SPAN)
    stats[clip4, 2] = 0.4
    (depth_root / "P2_0").mkdir(parents=True)
    np.save(depth_root / "P2_0" / "stats.npy", stats)
    monkeypatch.setattr(S, "video_info", lambda r, s: dict(width=1280, height=2048, n_frames=N_FRAMES, fps=FPS))
    monkeypatch.setattr(S, "frame_times", lambda r, s: np.arange(N_FRAMES) / FPS)
    return root, depth_root


def test_a_sequence_is_cut_into_clips_and_each_is_marked_by_its_inputs(clip_seq):
    """Clip 4 is static and shows no surface, clip 5 static and broken: the first reason listed is given."""
    cl = S.clips(*clip_seq, "P2_0")
    assert [c["name"] for c in cl] == [f"P2_0__clip_{k:04d}" for k in range(6)]
    assert [c["drop"] for c in cl] == ["", "", "gt_static", "gt_jump", "no_surface", "gt_jump"]
    assert [c["usable"] for c in cl] == [True, True, False, False, False, False]
    assert [c["stratum"] for c in cl] == ["moving", "slow", "slow", "moving", "slow", "slow"]
    assert [cl[k]["span_mm"] for k in (0, 1, 2, 4, 5)] == [20.0, 5.0, 0.5, 0.5, 0.5]
    # Clip 0 holds 45 maps, the one on its last position among them; clip 1 holds 45, 15 without surface.
    assert [c["depth_ok"] for c in cl] == [0.978, 0.667, 1.0, 1.0, 0.0, 1.0]
    first = cl[0]
    assert tuple(first) == S.CLIP_KEYS
    assert first["seq"] == "P2_0" and first["stride"] == STRIDE and first["gt_row_offset"] == -3
    assert first["frames"] == list(range(4, 4 + SPAN, STRIDE))
    assert S.gt_rows(first) == list(range(1, 1 + SPAN, STRIDE))
    assert first["times"] == [f / FPS for f in first["frames"]]
    assert first["masks"] == [None] * S.CLIP_FRAMES
    assert cl[1]["frames"][0] == 4 + SPAN


def test_a_video_at_59_94_frames_a_second_takes_every_12th_frame(clip_seq, monkeypatch):
    """Nine of the ten sequences declare 60000/1001 frames a second: 11.988 frames to 0.2 s, rounded to 12."""
    fps = 60000 / 1001
    monkeypatch.setattr(S, "video_info", lambda r, s: dict(width=1280, height=2048, n_frames=N_FRAMES, fps=fps))
    assert S.clip_stride(clip_seq[0], "P2_0") == 12
    first = S.clips(*clip_seq, "P2_0")[0]
    assert first["stride"] == 12 and first["frames"][:3] == [4, 16, 28]


def test_a_clip_takes_its_times_from_the_timestamps_not_from_the_frame_rate(clip_seq, monkeypatch):
    """The planted fault of the P2 videos: they declare 59.94 frames a second, and their frames come slower."""
    monkeypatch.setattr(S, "frame_times", lambda r, s: np.arange(N_FRAMES) * 0.0181)
    first = S.clips(*clip_seq, "P2_0")[0]
    assert first["times"] == [f * 0.0181 for f in first["frames"]]


@pytest.mark.parametrize("offset, first_name", [(0, "P2_0__clip_0000"), (-3, "P2_0__clip_0000"),
                                                (-6, "P2_0__clip_0001")])
def test_the_grid_starts_at_its_fixed_position_whatever_the_ground_truth_offset(clip_seq, monkeypatch, offset,
                                                                                   first_name):
    """At an offset of -6, the first place's first rows would be -2 and -1, so its clip is not returned."""
    monkeypatch.setitem(S.GT_ROW_OFFSET, "P2_0", offset)
    first = S.clips(*clip_seq, "P2_0")[0]
    assert first["name"] == first_name
    assert first["frames"][0] == S.CLIP_GRID_START + SPAN * int(first_name[-4:])
    assert first["gt_row_offset"] == offset


def test_a_clip_records_the_mask_nearest_each_frame(clip_seq):
    """Frame 4 has the mask of its own number, 5; frame 16 takes the one of frame 15; frame 28 has none near."""
    root, depth_root = clip_seq
    for number in (5, 16, 25):
        write_mask(root / "P2_0", number)
    assert S.clips(root, depth_root, "P2_0")[0]["masks"][:3] == [5, 16, None]


def test_a_clip_whose_rows_run_past_the_ground_truth_is_not_returned(clip_seq):
    root, depth_root = clip_seq
    write_gt(root / "P2_0", *circle_gt(5 * SPAN + 1000))
    cl = S.clips(root, depth_root, "P2_0")
    assert [c["name"] for c in cl] == [f"P2_0__clip_{k:04d}" for k in range(5)]


def test_a_map_on_a_clip_s_first_position_counts_for_that_clip(clip_seq):
    root, depth_root = clip_seq
    stats = np.load(depth_root / "P2_0" / "stats.npy")
    np.save(depth_root / "P2_0" / "stats.npy", np.vstack([stats, [4 + SPAN + 1, 10.0, 0.9]]))
    # Clip 1 now holds 46 maps, 16 without surface; clip 0, which ends one position before, is unchanged.
    assert [c["depth_ok"] for c in S.clips(root, depth_root, "P2_0")][:2] == [0.978, 0.652]


@pytest.mark.parametrize("median_mm, valid, shows", [
    (30.0, 0.9, True), (400.0, 0.9, True), (29.9, 0.9, False), (400.1, 0.9, False),
    (100.0, 0.5, False), (100.0, 0.51, True),
])
def test_a_map_shows_a_surface_with_its_median_from_30_to_400_mm_and_more_than_half_valid(median_mm, valid, shows):
    assert S.shows_surface(median_mm, valid) is shows


@pytest.mark.parametrize("depth_ok, jump, span_mm, drop", [
    (0.5, False, 1.0, ""), (0.49, False, 20.0, "no_surface"), (0.5, True, 20.0, "gt_jump"),
    (0.5, False, 0.99, "gt_static"), (0.4, True, 0.5, "no_surface"), (1.0, True, 0.5, "gt_jump"),
])
def test_a_clip_is_left_out_for_the_first_reason_that_holds_each_threshold_kept(depth_ok, jump, span_mm, drop):
    assert S.drop_of(depth_ok, jump, span_mm) == drop


@pytest.mark.parametrize("span_mm, stratum", [(10.0, "moving"), (9.99, "slow"), (1.0, "slow")])
def test_a_clip_moves_from_10_mm_up(span_mm, stratum):
    assert S.stratum_of(span_mm) == stratum


def test_a_depth_export_without_a_row_is_refused(clip_seq):
    root, depth_root = clip_seq
    np.save(depth_root / "P2_0" / "stats.npy", np.zeros((0, 5)))
    with pytest.raises(ValueError, match="stats.npy"):
        S.clips(root, depth_root, "P2_0")


def test_a_clip_without_a_depth_map_is_refused(clip_seq):
    """The planted fault: the export holds no map on clip 2, which would otherwise be left out as `no_surface`."""
    root, depth_root = clip_seq
    stats = np.load(depth_root / "P2_0" / "stats.npy")
    positions = stats[:, 0] - 1
    on_clip_2 = (positions >= 4 + 2 * SPAN) & (positions < 4 + 3 * SPAN)
    np.save(depth_root / "P2_0" / "stats.npy", stats[~on_clip_2])
    with pytest.raises(ValueError, match="P2_0__clip_0002 shows a surface cannot be measured"):
        S.clips(root, depth_root, "P2_0")


def test_a_depth_export_with_a_file_number_that_is_not_whole_is_refused(clip_seq):
    root, depth_root = clip_seq
    stats = np.load(depth_root / "P2_0" / "stats.npy")
    stats[3, 0] += 0.5
    np.save(depth_root / "P2_0" / "stats.npy", stats)
    with pytest.raises(ValueError, match="not whole"):
        S.clips(root, depth_root, "P2_0")


def test_a_depth_export_with_a_file_number_given_twice_is_refused(clip_seq):
    """The planted fault: a second row for one map, which `clips` would count once, by one of the two rows."""
    root, depth_root = clip_seq
    stats = np.load(depth_root / "P2_0" / "stats.npy")
    np.save(depth_root / "P2_0" / "stats.npy", np.vstack([stats, [stats[3, 0], 10.0, 0.9]]))
    with pytest.raises(ValueError, match="more than once"):
        S.clips(root, depth_root, "P2_0")


# --------------------------------------------------------------------------- the clip list


def test_a_clip_list_is_read_back_as_it_was_written_one_clip_a_line(clip_seq, tmp_path):
    cl = S.clips(*clip_seq, "P2_0")
    S.write_clip_list(cl, tmp_path / "clips.json")
    text = (tmp_path / "clips.json").read_text()
    assert text.startswith('[\n{"name": "P2_0__clip_0000", "seq": "P2_0", "usable": true,')
    assert text.endswith("}\n]\n") and len(text.splitlines()) == len(cl) + 2
    assert S.read_clip_list(tmp_path / "clips.json") == cl


def test_the_usable_clips_are_those_of_the_sequences_named_sequence_by_sequence():
    cl = [dict(name=f"{s}__clip_{k:04d}", seq=s, usable=u)
          for s, k, u in (("P2_0", 0, True), ("P1", 0, True), ("P1", 1, False), ("P2_1", 0, True))]
    assert [c["name"] for c in S.usable_clips(cl, ["P1", "P2_0"])] == ["P1__clip_0000", "P2_0__clip_0000"]
    assert [c["name"] for c in S.usable_clips(cl)] == ["P1__clip_0000", "P2_0__clip_0000", "P2_1__clip_0000"]


def planted(clip: dict, fault: str) -> dict:
    """Return a copy of `clip` with one fault planted."""
    c = json.loads(json.dumps(clip))
    if fault == "key missing":
        del c["masks"]
    elif fault == "keys out of order":
        c = {k: c[k] for k in reversed(S.CLIP_KEYS)}
    elif fault == "another sequence's name":
        c["name"] = "P2_1__clip_0000"
    elif fault == "unknown sequence":
        c["seq"], c["name"] = "P3", "P3__clip_0000"
    elif fault == "a frame short":
        c["frames"], c["times"], c["masks"] = c["frames"][:-1], c["times"][:-1], c["masks"][:-1]
    elif fault == "a time short":
        c["times"] = c["times"][:-1]
    elif fault == "a mask short":
        c["masks"] = c["masks"][:-1]
    elif fault == "a frame off the stride":
        c["frames"][5] += 1
    elif fault == "times that stall":
        c["times"][5] = c["times"][4]
    elif fault == "usable with a reason":
        c["drop"] = "gt_jump"
    elif fault == "left out without a reason":
        c["usable"] = False
    elif fault == "usable as a number":
        c["usable"] = 1
    return c


@pytest.mark.parametrize("fault, refusal", [
    ("key missing", "not a list of clips with the keys"),
    ("keys out of order", "not a list of clips with the keys"),
    ("another sequence's name", "is not <sequence>__clip_NNNN of a sequence"),
    ("unknown sequence", "is not <sequence>__clip_NNNN of a sequence"),
    ("a frame short", "has 111 frames, 111 times and 111 masks"),
    ("a time short", "has 112 frames, 111 times and 112 masks"),
    ("a mask short", "has 112 frames, 112 times and 111 masks"),
    ("a frame off the stride", "frames are not 12 apart, or its times do not increase"),
    ("times that stall", "frames are not 12 apart, or its times do not increase"),
    ("usable with a reason", "records usable True and the reason 'gt_jump'"),
    ("left out without a reason", "records usable False and the reason ''"),
    ("usable as a number", "records usable 1 and the reason ''"),
])
def test_a_clip_list_the_measurements_cannot_trust_is_refused(clip_seq, tmp_path, fault, refusal):
    cl = list(S.clips(*clip_seq, "P2_0"))
    cl[0] = planted(cl[0], fault)
    (tmp_path / "clips.json").write_text(json.dumps(cl))
    with pytest.raises(ValueError, match=re.escape(refusal)):
        S.read_clip_list(tmp_path / "clips.json")


def test_a_clip_list_that_names_a_clip_twice_is_refused(clip_seq, tmp_path):
    cl = S.clips(*clip_seq, "P2_0")
    (tmp_path / "clips.json").write_text(json.dumps([cl[0], cl[1], cl[0]]))
    with pytest.raises(ValueError, match="'P2_0__clip_0000' of 'P2_0' is not .* or comes twice"):
        S.read_clip_list(tmp_path / "clips.json")


def test_a_clip_list_that_is_not_a_list_is_refused(tmp_path):
    (tmp_path / "clips.json").write_text(json.dumps({"clips": []}))
    with pytest.raises(ValueError, match="not a list of clips"):
        S.read_clip_list(tmp_path / "clips.json")
