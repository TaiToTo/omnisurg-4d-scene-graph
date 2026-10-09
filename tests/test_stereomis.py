"""Test the StereoMIS reader on a sequence made up for the test.

The calibration is two identical cameras side by side, so a view of one grey
level stays that level when rectified. The ground truth moves the camera on
circles that pass through the origin, one turn per clip, so each clip's RMS
radius is the circle's radius. The video tests skip where FFmpeg is not
installed. Each refusal has a test that plants its fault.
"""

import math
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
    for f in (S.calib, S.video_path, S.video_info, S.load_gt, S.gt_jump_rows):
        f.cache_clear()
    yield
    for f in (S.calib, S.video_path, S.video_info, S.load_gt, S.gt_jump_rows):
        f.cache_clear()


def write_calibration(seq_dir: Path, res: tuple[int, int] = S.VIEW_SIZE, right_cc_y: float = 512) -> None:
    """Write a calibration of two cameras without distortion, the right one `BASELINE_MM` to the right.

    The cameras are identical unless the right one's principal point is moved down to `right_cc_y`.
    """
    seq_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for section, tx, cc_y in (("StereoLeft", 0.0, 512), ("StereoRight", -BASELINE_MM, right_cc_y)):
        lines.append(f"[{section}]")
        lines += [f"res_x = {res[0]}", f"res_y = {res[1]}", "fc_x = 1000", "fc_y = 1000", "cc_x = 640", f"cc_y = {cc_y}"]
        lines += [f"kc_{i} = 0" for i in range(5)]
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


def levels(i: int) -> tuple[int, int]:
    """The grey levels of frame `i`'s left and right views."""
    return 20 + 20 * i, 230 - 20 * i


@pytest.fixture(scope="module")
def video_file(tmp_path_factory):
    """Write, once, a video of `N_VIDEO_FRAMES` stacked frames at 60 frames a second, each view one grey level."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("FFmpeg is not installed")
    path = tmp_path_factory.mktemp("video") / "video.mp4"
    w, h = S.VIDEO_SIZE
    frames = []
    for i in range(N_VIDEO_FRAMES):
        f = np.empty((h, w, 3), np.uint8)
        f[: h // 2], f[h // 2:] = levels(i)
        frames.append(f)
    # Lossless H.264 in 4:4:4, so that a grey level comes back within one step of what was written.
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{w}x{h}", "-r", "60", "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0",
                    "-pix_fmt", "yuv444p", str(path)], input=b"".join(f.tobytes() for f in frames), check=True)
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
    write_calibration(tmp_path / "P1", right_cc_y=552)
    c = S.Calib(tmp_path, "P1")
    assert not np.array_equal(c.map_left[1], c.map_right[1])
    img = np.random.default_rng(0).integers(0, 256, (S.VIEW_SIZE[1], S.VIEW_SIZE[0], 3), dtype=np.uint8)
    for side, (mx, my) in (("l", c.map_left), ("r", c.map_right)):
        np.testing.assert_array_equal(c.rectify(img, side), cv2.remap(img, mx, my, cv2.INTER_LINEAR))


def test_a_side_other_than_left_or_right_is_refused(tmp_path):
    write_calibration(tmp_path / "P1")
    with pytest.raises(ValueError, match="side"):
        S.Calib(tmp_path, "P1").rectify(np.zeros((4, 4, 3), np.uint8), "left")


def test_a_sequence_without_a_calibration_is_refused(tmp_path):
    (tmp_path / "P1").mkdir()
    with pytest.raises(FileNotFoundError):
        S.Calib(tmp_path, "P1")


# --------------------------------------------------------------------------- video


@needs_ffmpeg
def test_each_position_decodes_its_own_frame_in_order_with_the_left_view_on_top(video_seq):
    got = list(S.iter_frame_set(video_seq, "P1", [5, 0, 3, 5, 7]))
    assert [f for f, _, _ in got] == [0, 3, 5, 7]
    for f, left, right in got:
        assert left.shape == right.shape == (S.HALF_SIZE[1], S.HALF_SIZE[0], 3)
        assert left.dtype == right.dtype == np.uint8
        lv, rv = levels(f)
        assert np.abs(left.astype(int) - lv).max() <= 1, f"frame {f}: the left view is not frame {f}'s top half"
        assert np.abs(right.astype(int) - rv).max() <= 1, f"frame {f}: the right view is not frame {f}'s bottom half"


@needs_ffmpeg
def test_the_video_is_probed_for_its_size_count_and_rate(video_seq):
    assert S.video_info(video_seq, "P1") == dict(width=1280, height=2048, n_frames=N_VIDEO_FRAMES, fps=60.0)
    assert S.clip_stride(video_seq, "P1") == 12


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
def test_a_calibration_for_another_view_size_is_refused(video_seq):
    write_calibration(video_seq / "P1", res=(1280, 1000))
    with pytest.raises(ValueError, match="calibration"):
        list(S.iter_frame_set(video_seq, "P1", [0]))


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


def test_a_frame_takes_the_ground_truth_row_four_before_its_position(tmp_path):
    write_gt(tmp_path / "P2_0", np.arange(30.0)[:, None] * [0.001, 0.0, 0.0])
    assert S.gt_row("P2_0", 10) == 6
    T = S.gt_pose(tmp_path, "P2_0", [4, 10, 33])
    assert T.shape == (3, 4, 4)
    np.testing.assert_array_equal(T[:, :3, 3], [[0.0, 0, 0], [6.0, 0, 0], [29.0, 0, 0]])
    np.testing.assert_array_equal(T[:, 3], np.tile([0.0, 0, 0, 1], (3, 1)))


def test_a_frame_without_a_ground_truth_row_is_refused(tmp_path):
    write_gt(tmp_path / "P1", np.zeros((30, 3)))
    with pytest.raises(IndexError):
        S.gt_pose(tmp_path, "P1", [3])
    with pytest.raises(IndexError):
        S.gt_pose(tmp_path, "P1", [34])
    with pytest.raises(ValueError, match="no frame"):
        S.gt_pose(tmp_path, "P1", [])


def test_a_sequence_without_a_measured_offset_is_refused():
    with pytest.raises(KeyError, match="not been measured"):
        S.gt_row("P3", 10)


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


@pytest.mark.parametrize("jump_row, expected", [(5, False), (6, True), (16, True), (17, False)])
def test_a_break_counts_for_a_clip_from_the_step_into_its_first_row_to_the_step_out_of_its_last(
        tmp_path, jump_row, expected):
    """Positions 11 to 20 take rows 7 to 16; a break is the step from its row to the next."""
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


def test_a_mask_is_read_at_its_file_number_with_tissue_true(tmp_path):
    write_mask(tmp_path / "P2_0", 10)
    m = S.load_mask(tmp_path, "P2_0", 9)
    assert m.shape == (S.HALF_SIZE[1], S.HALF_SIZE[0]) and m.dtype == bool
    assert m[:, :320].all() and not m[:, 320:].any()
    assert S.load_mask(tmp_path, "P2_0", 10) is None


def test_the_nearest_mask_within_two_frames_is_taken_the_earlier_on_a_tie(tmp_path):
    write_mask(tmp_path / "P1", 10, tissue=True)
    write_mask(tmp_path / "P1", 12, tissue=False)
    assert S.load_mask_nearest(tmp_path, "P1", 11)[:, :320].all()
    assert not S.load_mask_nearest(tmp_path, "P1", 13)[:, :320].any()
    assert not S.load_mask_nearest(tmp_path, "P1", 14)[:, :320].any()
    assert S.load_mask_nearest(tmp_path, "P1", 15) is None
    assert S.load_mask_nearest(tmp_path, "P1", 7) is None


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
# Seven grid places from position 4; the ground truth ends inside the seventh, so six clips are returned.
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
    assert first["seq"] == "P2_0" and first["stride"] == STRIDE
    assert first["frames"] == list(range(4, 4 + SPAN, STRIDE))
    assert first["times"] == [f / FPS for f in first["frames"]]
    assert cl[1]["frames"][0] == 4 + SPAN


def test_the_grid_starts_at_the_first_position_with_a_ground_truth_row(clip_seq, monkeypatch):
    monkeypatch.setitem(S.GT_ROW_OFFSET, "P2_0", 0)
    assert S.clips(*clip_seq, "P2_0")[0]["frames"][0] == 0


def test_a_clip_whose_rows_run_past_the_ground_truth_is_not_returned(clip_seq):
    root, depth_root = clip_seq
    write_gt(root / "P2_0", *circle_gt(5 * SPAN + 1000))
    cl = S.clips(root, depth_root, "P2_0")
    assert [c["name"] for c in cl] == [f"P2_0__clip_{k:04d}" for k in range(5)]


def test_a_depth_export_without_a_row_is_refused(clip_seq):
    root, depth_root = clip_seq
    np.save(depth_root / "P2_0" / "stats.npy", np.zeros((0, 5)))
    with pytest.raises(ValueError, match="stats.npy"):
        S.clips(root, depth_root, "P2_0")
