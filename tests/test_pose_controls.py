"""Test the floors and the ceiling on trajectories and stereo pairs whose answer is known.

- The static floor scores exactly 1.0, and the line floor runs at the true
  path's mean speed, so it fits a straight true path exactly.
- Disparity becomes depth by `bf / d`, inside the kept range only.
- The ceiling follows a camera sliding over a textured plane: the views are
  windows of one texture, the right one 24 px further along, and each frame
  moves the windows 2 px, which is 0.5 mm at the plane's 100 mm.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")

import cv2  # noqa: E402

import evalkit.tools.pose_controls as PC  # noqa: E402
import surgical_core.stereomis as stereomis  # noqa: E402
from evalkit.tools.pose_metrics import ate  # noqa: E402

# A camera of 400 px focal length on 320x256 views, and a 6 mm baseline: bf is 2400, so a plane at 100 mm is at
# a disparity of 24 px, and a step of 0.5 mm sideways moves it 2 px.
K = np.array([[400.0, 0.0, 160.0], [0.0, 400.0, 128.0], [0.0, 0.0, 1.0]])
BASELINE_MM = 6.0
VIEW_W, VIEW_H = 320, 256
DISPARITY = 24
STEP_PX = 2
STEP_MM = STEP_PX * 100.0 / 400.0


MARGIN = 80
ROLL_DEG = 4.0
ROLL_STEP_PX = 4


def texture() -> np.ndarray:
    rng = np.random.default_rng(3)
    t = rng.integers(0, 256, (VIEW_H + 2 * MARGIN, VIEW_W + 240)).astype(np.float32)
    return cv2.normalize(cv2.GaussianBlur(t, (0, 0), 1.5), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


TEXTURE = texture()


def views(plane: np.ndarray, x: int) -> tuple[np.ndarray, np.ndarray]:
    """Cut the RGB views whose left one starts at column `x` of `plane`; the right one starts `DISPARITY` later."""
    rows = slice(MARGIN, MARGIN + VIEW_H)
    left = plane[rows, x:x + VIEW_W]
    right = plane[rows, x + DISPARITY:x + DISPARITY + VIEW_W]
    return cv2.cvtColor(left, cv2.COLOR_GRAY2RGB), cv2.cvtColor(right, cv2.COLOR_GRAY2RGB)


def pair(k: int) -> tuple[np.ndarray, np.ndarray]:
    """Frame `k`'s RGB views: the camera has moved `k * STEP_MM` along x over a plane at 100 mm."""
    return views(TEXTURE, 80 + STEP_PX * k)


def rolled_pair(k: int) -> tuple[np.ndarray, np.ndarray]:
    """Frame `k`'s RGB views: the camera has moved `k * ROLL_STEP_PX` pixels' worth along the world's x, and
    turned `k * ROLL_DEG` about its optical axis, from x towards y.

    A camera turned by `Rz(a)` sees a point of the plane at `Rz(-a)` of where it saw it, about the principal
    point; OpenCV's rotation by `a` degrees moves the image so.
    """
    x = 80 + ROLL_STEP_PX * k
    M = cv2.getRotationMatrix2D((x + K[0, 2], MARGIN + K[1, 2]), k * ROLL_DEG, 1.0)
    return views(cv2.warpAffine(TEXTURE, M, TEXTURE.shape[::-1], flags=cv2.INTER_LINEAR), x)


def rz(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------- floors


def test_the_static_floor_scores_exactly_one_against_a_moving_truth():
    gt = np.stack([np.linspace(0, 30, 50), np.sin(np.linspace(0, 3, 50)) * 5, np.zeros(50)], 1)
    c, R = PC.static_floor(50)
    assert c.shape == (50, 3) and not c.any()
    np.testing.assert_array_equal(R, np.tile(np.eye(3), (50, 1, 1)))
    assert ate(c, gt)["ate_rel"] == pytest.approx(1.0, abs=1e-12)


def test_the_line_floor_moves_along_z_at_the_true_paths_mean_speed():
    """The true path is 5 mm then 12 mm in 3 s, so the line runs at 17/3 mm a second."""
    gt = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [3.0, 4.0, 12.0]])
    c, R = PC.line_floor(gt, np.array([10.0, 11.0, 13.0]))
    np.testing.assert_allclose(c, [[0, 0, 0], [0, 0, 17 / 3], [0, 0, 17]], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(R, np.tile(np.eye(3), (3, 1, 1)))


def test_the_line_floor_fits_a_straight_true_path_at_constant_speed_exactly():
    times = np.arange(20) * 0.2
    gt = np.outer(times, [3.0, -1.0, 2.0]) + [5.0, 5.0, 5.0]
    assert ate(PC.line_floor(gt, times)[0], gt)["ate_rel"] < 1e-9


def test_the_line_floor_refuses_times_that_do_not_fit_the_centres():
    gt = np.zeros((3, 3))
    with pytest.raises(ValueError, match="2 times for 3 centres"):
        PC.line_floor(gt, np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="no speed"):
        PC.line_floor(gt, np.array([2.0, 2.0, 2.0]))


# --------------------------------------------------------------------------- disparity


class FixedMatcher:
    """A matcher that returns one disparity map, in sixteenths of a pixel, whatever it is given."""

    def __init__(self, d16):
        self.d16 = np.asarray(d16, np.int16)

    def compute(self, left, right):
        return self.d16


def test_disparity_becomes_depth_by_bf_over_d_inside_the_kept_range():
    """With bf 2500: 5 px is 500 mm, 25 px 100 mm, 125 px 20 mm, both ends kept; 1 px and below, 12.5 mm and
    2353 mm are dropped."""
    d16 = [[-16, 0, 16, 17], [80, 400, 2000, 3200]]
    z = PC.sgbm_depth(FixedMatcher(d16), None, None, 2500.0)
    np.testing.assert_array_equal(z, [[0, 0, 0, 0], [500, 100, 20, 0]])


def test_a_disparity_of_one_pixel_or_less_is_unknown_even_inside_the_range():
    """With bf 400, 1 px would be 400 mm and half a pixel 800; only 2 px, 200 mm, is kept."""
    z = PC.sgbm_depth(FixedMatcher([[16, 8, 32]]), None, None, 400.0)
    np.testing.assert_array_equal(z, [[0, 0, 200]])


def test_the_matcher_is_set_for_the_clips_depth_range():
    m = PC.sgbm()
    got = (m.getMinDisparity(), m.getNumDisparities(), m.getBlockSize(), m.getP1(), m.getP2(),
           m.getUniquenessRatio(), m.getSpeckleWindowSize(), m.getSpeckleRange(), m.getDisp12MaxDiff(), m.getMode())
    assert got == (0, 96, 5, 600, 2400, 8, 120, 2, 1, cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def test_the_matcher_finds_a_plane_at_its_depth():
    left, right = (cv2.cvtColor(v, cv2.COLOR_RGB2GRAY) for v in pair(0))
    z = PC.sgbm_depth(PC.sgbm(), left, right, K[0, 0] * BASELINE_MM)
    inside = z[20:-20, 120:-20]
    assert (inside > 0).mean() > 0.95
    assert np.median(inside[inside > 0]) == pytest.approx(100.0, rel=0.01)


# --------------------------------------------------------------------------- ceiling


def run_vo(frames, masks=None) -> PC.StereoVO:
    vo = PC.StereoVO(K, BASELINE_MM)
    for i, k in enumerate(frames):
        vo.add(*pair(k), None if masks is None else masks[i])
    return vo


def test_the_ceiling_follows_a_camera_sliding_over_a_plane():
    vo = run_vo(range(6))
    c, R = vo.result()
    assert vo.n_fail == 0
    np.testing.assert_allclose(c, np.outer(np.arange(6) * STEP_MM, [1.0, 0.0, 0.0]), atol=0.02 * STEP_MM)
    np.testing.assert_allclose(R, np.tile(np.eye(3), (6, 1, 1)), atol=2e-3)


def test_the_ceiling_follows_a_camera_turning_about_its_axis_as_it_slides():
    """Each step is a turn and a slide, so the steps compose in the camera's frame, not the world's."""
    vo = PC.StereoVO(K, BASELINE_MM)
    for k in range(5):
        vo.add(*rolled_pair(k), None)
    c, R = vo.result()
    assert vo.n_fail == 0
    for k in range(5):
        np.testing.assert_allclose(R[k], rz(k * ROLL_DEG), atol=5e-3)
    # The turned views are resampled, which moves the centres by a fraction of a millimetre.
    step_mm = ROLL_STEP_PX * 100.0 / K[0, 0]
    np.testing.assert_allclose(c, np.outer(np.arange(5) * step_mm, [1.0, 0.0, 0.0]), atol=0.2)


def test_a_turn_then_a_step_along_the_cameras_x_moves_it_along_the_worlds_y(monkeypatch):
    """PnP is replaced by two steps: the camera turns 90 degrees from x towards y, then moves 1 mm along its x.

    PnP gives the earlier camera's frame in the later one's: a turn of the camera by `Rz(90)` is `Rz(-90)`
    there, and a move by +1 mm along x is a translation of -1 mm.
    """
    steps = iter([(np.array([0.0, 0.0, -np.pi / 2]), np.zeros(3)), (np.zeros(3), np.array([-1.0, 0.0, 0.0]))])

    def pnp(obj, img, K_, dist, **kw):
        rvec, tvec = next(steps)
        return True, rvec.reshape(3, 1), tvec.reshape(3, 1), np.arange(len(obj)).reshape(-1, 1)

    monkeypatch.setattr(PC.cv2, "solvePnPRansac", pnp)
    vo = run_vo([0, 0, 0])
    c, R = vo.result()
    np.testing.assert_allclose(R[1], rz(90), atol=1e-12)
    np.testing.assert_allclose(R[2], rz(90), atol=1e-12)
    np.testing.assert_allclose(c, [[0, 0, 0], [0, 0, 0], [0, 1, 0]], atol=1e-12)


def test_a_step_without_depth_repeats_the_pose_and_is_counted():
    """Frame 2's mask hides everything; a step takes depth from its earlier frame, so only the step 2 to 3 fails."""
    none = np.zeros((VIEW_H, VIEW_W), bool)
    vo = run_vo(range(5), masks=[None, None, none, None, None])
    c, _ = vo.result()
    assert vo.n_fail == 1
    assert c[2, 0] == pytest.approx(2 * STEP_MM, rel=0.02)
    np.testing.assert_array_equal(vo.poses[3], vo.poses[2])
    assert c[4, 0] - c[3, 0] == pytest.approx(STEP_MM, rel=0.02)


def test_a_reset_starts_a_new_trajectory_at_the_identity():
    vo = run_vo(range(3))
    vo.reset()
    assert len(vo.poses) == 1 and vo.n_fail == 0 and vo.prev is None
    np.testing.assert_array_equal(vo.poses[0], np.eye(4))


def test_each_clip_of_a_sequence_gets_its_own_trajectory_from_one_decode(monkeypatch, tmp_path):
    clips = (dict(name="P1__clip_0000", frames=[0, 1, 2, 3], usable=True),
             dict(name="P1__clip_0001", frames=[4, 5, 6], usable=False),
             dict(name="P1__clip_0002", frames=[7, 8, 9], usable=True))
    decoded, masked = [], []

    class Calib:
        K_half, baseline_mm = K, BASELINE_MM

    def iter_frame_set(root, seq, frames):
        decoded.append(list(frames))
        for f in frames:
            yield (f, *pair(f))

    def load_mask_nearest(root, seq, f):
        masked.append(f)
        return None

    monkeypatch.setattr(stereomis, "clips", lambda root, depth_root, seq: clips)
    monkeypatch.setattr(stereomis, "calib", lambda root, seq: Calib)
    monkeypatch.setattr(stereomis, "iter_frame_set", iter_frame_set)
    monkeypatch.setattr(stereomis, "load_mask_nearest", load_mask_nearest)
    out = PC.run_sequence(tmp_path, tmp_path, "P1")
    assert decoded == [[0, 1, 2, 3, 7, 8, 9]] and masked == [0, 1, 2, 3, 7, 8, 9]
    assert list(out) == ["P1__clip_0000", "P1__clip_0002"]
    for name, frames in (("P1__clip_0000", range(4)), ("P1__clip_0002", range(7, 10))):
        alone = run_vo(frames)
        c, R, n_fail = out[name]
        np.testing.assert_array_equal(c, alone.result()[0])
        np.testing.assert_array_equal(R, alone.result()[1])
        assert n_fail == alone.n_fail == 0


def test_a_sequence_without_a_usable_clip_decodes_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(stereomis, "clips", lambda root, depth_root, seq: (dict(name="x", frames=[0], usable=False),))
    monkeypatch.setattr(stereomis, "iter_frame_set", lambda *a: pytest.fail("decoded"))
    assert PC.run_sequence(tmp_path, tmp_path, "P2_3") == {}
