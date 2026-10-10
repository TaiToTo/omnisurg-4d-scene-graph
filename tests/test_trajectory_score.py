"""A clip's row, on trajectories whose answer is known, and the reading of a depth stage's bundle.

- The truth, written as a depth stage writes extrinsics and read back, scores
  0; its centres negated score badly. A wrong convention, row or unit on the
  way breaks the first.
- A camera that never moves scores exactly 1.0 on `ate_rel` and
  `rpe_trans_rel`.
"""

import numpy as np
import pytest

from trajectory_eval.code_sha import trajectory_code_sha
from trajectory_eval.score import METHOD_BUNDLES, method_trajectory, score

N = 112
TIMES = np.arange(N) * 0.2
CLIP = dict(name="P9__clip_0000", seq="P9", stratum="moving", span_mm=12.5, times=list(TIMES))
KEYS = ["clip", "seq", "stratum", "span_mm", "ate_rel", "ate_mm", "scale", "rpe_trans_rel", "rpe_rot_deg", "n_rpe",
        "scale_ratio", "cond", "trajectory_code_sha"]


def rot(axis: np.ndarray, deg: float) -> np.ndarray:
    """Return the rotation by `deg` degrees about the unit vector `axis`."""
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    a = np.radians(deg)
    return np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * k @ k


def true_poses() -> np.ndarray:
    """Return camera-to-world poses in mm of a camera that moves along a curve and turns as it goes."""
    T = np.tile(np.eye(4), (N, 1, 1))
    T[:, :3, 3] = np.stack([20 * np.sin(TIMES / 4), 8 * np.cos(TIMES / 3), 2 * TIMES], 1)
    axis = np.array([1.0, 2.0, 2.0]) / 3.0
    T[:, :3, :3] = [rot(axis, 1.5 * i) for i in range(N)]
    return T


def write_bundle(clip_dir, method: str, extrinsics: np.ndarray) -> None:
    """Write a bundle of `method` into `clip_dir` as the depth stage does, extrinsics in float32."""
    (clip_dir / "exports" / "mini_npz").mkdir(parents=True, exist_ok=True)
    np.savez(clip_dir / "exports" / "mini_npz" / METHOD_BUNDLES[method], extrinsics=extrinsics.astype(np.float32),
             depth=np.zeros((len(extrinsics), 2, 2), np.float32))


def world_to_cam(T: np.ndarray) -> np.ndarray:
    """Return the (N, 3, 4) world-to-camera extrinsics of camera-to-world poses."""
    E = np.zeros((len(T), 3, 4))
    E[:, :, :3] = np.transpose(T[:, :3, :3], (0, 2, 1))
    E[:, :, 3] = -np.einsum("nij,nj->ni", E[:, :, :3], T[:, :3, 3])
    return E


@pytest.mark.parametrize("method", sorted(METHOD_BUNDLES))
def test_the_truth_read_back_through_a_bundle_scores_0_and_negated_scores_badly(tmp_path, method):
    T = true_poses()
    # Scaled to metres, as a depth model's poses are on a scale of their own.
    T_m = T.copy()
    T_m[:, :3, 3] /= 1000.0
    write_bundle(tmp_path, method, world_to_cam(T_m))
    est_c, est_R = method_trajectory(tmp_path, method, N)
    row = score(CLIP, method, T, est_c, est_R)
    # float32 extrinsics leave a few thousandths of a degree; the camera turns 7.5 degrees between a pair.
    assert row["ate_rel"] < 1e-4 and row["rpe_trans_rel"] < 1e-4 and row["rpe_rot_deg"] < 0.05
    assert abs(row["scale"] - 1000.0) < 0.1
    assert score(CLIP, method, T, -est_c, est_R)["ate_rel"] > 0.1


def test_a_camera_that_never_moves_scores_exactly_1():
    row = score(CLIP, "floor_static", true_poses(), np.zeros((N, 3)), np.tile(np.eye(3), (N, 1, 1)))
    assert (row["ate_rel"], row["rpe_trans_rel"], row["scale"], row["scale_ratio"]) == (1.0, 1.0, 0.0, None)


def test_a_row_names_the_clip_and_the_condition_and_records_the_hash():
    row = score(CLIP, "da3", true_poses(), true_poses()[:, :3, 3], true_poses()[:, :3, :3])
    assert list(row) == KEYS
    assert (row["clip"], row["seq"], row["stratum"], row["span_mm"], row["cond"]) == (
        "P9__clip_0000", "P9", "moving", 12.5, "da3")
    assert row["trajectory_code_sha"] == trajectory_code_sha()


def test_an_estimate_of_another_length_than_the_truth_is_refused():
    with pytest.raises(ValueError, match="111 estimated frames against 112 true ones"):
        score(CLIP, "da3", true_poses(), np.zeros((N - 1, 3)), np.tile(np.eye(3), (N - 1, 1, 1)))


def test_a_bundle_holds_4x4_extrinsics_too(tmp_path):
    E = np.tile(np.eye(4), (N, 1, 1))
    E[:, :3, :] = world_to_cam(true_poses())
    write_bundle(tmp_path, "pi3x", E)
    est_c, est_R = method_trajectory(tmp_path, "pi3x", N)
    np.testing.assert_allclose(est_c, true_poses()[:, :3, 3], atol=1e-3)
    np.testing.assert_allclose(est_R, true_poses()[:, :3, :3], atol=1e-6)


def test_a_bundle_of_another_number_of_frames_than_the_clip_is_refused(tmp_path):
    write_bundle(tmp_path, "da3", world_to_cam(true_poses())[:-1])
    with pytest.raises(ValueError, match=r"extrinsics of shape \(111, 3, 4\), where 112 frames"):
        method_trajectory(tmp_path, "da3", N)


def test_a_clip_without_the_bundle_is_refused(tmp_path):
    write_bundle(tmp_path, "da3", world_to_cam(true_poses()))
    with pytest.raises(FileNotFoundError):
        method_trajectory(tmp_path, "pi3x", N)
