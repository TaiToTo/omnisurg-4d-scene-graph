"""The trajectory measures, on trajectories whose answer is known.

- A similar copy of the truth scores 0.
- A camera that never moves scores exactly 1.0.
- The truth read backwards, or mirrored, scores badly.
- A uniform scale keeps the scale ratio at 1; a scale that grows along the
  trajectory moves the ratio off 1.
"""

import numpy as np
import pytest

from evalkit.tools.pose_metrics import (ate, centers_from_cam_to_world, centers_from_world_to_cam, rpe,
                                        rot_from_world_to_cam, scale_consistency, umeyama)

N = 120
TIMES = np.linspace(0, 24, N)
GT_C = np.stack([30 * np.sin(TIMES / 3), 12 * np.cos(TIMES / 5), 4 * TIMES], 1)
ANG = 0.7
RZ = np.array([[np.cos(ANG), -np.sin(ANG), 0], [np.sin(ANG), np.cos(ANG), 0], [0, 0, 1]])
SIMILAR = (GT_C @ RZ.T) * 0.013 + np.array([5.0, -2.0, 1.0])


def test_a_similar_copy_of_the_truth_scores_0_and_recovers_the_scale():
    r = ate(SIMILAR, GT_C)
    assert r["ate_rel"] < 1e-9
    assert abs(r["scale"] - 1 / 0.013) < 1e-6


def test_a_camera_that_never_moves_scores_exactly_1():
    assert abs(ate(np.zeros((N, 3)), GT_C)["ate_rel"] - 1.0) < 1e-12


def test_the_truth_read_backwards_scores_badly():
    assert ate(GT_C[::-1].copy(), GT_C)["ate_rel"] > 0.3


def test_a_mirrored_trajectory_is_not_fitted_as_a_rotation():
    assert ate(SIMILAR * np.array([1.0, 1.0, -1.0]), GT_C)["ate_rel"] > 0.1


def test_rpe_of_a_similar_copy_is_0_and_nan_without_pairs():
    # The true camera turns as it moves, so a relative translation must be read in the camera's own frame.
    a = TIMES / 10
    turning = np.stack([np.stack([np.cos(a), np.zeros(N), np.sin(a)], 1), np.tile([0.0, 1.0, 0.0], (N, 1)),
                        np.stack([-np.sin(a), np.zeros(N), np.cos(a)], 1)], 1)
    s = ate(SIMILAR, GT_C)["scale"]
    est_R = np.einsum("ij,njk->nik", RZ, turning)
    q = rpe(SIMILAR, est_R, GT_C, turning, TIMES, s, dt=1.0)
    assert q["rpe_trans_rel"] < 1e-9 and q["rpe_rot_deg"] < 1e-9 and q["n_pairs"] > 3
    q = rpe(SIMILAR, est_R, GT_C, turning, TIMES, s, dt=1e6)
    assert q["n_pairs"] == 0 and np.isnan(q["rpe_trans_rel"])


def test_a_uniform_scale_is_consistent_and_a_growing_one_is_not():
    assert abs(scale_consistency(SIMILAR, GT_C)["scale_ratio"] - 1.0) < 1e-9
    assert scale_consistency(SIMILAR * np.linspace(1.0, 2.0, N)[:, None], GT_C)["scale_ratio"] > 1.3


def test_the_two_directions_of_a_pose_give_one_centre():
    rng = np.random.default_rng(0)
    R = np.linalg.qr(rng.normal(size=(5, 3, 3)))[0]
    c = rng.normal(size=(5, 3))
    T = np.tile(np.eye(4), (5, 1, 1))
    T[:, :3, :3], T[:, :3, 3] = R, c
    E = np.linalg.inv(T)[:, :3, :]
    assert np.allclose(centers_from_world_to_cam(E), centers_from_cam_to_world(T))
    assert np.allclose(rot_from_world_to_cam(E), R)


def test_inputs_a_fit_cannot_take_are_refused():
    with pytest.raises(ValueError, match="two \\(N, 3\\) arrays"):
        umeyama(np.zeros((5, 3)), np.zeros((5, 2)))
    with pytest.raises(ValueError, match="needs 3"):
        umeyama(np.zeros((2, 3)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="estimated frames"):
        ate(np.zeros((4, 3)), GT_C)
    with pytest.raises(ValueError, match="does not move"):
        ate(SIMILAR, np.ones((N, 3)))
