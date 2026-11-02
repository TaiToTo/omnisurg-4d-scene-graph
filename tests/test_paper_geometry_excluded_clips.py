"""The StereoMIS exclusion rule keeps a clip whose kinematics explain its depth, and excludes one whose do not.

The scene is a tilted plane seen by a camera that slides sideways, so that holding the camera still changes the
depth at every pixel by more than the 2 % a pixel may differ. Planted faults: kinematics that record a motion the
images do not show, and a clip whose depth maps never overlap.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "paper" / "ipcai2027" / "geometry" / "excluded_clips.py"

K = np.array([[60.0, 0.0, 40.0], [0.0, 60.0, 32.0], [0.0, 0.0, 1.0]])
H, W, N = 64, 80, 12


@pytest.fixture(scope="module")
def rule():
    pytest.importorskip("scipy", reason="the StereoMIS reader needs the `stereomis` extra")
    spec = importlib.util.spec_from_file_location("excluded_clips", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["excluded_clips"] = module
    spec.loader.exec_module(module)
    yield module
    del sys.modules["excluded_clips"]


def poses(step_mm: float) -> np.ndarray:
    """Return camera-to-world poses that slide along x by `step_mm` a frame, without turning."""
    T = np.tile(np.eye(4), (N, 1, 1))
    T[:, 0, 3] = step_mm * np.arange(N)
    return T


def plane_depth(T: np.ndarray) -> np.ndarray:
    """Return each camera's depth of the plane z = 100 + 0.5 x (mm, world frame)."""
    v, u = np.mgrid[0:H, 0:W].astype(float)
    dx = (u - K[0, 2]) / K[0, 0]
    return np.stack([(100 + 0.5 * t[0, 3]) / (1 - 0.5 * dx) for t in T]).astype(np.float32)


def test_kinematics_that_match_the_images_keep_the_clip(rule):
    T = poses(6.0)
    check = rule.check_clip(plane_depth(T), K, T)
    assert check["kinematics"] > 0.9 and check["still"] < 0.1
    assert not check["excluded"]


def test_kinematics_that_record_a_motion_the_images_do_not_show_exclude_the_clip(rule):
    # The camera barely moves, but the record says it slides 6 mm a frame: a break in the kinematic record.
    check = rule.check_clip(plane_depth(poses(0.05)), K, poses(6.0))
    assert check["still"] > 0.9 and check["difference"] < -rule.MARGIN
    assert check["excluded"]


def test_a_clip_whose_depth_never_lands_on_a_depth_cannot_be_checked(rule):
    depth = plane_depth(poses(6.0))
    depth[::2] = 0
    with pytest.raises(ValueError, match="no frame pair"):
        rule.check_clip(depth, K, poses(6.0))


def test_depth_and_poses_of_different_lengths_are_refused(rule):
    T = poses(6.0)
    with pytest.raises(ValueError, match="depth maps for"):
        rule.check_clip(plane_depth(T)[:-1], K, T)
