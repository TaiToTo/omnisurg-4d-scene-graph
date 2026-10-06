"""`surgical_core.geometry.camera`: pixels to camera, world and glTF space, exactly, and `project.backproject` on it.

A sign or a transpose wrong here moves every point cloud and every warped
label without an error, so each transform is pinned against its formula,
and the round trip through `project.project_world_to_frame` lands back on
the pixel grid.
"""

import numpy as np
import pytest

from surgical_core.geometry import project
from surgical_core.geometry.camera import GLTF_FLIP, backproject_depth, cam_to_world, world_to_gltf


def _intrinsics(fx=320.0, fy=240.0, cx=160.0, cy=120.0):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _rotation(rx, ry, rz):
    """`Rz @ Ry @ Rx` from three angles in radians."""
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _ext(R, t):
    return np.hstack([R, np.asarray(t)[:, None]])


# ------------------------------------------------------------- glTF


def test_the_flip_turns_y_and_z_and_is_its_own_inverse():
    assert np.array_equal(GLTF_FLIP, [1, -1, -1])
    assert np.array_equal(world_to_gltf(np.array([2.0, 3.0, 5.0])), [2.0, -3.0, -5.0])
    pts = np.random.default_rng(0).normal(size=(50, 3))
    assert np.array_equal(world_to_gltf(world_to_gltf(pts)), pts)


# ------------------------------------------------------------- backproject_depth


def test_the_principal_point_back_projects_onto_the_axis():
    depth = np.zeros((3, 4))
    depth[1, 2] = 7.5
    pts = backproject_depth(depth, _intrinsics(cx=2.0, cy=1.0))
    assert np.array_equal(pts[1 * 4 + 2], [0.0, 0.0, 7.5])


def test_a_pixel_off_centre_back_projects_through_the_intrinsics():
    depth = np.zeros((25, 15))
    depth[22, 13] = 4.0
    pts = backproject_depth(depth, _intrinsics(fx=100.0, fy=200.0, cx=10.0, cy=20.0))
    assert np.allclose(pts[22 * 15 + 13], [3 * 4.0 / 100.0, 2 * 4.0 / 200.0, 4.0])


def test_the_points_follow_the_order_of_the_flattened_depth():
    depth = np.random.default_rng(1).uniform(0.5, 3.0, size=(5, 7))
    pts = backproject_depth(depth, _intrinsics())
    assert pts.shape == (35, 3)
    assert np.array_equal(pts[:, 2], depth.reshape(-1))


def test_intrinsics_of_another_shape_are_refused():
    with pytest.raises(ValueError, match="K must have shape"):
        backproject_depth(np.ones((4, 4)), np.eye(4))


# ------------------------------------------------------------- cam_to_world


def test_cam_to_world_is_the_column_formula():
    R, t = _rotation(0.3, -0.7, 1.1), np.array([0.5, -1.2, 4.0])
    pts = np.random.default_rng(2).normal(size=(20, 3))
    assert np.allclose(cam_to_world(pts, R, t), np.array([R.T @ (p - t) for p in pts]))
    assert np.allclose(cam_to_world(pts, np.eye(3), np.zeros(3)), pts)
    assert np.allclose(cam_to_world(np.array([10.0, 10.0, 10.0]), np.eye(3), np.array([1.0, 2.0, 3.0])), [9, 8, 7])


def test_the_camera_origin_lands_at_minus_r_transposed_t():
    R, t = _rotation(0.4, 0.2, -0.5), np.array([1.5, -2.0, 3.3])
    out = cam_to_world(np.zeros(3), R, t)
    assert out.shape == (3,)
    assert np.allclose(out, -R.T @ t)


@pytest.mark.parametrize("R, t", [(np.eye(4)[:, :3], np.zeros(4)), (np.eye(3), np.zeros(4)), (np.eye(3), np.zeros((3, 1)))])
def test_extrinsics_of_another_shape_are_refused(R, t):
    with pytest.raises(ValueError, match="must have shape"):
        cam_to_world(np.zeros((2, 3)), R, t)


# ------------------------------------------------------------- the round trip, and project.backproject


def test_back_projection_then_projection_lands_on_the_pixel_grid():
    K, R, t = _intrinsics(), _rotation(0.2, -0.3, 0.5), np.array([0.1, -0.4, 2.0])
    depth = np.full((10, 12), 3.0)
    gltf = world_to_gltf(cam_to_world(backproject_depth(depth, K), R, t))
    u, v, Z = project.project_world_to_frame(world_to_gltf(gltf), K, _ext(R, t))
    ug, vg = np.meshgrid(np.arange(12), np.arange(10))
    assert np.allclose(u, ug.reshape(-1)) and np.allclose(v, vg.reshape(-1)) and np.allclose(Z, 3.0)


def test_project_backproject_is_the_camera_transforms_on_the_valid_pixels():
    rng = np.random.default_rng(3)
    depth = rng.uniform(0.5, 4.0, size=(9, 11))
    depth[2, 3], depth[4, 5], depth[6, 7] = 0.0, np.nan, 1e-7
    K, R, t = _intrinsics(), _rotation(0.15, -0.25, 0.7), np.array([0.3, -0.6, 1.8])
    Pw, m, ys, xs = project.backproject(depth, K, _ext(R, t))
    assert m.sum() == depth.size - 3
    assert np.array_equal(Pw, cam_to_world(backproject_depth(depth, K)[m.reshape(-1)], R, t))
    assert np.allclose((Pw @ R.T + t)[:, 2], depth[ys, xs]), "each point back in the camera has its pixel's depth"


@pytest.mark.parametrize("ext", [np.eye(4), np.eye(3), np.hstack([np.eye(3), np.zeros((3, 2))])])
def test_project_backproject_refuses_extrinsics_of_another_shape(ext):
    # The (3, 5) case is the planted fault: its slices would silently drop the last column.
    with pytest.raises(ValueError, match="ext_w2c must have shape"):
        project.backproject(np.ones((4, 4)), _intrinsics(), ext)
