"""What the depth stages write besides depth: the colormap, the GLB point cloud and the camera's axes in glTF.

The colormap and the GLB writer need the `render` extra and skip without it,
as the render tests do. The GLB writer refuses each input that would write a
file that opens and shows the wrong cloud.
"""

import numpy as np
import pytest

from surgical_core.viewer.camera_axes import camera_axes_in_gltf


# ------------------------------------------------------------- the colormap


def test_near_is_warm_far_is_cool_and_an_invalid_pixel_is_far():
    pytest.importorskip("matplotlib")
    from surgical_core.viewer.depth_vis import depth_to_colormap

    depth = np.tile(np.linspace(0.5, 5.0, 40), (30, 1))
    depth[0, 0], depth[0, 1], depth[0, 2] = 0.0, np.nan, np.inf
    rgb = depth_to_colormap(depth)
    assert rgb.shape == (30, 40, 3) and rgb.dtype == np.uint8
    near, far = rgb[5, 3].astype(int), rgb[5, -1].astype(int)
    assert near[0] > near[2] and far[2] > far[0], "Spectral: red near, blue far"
    for col in range(3):
        assert np.array_equal(rgb[0, col], rgb[5, -1]), "a pixel with no depth is coloured as the farthest"
    assert np.array_equal(depth_to_colormap(depth), rgb)


def test_a_depth_at_or_below_depth_min_counts_as_none():
    pytest.importorskip("matplotlib")
    from surgical_core.geometry.valid import DEPTH_MIN
    from surgical_core.viewer.depth_vis import depth_to_colormap

    depth = np.tile(np.linspace(0.5, 5.0, 40), (30, 1))
    with_tiny = depth.copy()
    with_tiny[0, 0] = DEPTH_MIN / 2
    with_zero = depth.copy()
    with_zero[0, 0] = 0.0
    assert np.array_equal(depth_to_colormap(with_tiny), depth_to_colormap(with_zero))


# ------------------------------------------------------------- the GLB writer


def _cloud(n=50, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)) + [1.0, 2.0, 3.0], rng.integers(0, 256, (n, 3), dtype=np.uint8)


def test_the_glb_holds_the_cloud_centred_and_the_centroid_is_returned(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    from surgical_core.viewer.glb import write_point_cloud_glb

    v, c = _cloud()
    centroid = write_point_cloud_glb(tmp_path / "f.glb", v, c)
    assert np.allclose(centroid, v.mean(axis=0))
    loaded = trimesh.load(tmp_path / "f.glb", file_type="glb", process=False)
    cloud = next(iter(loaded.geometry.values()))
    assert np.allclose(cloud.vertices, v - centroid)
    assert np.array_equal(np.asarray(cloud.colors)[:, :3], c)
    assert np.array_equal(write_point_cloud_glb(tmp_path / "g.glb", v, c, recenter=False), np.zeros(3))
    write_point_cloud_glb(tmp_path / "h.glb", v, c)
    assert (tmp_path / "f.glb").read_bytes() == (tmp_path / "h.glb").read_bytes(), "the same input, the same bytes"


@pytest.mark.parametrize("vertices, colors, match", [
    (np.zeros((5, 2)), np.zeros((5, 3), np.uint8), "vertices must be"),
    (np.zeros((0, 3)), np.zeros((0, 3), np.uint8), "empty"),
    (np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 2.0]]), np.zeros((2, 3), np.uint8), "finite"),
    (np.array([[0.0, 0.0, 0.0], [np.inf, 1.0, 2.0]]), np.zeros((2, 3), np.uint8), "finite"),
    (np.zeros((5, 3)), np.zeros((5, 2), np.uint8), "colors must be"),
    (np.zeros((5, 3)), np.zeros((5, 3), np.float32), "uint8"),
    (np.zeros((5, 3)), np.zeros((4, 3), np.uint8), "4 colors for 5 vertices"),
])
def test_a_cloud_the_viewer_would_show_wrong_is_refused(tmp_path, vertices, colors, match):
    pytest.importorskip("trimesh")
    from surgical_core.viewer.glb import write_point_cloud_glb

    with pytest.raises(ValueError, match=match):
        write_point_cloud_glb(tmp_path / "f.glb", vertices, colors)
    assert not (tmp_path / "f.glb").exists()


# ------------------------------------------------------------- the camera's axes


def test_an_identity_camera_looks_down_minus_z_with_y_up_in_gltf():
    axes = camera_axes_in_gltf(np.eye(3), np.zeros(3))
    assert axes == {"camera_pos_glb": [0.0, -0.0, -0.0], "camera_forward_glb": [0.0, -0.0, -1.0],
                    "camera_up_glb": [0.0, 1.0, -0.0]}


def test_the_camera_sits_and_aims_where_its_extrinsics_say():
    rng = np.random.default_rng(4)
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    t = rng.normal(size=3)
    axes = camera_axes_in_gltf(R, t)
    pos = np.array(axes["camera_pos_glb"]) * [1, -1, -1]
    fwd = np.array(axes["camera_forward_glb"]) * [1, -1, -1]
    up = np.array(axes["camera_up_glb"]) * [1, -1, -1]
    assert np.allclose(R @ pos + t, 0.0)
    # Orthogonality survives a transposed rotation; sending each axis back through R does not.
    assert np.allclose(R @ fwd, [0.0, 0.0, 1.0]), "forward is the camera's +Z"
    assert np.allclose(R @ up, [0.0, -1.0, 0.0]), "up is the camera's -Y"


@pytest.mark.parametrize("R, t", [(np.eye(4), np.zeros(3)), (np.eye(3), np.zeros((3, 1))), (np.eye(3), np.zeros(4))])
def test_extrinsics_of_another_shape_are_refused(R, t):
    with pytest.raises(ValueError, match="must have shape"):
        camera_axes_in_gltf(R, t)
