"""From a depth map's pixels to camera space, to world space, to the viewer's glTF space.

Extrinsics map world to camera, split into a (3, 3) rotation `R` and a (3,)
translation `t`. glTF is +Y up and the world is +Y down and +Z forward, so
`world_to_gltf` flips Y and Z (`GLTF_FLIP`); the flip is its own inverse.
Points are rows: arrays end in a length-3 axis. numpy only.
"""

import numpy as np

# The axis flip from the world's frame into glTF's.
GLTF_FLIP = np.array([1, -1, -1])


def _check_matrix(name: str, a: np.ndarray, shape: tuple[int, ...]) -> None:
    if np.shape(a) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {np.shape(a)}")


def backproject_depth(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Camera-space points of every pixel, in the order of `depth.reshape(-1)`, so that both mask alike.

    Nothing is filtered: a pixel with no depth gives a point the caller masks out.

    Args:
        depth: (H, W) depth.
        K: (3, 3) intrinsics.

    Returns:
        (H * W, 3) points.

    Raises:
        ValueError: `K` is not (3, 3); a (4, 4) one would be read as fx and cx without a word.
    """
    _check_matrix("K", K, (3, 3))
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


def cam_to_world(pts_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World points of camera-space points, `R.T @ (p - t)` for each, as rows: `(p - t) @ R`.

    Args:
        pts_cam: (..., 3) points.
        R: (3, 3) rotation of the world-to-camera extrinsics.
        t: (3,) translation of the world-to-camera extrinsics.

    Raises:
        ValueError: `R` or `t` has another shape.
    """
    _check_matrix("R", R, (3, 3))
    _check_matrix("t", t, (3,))
    return (np.asarray(pts_cam) - t) @ R


def world_to_gltf(pts_world: np.ndarray) -> np.ndarray:
    """Points in glTF's frame; given points in glTF's frame, the world's again."""
    return np.asarray(pts_world) * GLTF_FLIP
