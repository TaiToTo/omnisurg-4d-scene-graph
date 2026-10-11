"""Back-project depth into world points, and project world points into a frame.

Extrinsics are world-to-camera (w2c) throughout. numpy only.
"""

import numpy as np

from surgical_core.geometry.camera import _check_shape, backproject_depth, cam_to_world
from surgical_core.geometry.valid import valid_depth_mask


def backproject(depth, K, ext_w2c):
    """Depth to world points.

    For camera-to-world extrinsics the formula would be `Pw = (R @ Pc.T + t).T`.

    Args:
        depth: (H, W) depth. 0 and NaN are invalid.
        K: (3, 3) intrinsics.
        ext_w2c: (3, 4) world-to-camera extrinsics.

    Returns:
        `(Pw, m, ys, xs)`: the (N, 3) world points of the valid pixels, the
        (H, W) valid mask, and the row and column of each point.

    Raises:
        ValueError: `K` is not (3, 3), or `ext_w2c` is not (3, 4). The slices
            below would read a wider matrix without a word, dropping its last
            columns, and give a (3, 3) one an IndexError instead of an answer.
    """
    _check_shape("ext_w2c", ext_w2c, (3, 4))
    H, W = depth.shape
    ys, xs = np.mgrid[0:H, 0:W]
    m = valid_depth_mask(depth)
    R, t = ext_w2c[:, :3], ext_w2c[:, 3]
    with np.errstate(all="ignore"):
        Pw = cam_to_world(backproject_depth(depth, K)[m.reshape(-1)], R, t)
    return Pw, m, ys[m], xs[m]


def project_world_to_frame(Pw, K, ext_w2c):
    """Project world points to pixel coordinates `(u, v)` and camera depth `Z`.

    Args:
        Pw: (N, 3) world points.
        K: (3, 3) intrinsics.
        ext_w2c: (3, 4) world-to-camera extrinsics.

    Raises:
        ValueError: `Pw` is not (N, 3), `K` is not (3, 3), or `ext_w2c` is not
            (3, 4).
    """
    # The slices below would read a (4, 4) matrix too, and a (3,) point would broadcast into three points.
    # The number of points is free, so the expected shape takes it from `Pw`.
    _check_shape("Pw", Pw, (*np.shape(Pw)[:1], 3))
    _check_shape("K", K, (3, 3))
    _check_shape("ext_w2c", ext_w2c, (3, 4))
    R, t = ext_w2c[:, :3], ext_w2c[:, 3]
    with np.errstate(all="ignore"):
        Pc = (R @ Pw.T + t[:, None]).T
    Z = Pc[:, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = Pc[:, 0] * fx / Z + cx
    v = Pc[:, 1] * fy / Z + cy
    return u, v, Z
