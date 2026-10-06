"""A camera's position and axes in glTF's frame, as the manifest's `camera_*_glb` keys the viewer places it by.

Every depth source writes them through this one function, so that two sources
side by side differ in their geometry and not in how the camera was placed.
"""

import numpy as np

from surgical_core.geometry.camera import GLTF_FLIP


def camera_axes_in_gltf(R: np.ndarray, t: np.ndarray) -> dict:
    """The camera's position, forward and up in glTF's frame, from world-to-camera extrinsics.

    The position is absolute, not relative to a point cloud's centroid: the viewer subtracts the first frame's
    centroid itself.

    Args:
        R: (3, 3) rotation.
        t: (3,) translation.

    Returns:
        `camera_pos_glb`, `camera_forward_glb` and `camera_up_glb`, each a list of three floats.
    """
    # The camera sits where R @ p + t = 0; it looks along its +Z and its up is its -Y.
    cam_world = -R.T @ t
    fwd_world = R.T @ np.array([0.0, 0.0, 1.0])
    up_world = R.T @ np.array([0.0, -1.0, 0.0])
    return {
        "camera_pos_glb": (cam_world * GLTF_FLIP).tolist(),
        "camera_forward_glb": (fwd_world * GLTF_FLIP).tolist(),
        "camera_up_glb": (up_world * GLTF_FLIP).tolist(),
    }
