"""A coloured point cloud written as a GLB file, the format the viewer loads per frame.

Points are in glTF's frame (`surgical_core.geometry.camera.world_to_gltf`).
Needs trimesh, the `render` extra.
"""

from pathlib import Path

import numpy as np


def write_point_cloud_glb(path: str | Path, vertices_gltf: np.ndarray, colors_uint8: np.ndarray,
                          recenter: bool = True) -> np.ndarray:
    """Write a point cloud, centred on its centroid when `recenter`, and return the centroid taken off.

    Args:
        path: the `.glb` to write.
        vertices_gltf: (N, 3) points.
        colors_uint8: (N, 3) or (N, 4) uint8 colours, one per point.
        recenter: subtract the centroid, which the manifest records as `glb_centroid`.

    Returns:
        The (3,) centroid subtracted, zeros without `recenter`.

    Raises:
        ValueError: points not (N, 3), none or not finite, colours of another shape, count or dtype. Each of
            these writes a file that opens and shows the wrong cloud: colours shifted, all black, or nothing.
    """
    import trimesh

    v = np.asarray(vertices_gltf, dtype=np.float64)
    c = np.asarray(colors_uint8)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must be (N, 3), got {v.shape}")
    if len(v) == 0:
        raise ValueError("no vertices: the point cloud is empty")
    if not np.isfinite(v).all():
        raise ValueError("vertices must be finite: one NaN or infinity poisons the centroid, and with it every centred point")
    if c.ndim != 2 or c.shape[1] not in (3, 4):
        raise ValueError(f"colors must be (N, 3) or (N, 4), got {c.shape}")
    if c.dtype != np.uint8:
        raise ValueError(f"colors must be uint8, 0 to 255, got {c.dtype}")
    if c.shape[0] != v.shape[0]:
        raise ValueError(f"{c.shape[0]} colors for {v.shape[0]} vertices")
    centroid = v.mean(axis=0) if recenter else np.zeros(3)
    trimesh.Scene([trimesh.PointCloud(vertices=v - centroid, colors=c)]).export(str(path), file_type="glb")
    return centroid
