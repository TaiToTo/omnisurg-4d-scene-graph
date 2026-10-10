"""Write a coloured point cloud as a GLB file, the format the viewer loads per frame, and read its points back.

Points are in glTF's frame (`surgical_core.geometry.camera.world_to_gltf`).
Writing needs trimesh, the `render` extra; reading needs numpy only.
"""

import json
import struct
from pathlib import Path

import numpy as np

_GLB_MAGIC = b"glTF"
_JSON_CHUNK = b"JSON"
_BIN_CHUNK = b"BIN\x00"
_FLOAT = 5126


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


def read_point_cloud_glb(path: str | Path) -> np.ndarray:
    """Read the points of a point-cloud GLB as (N, 3) float32, in the order the file stores them.

    The file must hold one mesh of one primitive whose `POSITION` is tightly packed float32, as
    `write_point_cloud_glb` writes it. The viewer indexes per-point labels by this order.

    Raises:
        ValueError: the file is not a GLB of that shape. Reading another layout as this one would
            return points in another order, and every label would land on the wrong point.
    """
    data = Path(path).read_bytes()
    if len(data) < 20 or data[:4] != _GLB_MAGIC:
        raise ValueError(f"{path}: not a GLB file")
    version, length = struct.unpack_from("<II", data, 4)
    if version != 2 or length != len(data):
        raise ValueError(f"{path}: GLB version {version} of {length} bytes in a file of {len(data)}")
    json_len, json_type = struct.unpack_from("<I4s", data, 12)
    if json_type != _JSON_CHUNK:
        raise ValueError(f"{path}: the first chunk is not JSON")
    gltf = json.loads(data[20:20 + json_len])
    bin_at = 20 + json_len
    bin_len, bin_type = struct.unpack_from("<I4s", data, bin_at)
    if bin_type != _BIN_CHUNK:
        raise ValueError(f"{path}: the second chunk is not binary")
    blob = data[bin_at + 8:bin_at + 8 + bin_len]

    meshes = gltf.get("meshes", [])
    if len(meshes) != 1 or len(meshes[0].get("primitives", [])) != 1:
        raise ValueError(f"{path}: expected one mesh of one primitive")
    accessor = gltf["accessors"][meshes[0]["primitives"][0]["attributes"]["POSITION"]]
    view = gltf["bufferViews"][accessor["bufferView"]]
    if accessor["componentType"] != _FLOAT or accessor["type"] != "VEC3" or view.get("byteStride", 12) != 12:
        raise ValueError(f"{path}: POSITION is not tightly packed float32 VEC3")
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    count = accessor["count"]
    if start + 12 * count > len(blob):
        raise ValueError(f"{path}: POSITION runs past the end of the binary chunk")
    return np.frombuffer(blob, dtype="<f4", count=3 * count, offset=start).reshape(count, 3).copy()
