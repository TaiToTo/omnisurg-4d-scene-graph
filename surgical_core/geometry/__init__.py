"""Geometry shared by the pipeline and the evaluation toolkit.

Everything here assumes depth, intrinsics and extrinsics that map world to
camera (w2c). If the extrinsics of a dataset map camera to world, the
back-projection formula has to be switched, because feeding c2w through the
w2c formula raises nothing: the point cloud simply comes out on the far side
of the origin.

The pipeline builds its segmenter inputs from these normals and edge maps,
and the viewer draws its point clouds with these transforms, so the code
lives in neither and is imported by both. The toolkit imports none of it; it
keeps its own `DEPTH_MIN`, and a test refuses the two drifting apart.

The package is split by dependency:

- `normals`: normals from depth, the geometric edge maps and the edge-burnt
  normal image. numpy and OpenCV only. This is what `render` builds the
  segmenter inputs from.
- `camera`: pixels to camera, world and the viewer's glTF space, the
  transforms the depth stages write their point clouds with. numpy only.
- `project`: back-projection and projection between frames, on `camera`'s
  transforms. numpy only.
- `valid`: which depth values count, `DEPTH_MIN` and `valid_depth_mask`.
  numpy only.
- `render`: the images the segmenter is prompted with (colormapped depth,
  relighting, Retinex, the input-mode table). Needs scipy and matplotlib,
  the `render` extra, so it is not imported here; import it by name.

The functions of `normals`, `project` and `valid`, and `DEPTH_MIN`, are
re-exported so that `from surgical_core import geometry` keeps working for
them. `DEPTH_MIN` is never set, so a copy of it cannot drift. The flags
`EDGE_MASK_RING` and `EDGE_RING_PX` are not: an import copies a value, so a
copy here could be set to False while `geom_edge_map` kept reading True from
`normals`, and the provenance record would then say the ring was left in
when it was not. They live in `surgical_core.geometry.normals` only.
"""

from surgical_core.geometry.normals import (  # noqa: F401
    burn_geom_edge, camera_normals, edge_reliable_mask, geom_edge_map, normal_edge_map,
    normal_map)
from surgical_core.geometry.project import (  # noqa: F401
    backproject, project_world_to_frame)
from surgical_core.geometry.valid import DEPTH_MIN, valid_depth_mask  # noqa: F401
