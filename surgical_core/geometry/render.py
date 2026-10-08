"""Make the images the segmenter is prompted with, one per input mode.

A segmenter input is the RGB frame, the colormapped depth or the camera
normals, alone or with the geometric edges burnt in as dark lines
(`rgb_edge`, `normal_edge`). `sam_input_image` builds any mode in
`SAM_INPUT_MODES`.

Needs matplotlib (the depth colormap): the `render` extra. The evaluator
never imports this module.
"""

import matplotlib
import numpy as np

from surgical_core.geometry.normals import burn_geom_edge, normal_edge_map, normal_map
from surgical_core.geometry.valid import valid_depth_mask


def global_depth01(depth_all, lo_p=1, hi_p=99):
    """Normalise a clip's depth to [0, 1] with one scale for every frame, so
    that the colormapped depth is stable over time."""
    # Invalid pixels (depth 0) map below 0 and clip to 0, the colormap's
    # nearest colour, so in the segmenter input they look like a near valid
    # surface (NaN goes to black). The valid mask removes them downstream, so
    # the harm is small; blacken them here if that changes.
    finite = depth_all[valid_depth_mask(depth_all)]
    lo, hi = np.percentile(finite, lo_p), np.percentile(finite, hi_p)
    # The 1e-6 keeps the division finite on a clip of one depth. It is not a depth test.
    return np.clip((depth_all - lo) / (hi - lo + 1e-6), 0, 1)


def depth_to_colormapped(gray01, cmap="viridis"):
    """Normalised depth (H, W) to RGB uint8; the segmenter takes 3 channels."""
    colored = matplotlib.colormaps[cmap](gray01)
    return (colored[:, :, :3] * 255).astype(np.uint8)


# This tuple lists, in one place, the modes `sam_input_image` accepts: the inputs the paper's conditions ran.
# Pass it to argparse as the choices: when callers each kept their own literal list, one of them was extended and
# the other was not, and a whole propagation stage failed on every clip after the per-frame stage had passed.
SAM_INPUT_MODES = ("depth", "normal", "normal_edge", "rgb", "rgb_edge")


def uses_geom_edge(mode):
    """Return whether `sam_input_image(mode, ...)` burns geometric edges in.

    For the provenance record: `edge_ring_masked` records the `mask_ring`
    the edges were made with, so writing True or False for an input that
    burns no edges (`rgb`, `normal`, `depth`) would read as "made with that
    setting". For those inputs the record writes `None`, and this is how it
    tells.

    Args:
        mode: one of `SAM_INPUT_MODES`.

    Raises:
        ValueError: `mode` is not one of `SAM_INPUT_MODES`.
    """
    if mode not in SAM_INPUT_MODES:
        raise ValueError(f"unknown input mode {mode!r}; one of {SAM_INPUT_MODES}")
    return mode in ("normal_edge", "rgb_edge")


def sam_input_image(mode, depth, K, gray01, rgb, edge_gain=0.85, smooth=True, mask_ring=True):
    """Return the 3-channel image the segmenter is prompted with, per input mode.

    Args:
        mode: `"rgb"`, `"depth"` (colormapped depth), `"normal"`,
            `"normal_edge"` (normals with the geometric edges burnt in) or
            `"rgb_edge"` (the same edges burnt into RGB). Anything else
            raises: there is no default mode.
        depth: (H, W) depth.
        K: (3, 3) intrinsics.
        gray01: (H, W) depth normalised by `global_depth01`, for `depth`.
        rgb: (H, W, 3) uint8.
        edge_gain: how dark the edge lines are, 0 to 1.
        smooth: bilateral-filter the normal image in `normal_edge`.
        mask_ring: zero the ring of the burnt-in edges along the image border
            and around invalid depth (`geom_edge_map`), in the `_edge` modes.

    Returns:
        (H, W, 3) uint8.

    Raises:
        ValueError: `mode` is not one of `SAM_INPUT_MODES`.
    """
    if mode == "rgb":
        return rgb
    if mode == "depth":
        return depth_to_colormapped(gray01)
    if mode == "normal":
        return normal_map(depth, K)
    if mode == "normal_edge":
        return normal_edge_map(depth, K, edge_gain=edge_gain, smooth=smooth, mask_ring=mask_ring)
    if mode == "rgb_edge":
        return burn_geom_edge(rgb, depth, K, edge_gain=edge_gain, mask_ring=mask_ring)
    # Matched by name like every other mode, never as a default: a misspelt
    # mode in a config would otherwise run the depth condition under another
    # name, and nothing downstream could tell.
    raise ValueError(f"unknown input mode {mode!r}; one of {SAM_INPUT_MODES}")
