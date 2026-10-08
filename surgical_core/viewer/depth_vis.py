"""A depth map as an RGB image, near warm and far cool, the same for every depth source.

Disparity, normalised per frame between percentiles: DA3 and Pi3X are both
scale-invariant, so only the shape within a frame carries meaning, and one
shared colouring lets a viewer set two sources side by side. Needs matplotlib,
the `render` extra.
"""

import numpy as np

from surgical_core.geometry.valid import valid_depth_mask


def depth_to_colormap(depth: np.ndarray, cmap: str = "Spectral", percentile: float = 2.0) -> np.ndarray:
    """Colour one depth map.

    Args:
        depth: (H, W) depth. A pixel `valid_depth_mask` refuses goes to the far end of the ramp and stays out of the
            percentiles; `depth > 0` alone would admit `+inf`, whose disparity of 0 drags the range down.
        cmap: a matplotlib colormap name.
        percentile: clipped from each end of the disparity range, so that a few outliers cannot flatten the rest.

    Returns:
        (H, W, 3) uint8 RGB.
    """
    # Imported here so that the module imports without the extra. Only the colormap tables are used, no backend.
    import matplotlib

    disp = np.zeros_like(depth)
    valid = valid_depth_mask(depth)
    disp[valid] = 1.0 / depth[valid]
    if valid.sum() > 10:
        lo = np.percentile(disp[valid], percentile)
        hi = np.percentile(disp[valid], 100 - percentile)
    else:
        lo, hi = 0.0, 1.0
    if hi == lo:
        hi = lo + 1e-6
    normed = 1.0 - ((disp - lo) / (hi - lo)).clip(0, 1)
    rgb = matplotlib.colormaps[cmap](normed, bytes=False)[:, :, :3]
    return (rgb * 255).astype(np.uint8)
