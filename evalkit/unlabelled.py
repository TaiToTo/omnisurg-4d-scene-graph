"""Measure `unlabelled_share`: the share of the scored regions' pixels that lie on unlabelled tissue.

Every view removes background, so no other key sees the pixels a region
has on unlabelled tissue. `unlabelled_share` counts those pixels.
`docs/evaluation.md` defines the key ("The spill no other key sees:
`unlabelled_share`") and says why background is removed ("Background").
The key reads no GT class, only where the GT is unlabelled. It is a
reference value, and no verdict reads it.

`docs/figures/unlabelled_share.png` shows this on a drawn scene, with the
numbers the module gives for it.
"""
from __future__ import annotations

import numpy as np


def unlabelled_share(regions: np.ndarray, scored: np.ndarray, background: np.ndarray) -> float | None:
    """Return the share of the scored regions' pixels that lie on unlabelled tissue.

    Args:
        regions: An (H, W) integer map of region ids, -1 for no region.
        scored: An (H, W) bool mask of the pixels the view scores.
        background: An (H, W) bool mask of the valid pixels whose GT is
            background. Disjoint from `scored`.

    Returns:
        The share in [0, 1], or None when no region has a scored pixel.
        There is then no region to measure, and the caller counts the
        frame.

    Raises:
        ValueError: A map is not an (H, W) array of the right dtype, the
            three do not share one shape, a region id is below -1, or
            `scored` and `background` overlap: a pixel is scored or
            unlabelled, never both, and an overlap means the masks were
            made from different frames or views.
    """
    regions, scored, background = np.asarray(regions), np.asarray(scored), np.asarray(background)
    if regions.ndim != 2 or not np.issubdtype(regions.dtype, np.integer):
        raise ValueError(f"a region map is an (H, W) integer array, got {regions.dtype} of shape {regions.shape}")
    for name, m in (("scored", scored), ("background", background)):
        if m.dtype != np.bool_ or m.shape != regions.shape:
            raise ValueError(f"`{name}` must be a bool array of shape {regions.shape}, got {m.dtype} {m.shape}")
    if regions.min(initial=0) < -1:
        raise ValueError("a region map holds region ids >= 0 and -1 for no region")
    if (scored & background).any():
        raise ValueError(
            f"{int((scored & background).sum())} pixels are both scored and background; "
            f"the two masks must come from one frame and one view"
        )
    ids = np.unique(regions[scored & (regions >= 0)])
    if ids.size == 0:
        return None
    member = np.isin(regions, ids)
    on_background = int((member & background).sum())
    counted = int((member & (scored | background)).sum())
    return on_background / counted
