"""`unlabelled_share`: how much of the scored regions lies on unlabelled tissue.

Background is removed from every view, so a region that spills from a
labelled organ into unlabelled tissue is scored as if it stopped at the
organ's edge: no key sees the spill. That is the convention of panoptic
quality, where a segment's void pixels leave the union and a segment lying
mostly on void is no false positive, and it is kept because ATLAS-120k's
unlabelled pixels include organs painted only in part, where a region that
follows the organ past the paint is right, not wrong.

What the scores cannot see is counted here instead. Over the regions that
have at least one scored pixel, `unlabelled_share` is the share of their
pixels that lie on background, among their pixels on scored or background
pixels. A region lying on background alone is not among them, as it is no
object. Pixels the view removes for another reason (ignored, invalid depth,
a class the view leaves out) are in neither the numerator nor the
denominator: a region's pixels under a tool say nothing about spill.

It reads no GT class, only where the GT is unlabelled, and it is a reference
value: it never gets a star. It is reported beside a comparison whose two
conditions differ in it by much, as the pilot measurements did for the share
of pixels left without a region.

`docs/figures/unlabelled_share.png` shows this on a drawn scene, with the numbers the module
gives for it.
"""
from __future__ import annotations

import numpy as np


def unlabelled_share(regions: np.ndarray, scored: np.ndarray, background: np.ndarray) -> float | None:
    """The share of the scored regions' pixels that lie on unlabelled tissue.

    Args:
        regions: An (H, W) integer map of region ids, -1 for no region.
        scored: An (H, W) bool mask of the pixels the view scores.
        background: An (H, W) bool mask of the valid pixels whose GT is
            background. Disjoint from `scored`.

    Returns:
        The share in [0, 1], or None when no region has a scored pixel: then
        there is nothing whose spill could be measured, and the frame is
        counted.

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
