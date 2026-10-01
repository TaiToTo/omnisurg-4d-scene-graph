"""`time_IoU`: a region id's IoU with itself in the next frame, pooled over a clip.

It is a reference value only and never gets a star. It uses no GT, and the
workbench measured three faults in it: it rises as regions get coarser, and
one region covering the whole frame scores 1.0; an id that disappears costs
nothing; and for a condition segmented frame by frame, whose ids do not
carry over, it means nothing. It is kept because the pilot evaluator wrote
it, and the measure of consistency over time that is to replace it is not
decided yet (`docs/evaluation.md`, "Consistency over time").

Unlike the other metrics it is one number per clip: the IoUs of every
(id, frame pair) are pooled over all tracked frames, with or without GT, and
averaged. The frames come in time order, which the caller settles from the
timestamps: in 11 of the 27 CholecSeg8k clips the frame numbers do not
follow time. Pilot mode orders them by file name instead, and that too is
the caller's.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def time_iou(regions: Sequence[np.ndarray], valid: Sequence[np.ndarray]) -> float | None:
    """The mean IoU of each region id with itself one frame later.

    For every pair of consecutive frames, every id present in both (anywhere
    in the frame) contributes the IoU of its two masks over the pixels valid
    in both frames; a pair with an empty union is skipped, since there is
    nothing to compare. The pilot evaluator pooled the same way.

    Args:
        regions: The clip's region maps, (H, W) integer arrays in time
            order, -1 for no region.
        valid: The pixels with valid depth in each frame, (H, W) bool.

    Returns:
        The mean over every (id, frame pair), or None when nothing was
        pooled: a clip of one frame, or one whose ids never carry over.
        Pilot mode writes 0 there, and that is its own rule.
    """
    if len(regions) != len(valid):
        raise ValueError(f"{len(regions)} region maps but {len(valid)} valid masks")
    for i, (r, v) in enumerate(zip(regions, valid)):
        r, v = np.asarray(r), np.asarray(v)
        if r.ndim != 2 or not np.issubdtype(r.dtype, np.integer):
            raise ValueError(f"frame {i}: a region map is an (H, W) integer array, got {r.dtype} of shape {r.shape}")
        if v.dtype != np.bool_ or v.shape != r.shape:
            raise ValueError(f"frame {i}: `valid` must be a bool array of shape {r.shape}, got {v.dtype} {v.shape}")
        if i and r.shape != np.asarray(regions[0]).shape:
            raise ValueError(f"frame {i} has shape {r.shape}, frame 0 has {np.asarray(regions[0]).shape}")
    values = []
    for t in range(len(regions) - 1):
        a, b = np.asarray(regions[t]), np.asarray(regions[t + 1])
        both = np.asarray(valid[t]) & np.asarray(valid[t + 1])
        ids = np.intersect1d(np.unique(a), np.unique(b))
        for r in ids[ids >= 0].tolist():
            in_a, in_b = (a == r) & both, (b == r) & both
            union = int((in_a | in_b).sum())
            if union:
                values.append(int((in_a & in_b).sum()) / union)
    return float(np.mean(values)) if values else None
