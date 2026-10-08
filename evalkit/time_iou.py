"""`time_IoU`: a region id's IoU with itself in the next frame, pooled over a clip.

It is a reference value only and never gets a star. It uses no GT, and the
workbench measured three faults in it: it rises as regions get coarser, and
one region covering the whole frame scores 1.0; an id that disappears costs
nothing; and for a condition segmented frame by frame, whose ids do not
carry over, it means nothing. It is kept because the pilot evaluator wrote
it, and the evaluator computes no other measure over time
(`docs/evaluation.md`, "Consistency over time: reference values only").

Unlike the other metrics it is one number per clip: the IoUs of every
(id, frame pair) are pooled over all tracked frames, with or without GT, and
averaged. The frames come in time order, which the caller settles from the
timestamps: in 11 of the 27 CholecSeg8k clips the frame numbers do not
follow time.

Pilot mode orders the frames exactly as the pilot evaluator did, by
`sorted()` of the `label_*.npy` file names. That sorts the names as text,
so `label_10` comes before `label_2` unless the numbers are zero-padded.
This metric reads no GT and no view, so the entry point computes it once
per clip.

`docs/figures/time_iou.png` shows this on a drawn scene, with the numbers the module
gives for it.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def time_iou(regions: Sequence[np.ndarray], valid: Sequence[np.ndarray]) -> float | None:
    """The mean IoU of each region id with itself one frame later.

    For every pair of consecutive frames, every id present in both frames
    contributes the IoU of its two masks over the pixels valid in both
    frames; a pair with an empty union is skipped, since there is nothing to
    compare. "Present" is judged on the whole frame, valid or not, as the
    pilot evaluator judged it. So an id that lay on shared valid pixels and
    in the next frame survives only outside them, on invalid pixels, still
    counts, with an IoU of 0, while one gone from the whole frame costs
    nothing: whether a disappearance costs anything depends on where the
    depth is valid. It is a reference value, so the rule is kept and named
    rather than fixed.

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
    regions = [np.asarray(r) for r in regions]
    valid = [np.asarray(v) for v in valid]
    for i, (r, v) in enumerate(zip(regions, valid)):
        if r.ndim != 2 or not np.issubdtype(r.dtype, np.integer):
            raise ValueError(f"frame {i}: a region map is an (H, W) integer array, got {r.dtype} of shape {r.shape}")
        if v.dtype != np.bool_ or v.shape != r.shape:
            raise ValueError(f"frame {i}: `valid` must be a bool array of shape {r.shape}, got {v.dtype} {v.shape}")
        if r.shape != regions[0].shape:
            raise ValueError(f"frame {i} has shape {r.shape}, frame 0 has {regions[0].shape}")
    values = []
    for t in range(len(regions) - 1):
        a, b = regions[t], regions[t + 1]
        both = valid[t] & valid[t + 1]
        ids = np.intersect1d(np.unique(a), np.unique(b), assume_unique=True)
        # A negative id is no region, so it has no IoU with itself.
        for r in ids[ids >= 0].tolist():
            in_a, in_b = (a == r) & both, (b == r) & both
            union = int((in_a | in_b).sum())
            if union:
                values.append(int((in_a & in_b).sum()) / union)
    return float(np.mean(values)) if values else None
