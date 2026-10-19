"""Fill the pixels of a GT mask's `filled` classes from the nearest region, and mark which pixels were filled.

A `filled` class is no class: a line or gap the annotation leaves between
two regions (`docs/evaluation.md`, "Class types"). `fill_from_nearest`
gives each of its pixels the id of the nearest pixel of another class.
Nearness counts steps between 4-neighbours. Of several ids equally near,
the pixel takes the smallest, so a line 1 px wide takes the smaller of the
ids on its two sides, whole. The reader fills a mask as stored, before it
is resized, so that the fill depends on the annotation alone, not on the
depth map's shape.
"""
from __future__ import annotations

import numpy as np


def fill_from_nearest(mask_ids: np.ndarray, filled_ids: frozenset[int]) -> tuple[np.ndarray, np.ndarray]:
    """Give each pixel of a `filled` id the id of the nearest pixel of another id.

    The fill grows one step at a time from the pixels of other ids. Each
    step gives every pixel still to fill that has a 4-neighbour already
    filled or never to fill the smallest of those neighbours' ids. A pixel
    filled at step d is d steps from its nearest pixel of another id, and
    the smallest of its neighbours' ids is the smallest id among those
    equally near, since each neighbour holds the smallest of its own.

    Args:
        mask_ids: An (H, W) integer map of the ids a GT mask holds, as
            `ClassTable.mask_ids` returns them.
        filled_ids: The ids to fill, `ClassTable.filled_mask_ids`.

    Returns:
        The filled map, and an (H, W) bool array, True on the pixels filled.

    Raises:
        ValueError: `mask_ids` is not an (H, W) integer map, or every one of
            its pixels is of a `filled` id, so that nothing is near.
    """
    mask_ids = np.asarray(mask_ids)
    if mask_ids.ndim != 2 or not np.issubdtype(mask_ids.dtype, np.integer):
        raise ValueError(f"a mask id map is an (H, W) integer array, got {mask_ids.dtype} of shape {mask_ids.shape}")
    todo = np.isin(mask_ids, sorted(filled_ids))
    was_filled = todo.copy()
    if not todo.any():
        return mask_ids.copy(), was_filled
    if todo.all():
        raise ValueError(f"every pixel holds a filled id {sorted(filled_ids)}, so no region is near to fill it from")
    out = mask_ids.astype(np.int64)
    none = np.iinfo(np.int64).max
    while todo.any():
        known = np.where(todo, none, out)
        nearest = np.full(out.shape, none, dtype=np.int64)
        np.minimum(nearest[1:, :], known[:-1, :], out=nearest[1:, :])
        np.minimum(nearest[:-1, :], known[1:, :], out=nearest[:-1, :])
        np.minimum(nearest[:, 1:], known[:, :-1], out=nearest[:, 1:])
        np.minimum(nearest[:, :-1], known[:, 1:], out=nearest[:, :-1])
        step = todo & (nearest != none)
        out[step] = nearest[step]
        todo &= ~step
    return out.astype(mask_ids.dtype), was_filled
