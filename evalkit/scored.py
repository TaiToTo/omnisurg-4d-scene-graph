"""The scored pixels of a frame: what a view keeps, and a count of what it removes.

Every metric is computed over the scored pixels of a view, the same way: a
pixel is scored when its depth is valid and its GT class is one the view
includes. The rest is removed from the GT and the prediction alike, so a
region lying there is neither an object nor a false positive, and no
boundary is marked against it. This module computes that mask once per
frame and view; the metric modules take it as `scored`.

Removed pixels are counted, by the first reason that removes them: invalid
depth, then an `ignored` class, then `background`, then a class the view
leaves out. The counts are disjoint and sum to the frame, so a JSON reader
can see where every pixel went.

An `excluded` class is a marker that takes the whole frame out. It is
checked on the mask's own ids, before any mapping, because ATLAS-120k's
benchmark set maps the marker to background; a frame that carries it is
skipped and counted by the caller, and never reaches `scored_pixels`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evalkit.classes import VIEWS, ClassTable, ClassType


@dataclass(frozen=True)
class PixelCounts:
    """Where each pixel of a frame went, by the first reason that removed it.

    Attributes:
        invalid_depth: Depth not finite or not above the threshold.
        ignored: An `ignored` class: outside the view, or not a class.
        background: Labelled as nothing.
        left_out: A scored type the view leaves out (a tool in the `tissue`
            view, blood in the `geometric` view).
        scored: What the metrics see.
    """

    invalid_depth: int
    ignored: int
    background: int
    left_out: int
    scored: int

    @property
    def total(self) -> int:
        return self.invalid_depth + self.ignored + self.background + self.left_out + self.scored


@dataclass(frozen=True)
class Scored:
    """The scored pixels of one frame in one view.

    Attributes:
        mask: An (H, W) bool array, True on the pixels the metrics score.
        counts: Where every pixel of the frame went.
        view: The view's name.
    """

    mask: np.ndarray
    counts: PixelCounts
    view: str


def frame_is_excluded(mask_ids: np.ndarray, table: ClassTable) -> bool:
    """Whether the frame carries the marker that takes it out of evaluation.

    Checked on the mask ids as read, not on the class set's ids, so that the
    benchmark set, which maps the marker to background, still sees it.
    """
    mask_ids = np.asarray(mask_ids)
    if mask_ids.ndim != 2 or not np.issubdtype(mask_ids.dtype, np.integer):
        raise ValueError(f"a mask id map is an (H, W) integer array, got {mask_ids.dtype} of shape {mask_ids.shape}")
    present = np.unique(mask_ids).tolist()
    for v in present:
        table.class_of(v)
    return any(v in table.excluded_mask_ids for v in present)


def scored_pixels(classes: np.ndarray, table: ClassTable, view: str, valid: np.ndarray) -> Scored:
    """The pixels a view scores, and a count of the ones it does not.

    Args:
        classes: An (H, W) integer map of class ids of `table`'s class set,
            as `ClassTable.classes_of` returns them.
        table: The dataset's class table.
        view: One of `VIEWS`.
        valid: An (H, W) bool mask of the pixels with valid depth.

    Raises:
        KeyError: A class id the table does not have, or a view name
            `VIEWS` does not.
        ValueError: A class of type `excluded` is present: the frame should
            have been skipped before any pixel of it was scored.
    """
    classes, valid = np.asarray(classes), np.asarray(valid)
    if classes.ndim != 2 or not np.issubdtype(classes.dtype, np.integer):
        raise ValueError(f"a class map is an (H, W) integer array, got {classes.dtype} of shape {classes.shape}")
    if valid.dtype != np.bool_ or valid.shape != classes.shape:
        raise ValueError(f"`valid` must be a bool array of shape {classes.shape}, got {valid.dtype} {valid.shape}")
    if view not in VIEWS:
        raise KeyError(f"unknown view {view!r}; views are {sorted(VIEWS)}")

    # Every class present is typed through the table, so an id it does not
    # know raises here rather than being scored as something.
    types = {int(c): table.type_of(int(c)) for c in np.unique(classes).tolist()}
    excluded = sorted(c for c, t in types.items() if t is ClassType.EXCLUDED)
    if excluded:
        raise ValueError(
            f"{table.dataset}: class {excluded} marks the frame as excluded; it is skipped "
            f"as a whole, not scored pixel by pixel"
        )

    def of_type(*wanted: ClassType) -> np.ndarray:
        ids = [c for c, t in types.items() if t in wanted]
        return np.isin(classes, ids) if ids else np.zeros(classes.shape, dtype=bool)

    ignored = valid & of_type(ClassType.IGNORED)
    background = valid & of_type(ClassType.BACKGROUND)
    in_view = valid & of_type(*VIEWS[view])
    left_out = valid & ~ignored & ~background & ~in_view
    counts = PixelCounts(
        invalid_depth=int((~valid).sum()),
        ignored=int(ignored.sum()),
        background=int(background.sum()),
        left_out=int(left_out.sum()),
        scored=int(in_view.sum()),
    )
    return Scored(mask=in_view, counts=counts, view=view)
