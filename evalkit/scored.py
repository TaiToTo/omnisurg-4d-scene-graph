"""The scored pixels of a frame: what a view keeps, and a count of what it removes.

Every metric is computed over the scored pixels of a view, the same way: a
pixel is scored when its depth is valid and its GT class is one the view
includes. The rest is removed from the GT and the prediction alike, so a
region lying there is neither an object nor a false positive, and no
boundary is marked against it. This module computes that mask once per
frame and view, from the mask ids as `ClassTable.mask_ids` read them, and
the class map that goes with it; the metric modules take them as `scored`
and the GT.

Removed pixels are counted, by the first reason that removes them: invalid
depth, then an `ignored` class, then `background`, then a class the view
leaves out. The counts are disjoint and sum to the frame, so a JSON reader
can see where every pixel went. `docs/evaluation.md` fixes the order.

An `excluded` class is a marker that takes the whole frame out. It is
checked on the mask's own ids, before the mapping to a class set, because
ATLAS-120k's benchmark set maps the marker to background and would hide it.
The caller asks `frame_is_excluded` first and skips and counts the frame;
`scored_pixels` runs the same check on the same ids and refuses the frame,
in every class set, so a caller that forgot cannot score it.

The per-frame driver that skips and counts an excluded frame is not
written yet, so `frame_is_excluded` has no caller but the tests. It stays
public for that driver.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evalkit.classes import ClassTable, ClassType


@dataclass(frozen=True)
class PixelCounts:
    """Where each pixel of a frame went, by the first reason that removed it.

    Attributes:
        invalid_depth: Depth not finite or not above the threshold of
            `docs/evaluation.md`, "Valid pixels".
        ignored: An `ignored` class: outside the view, or not a class.
        background: Labelled as nothing.
        left_out: A scored type the view leaves out (a tool in the `tissue`
            view, blood in the `geometric` view).
        scored: What the metrics see.
    """

    # `valid` is computed by the caller. The function that takes a depth map
    # to it does not exist yet; when it does, it is the one place the
    # threshold lives, and this docstring names it.
    invalid_depth: int
    ignored: int
    background: int
    left_out: int
    scored: int

    @property
    def total(self) -> int:
        """Every pixel of the frame: the counts are disjoint, so their sum."""
        return self.invalid_depth + self.ignored + self.background + self.left_out + self.scored


@dataclass(frozen=True)
class Scored:
    """The scored pixels of one frame in one view, with the GT they are scored against.

    Attributes:
        mask: An (H, W) bool array, True on the pixels the metrics score.
        classes: The (H, W) int32 GT class map, in the table's class set, on
            every pixel scored or not; the metrics read it under `mask`.
        counts: Where every pixel of the frame went.
        view: The view's name.
    """

    mask: np.ndarray
    classes: np.ndarray
    counts: PixelCounts
    view: str


def _mask_id_map(mask_ids: np.ndarray) -> np.ndarray:
    mask_ids = np.asarray(mask_ids)
    if mask_ids.ndim != 2 or not np.issubdtype(mask_ids.dtype, np.integer):
        raise ValueError(f"a mask id map is an (H, W) integer array, got {mask_ids.dtype} of shape {mask_ids.shape}")
    return mask_ids


def _excluded_ids_present(mask_ids: np.ndarray, table: ClassTable) -> list[int]:
    """The excluded markers the frame holds; every id present is checked against the table first."""
    present = np.unique(mask_ids).tolist()
    for v in present:
        table.class_of(v)
    return sorted(v for v in present if v in table.excluded_mask_ids)


def frame_is_excluded(mask_ids: np.ndarray, table: ClassTable) -> bool:
    """Whether the frame carries the marker that takes it out of evaluation.

    Checked on the mask ids as read, not on the class set's ids, so that the
    benchmark set, which maps the marker to background, still sees it.

    Raises:
        ValueError: `mask_ids` is not an (H, W) integer array.
        KeyError: An id the table does not have.
    """
    return bool(_excluded_ids_present(_mask_id_map(mask_ids), table))


def scored_pixels(mask_ids: np.ndarray, table: ClassTable, view: str, valid: np.ndarray) -> Scored:
    """The pixels a view scores, the GT classes, and a count of the pixels it does not score.

    Args:
        mask_ids: An (H, W) integer map of the ids the GT mask holds, as
            `ClassTable.mask_ids` returns them, before any mapping.
        table: The dataset's class table, in the class set to score.
        view: One of `VIEWS`.
        valid: An (H, W) bool mask of the pixels with valid depth.

    Raises:
        KeyError: A mask id the table does not have, or a view name
            `VIEWS` does not.
        ValueError: An input is not a map of the frame's shape, or a class
            of type `excluded` is present: the frame should have been skipped
            before any pixel of it was scored.
    """
    mask_ids, valid = _mask_id_map(mask_ids), np.asarray(valid)
    if valid.dtype != np.bool_ or valid.shape != mask_ids.shape:
        raise ValueError(f"`valid` must be a bool array of shape {mask_ids.shape}, got {valid.dtype} {valid.shape}")
    in_view_ids = table.ids_in_view(view)  # raises on a view `VIEWS` does not have

    # On the mask ids, before the mapping: in the benchmark set the marker
    # has become background by the time the classes are typed.
    excluded = _excluded_ids_present(mask_ids, table)
    if excluded:
        raise ValueError(
            f"{table.dataset}: mask id {excluded} marks the frame as excluded; it is skipped "
            f"as a whole, not scored pixel by pixel"
        )
    # Every id present is mapped through the table, so one it does not know
    # raises here rather than being scored as something.
    classes = table.classes_of(mask_ids)

    def valid_and_of(ids: frozenset[int]) -> np.ndarray:
        return valid & np.isin(classes, sorted(ids))

    ignored = valid_and_of(table.ids_of_type(ClassType.IGNORED))
    background = valid_and_of(table.ids_of_type(ClassType.BACKGROUND))
    in_view = valid_and_of(in_view_ids)
    left_out = valid & ~ignored & ~background & ~in_view
    counts = PixelCounts(
        invalid_depth=int((~valid).sum()),
        ignored=int(ignored.sum()),
        background=int(background.sum()),
        left_out=int(left_out.sum()),
        scored=int(in_view.sum()),
    )
    return Scored(mask=in_view, classes=classes, counts=counts, view=view)
