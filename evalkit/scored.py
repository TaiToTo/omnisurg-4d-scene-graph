"""Compute the scored pixels of a frame in a view, and count the pixels the view removes.

A pixel is scored when its depth is valid and its GT class is one the view
includes. `scored_pixels` computes that mask from the mask ids as
`ClassTable.mask_ids` read them, with the class map that goes with it. It
counts each removed pixel under the first reason that removes it, so the
counts sum to the frame. `frame_is_excluded` finds a frame that an
`excluded` class takes out, and `scored_pixels` refuses such a frame too.
`docs/evaluation.md` fixes the order of the reasons and the depth rule
("Fail closed"). `docs/figures/scored_pixels.png` shows this on a drawn
scene, in pilot mode, since the scene has a patch without depth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evalkit.classes import ClassTable, ClassType

# The pilot evaluator's test of a depth value: finite and above this.
# `surgical_core.geometry.valid` has an equal copy, and
# `tests/test_depth_min_agrees.py` fails if the two differ.
DEPTH_MIN = 1e-6


@dataclass(frozen=True)
class PixelCounts:
    """Where each pixel of a frame went, by the first reason that removed it.

    Attributes:
        invalid_depth: Depth not finite or not above `DEPTH_MIN`. Always 0
            in normal mode, which refuses such a frame; pilot mode masks
            the pixels and counts them here.
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
        """Every pixel of the frame: the counts are disjoint, so their sum."""
        return self.invalid_depth + self.ignored + self.background + self.left_out + self.scored


@dataclass(frozen=True)
class Scored:
    """The scored pixels of one frame in one view, with the GT they are scored against.

    Attributes:
        mask: An (H, W) bool array, True on the pixels the metrics score.
        classes: The (H, W) int32 GT class map, in the table's class set, on
            every pixel scored or not; the metrics read it under `mask`.
        background: An (H, W) bool array, True on the valid pixels whose GT
            is `background`: disjoint from `mask`, and what
            `unlabelled_share` reads to see the spill the metrics cannot.
        counts: Where every pixel of the frame went.
        view: The view's name.
    """

    mask: np.ndarray
    classes: np.ndarray
    background: np.ndarray
    counts: PixelCounts
    view: str


def _mask_id_map(mask_ids: np.ndarray) -> np.ndarray:
    mask_ids = np.asarray(mask_ids)
    if mask_ids.ndim != 2 or not np.issubdtype(mask_ids.dtype, np.integer):
        raise ValueError(f"a mask id map is an (H, W) integer array, got {mask_ids.dtype} of shape {mask_ids.shape}")
    return mask_ids


def _excluded_ids_present(mask_ids: np.ndarray, table: ClassTable) -> list[int]:
    """Return the excluded markers the frame holds; every id present is checked against the table first."""
    present = np.unique(mask_ids).tolist()
    for v in present:
        table.class_of(v)
    return sorted(v for v in present if v in table.excluded_mask_ids)


def frame_is_excluded(mask_ids: np.ndarray, table: ClassTable) -> bool:
    """Return whether the frame carries the marker that takes it out of evaluation.

    Checked on the mask ids as read, not on the class set's ids, so that the
    benchmark set, which maps the marker to background, still sees it.

    Raises:
        ValueError: `mask_ids` is not an (H, W) integer array.
        KeyError: An id the table does not have.
    """
    return bool(_excluded_ids_present(_mask_id_map(mask_ids), table))


def valid_depth(depth: np.ndarray, *, pilot: bool = False) -> np.ndarray:
    """Return the pixels whose depth is finite and above `DEPTH_MIN`, as a bool mask.

    Args:
        depth: The (H, W) float depth map of the frame.
        pilot: In pilot mode the mask is returned as it is, which is what the
            pilot evaluator scored over. Otherwise every pixel must pass.

    Returns:
        An (H, W) bool array; all True in normal mode.

    Raises:
        ValueError: `depth` is not an (H, W) float array; or, in normal mode,
            a pixel has no valid depth. That is a fault in the data, not
            something to score around: the depth maps the pipeline runs on
            have a finite positive value on every pixel.
    """
    depth = np.asarray(depth)
    if depth.ndim != 2 or not np.issubdtype(depth.dtype, np.floating):
        raise ValueError(f"a depth map is an (H, W) float array, got {depth.dtype} of shape {depth.shape}")
    valid = np.isfinite(depth) & (depth > DEPTH_MIN)
    if not pilot and not valid.all():
        n = int((~valid).sum())
        raise ValueError(
            f"{n} of {depth.size} pixels have no valid depth (not finite or not above {DEPTH_MIN}); "
            f"the depth map is faulty, and the frame is refused rather than scored on the rest"
        )
    return valid


def scored_pixels(
    mask_ids: np.ndarray, table: ClassTable, view: str, valid: np.ndarray, *, pilot: bool = False,
) -> Scored:
    """Return the pixels a view scores, the GT classes, and a count of the pixels it does not score.

    Args:
        mask_ids: An (H, W) integer map of the ids the GT mask holds, as
            `ClassTable.mask_ids` returns them, before any mapping.
        table: The dataset's class table, in the class set to score.
        view: One of `VIEWS`.
        valid: The (H, W) bool mask `valid_depth` returned. In normal mode
            it is all True, and a False in it is refused here as well, so
            that no caller scores around a pixel without depth.
        pilot: Pilot mode, where `valid` may mask pixels out.

    Raises:
        KeyError: A mask id the table does not have, or a view name
            `VIEWS` does not.
        ValueError: An input is not a map of the frame's shape; `valid` has
            a False in normal mode; or a class of type `excluded` is
            present: the frame should have been skipped before any pixel of
            it was scored.
    """
    mask_ids, valid = _mask_id_map(mask_ids), np.asarray(valid)
    if valid.dtype != np.bool_ or valid.shape != mask_ids.shape:
        raise ValueError(f"`valid` must be a bool array of shape {mask_ids.shape}, got {valid.dtype} {valid.shape}")
    if not pilot and not valid.all():
        raise ValueError(
            f"`valid` leaves {int((~valid).sum())} pixels out, which only pilot mode may do; in normal "
            f"mode a pixel without valid depth is a data fault, refused by `valid_depth`"
        )
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
    return Scored(mask=in_view, classes=classes, background=background, counts=counts, view=view)
