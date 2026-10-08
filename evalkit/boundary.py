"""Mark where a label map changes, and score how well two sets of boundaries agree.

`boundary_F`, `boundary_R_raw` and `inst_BF` are all computed with
`boundary_pixels` and `boundary_score`. `boundary_pixels` marks the boundary
pixels of a label map over the scored pixels. `boundary_score` returns the
precision, recall and F of a predicted boundary against a GT boundary,
within `tol` pixels, `BOUNDARY_TOL_PX` by default. `docs/evaluation.md`
defines a boundary pixel and the tolerance ("Boundaries: `boundary_F`,
`boundary_R_raw`"). `docs/figures/boundary.png` shows the boundary pixels
on a drawn scene, and `docs/figures/boundary_tolerance.png` shows the
tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# The pilot evaluator's value for its main boundary keys. `docs/evaluation.md`
# says what it allows ("Boundaries: `boundary_F`, `boundary_R_raw`").
BOUNDARY_TOL_PX = 2


@dataclass(frozen=True)
class BoundaryScore:
    """How well a predicted boundary agrees with a GT boundary.

    Attributes:
        precision: The share of predicted boundary pixels within the tolerance
            of a GT boundary pixel.
        recall: The share of GT boundary pixels within the tolerance of a
            predicted boundary pixel.
        f: Their harmonic mean; 0 when both are 0. Pilot mode is to compute
            the pilot evaluator's own F from `precision` and `recall`
            (`docs/porting.md`, "Pilot mode's own rules").
    """

    precision: float
    recall: float
    f: float


def _check_map(labels: np.ndarray, scored: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    if labels.ndim != 2:
        raise ValueError(f"a label map is (H, W), got shape {labels.shape}")
    # bool is a two-label map: an object's mask from `Objects.mask` arrives so.
    if labels.dtype != np.bool_ and not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"a label map holds integer or bool labels, got dtype {labels.dtype}")
    if scored is None:
        return labels, np.ones(labels.shape, dtype=bool)
    scored = np.asarray(scored)
    if scored.dtype != np.bool_ or scored.shape != labels.shape:
        raise ValueError(
            f"`scored` must be a bool array of the label map's shape {labels.shape}, "
            f"got {scored.dtype} of shape {scored.shape}"
        )
    return labels, scored


def boundary_pixels(labels: np.ndarray, scored: np.ndarray | None = None) -> np.ndarray:
    """Mark the pixels where `labels` changes between two scored neighbours.

    Args:
        labels: An (H, W) integer or bool label map: GT classes, a class map,
            region ids or one object's mask. A region map's -1 (no region)
            is a label like any other, as it was for the pilot evaluator.
        scored: An (H, W) bool mask of the pixels the metric scores. An edge
            between a scored pixel and one that is not is not a boundary.
            None scores every pixel, which is what pilot mode wants.

    Returns:
        An (H, W) bool array, True on both sides of every edge.
    """
    labels, scored = _check_map(labels, scored)
    boundary = np.zeros(labels.shape, dtype=bool)
    across = (labels[:, :-1] != labels[:, 1:]) & scored[:, :-1] & scored[:, 1:]
    boundary[:, :-1] |= across
    boundary[:, 1:] |= across
    down = (labels[:-1, :] != labels[1:, :]) & scored[:-1, :] & scored[1:, :]
    boundary[:-1, :] |= down
    boundary[1:, :] |= down
    return boundary


def within_tolerance(boundary: np.ndarray, tol: int = BOUNDARY_TOL_PX) -> np.ndarray:
    """The pixels within `tol` of a boundary pixel, as a square neighbourhood.

    A square rather than a disk because the pilot evaluator dilated with a
    (2·tol + 1)² box, and both modes share one rule.
    """
    boundary = np.asarray(boundary)
    if boundary.dtype != np.bool_ or boundary.ndim != 2:
        raise ValueError(f"a boundary is an (H, W) bool array, got {boundary.dtype} of shape {boundary.shape}")
    if not isinstance(tol, (int, np.integer)) or isinstance(tol, (bool, np.bool_)) or tol < 0:
        raise ValueError(f"the tolerance is a non-negative integer number of pixels, got {tol!r}")
    tol = int(tol)
    if tol == 0:
        return boundary.copy()
    # cv2.dilate with a square kernel of ones. A numpy shift-or over the same
    # (2*tol + 1)^2 offsets agrees on every mask tried, borders included;
    # whether a hashed file should depend on a library's behaviour at all,
    # while OpenCV is unpinned, is decided before the freeze (an open question
    # of the port), not here.
    kernel = np.ones((2 * tol + 1, 2 * tol + 1), dtype=np.uint8)
    return cv2.dilate(boundary.astype(np.uint8), kernel).astype(bool)


def boundary_score(
    predicted: np.ndarray, gt: np.ndarray, tol: int = BOUNDARY_TOL_PX,
) -> BoundaryScore | None:
    """Precision, recall and F of a predicted boundary against a GT boundary.

    Args:
        predicted: The predicted boundary pixels, from `boundary_pixels`.
        gt: The GT boundary pixels, from `boundary_pixels` with the same
            `scored` mask.
        tol: The tolerance in pixels.

    Returns:
        The score, or None when the GT has no boundary pixel: then there is
        nothing to recover and the metric is not defined. A frame whose
        scored pixels are all one class is the common case, once a view has
        removed the rest. When only the prediction's boundary is empty, the
        score is 0.
    """
    predicted = np.asarray(predicted)
    gt = np.asarray(gt)
    if predicted.dtype != np.bool_ or gt.dtype != np.bool_ or predicted.shape != gt.shape or gt.ndim != 2:
        raise ValueError(
            f"both boundaries must be (H, W) bool arrays of one shape, got "
            f"{predicted.dtype} {predicted.shape} and {gt.dtype} {gt.shape}"
        )
    n_gt = int(gt.sum())
    if n_gt == 0:
        return None
    n_pred = int(predicted.sum())
    if n_pred == 0:
        return BoundaryScore(0.0, 0.0, 0.0)
    precision = int((predicted & within_tolerance(gt, tol)).sum()) / n_pred
    recall = int((gt & within_tolerance(predicted, tol)).sum()) / n_gt
    f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return BoundaryScore(precision, recall, f)
