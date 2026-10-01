"""Boundaries: where a label map changes, and how well two sets of boundaries agree.

Three of the evaluator's metrics read boundaries: `boundary_F` compares the
class map's boundaries with the GT's, `boundary_R_raw` asks how much of the
GT's boundary the regions recover before any class is assigned, and
`inst_BF` compares the contour of each paired object. All three use the same
two steps, defined here once: mark the boundary pixels of a label map, then
count how many of one map's boundary pixels lie within `BOUNDARY_TOL_PX` of
the other's.

A boundary pixel is a scored pixel whose left, right, upper or lower
neighbour is a scored pixel with another label. Both sides of an edge are
marked, so a boundary is 2 px wide. An edge against a pixel that is not
scored (invalid depth, ignored, background, or a class the view leaves out)
is not a boundary: a region is neither rewarded nor penalised for where it
ends against them. A pixel with no label (a region map's -1, say) is a
label like any other: an edge against it, between two scored pixels, is a
boundary, as it was for the pilot evaluator.

The tolerance is a square dilation by `BOUNDARY_TOL_PX`. Because the GT marks
both sides of its edge, a predicted pixel `BOUNDARY_TOL_PX` + 1 px from the
GT edge still reaches the GT pixel on its own side; so an edge shifted by
`BOUNDARY_TOL_PX` px scores 1, one shifted by `BOUNDARY_TOL_PX` + 1 px scores
1/2 (only its near column is within reach), and one further off scores 0.

The pilot evaluator marked boundaries on the whole map and then dropped the
invalid pixels, so its edge against background counted as a boundary. Pilot
mode gets that by passing no `scored` mask to `boundary_pixels` and masking
the result itself. Its F had a `1e-9` in the denominator and it wrote 0
where this module returns None; pilot mode takes both from the `precision`
and `recall` returned here, 2·p·r / (p + r + 1e-9), rather than counting
boundary pixels a second time.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# The pilot evaluator's value for its main boundary keys. What it allows is
# given with the boundary definition above.
BOUNDARY_TOL_PX = 2


@dataclass(frozen=True)
class BoundaryScore:
    """How well a predicted boundary agrees with a GT boundary.

    Attributes:
        precision: The share of predicted boundary pixels within the tolerance
            of a GT boundary pixel.
        recall: The share of GT boundary pixels within the tolerance of a
            predicted boundary pixel.
        f: Their harmonic mean; 0 when both are 0.
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
            region ids or one object's mask.
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
