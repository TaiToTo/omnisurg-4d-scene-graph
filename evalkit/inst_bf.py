"""`inst_BF`: how well the found objects' contours fit.

`SQ` says how well each hit fits by area; `inst_BF` says it by contour. For
every hit (a GT object paired with a predicted object at IoU >= `MATCH_IOU`),
the predicted object's boundary is scored against the GT object's with the
boundary rule of `evalkit.boundary`, both taken over the scored pixels, and
the frame's value is the mean F over the hits.

An object's boundary is the edge between its pixels and the other scored
pixels. An edge against a pixel the view removed is no boundary, as
everywhere else, so a GT object whose contour lies wholly against removed
pixels has none: the one GT object of a frame that fills the scored pixels,
or an object cut off from every other class by a tool or by unlabelled
tissue. There is no contour to recover there, so such a hit enters no mean
and is counted, and `inst_BF` is not defined on a frame where no hit remains.
That is the rule the frame-level boundary metrics follow on a frame without
a GT boundary, applied per object. When only the predicted object's boundary
is empty, the hit scores 0, as the frame-level metrics do.

The pilot evaluator marked each object's boundary on the whole map and
masked it afterwards, so an object's edge against a removed pixel (invalid
depth, or background and instruments in its `labeled` and `tissue` domains)
was a boundary on the object's side; and it scored such a hit 0 and used a
1e-9 in its F. Pilot mode asks for the pilot's marking with `pilot=True`,
and reads the per-hit scores returned here to apply its own zeros and its
own F, rather than counting boundary pixels a second time.

`docs/figures/inst_bf.png` shows this on a drawn scene, with the numbers the module
gives for it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evalkit.boundary import BOUNDARY_TOL_PX, BoundaryScore, boundary_pixels, boundary_score
from evalkit.objects import InstanceScores, Objects


@dataclass(frozen=True)
class InstanceBoundary:
    """`inst_BF` of one frame, with the score of every hit behind it.

    Attributes:
        inst_bf: The mean F over the hits whose GT object has a boundary;
            None when no hit has one, which includes a frame with no hit.
        scores: One entry per hit, in the order of `InstanceScores.hits`:
            the hit's boundary score, or None when its GT object has no
            boundary pixel within the scored pixels.
        n_hits: The hits of the frame.
        n_entered: The hits that entered the mean.
    """

    inst_bf: float | None
    scores: tuple[BoundaryScore | None, ...]
    n_hits: int
    n_entered: int


def _object_boundary(mask: np.ndarray, scored: np.ndarray, pilot: bool) -> np.ndarray:
    # An object's mask is a two-label map, so its boundary is the edge between
    # the object and the rest, marked on both sides. Under the evaluator's
    # rule an edge against a removed pixel is no boundary; the pilot marked
    # it and then dropped the removed side.
    if pilot:
        return boundary_pixels(mask) & scored
    return boundary_pixels(mask, scored)


def instance_boundary_f(
    gt: Objects, pred: Objects, scores: InstanceScores, tol: int = BOUNDARY_TOL_PX, *, pilot: bool = False,
) -> InstanceBoundary:
    """Score the contour of every hit, and take `inst_BF` as the mean.

    Args:
        gt: The GT objects of the frame, from `gt_objects`.
        pred: The predicted objects, from `predicted_objects`, taken over the
            same scored mask.
        scores: The pairing of the two, from `instance_scores`; its hits are
            the pairs scored here.
        tol: The tolerance in pixels.
        pilot: Mark each object's boundary on the whole map and mask it
            afterwards, as the pilot evaluator did, so that an object's edge
            against a removed pixel is a boundary on the object's side.

    Raises:
        ValueError: The two object maps were taken over different scored
            masks, or `scores` pairs other numbers of objects than were
            given: each hit's contour would then be read off the wrong
            pixels without a crash. A pairing of other objects in the same
            numbers cannot be told apart here.
    """
    if gt.index.shape != pred.index.shape or not np.array_equal(gt.scored, pred.scored):
        raise ValueError(
            "the GT and predicted objects were taken over different scored masks; "
            "take both over one mask, and pair them with `instance_scores`"
        )
    if scores.n_gt != len(gt) or scores.n_pred != len(pred):
        raise ValueError(
            f"`scores` pairs {scores.n_gt} GT and {scores.n_pred} predicted objects, but "
            f"{len(gt)} and {len(pred)} were given; it was computed from other objects"
        )
    scored = gt.scored
    per_hit = []
    for hit in scores.hits:
        gt_boundary = _object_boundary(gt.mask(hit.gt), scored, pilot)
        pred_boundary = _object_boundary(pred.mask(hit.pred), scored, pilot)
        per_hit.append(boundary_score(pred_boundary, gt_boundary, tol))
    entered = [s.f for s in per_hit if s is not None]
    inst_bf = sum(entered) / len(entered) if entered else None
    return InstanceBoundary(
        inst_bf=inst_bf, scores=tuple(per_hit), n_hits=len(per_hit), n_entered=len(entered),
    )
