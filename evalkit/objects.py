"""Objects, their pairing, and the two primary metrics `F1_50` and `SQ`.

The datasets label classes, not individual things, so in the GT an object is
one class's whole region in one frame; in the prediction an object is one
region, one id in one frame. Both are taken over the scored pixels only: a
view removes the pixels of the classes it leaves out from the GT and the
prediction alike, and a region with no scored pixel is no object.

Objects are paired greedily, highest IoU first, each object at most once,
whatever its class. Among pairs of equal IoU the one with the higher GT
object index is taken first, then the higher predicted index, with GT
objects indexed by class id and predicted objects by region id, both
ascending. That is the pilot evaluator's order, kept so that both modes
share one rule. The first two keys decide which objects are found: at
exactly `MATCH_IOU` a tie between two regions over one class, or two classes
under one region, is settled by them, and an ascending order would take
different pairs. The third key, the predicted index, only fixes the order
in which the pairs are listed: with the first two held, taking the higher
predicted index first or the higher GT index first gives the same pairs
(checked exhaustively up to 4 GT and 5 predicted objects). A pair with
IoU >= `MATCH_IOU` is a hit.

`F1_50` is 2 · hits / (GT objects + predicted objects), so every extra region
counts against it; `SQ` is the mean IoU of the hits, so it says how well the
found objects fit. `F1_50` is not defined on a frame with no GT object, `SQ`
on a frame with no hit; the caller leaves those frames out of the mean and
counts them. Pilot mode's GT objects are per-class connected components of at
least `PILOT_MIN_CC_PX` and come from its own module; the pairing and the two
formulas here are shared.

`docs/figures/objects.png` shows this on a drawn scene, with the numbers the module
gives for it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The pilot evaluator's rule: a pair is a hit at IoU >= 0.5. PQ's convention
# (IoU > 0.5) would make a pairing unique; at exactly 0.5 the greedy order
# given in `pair` decides.
MATCH_IOU = 0.5


# `eq=False`: the generated `__eq__` would compare the arrays with `==` and
# raise on the ambiguous result, and the generated `__hash__` would try to
# hash them. Two `Objects` are compared by their fields where a test needs it.
@dataclass(frozen=True, eq=False)
class Objects:
    """The objects of one frame, as an index map.

    Attributes:
        index: An (H, W) int32 map: 0 on a pixel that belongs to no object,
            k on the pixels of object k, numbered from 1 in the order of
            `ids`.
        ids: The class id (GT) or region id (prediction) of each object,
            ascending; object k has id `ids[k - 1]`.
        areas: The scored pixel count of each object, in the same order.
        scored: The (H, W) bool mask the objects were taken over. The GT
            and the prediction must be taken over the same mask, or the
            unions in `pair` are wrong without anything crashing; `intersections`
            compares the two masks and refuses a difference.
    """

    index: np.ndarray
    ids: tuple[int, ...]
    areas: tuple[int, ...]
    scored: np.ndarray

    def __len__(self) -> int:
        return len(self.ids)

    def mask(self, k: int) -> np.ndarray:
        """The pixels of object `k` (numbered from 1)."""
        if not 1 <= k <= len(self.ids):
            raise IndexError(f"object {k} does not exist; objects are 1..{len(self.ids)}")
        return self.index == k


@dataclass(frozen=True)
class Pair:
    """One greedy pairing of a GT object with a predicted object.

    Attributes:
        iou: Their IoU over the scored pixels.
        gt: The GT object's number (from 1).
        pred: The predicted object's number (from 1).
    """

    iou: float
    gt: int
    pred: int

    @property
    def hit(self) -> bool:
        return self.iou >= MATCH_IOU


def _check(labels: np.ndarray, scored: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # TODO: the class map module checks its inputs the same way; once both are
    # in main the two checks become one, so a label map is refused for one
    # reason everywhere.
    labels = np.asarray(labels)
    scored = np.asarray(scored)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"a label map is an (H, W) integer array, got {labels.dtype} of shape {labels.shape}")
    if scored.dtype != np.bool_ or scored.shape != labels.shape:
        raise ValueError(
            f"`scored` must be a bool array of the label map's shape {labels.shape}, "
            f"got {scored.dtype} of shape {scored.shape}"
        )
    return labels, scored


def _objects(labels: np.ndarray, scored: np.ndarray, present: np.ndarray) -> Objects:
    """One object per label in `present` (sorted ascending), numbered in that order."""
    # TODO: one pass over the frame per object. With the five to seven regions
    # a frame had in the pilot measurements that is nothing; at a few hundred
    # regions it is half a second per 1080p frame, where
    # `np.unique(labels[member], return_inverse=True, return_counts=True)`
    # builds the same map and areas in one pass (checked equal on 1080p frames).
    index = np.zeros(labels.shape, dtype=np.int32)
    areas = []
    for k, label in enumerate(present.tolist(), start=1):
        member = scored & (labels == label)
        index[member] = k
        areas.append(int(member.sum()))
    return Objects(
        index=index, ids=tuple(int(v) for v in present.tolist()), areas=tuple(areas), scored=scored,
    )


def gt_objects(classes: np.ndarray, scored: np.ndarray) -> Objects:
    """One GT object per class present in the scored pixels.

    Args:
        classes: An (H, W) integer map of GT class ids. Every class in a
            scored pixel becomes an object: the caller has already removed
            background, ignored and left-out classes from `scored`.
        scored: An (H, W) bool mask of the pixels the metric scores.

    Returns:
        The objects, indexed by class id ascending.
    """
    classes, scored = _check(classes, scored)
    present = np.unique(classes[scored])
    if present.size and present.min() < 0:
        raise ValueError("a GT class map holds no negative ids; -1 is a prediction's 'no region'")
    return _objects(classes, scored, present)


def predicted_objects(regions: np.ndarray, scored: np.ndarray) -> Objects:
    """One predicted object per region id present in the scored pixels.

    Args:
        regions: An (H, W) integer map of region ids, -1 for no region.
        scored: An (H, W) bool mask of the pixels the metric scores.

    Returns:
        The objects, indexed by region id ascending. A region with no scored
        pixel is no object; a region with some is the object of its scored
        pixels only.
    """
    regions, scored = _check(regions, scored)
    if regions.min(initial=0) < -1:
        raise ValueError("a region map holds region ids >= 0 and -1 for no region")
    present = np.unique(regions[scored & (regions >= 0)])
    return _objects(regions, scored, present)


def intersections(gt: Objects, pred: Objects) -> np.ndarray:
    """The pixel count of every (GT object, predicted object) overlap, as a (G, P) table.

    Raises:
        ValueError: If the two were not taken over the same scored mask. A
            region's area would then be counted over other pixels than the
            class's, and every union would be off without a crash.
    """
    if gt.index.shape != pred.index.shape:
        raise ValueError(f"the two object maps differ in shape: {gt.index.shape} and {pred.index.shape}")
    if not np.array_equal(gt.scored, pred.scored):
        raise ValueError(
            "the GT and predicted objects were taken over different scored masks "
            f"({int(gt.scored.sum())} and {int(pred.scored.sum())} scored pixels); use one mask for both"
        )
    n_g, n_p = len(gt), len(pred)
    joint = np.bincount(
        (gt.index.astype(np.int64) * (n_p + 1) + pred.index).ravel(),
        minlength=(n_g + 1) * (n_p + 1),
    ).reshape(n_g + 1, n_p + 1)
    return joint[1:, 1:]


def pair(gt: Objects, pred: Objects) -> list[Pair]:
    """Pair the objects greedily, highest IoU first, each object at most once.

    Only overlapping pairs are candidates. Among pairs of equal IoU the one
    with the higher GT index is taken first, then the higher predicted index:
    the pilot evaluator sorted (iou, gt, pred) descending. The predicted
    index decides only the order of the returned list, never which pairs are
    in it (see the module docstring).

    Returns:
        The pairs in the order they were taken.
    """
    inter = intersections(gt, pred)
    candidates = []
    for gi in range(len(gt)):
        for pj in range(len(pred)):
            i = int(inter[gi, pj])
            if i > 0:
                union = gt.areas[gi] + pred.areas[pj] - i
                candidates.append((i / union, gi + 1, pj + 1))
    candidates.sort(reverse=True)
    gt_used, pred_used, pairs = set(), set(), []
    for iou, g, p in candidates:
        if g in gt_used or p in pred_used:
            continue
        gt_used.add(g)
        pred_used.add(p)
        pairs.append(Pair(iou=iou, gt=g, pred=p))
    return pairs


@dataclass(frozen=True)
class InstanceScores:
    """`F1_50` and `SQ` of one frame, with the counts behind them.

    Attributes:
        f1_50: 2 · hits / (GT objects + predicted objects); None on a frame
            with no GT object.
        sq: The mean IoU of the hits; None on a frame with no hit.
        n_gt: GT objects.
        n_pred: Predicted objects.
        n_hits: Pairs with IoU >= `MATCH_IOU`.
        pairs: Every pair taken, hits and misses.
    """

    f1_50: float | None
    sq: float | None
    n_gt: int
    n_pred: int
    n_hits: int
    pairs: tuple[Pair, ...]

    @property
    def hits(self) -> tuple[Pair, ...]:
        return tuple(p for p in self.pairs if p.hit)


def instance_scores(gt: Objects, pred: Objects) -> InstanceScores:
    """Pair the objects and take `F1_50` and `SQ` over the result."""
    pairs = tuple(pair(gt, pred))
    hits = [p for p in pairs if p.hit]
    n_gt, n_pred, n_hits = len(gt), len(pred), len(hits)
    f1_50 = 2 * n_hits / (n_gt + n_pred) if n_gt else None
    sq = sum(p.iou for p in hits) / n_hits if n_hits else None
    return InstanceScores(f1_50=f1_50, sq=sq, n_gt=n_gt, n_pred=n_pred, n_hits=n_hits, pairs=pairs)
