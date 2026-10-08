"""Name each region from the GT, and compute `mIoU` over the names.

The pipeline gives regions without classes, so `mIoU` and `boundary_F` need
a class for each region. Each region takes the class that most of its scored
pixels have in the GT. `docs/evaluation.md` defines the class map and its
ties ("Naming the regions: the class map"), and says why `mIoU` over it is
an oracle value. Pilot mode takes the same vote over its own `scored` mask,
the `full` domain, where background votes too.

`docs/figures/class_map.png` shows this on a drawn scene, with the numbers
the module gives for it.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping

import numpy as np

NO_CLASS = -1


@dataclass(frozen=True)
class ClassMap:
    """The prediction's regions named from the GT.

    Attributes:
        classes: An (H, W) int32 map: the name of the region on each scored
            pixel, `NO_CLASS` where there is no region, where the region has
            no name, or where the pixel is not scored. `boundary_F` takes
            this map as it is: under the scored mask, `NO_CLASS` on a pixel
            with no region is a label like any other, so the edge between a
            named region and the pixels it left out is a predicted boundary,
            as it was in the pilot evaluator, which put background there.
        names: Each named region's id to its class. Read-only.
    """

    classes: np.ndarray
    names: Mapping[int, int]


def _check(gt: np.ndarray, regions: np.ndarray, scored: np.ndarray) -> None:
    for name, arr in (("gt", gt), ("regions", regions)):
        if arr.ndim != 2 or not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"`{name}` is an (H, W) integer array, got {arr.dtype} of shape {arr.shape}")
    if scored.dtype != np.bool_ or scored.shape != gt.shape or regions.shape != gt.shape:
        raise ValueError(
            f"`gt`, `regions` and the bool `scored` mask must share one shape, got "
            f"{gt.shape}, {regions.shape} and {scored.dtype} {scored.shape}"
        )
    if scored.any() and gt[scored].min() < 0:
        raise ValueError("a GT class map holds no negative ids on a scored pixel")
    if regions.min(initial=0) < -1:
        raise ValueError("a region map holds region ids >= 0 and -1 for no region")


def class_map(gt: np.ndarray, regions: np.ndarray, scored: np.ndarray) -> ClassMap:
    """Name every region from the GT by majority vote over its scored pixels.

    Args:
        gt: An (H, W) integer map of GT class ids.
        regions: An (H, W) integer map of region ids, -1 for no region.
        scored: An (H, W) bool mask of the pixels the metric scores.

    Returns:
        The class map. A tie goes to the smaller class id.
    """
    gt, regions, scored = np.asarray(gt), np.asarray(regions), np.asarray(scored)
    _check(gt, regions, scored)
    voting = scored & (regions >= 0)
    classes = np.full(gt.shape, NO_CLASS, dtype=np.int32)
    if not voting.any():
        return ClassMap(classes=classes, names=MappingProxyType({}))
    region_ids, region_index = np.unique(regions[voting], return_inverse=True)
    class_ids, class_index = np.unique(gt[voting], return_inverse=True)
    # One vote per pixel, as integer counts; bincount gives exactly what
    # adding one at a time would.
    votes = np.bincount(
        region_index * class_ids.size + class_index, minlength=region_ids.size * class_ids.size,
    ).reshape(region_ids.size, class_ids.size)
    # `argmax` takes the first of equal counts, and `class_ids` is sorted
    # ascending, so a tie goes to the smaller id.
    winner = class_ids[votes.argmax(axis=1)]
    names = MappingProxyType({int(r): int(c) for r, c in zip(region_ids.tolist(), winner.tolist())})
    classes[voting] = winner[region_index]
    return ClassMap(classes=classes, names=names)


@dataclass(frozen=True)
class ClassScores:
    """`mIoU` of one frame, with the IoU of every class behind it.

    Attributes:
        miou: The mean of `ious`, summed in class id order; None when no
            class is present. Pilot mode averages `ious` itself: the pilot
            evaluator's `np.mean` in set order differs in the last bit.
        ious: Each class present in the GT or the class map, over the scored
            pixels, to its IoU.
    """

    miou: float | None
    ious: dict[int, float]


def class_scores(gt: np.ndarray, cmap: ClassMap, scored: np.ndarray) -> ClassScores:
    """Compute the IoU per class between the class map and the GT, and their mean.

    The classes are those present on a scored pixel in the GT or in the
    class map. Background is not among them, because the caller removed it
    from `scored`; a pixel with no class counts against its GT class's IoU
    like any other mislabelled pixel.
    """
    gt, scored = np.asarray(gt), np.asarray(scored)
    _check(gt, cmap.classes, scored)
    g = gt[scored]
    p = cmap.classes[scored]
    # A name is a GT class seen on a scored pixel, so with the mask the map
    # was voted under, the class map adds no class to the GT's and the union
    # is the GT's set. It is kept as the specification words it, and it
    # matters when a caller scores under another mask than it voted under.
    present = np.union1d(np.unique(g), np.unique(p))
    present = present[present != NO_CLASS]
    ious: dict[int, float] = {}
    for c in present.tolist():
        in_g, in_p = g == c, p == c
        union = int((in_g | in_p).sum())
        ious[int(c)] = int((in_g & in_p).sum()) / union
    miou = sum(ious.values()) / len(ious) if ious else None
    return ClassScores(miou=miou, ious=ious)
