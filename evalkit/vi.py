"""`VI_split` and `VI_merge`: the two halves of the variation of information.

Over the scored pixels, with X the region id and Y the GT class of a pixel,
`VI_split` = H(X | Y) and `VI_merge` = H(Y | X), in bits. The first is zero
when every GT class is covered by one region and grows as classes are cut
into pieces; the second is zero when every region lies in one class and
grows as regions run across classes. Both are taken over pixels, so a
sliver cut off a class costs little, and a class halved costs one bit on
that class's pixels, weighted by their share of the frame.

The pixels with no region count together as one region: leaving them out
would let a prediction improve `VI_merge` by abandoning the pixels it is
unsure of. The pilot evaluator counted them the same way and left out
background by id; here background is already gone from the scored pixels,
so the function is shared by both modes as it stands. Pilot mode passes the
pilot's own mask: the valid pixels whose GT is not background, which is not
its `full` domain (`docs/evaluation.md`, "Checked against the pilot
evaluator").

The pilot evaluator also removed the ids in its `EXTRA_IGNORE`, set from
the command line and written to each score as `extra_ignore`. Every score
file in the workbench that records it has it empty, but those are the
workshop's; pilot mode assumes an empty set only once the 38 conditions'
JSONs have been checked (an open question of the port).

`docs/figures/vi.png` shows this on a drawn scene, with the numbers the module
gives for it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VIScores:
    """The two halves of one frame's variation of information, in bits.

    Attributes:
        split: `VI_split`, H(regions | GT): the information the regions add
            to the GT, which is what cutting one class into several costs.
        merge: `VI_merge`, H(GT | regions): the information the regions
            lose of the GT, which is what one region over several classes costs.
    """

    split: float
    merge: float


def _entropy_bits(p: np.ndarray) -> float:
    """The entropy of the distribution `p`, in bits; zero entries contribute nothing."""
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def variation_of_information(
    gt: np.ndarray, regions: np.ndarray, scored: np.ndarray,
) -> VIScores | None:
    """`VI_split` and `VI_merge` of one frame.

    Args:
        gt: An (H, W) integer map of GT class ids, non-negative on every
            scored pixel.
        regions: An (H, W) integer map of region ids, -1 for no region,
            which counts as a region of its own.
        scored: An (H, W) bool mask of the pixels the metric scores.

    Returns:
        `VI_split` and `VI_merge` in bits, or None when no pixel is scored:
        the metric is not defined there, and the caller counts the frame.
        A value that is zero in exact arithmetic can come out as a rounding
        residue of either sign, about 1e-16; the pilot evaluator wrote the
        same residues, so they are not clamped.

    Raises:
        ValueError: A map is not an (H, W) integer array, the three do not
            share one shape, or a scored pixel has a negative GT id.
    """
    gt, regions, scored = np.asarray(gt), np.asarray(regions), np.asarray(scored)
    for name, arr in (("gt", gt), ("regions", regions)):
        if arr.ndim != 2 or not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"`{name}` is an (H, W) integer array, got {arr.dtype} of shape {arr.shape}")
    if scored.dtype != np.bool_ or scored.shape != gt.shape or regions.shape != gt.shape:
        raise ValueError(
            f"`gt`, `regions` and the bool `scored` mask must share one shape, got "
            f"{gt.shape}, {regions.shape} and {scored.dtype} {scored.shape}"
        )
    x = regions[scored]
    y = gt[scored]
    if x.size == 0:
        return None
    # A negative GT id is no class; it would be counted as one here, so it
    # is refused, as the class map refuses it.
    if y.min() < 0:
        raise ValueError("a GT class map holds no negative ids on a scored pixel")
    # The joint distribution of (region, class). The counts are integers, so
    # bincount gives exactly what adding one per pixel would.
    xs, x = np.unique(x, return_inverse=True)
    ys, y = np.unique(y, return_inverse=True)
    joint = np.bincount(x * len(ys) + y, minlength=len(xs) * len(ys)).reshape(len(xs), len(ys)) / x.size
    h_xy = _entropy_bits(joint.ravel())
    h_x = _entropy_bits(joint.sum(axis=1))
    h_y = _entropy_bits(joint.sum(axis=0))
    return VIScores(split=h_xy - h_y, merge=h_xy - h_x)
