"""Small synthetic frames whose metric values can be worked out by hand.

The evaluator's hand-derived tests are computed on these scenes: each one is
a ground truth and a prediction with one controlled fault, and the areas are
round numbers, so the value of every metric on it can be derived on paper and
pinned. A metric guide, if one is built here, draws these same frames rather
than frames of its own, so that its pictures and the tests cannot drift apart.

This module holds the frames only. What each metric must give on them is
derived under the evaluator's rules, in the tests that come with the metrics;
the pilot evaluator's values are pinned by its own tests, in pilot mode.

Every scene is 60 x 100 px. That keeps the arithmetic readable while every
region stays above the pilot evaluator's 300 px cut, which it needs to
reproduce the pilot numbers; the one exception is `sliver`, whose region is
made to sit on either side of that cut. GT class 1 fills the left half and
class 2 the right half unless a scene says otherwise. Predicted regions are
ids 0, 1, 2, ..., and -1 is "no region".
"""
from dataclasses import dataclass, field

import numpy as np

H, W = 60, 100
HALF = W // 2


@dataclass
class Scene:
    """One frame: ground truth, prediction and the pixels that are evaluated.

    Attributes:
        key: Short identifier used by the tests.
        title: One-line description.
        gt: (H, W) GT class id map (0 = background).
        lab: (H, W) predicted region ids (-1 = no region).
        valid: (H, W) evaluated pixels.
        next_lab: The prediction one frame later, for the temporal scenes.
    """
    key: str
    title: str
    gt: np.ndarray
    lab: np.ndarray
    valid: np.ndarray = field(default_factory=lambda: np.ones((H, W), bool))
    next_lab: np.ndarray | None = None


def _two_classes() -> np.ndarray:
    gt = np.zeros((H, W), np.int32)
    gt[:, :HALF] = 1
    gt[:, HALF:] = 2
    return gt


def _split_at(col: int, ids=(0, 1)) -> np.ndarray:
    lab = np.empty((H, W), np.int32)
    lab[:, :col] = ids[0]
    lab[:, col:] = ids[1]
    return lab


def exact() -> Scene:
    return Scene("exact", "Prediction equals the ground truth", _two_classes(), _split_at(HALF))


def shifted(k: int = 3) -> Scene:
    """The predicted boundary sits `k` px right of the true one."""
    return Scene(f"shift{k}", f"Boundary {k} px off", _two_classes(), _split_at(HALF + k))


def split() -> Scene:
    """Class 1 is cut into two regions of equal size (over-segmentation)."""
    lab = _split_at(HALF, (0, 2))
    lab[:, HALF // 2:HALF] = 1
    return Scene("split", "One class split in two", _two_classes(), lab)


def merged() -> Scene:
    """One region covers both classes (under-segmentation)."""
    return Scene("merged", "Two classes merged into one region", _two_classes(),
                 np.zeros((H, W), np.int32))


def gap(width: int = 10, painted: bool = False) -> Scene:
    """A band of `width` px around the boundary is left without a region, or painted as a third one.

    Half of the band lies on each class. `width` is even: an odd one gives
    `width - 1` columns, since the band is `width // 2` to each side.
    """
    lab = _split_at(HALF)
    lab[:, HALF - width // 2:HALF + width // 2] = 2 if painted else -1
    key, title = ("band", "The gap painted as a third region") if painted else \
        ("gap", "A band around the boundary left without a region")
    return Scene(key, title, _two_classes(), lab)


def on_background(rows: int = 10) -> Scene:
    """The top `rows` rows are unannotated background, and a region covers them."""
    gt = _two_classes()
    gt[:rows] = 0
    lab = _split_at(HALF)
    lab[:rows] = 2
    return Scene("background", "A region on unannotated background", gt, lab)


def sliver(px: int) -> Scene:
    """A predicted region of `px` pixels, filling the top rows from the left."""
    lab = _split_at(HALF)
    lab.reshape(-1)[:px] = 3
    return Scene(f"sliver{px}", f"A {px} px sliver", _two_classes(), lab)


def spill(width: int, small: int = 16) -> Scene:
    """Region 0 covers GT class 1 (the left `small` columns) and spills `width` columns into class 2.

    Class 2 is 60 x (100 - `small`) = 5,040 px, so 5 % of it is 252 px, and
    the spill is 60 x `width` px. The pilot evaluator counts a piece towards
    over-segmentation only when it covers both 5 % of the class and 200 px,
    though its comment said either; a 4-column spill (240 px) is the case that
    tells the two readings apart.
    """
    gt = np.zeros((H, W), np.int32)
    gt[:, :small] = 1
    gt[:, small:] = 2
    return Scene(f"spill{width}", f"Region spills {width} px into a neighbour", gt,
                 _split_at(small + width))


def moved(k: int = 5) -> Scene:
    """Both regions keep their ids and move `k` px right in the next frame."""
    lab = _split_at(HALF)
    return Scene(f"moved{k}", f"Regions move {k} px between frames", _two_classes(), lab,
                 next_lab=_split_at(HALF + k))


def swapped() -> Scene:
    """The two regions exchange ids between frames."""
    return Scene("swapped", "Regions exchange ids between frames", _two_classes(),
                 _split_at(HALF), next_lab=_split_at(HALF, (1, 0)))
