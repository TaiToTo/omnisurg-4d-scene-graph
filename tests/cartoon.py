"""A cartoon of a laparoscopic frame, for explaining the metrics by picture.

A metric guide, when one is built, is to draw this scene: liver, gallbladder,
fat and an instrument, and a prediction that gets each of them wrong in one
typical way. The guide's numbers are to be what the evaluator computes, not
values worked out by hand; the hand-checked arithmetic lives in `scenes.py`
and the tests that come with the metrics.

The class ids are the cartoon's own, not a dataset's: the point is the shape
of each mistake, not the class table.
"""
import cv2
import numpy as np

H, W = 150, 240

BACKGROUND, LIVER, GALLBLADDER, FAT, TOOL = 0, 1, 2, 3, 4
CLASS_NAMES = {LIVER: "liver", GALLBLADDER: "gallbladder", FAT: "fat", TOOL: "instrument"}


def _tool_mask(offset: int = 0, grow: int = 0) -> np.ndarray:
    """An instrument shaft entering from the right edge, `offset` px further left."""
    m = np.zeros((H, W), np.uint8)
    cv2.line(m, (W + 10, 18), (118 - offset, 88), 1, thickness=16 + 2 * grow)
    cv2.circle(m, (118 - offset, 88), 11 + grow, 1, -1)
    return m.astype(bool)


def ground_truth() -> np.ndarray:
    """GT class map. The instrument crosses the liver and cuts it in two."""
    gt = np.zeros((H, W), np.int32)
    liver = np.zeros((H, W), np.uint8)
    cv2.ellipse(liver, (95, 22), (135, 62), 0, 0, 360, 1, -1)
    gt[liver.astype(bool)] = LIVER
    fat = np.zeros((H, W), np.uint8)
    cv2.fillPoly(fat, [np.array([[0, 150], [0, 108], [70, 98], [150, 112], [205, 128], [205, 150]])], 1)
    gt[fat.astype(bool)] = FAT
    gb = np.zeros((H, W), np.uint8)
    cv2.ellipse(gb, (78, 92), (40, 20), -12, 0, 360, 1, -1)
    gt[gb.astype(bool)] = GALLBLADDER
    gt[_tool_mask()] = TOOL
    return gt


def prediction(gt: np.ndarray, tool_offset: int = 0, swap_liver: bool = False) -> np.ndarray:
    """Predicted regions, each mistake a typical one:

    - region 0 and 1: the liver, split in two by a cut of its own (over-segmentation);
    - region 2: the gallbladder merged with the fat below it (under-segmentation);
    - region 3: the instrument, its outline 3 px too wide;
    - region 4: a region on unannotated background (a false positive);
    - -1: a band between liver and gallbladder that no region covers.

    Args:
        gt: The GT class map from `ground_truth()`.
        tool_offset: Move the instrument this many px left (a later frame).
        swap_liver: Give the two liver regions each other's ids (a tracking id switch).
    """
    lab = np.full((H, W), -1, np.int32)
    liver = gt == LIVER
    cols = np.arange(W)[None, :].repeat(H, 0)
    a, b = (1, 0) if swap_liver else (0, 1)
    lab[liver & (cols < 60)] = a
    lab[liver & (cols >= 60)] = b
    lab[(gt == GALLBLADDER) | (gt == FAT)] = 2
    band = np.zeros((H, W), np.uint8)
    cv2.ellipse(band, (78, 92), (44, 24), -12, 180, 360, 1, 5)
    lab[band.astype(bool) & (lab != 2)] = -1
    lab[_tool_mask(tool_offset, grow=3)] = 3
    spur = np.zeros((H, W), np.uint8)
    cv2.ellipse(spur, (222, 132), (15, 12), 0, 0, 360, 1, -1)
    lab[spur.astype(bool) & (gt == BACKGROUND)] = 4
    return lab


def valid() -> np.ndarray:
    return np.ones((H, W), bool)
