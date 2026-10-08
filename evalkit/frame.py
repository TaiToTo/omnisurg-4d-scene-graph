"""Score one frame in every view, with the modules that define each metric.

`score_frame` is the normal-mode driver of `docs/evaluation.md`, "How a
frame is scored". It computes no metric itself. It checks the depth map,
then the excluded marker, and returns an excluded frame with no view for the
clip driver to count. It takes each view's scored pixels once, and passes
the same GT, region map and mask to every metric module. A key is None on a
frame its metric is not defined on. The counts behind every key are kept for
the clip driver. `time_IoU` is per clip and is not computed here. Pilot mode
has a driver of its own, and shares the metric modules, not this composition.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from evalkit.boundary import boundary_pixels, boundary_score
from evalkit.classes import VIEWS, ClassTable
from evalkit.classmap import class_map, class_scores
from evalkit.inst_bf import instance_boundary_f
from evalkit.keys import FRAME_METRICS
from evalkit.objects import gt_objects, instance_scores, predicted_objects
from evalkit.scored import PixelCounts, frame_is_excluded, scored_pixels, valid_depth
from evalkit.unlabelled import unlabelled_share
from evalkit.vi import variation_of_information

# The specification's key for each per-frame metric, to the field of `ViewScores`
# that holds it, in the specification's order. Each field is its key in lower
# case. The JSON is written through this mapping.
KEYS: Mapping[str, str] = MappingProxyType({key: key.lower() for key in FRAME_METRICS})


@dataclass(frozen=True)
class ViewScores:
    """Every per-frame metric of one frame in one view, with the counts behind them.

    Attributes:
        view: The view's name.
        counts: Where every pixel of the frame went.
        f1_50: `F1_50`; None on a frame with no GT object.
        sq: `SQ`; None on a frame with no hit.
        inst_bf: `inst_BF`; None when no hit has a GT contour to recover.
        miou: `mIoU`; None on a frame with no class.
        boundary_f: `boundary_F`; None on a frame with no GT boundary.
        boundary_r_raw: `boundary_R_raw`; None likewise.
        vi_split: `VI_split`; None on a frame with no scored pixel.
        vi_merge: `VI_merge`; None likewise.
        unlabelled_share: None when no region has a scored pixel.
        n_gt_objects: GT objects.
        n_pred_objects: Predicted objects.
        n_hits: Pairs at IoU >= `MATCH_IOU`.
        n_inst_bf_hits: The hits that entered `inst_BF`.
        ious: The IoU of each class behind `mIoU`. Read-only.
    """

    view: str
    counts: PixelCounts
    f1_50: float | None
    sq: float | None
    inst_bf: float | None
    miou: float | None
    boundary_f: float | None
    boundary_r_raw: float | None
    vi_split: float | None
    vi_merge: float | None
    unlabelled_share: float | None
    n_gt_objects: int
    n_pred_objects: int
    n_hits: int
    n_inst_bf_hits: int
    ious: Mapping[int, float]

    def metrics(self) -> dict[str, float | None]:
        """The metric values under the specification's keys, in its order."""
        return {key: getattr(self, field) for key, field in KEYS.items()}


@dataclass(frozen=True)
class FrameScores:
    """One frame, scored in every view, or skipped.

    Attributes:
        excluded: The frame carries the marker that takes it out of
            evaluation. Then `views` is empty, and the clip driver counts
            the frame.
        views: Each view's name to its scores. Read-only.
    """

    excluded: bool
    views: Mapping[str, ViewScores]


def _check_maps(mask_ids: np.ndarray, regions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask_ids, regions = np.asarray(mask_ids), np.asarray(regions)
    for name, arr in (("mask_ids", mask_ids), ("regions", regions)):
        if arr.ndim != 2 or not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"`{name}` is an (H, W) integer array, got {arr.dtype} of shape {arr.shape}")
    if regions.shape != mask_ids.shape:
        # The clip driver resizes a prediction of another shape to the depth
        # map's, with nearest neighbour; a frame arrives here in one shape.
        raise ValueError(f"`regions` has shape {regions.shape}, the GT mask {mask_ids.shape}")
    return mask_ids, regions


def score_view(
    mask_ids: np.ndarray, regions: np.ndarray, valid: np.ndarray, table: ClassTable, view: str,
) -> ViewScores:
    """Score one frame in one view.

    Args:
        mask_ids: An (H, W) integer map of the ids the GT mask holds, as
            `ClassTable.mask_ids` returns them.
        regions: An (H, W) integer map of region ids, -1 for no region.
        valid: The (H, W) bool mask `valid_depth` returned: all True in
            normal mode, which `scored_pixels` enforces.
        table: The dataset's class table, in the class set to score.
        view: One of `VIEWS`.

    Raises:
        ValueError: An input is not a map of one shape; a region id is
            below -1; the frame carries the excluded marker; or `valid`
            has a False in it.
        KeyError: A mask id the table does not have, or an unknown view.
    """
    # Which pixels this view scores, and the GT class on each. Every metric
    # below reads the same `gt` under the same `mask`.
    mask_ids, regions = _check_maps(mask_ids, regions)
    scored = scored_pixels(mask_ids, table, view, valid)
    gt, mask = scored.classes, scored.mask

    # Objects: the GT's and the regions', paired at IoU >= MATCH_IOU, give
    # F1_50 and SQ; the contours of the pairs give inst_BF.
    gt_objs = gt_objects(gt, mask)
    pred_objs = predicted_objects(regions, mask)
    inst = instance_scores(gt_objs, pred_objs)
    contours = instance_boundary_f(gt_objs, pred_objs, inst)

    # Classes: each region named after the GT class it covers most, then
    # the IoU of every class and mIoU over them.
    cmap = class_map(gt, regions, mask)
    classes = class_scores(gt, cmap, mask)

    # Boundaries, both against the GT's: the named regions' (boundary_F)
    # and the regions' as drawn (boundary_R_raw).
    gt_boundary = boundary_pixels(gt, mask)
    # The class map as it is: a pixel with no region carries `NO_CLASS`,
    # which is a label like any other, so the edge between a named region
    # and the pixels it left out is a predicted boundary.
    boundary_f = boundary_score(boundary_pixels(cmap.classes, mask), gt_boundary)
    boundary_raw = boundary_score(boundary_pixels(regions, mask), gt_boundary)

    # Agreement of the partitions (VI), and how much of the regions fell on
    # background the view does not score.
    vi = variation_of_information(gt, regions, mask)
    share = unlabelled_share(regions, mask, scored.background)

    return ViewScores(
        view=view,
        counts=scored.counts,
        f1_50=inst.f1_50,
        sq=inst.sq,
        inst_bf=contours.inst_bf,
        miou=classes.miou,
        boundary_f=None if boundary_f is None else boundary_f.f,
        boundary_r_raw=None if boundary_raw is None else boundary_raw.recall,
        vi_split=None if vi is None else vi.split,
        vi_merge=None if vi is None else vi.merge,
        unlabelled_share=share,
        n_gt_objects=inst.n_gt,
        n_pred_objects=inst.n_pred,
        n_hits=inst.n_hits,
        n_inst_bf_hits=contours.n_entered,
        ious=MappingProxyType(dict(classes.ious)),
    )


def score_frame(
    mask_ids: np.ndarray, regions: np.ndarray, depth: np.ndarray, table: ClassTable,
) -> FrameScores:
    """Score one frame in every view, or return it skipped.

    Args:
        mask_ids: An (H, W) integer map of the ids the GT mask holds, as
            `ClassTable.mask_ids` returns them, cut and resized to the
            depth map's shape by the caller.
        regions: An (H, W) integer map of region ids, -1 for no region, of
            the same shape.
        depth: The (H, W) float depth map of the frame. In normal mode every
            pixel must have valid depth, or the frame is refused.
        table: The dataset's class table, in the class set to score.

    Returns:
        The scores of every view in `VIEWS`, or, when the frame carries the
        excluded marker, `excluded=True` and no view.

    Raises:
        ValueError: An input is not a map of the depth map's shape, a
            region id is below -1, or a pixel has no valid depth.
        KeyError: A mask id the table does not have.
    """
    mask_ids, regions = _check_maps(mask_ids, regions)
    depth = np.asarray(depth)
    if depth.shape != mask_ids.shape:
        raise ValueError(f"`depth` has shape {depth.shape}, the GT mask {mask_ids.shape}")
    # The depth map before the marker: a pixel without valid depth is a
    # fault in the pipeline's output, and skipping the frame would hide it.
    valid = valid_depth(depth)
    # On the mask ids, before any pixel is scored: a frame the annotators
    # took out is skipped and counted, never scored in part.
    if frame_is_excluded(mask_ids, table):
        return FrameScores(excluded=True, views=MappingProxyType({}))
    views = {view: score_view(mask_ids, regions, valid, table, view) for view in VIEWS}
    return FrameScores(excluded=False, views=MappingProxyType(views))
