"""Summarise a clip: each key's mean over the frames it is defined on, and the number of those frames.

`summarize_clip` averages the per-frame keys of `evalkit.frame` over the
clip's GT frames, view by view. Each mean covers only the frames on which
its key is defined, and the summary records the number of those frames.
The excluded frames are counted, and the pixel and object counts are
summed. A clip with no scored frame is refused. `time_IoU` is passed in.
It pools every tracked frame, GT or not, so the entry point computes it.
PQ is not stored; `pq` gives it per frame. Pilot mode does not call
`summarize_clip`.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from types import MappingProxyType

from evalkit.classes import VIEWS
from evalkit.frame import KEYS, FrameScores, ViewScores
from evalkit.scored import PixelCounts


@dataclass(frozen=True)
class ScoredFrame:
    """One GT frame of a clip, scored or skipped.

    Attributes:
        frame: The frame number in the source video. The clip's frames are
            given in time order, which the caller settles from the
            timestamps; in some CholecSeg8k clips the numbers do not rise
            with it, so the order is not checked against them.
        scores: What `score_frame` returned for the frame.
    """

    frame: int
    scores: FrameScores


@dataclass(frozen=True)
class ViewSummary:
    """One clip in one view: each key's mean over the frames it is defined on.

    Attributes:
        view: The view's name.
        means: Each key of `KEYS` to its mean; None when the key is defined
            on no frame of the clip. Read-only.
        n_frames: Each key to the number of frames its mean covers. Read-only.
        pixels: Where every pixel of every scored frame went, summed.
        n_gt_objects: GT objects, summed over the scored frames.
        n_pred_objects: Predicted objects, summed likewise.
        n_hits: Pairs at IoU >= `MATCH_IOU`, summed likewise.
        n_inst_bf_hits: The hits that entered `inst_BF`, summed likewise.
    """

    view: str
    means: Mapping[str, float | None]
    n_frames: Mapping[str, int]
    pixels: PixelCounts
    n_gt_objects: int
    n_pred_objects: int
    n_hits: int
    n_inst_bf_hits: int


@dataclass(frozen=True)
class ClipScores:
    """One clip, scored in every view.

    Attributes:
        views: Each view's name to its summary. Read-only.
        time_iou: The clip's `time_IoU`, or None where nothing was pooled.
        n_scored_frames: GT frames scored.
        n_excluded_frames: GT frames the excluded marker took out.
        frames: Every GT frame given, scored or skipped, in time order.
    """

    views: Mapping[str, ViewSummary]
    time_iou: float | None
    n_scored_frames: int
    n_excluded_frames: int
    frames: tuple[ScoredFrame, ...]


def pq(scores: ViewScores) -> float | None:
    """Return PQ of one frame in one view: `SQ` x `F1_50`, and 0 on a frame with GT objects but no hit.

    None on a frame with no GT object, where `F1_50` is not defined either.
    `docs/evaluation.md` ("From frames to clips") says why a frame with no hit counts as 0.
    """
    if scores.f1_50 is None:
        return None
    if scores.sq is None:
        return 0.0
    return scores.sq * scores.f1_50


def _sum_pixels(views: Sequence[ViewScores]) -> PixelCounts:
    return PixelCounts(**{f.name: sum(getattr(v.counts, f.name) for v in views) for f in fields(PixelCounts)})


def _summarize_view(view: str, scored: Sequence[ViewScores]) -> ViewSummary:
    means: dict[str, float | None] = {}
    n_frames: dict[str, int] = {}
    for key in KEYS:
        values = [v for s in scored if (v := s.metrics()[key]) is not None]
        means[key] = sum(values) / len(values) if values else None
        n_frames[key] = len(values)
    return ViewSummary(
        view=view,
        means=MappingProxyType(means),
        n_frames=MappingProxyType(n_frames),
        pixels=_sum_pixels(scored),
        n_gt_objects=sum(s.n_gt_objects for s in scored),
        n_pred_objects=sum(s.n_pred_objects for s in scored),
        n_hits=sum(s.n_hits for s in scored),
        n_inst_bf_hits=sum(s.n_inst_bf_hits for s in scored),
    )


def summarize_clip(frames: Sequence[ScoredFrame], time_iou: float | None) -> ClipScores:
    """Average the per-frame keys over the frames each is defined on, per view.

    Args:
        frames: The clip's GT frames in time order, each given once, the
            excluded ones among them as `score_frame` returned them.
        time_iou: The clip's `time_IoU` from `evalkit.time_iou`, or None
            where it pooled nothing.

    Returns:
        The clip's `ClipScores`: a `ViewSummary` per view of `VIEWS`, the
        `time_iou` as given, the scored and excluded frame counts, and the
        frames as given.

    Raises:
        ValueError: No frame; a frame number given twice; a scored frame
            without every view of `VIEWS`; every frame excluded; or a
            `time_iou` that is not None or a number in [0, 1].
    """
    # The frames, once each, and the excluded ones counted out of them.
    if not frames:
        raise ValueError("a clip with no GT frame scores nothing; it is not summarised as zeros")
    seen: set[int] = set()
    for f in frames:
        if f.frame in seen:
            raise ValueError(f"frame {f.frame} is given twice")
        seen.add(f.frame)
    scored = [f for f in frames if not f.scores.excluded]
    if not scored:
        raise ValueError(f"every one of the {len(frames)} GT frames is excluded; the clip scores nothing")
    for f in scored:
        if set(f.scores.views) != set(VIEWS):
            raise ValueError(f"frame {f.frame} is scored in {sorted(f.scores.views)}, not in {sorted(VIEWS)}")
    if isinstance(time_iou, bool) or (time_iou is not None and not 0 <= time_iou <= 1):
        raise ValueError(f"`time_iou` is None or an IoU in [0, 1], got {time_iou!r}")

    # Per view, the means and the counts behind them.
    views = {view: _summarize_view(view, [f.scores.views[view] for f in scored]) for view in VIEWS}
    return ClipScores(
        views=MappingProxyType(views),
        time_iou=None if time_iou is None else float(time_iou),
        n_scored_frames=len(scored),
        n_excluded_frames=len(frames) - len(scored),
        frames=tuple(frames),
    )
