"""Sample each reference track of one clip: the tracked region's IoU and the fixed mask's IoU on later frames.

A reference track is one class's whole region over the clip's scored reference frames, in one view. The
tracked region of a track is the region that the seed frame's pairing (`evalkit.objects.pair`, IoU at
least 0.5) hits the class's object with; a track that no region hits is not picked. Each other scored
reference frame that holds the class gives the track one sample: the IoU of the tracked region with the
class's object there, 0 when the region has vanished. The fixed-mask sample pastes the class's seed-frame
object onto that frame, keeps its scored pixels, and takes the IoU with the class's object.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from evalkit.classes import ClassTable
from evalkit.evaluate import SEED_INFO
from evalkit.inputs import read_clip
from evalkit.objects import gt_objects
from evalkit.scored import frame_is_excluded, scored_pixels, valid_depth
from evalkit.tools.paired_stats import video_of
from evalkit.tools.track_metrics import tracks_on_frame


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Return the IoU of two boolean masks, 0 when both are empty."""
    inter = int((a & b).sum())
    union = int(a.sum()) + int(b.sum()) - inter
    return inter / union if union else 0.0


def clip_tracks(data_root: str | Path, tracks_root: str | Path, tag: str, clip: str, table: ClassTable,
                views: tuple[str, ...]) -> dict:
    """Return the clip's reference tracks in each view, with their samples.

    Args:
        data_root: The extracted clips, one directory each.
        tracks_root: The tracked labels, one directory per clip holding one directory per condition.
        tag: The condition's directory under each clip, with the tracking stage's `seed_info.json`.
        clip: The clip's name.
        table: The dataset's class table.
        views: The views to sample, from `evalkit.classes.VIEWS`.

    Returns:
        `clip`, `video`, and under `views` one list per view of the classes on the seed frame, by class id:
        `class`, `type`, `picked`, `tracked` (one IoU per later frame that holds the class, in time order)
        and `fixed` (the fixed mask's IoU on the same frames).

    Raises:
        ValueError: The seed frame is not a scored reference frame, or `read_clip` refuses the clip.
    """
    # The clip as the evaluator reads it, and the seed frame the tracking stage recorded.
    inputs = read_clip(data_root, tracks_root, tag, clip, table)
    seed = int(json.loads((Path(tracks_root) / clip / tag / SEED_INFO).read_text(encoding="utf-8"))["seed_frame"])

    # The scored reference frames in time order: every depth map checked, the excluded frames left out.
    reference = [i for i in inputs.order if i in inputs.gt]
    valid = {i: valid_depth(inputs.depth[i]) for i in reference}
    kept = [i for i in reference if not frame_is_excluded(inputs.gt[i], table)]
    if seed not in kept:
        raise ValueError(f"{clip}: the seed frame {seed} is not a scored reference frame")
    start = kept.index(seed)

    out: dict = {"clip": clip, "video": video_of(clip), "views": {}}
    for view in views:
        # Each frame's objects, paired with its regions. The samples need no time, so none is passed.
        objects = [gt_objects(s.classes, s.mask)
                   for s in (scored_pixels(inputs.gt[i], table, view, valid[i]) for i in kept)]
        frames = [tracks_on_frame(o, o.ids, inputs.regions[i], 0, 0.0) for o, i in zip(objects, kept)]

        # One track per class on the seed frame, sampled on every other frame that holds the class.
        first, seed_objects = frames[start], objects[start]
        tracks = []
        for g, c in enumerate(first.tracks):
            region = first.hits.get(c)
            fixed_mask = seed_objects.mask(g + 1)
            tracked, fixed = [], []
            for k, (f, o) in enumerate(zip(frames, objects)):
                if k == start or c not in f.tracks:
                    continue
                tracked.append(f.ious.get((c, region), 0.0) if region is not None else 0.0)
                fixed.append(mask_iou(fixed_mask & o.scored, o.mask(o.ids.index(c) + 1)))
            tracks.append({"class": int(c), "type": str(table.type_of(c)), "picked": region is not None,
                           "tracked": tracked, "fixed": fixed})
        out["views"][view] = tracks
    return out
