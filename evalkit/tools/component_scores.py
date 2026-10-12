"""Score a condition on component objects at each minimum size of `MIN_PX`: F1_50, SQ and inst_BF per view.

A component object is one 8-connected piece of one class's scored pixels
or one region, kept at a minimum size on both sides; `docs/evaluation.md`
defines it ("Component objects"). The tool reads each clip, takes each
view's scored pixels, and pairs and scores the objects with the
evaluator's own modules. Only the objects differ from the evaluator's. The
values are reference values and decide no claim. The JSON records the
hash of this file beside `eval_code_sha`, since no hash covers a tool.

Usage:
    python -m evalkit.tools.component_scores --dataset atlas120k --clips atlas120k_meta/clips.txt \\
        --data-root <dir of clips> --tracks-root <dir of predictions> --tag <condition> \\
        --out <condition>.components.json [--propagation <rule>]
"""
from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from evalkit.classes import CLASS_SETS, VIEWS, ClassTable, load_table
from evalkit.code_sha import eval_code_sha
from evalkit.evaluate import PROPAGATION_RULES, check_table, condition_rule, read_population, versions, write_scores
from evalkit.inputs import ClipInputs, read_clip
from evalkit.inst_bf import instance_boundary_f
from evalkit.keys import metric_key
from evalkit.objects import Objects, instance_scores
from evalkit.scored import frame_is_excluded, scored_pixels, valid_depth

# The minimum sizes of a component object, in pixels at the evaluation resolution. Fixed before any condition is
# scored on them: 0 keeps every piece, 300 is the pilot evaluator's size, 400 the workbench's tracking measures'
# and the tracking stage's smallest seed region, on the same pixel grid.
MIN_PX = (0, 100, 300, 400)

# The keys a row holds per minimum size, as `measure/view`, in this order.
MEASURES = ("F1_50", "SQ", "inst_BF")


def tool_sha(*paths: str | Path) -> dict[str, str]:
    """Return the sha256 of each file's bytes, by file name, for a JSON to say which tool code wrote it."""
    return {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


def component_objects(classes: np.ndarray, scored: np.ndarray, min_px: int) -> Objects:
    """Return each class's 8-connected pieces of scored pixels with at least `min_px` pixels, as GT objects.

    Args:
        classes: The (H, W) GT class map, read under `scored`.
        scored: The (H, W) bool mask of the view's scored pixels.
        min_px: The fewest pixels a piece keeps.

    Returns:
        The objects, class by class ascending and, within a class, in `cv2.connectedComponents` label order.
        `ids` holds each object's class id, so the pieces of one class share an id.
    """
    index = np.zeros(classes.shape, dtype=np.int32)
    ids, areas = [], []
    for c in np.unique(classes[scored]).tolist():
        n, labels = cv2.connectedComponents(((classes == c) & scored).astype(np.uint8), connectivity=8)
        counts = np.bincount(labels.ravel(), minlength=n)
        for piece in range(1, n):
            if int(counts[piece]) >= min_px:
                ids.append(int(c))
                areas.append(int(counts[piece]))
                index[labels == piece] = len(areas)
    return Objects(index=index, ids=tuple(ids), areas=tuple(areas), scored=scored)


def region_objects(regions: np.ndarray, scored: np.ndarray, min_px: int) -> Objects:
    """Return each region with at least `min_px` scored pixels, as a predicted object of its scored pixels.

    Raises:
        ValueError: A region id below -1.
    """
    if regions.min(initial=0) < -1:
        raise ValueError("a region map holds region ids >= 0 and -1 for no region")
    member = scored & (regions >= 0)
    present, counts = np.unique(regions[member], return_counts=True)
    keep = counts >= min_px
    index = np.zeros(regions.shape, dtype=np.int32)
    for k, r in enumerate(present[keep].tolist(), start=1):
        index[member & (regions == r)] = k
    return Objects(index=index, ids=tuple(int(r) for r in present[keep].tolist()),
                   areas=tuple(int(a) for a in counts[keep].tolist()), scored=scored)


def score_view(mask_ids: np.ndarray, regions: np.ndarray, valid: np.ndarray, table: ClassTable, view: str,
               min_px: Sequence[int]) -> dict[int, dict]:
    """Score one frame in one view at each minimum size.

    Returns:
        Each minimum size to `F1_50`, `SQ` and `inst_BF`, each None where the evaluator leaves it undefined, and
        the counts behind them: `gt`, `predicted`, `hits` and `inst_bf_hits`.

    Raises:
        ValueError, KeyError: `scored_pixels` refuses the frame, as the evaluator does.
    """
    scored = scored_pixels(mask_ids, table, view, valid)
    out = {}
    for m in min_px:
        gt = component_objects(scored.classes, scored.mask, m)
        pred = region_objects(regions, scored.mask, m)
        inst = instance_scores(gt, pred)
        contours = instance_boundary_f(gt, pred, inst)
        out[m] = {"F1_50": inst.f1_50, "SQ": inst.sq, "inst_BF": contours.inst_bf, "gt": inst.n_gt,
                  "predicted": inst.n_pred, "hits": inst.n_hits, "inst_bf_hits": contours.n_entered}
    return out


def measure_clip(inputs: ClipInputs, table: ClassTable, min_px: Sequence[int] = MIN_PX) -> dict:
    """Score every GT frame of one clip, and average each key over the frames it is defined on.

    Returns:
        The clip's row: its name, the GT frames scored and excluded, and per minimum size each key of `MEASURES`
        as `measure/view`, the frames each mean covers, and the object counts summed over the frames.

    Raises:
        ValueError, KeyError: A frame's depth or GT is refused, as the evaluator refuses it; or every GT frame
            is excluded.
    """
    # Every GT frame in time order: its depth checked, then skipped if excluded, else scored in every view.
    frames, n_excluded = [], 0
    for i in (i for i in inputs.order if i in inputs.gt):
        valid = valid_depth(inputs.depth[i])
        if frame_is_excluded(inputs.gt[i], table):
            n_excluded += 1
            continue
        frames.append({view: score_view(inputs.gt[i], inputs.regions[i], valid, table, view, min_px)
                       for view in VIEWS})
    if not frames:
        raise ValueError(f"{inputs.clip}: every GT frame is excluded, so nothing is scored")

    # Each key's mean over the frames it is defined on, and the counts summed, per minimum size and view.
    row: dict = {"clip": inputs.clip, "n_scored_frames": len(frames), "n_excluded_frames": n_excluded,
                 "by_min_px": {}}
    for m in min_px:
        means, n_frames, objects = {}, {}, {}
        for view in VIEWS:
            for measure in MEASURES:
                values = [v for f in frames if (v := f[view][m][measure]) is not None]
                means[metric_key(measure, view)] = float(np.mean(values)) if values else None
                n_frames[metric_key(measure, view)] = len(values)
            objects[view] = {k: sum(f[view][m][k] for f in frames) for k in ("gt", "predicted", "hits", "inst_bf_hits")}
        row["by_min_px"][str(m)] = {**means, "n_frames": n_frames, "objects": objects}
    return row


def measure_condition(dataset: str, class_set: str | None, clips: Sequence[str], data_root: str | Path,
                      tracks_root: str | Path, tag: str, propagation: str | None = None,
                      min_px: Sequence[int] = MIN_PX) -> dict:
    """Score one condition on every clip, and return its JSON as a dict.

    The JSON records what the evaluator records with a score, and the hash of this file. It lacks `pilot`, so the
    score tools refuse it as a score.

    Raises:
        ValueError, KeyError, FileNotFoundError: The condition's propagation rule is unknown (`condition_rule`), or a
            clip cannot be read or scored. Nothing is returned for the others.
    """
    table = load_table(dataset, class_set)
    check_table(table)
    rule = condition_rule(tracks_root, tag, clips, propagation)
    # Both hashes before any clip is scored: a source file edited during a long run would otherwise label the
    # values of the code in memory with the hash of the code on disk.
    tool, sha = tool_sha(__file__), eval_code_sha()
    rows, shas = [], {}
    for clip in clips:
        inputs = read_clip(data_root, tracks_root, tag, clip, table)
        rows.append(measure_clip(inputs, table, min_px))
        shas[clip] = dict(inputs.shas)
    return {
        "tool_sha": tool, "eval_code_sha": sha, "dataset": table.dataset,
        "class_set": table.class_set, "views": list(VIEWS), "min_px": list(min_px), "track_dir_name": tag,
        "clips": list(clips), "propagation": rule, "input_shas": shas, "versions": versions(), "per_clip": rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Score the condition the arguments name, and write its JSON."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, choices=sorted(CLASS_SETS))
    ap.add_argument("--class-set", default=None, help="original (default) or, for atlas120k, benchmark")
    ap.add_argument("--clips", required=True, help="the population: one clip name per line")
    ap.add_argument("--data-root", required=True, help="one directory per clip: manifest, depth, GT masks")
    ap.add_argument("--tracks-root", required=True, help="one directory per clip, one per condition inside")
    ap.add_argument("--tag", required=True, help="the condition's directory name under each clip")
    ap.add_argument("--out", required=True, help="the JSON to write")
    ap.add_argument("--propagation", choices=PROPAGATION_RULES, default=None,
                    help="the condition's propagation rule, for labels that carry no seed_info.json")
    args = ap.parse_args(argv)
    clips = read_population(args.clips)
    write_scores(measure_condition(args.dataset, args.class_set, clips, args.data_root, args.tracks_root, args.tag,
                                   args.propagation), args.out)
    print(f"{args.tag}: {len(clips)} clips scored on component objects, {args.out}")


if __name__ == "__main__":
    main()
