"""Measure how a condition keeps a region on each GT class over time: hold, IDF1, ID switches, fragmentation, re-entry.

The tool follows each class over a clip's GT frames, as one GT track. It
reads the clip, takes each frame's objects and pairs them with the
evaluator's own modules, in each view, with no minimum object size. A hit
links a class to a region on that frame. `docs/evaluation.md` defines the
measures ("The measures of identity"). They are reference values and
decide no claim. The tool refuses a `per_frame` condition, and a clip whose
seed frame is not a scored GT frame.

Usage:
    python -m evalkit.tools.track_metrics --dataset atlas120k --clips atlas120k_meta/clips.txt \\
        --data-root <dir of clips> --tracks-root <dir of predictions> --tag <condition> --out <condition>.json
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from evalkit.classes import CLASS_SETS, VIEWS, ClassTable, load_table
from evalkit.code_sha import eval_code_sha
from evalkit.evaluate import SEED_INFO, read_population, rule_of_seed_info, versions, write_scores
from evalkit.inputs import ClipInputs, read_clip
from evalkit.keys import metric_key
from evalkit.objects import gt_objects, intersections, pair, predicted_objects
from evalkit.scored import frame_is_excluded, scored_pixels, valid_depth
from evalkit.tools.scores import PER_FRAME
from surgical_core.clip_time import frame_times

# The upper edges of the hold curve's bins, in seconds from the seed frame.
# A bin holds the samples above the edge before it, up to its own edge.
HOLD_BINS_SEC = (0.5, 1.0, 2.0, 4.0, 8.0, float("inf"))

# The measures a row holds once per view, as `measure/view`, in this order.
MEASURES = ("hold_mean", "hold_all_mean", "picked_share", "IDF1", "ID_switches", "fragmentations", "reentry")


@dataclass(frozen=True)
class FrameTracks:
    """One GT frame in one view: the classes present, the region each class hits, and every overlap's IoU.

    Attributes:
        time: The frame's time in seconds.
        classes: The classes present, which are the frame's GT objects, ascending.
        hits: Each class to the region the evaluator's pairing hits it with. Read-only.
        n_regions: The frame's predicted objects.
        ious: Each overlapping (class, region) to their IoU over the view's scored pixels. Read-only.
    """

    time: float
    classes: tuple[int, ...]
    hits: Mapping[int, int]
    n_regions: int
    ious: Mapping[tuple[int, int], float]


def frame_tracks(mask_ids: np.ndarray, regions: np.ndarray, valid: np.ndarray, table: ClassTable, view: str,
                 time: float) -> FrameTracks:
    """Pair one frame's classes with its regions in one view, as the evaluator pairs its objects.

    Args:
        mask_ids: The (H, W) ids the GT mask holds, as `ClassTable.mask_ids` returns them.
        regions: The (H, W) region ids, -1 for no region.
        valid: The (H, W) mask `valid_depth` returned.
        table: The dataset's class table, in the class set to measure.
        view: One of `VIEWS`.
        time: The frame's time in seconds.

    Raises:
        ValueError, KeyError: `scored_pixels` or the object modules refuse the frame, as the evaluator does.
    """
    scored = scored_pixels(mask_ids, table, view, valid)
    gt = gt_objects(scored.classes, scored.mask)
    pred = predicted_objects(regions, scored.mask)
    inter = intersections(gt, pred)
    ious = {(c, r): int(inter[g, p]) / (gt.areas[g] + pred.areas[p] - int(inter[g, p]))
            for g, c in enumerate(gt.ids) for p, r in enumerate(pred.ids) if inter[g, p]}
    hits = {gt.ids[x.gt - 1]: pred.ids[x.pred - 1] for x in pair(gt, pred) if x.hit}
    return FrameTracks(time=float(time), classes=gt.ids, hits=MappingProxyType(hits), n_regions=len(pred),
                       ious=MappingProxyType(ious))


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def hold(frames: Sequence[FrameTracks], seed: int) -> dict:
    """Follow every class present on the seed frame with the region picked for it there.

    The region picked for a class is the one the seed frame's pairing hits it with. On every other frame on
    which the class is present, the class gives one sample: the IoU of the picked region with the class, 0
    where the two do not overlap, and 0 for a class with no picked region.

    Args:
        frames: The clip's GT frames in one view, in time order.
        seed: The seed frame's place in `frames`.

    Returns:
        `hold_mean`, the mean over the picked classes' samples; `hold_all_mean`, over every sample;
        `picked_share`, the share of the seed frame's classes that have a picked region; each None when it
        averages nothing. `hold_curve`, the samples of `hold_mean` in `HOLD_BINS_SEC` by their distance in
        seconds from the seed frame. And the counts behind them.
    """
    # One sample per class present on the seed frame, on every other frame the class is present on.
    start = frames[seed]
    picked = dict(start.hits)
    samples: list[tuple[float, float, bool]] = []
    for k, f in enumerate(frames):
        if k == seed:
            continue
        for c in start.classes:
            if c in f.classes:
                iou = f.ious.get((c, picked[c]), 0.0) if c in picked else 0.0
                samples.append((abs(f.time - start.time), iou, c in picked))

    # The picked classes' samples, binned by their distance in seconds from the seed frame.
    of_picked = [(dt, iou) for dt, iou, is_picked in samples if is_picked]
    curve, low = [], 0.0
    for high in HOLD_BINS_SEC:
        in_bin = [iou for dt, iou in of_picked if low < dt <= high]
        curve.append({"above_sec": low, "to_sec": None if np.isinf(high) else high, "n": len(in_bin),
                      "mean": _mean(in_bin)})
        low = high
    return {
        "hold_mean": _mean([iou for _, iou in of_picked]),
        "hold_all_mean": _mean([iou for _, iou, _ in samples]),
        "picked_share": len(picked) / len(start.classes) if start.classes else None,
        "hold_curve": curve,
        "n_seed_classes": len(start.classes), "n_picked": len(picked),
        "n_hold_samples": len(of_picked), "n_hold_all_samples": len(samples),
    }


def id_true_positives(frames: Sequence[FrameTracks]) -> int:
    """Return the most frames on which a class is hit by its region, with one region per class over the clip."""
    together: dict[tuple[int, int], int] = {}
    for f in frames:
        for c, r in f.hits.items():
            together[(c, r)] = together.get((c, r), 0) + 1
    if not together:
        return 0
    # scipy is the `tools` extra; the module imports without it.
    from scipy.optimize import linear_sum_assignment

    classes = sorted({c for c, _ in together})
    regions = sorted({r for _, r in together})
    counts = np.zeros((len(classes), len(regions)), dtype=np.int64)
    for (c, r), n in together.items():
        counts[classes.index(c), regions.index(r)] = n
    rows, cols = linear_sum_assignment(counts, maximize=True)
    return int(counts[rows, cols].sum())


def identity(frames: Sequence[FrameTracks]) -> dict:
    """Count each class's ID switches and fragmentations, and compute IDF1 over the clip.

    An ID switch is a hit by another region than the class's previous hit. A fragmentation is a frame on
    which the class is present and not hit, after one on which it was hit. Both walk the frames on which the
    class is present, in time order.

    Returns:
        `IDF1` = 2 x `id_true_positives` / (GT objects + predicted objects, summed over the frames), None
        when no frame has a GT object; `ID_switches` and `fragmentations`, summed over the classes; and the
        counts behind them.
    """
    # ID switches and fragmentations, class by class.
    classes = sorted({c for f in frames for c in f.classes})
    switches = fragmentations = 0
    for c in classes:
        last_region, last_hit = None, False
        for f in frames:
            if c not in f.classes:
                continue
            region = f.hits.get(c)
            if region is None:
                fragmentations += last_hit
            else:
                switches += last_region is not None and region != last_region
                last_region = region
            last_hit = region is not None

    # IDF1, over every frame's objects.
    n_gt = sum(len(f.classes) for f in frames)
    n_pred = sum(f.n_regions for f in frames)
    tp = id_true_positives(frames)
    return {
        "IDF1": 2 * tp / (n_gt + n_pred) if n_gt else None,
        "ID_switches": switches, "fragmentations": fragmentations,
        "n_classes": len(classes), "n_gt_objects": n_gt, "n_pred_objects": n_pred, "id_true_positives": tp,
    }


def reentry(frames: Sequence[FrameTracks]) -> dict:
    """Find each class's gaps, and the share of them whose two ends are hit by the same region.

    A gap is a run of frames on which a class is absent, between two frames on which it is present. A gap
    whose two ends are not both hits is counted apart and left out of the share.

    Returns:
        `reentry`, that share, None when no gap has two hit ends; `n_gaps`, the gaps with two hit ends;
        `n_gaps_recovered`, those hit by one region at both ends; and `n_gaps_unmatched`, the others.
    """
    gaps = recovered = unmatched = 0
    for c in sorted({c for f in frames for c in f.classes}):
        present = [k for k, f in enumerate(frames) if c in f.classes]
        for a, b in zip(present, present[1:]):
            if b == a + 1:
                continue
            before, after = frames[a].hits.get(c), frames[b].hits.get(c)
            if before is None or after is None:
                unmatched += 1
                continue
            gaps += 1
            recovered += before == after
    return {"reentry": recovered / gaps if gaps else None, "n_gaps": gaps, "n_gaps_recovered": recovered,
            "n_gaps_unmatched": unmatched}


def measure_clip(inputs: ClipInputs, table: ClassTable, seed: int, times: np.ndarray) -> dict:
    """Measure one clip in every view, from its seed frame, and return its row.

    Args:
        inputs: The clip, as `read_clip` read it.
        table: The class table it was read with.
        seed: The index of the seed frame, as `seed_info.json` records it.
        times: Every frame's time in seconds, by index, from `surgical_core.clip_time`.

    Returns:
        The clip's name, `seed_frame`, the GT frames measured and excluded, each measure of `MEASURES` as
        `measure/view`, and by view the `hold_curve` and the counts behind the measures.

    Raises:
        ValueError: The seed frame is not a GT frame, or is excluded; or a frame is refused as the evaluator
            refuses it.
    """
    # The GT frames in time order: every depth map checked, then the excluded frames left out.
    gt = [i for i in inputs.order if i in inputs.gt]
    valid = {i: valid_depth(inputs.depth[i]) for i in gt}
    kept = [i for i in gt if not frame_is_excluded(inputs.gt[i], table)]
    if seed not in kept:
        why = "an excluded frame" if seed in gt else "not a GT frame"
        raise ValueError(f"{inputs.clip}: the seed frame {seed} is {why}; hold starts on a scored GT frame")

    # Each view's frames, and the measures over them.
    row: dict = {"clip": inputs.clip, "seed_frame": seed, "n_gt_frames": len(kept),
                 "n_excluded_frames": len(gt) - len(kept), "hold_curve": {}, "counts": {}}
    for view in VIEWS:
        frames = [frame_tracks(inputs.gt[i], inputs.regions[i], valid[i], table, view, times[i]) for i in kept]
        values = {**hold(frames, kept.index(seed)), **identity(frames), **reentry(frames)}
        row.update({metric_key(m, view): values.pop(m) for m in MEASURES})
        row["hold_curve"][view] = values.pop("hold_curve")
        row["counts"][view] = values
    return row


def seed_frames(tracks_root: str | Path, tag: str, clips: Sequence[str]) -> tuple[dict[str, int], str]:
    """Return each clip's seed frame and the condition's propagation rule, from every clip's `seed_info.json`.

    Raises:
        FileNotFoundError: A clip has no `seed_info.json`.
        ValueError: A record holds no rule (`rule_of_seed_info`); the rule is `per_frame`, which has no seed
            frame and whose ids do not carry over; or two clips hold two rules.
    """
    seeds, rules = {}, set()
    for clip in clips:
        info = json.loads((Path(tracks_root) / clip / tag / SEED_INFO).read_text(encoding="utf-8"))
        try:
            rule = rule_of_seed_info(info)
        except ValueError as e:
            raise ValueError(f"{clip}: {tag}'s {SEED_INFO} holds no propagation rule: {e}") from None
        if rule == PER_FRAME:
            raise ValueError(f"{clip}: {tag} is segmented frame by frame; it has no seed frame to hold from, "
                             "and its ids do not carry over")
        rules.add(rule)
        seeds[clip] = int(info["seed_frame"])
    if len(rules) > 1:
        raise ValueError(f"{tag}: the clips hold {sorted(rules)}; a condition has one rule")
    return seeds, rules.pop()


def measure_condition(dataset: str, class_set: str | None, clips: Sequence[str], data_root: str | Path,
                      tracks_root: str | Path, tag: str) -> dict:
    """Measure one condition on every clip, and return its JSON as a dict.

    The JSON records what the evaluator records with a score: `eval_code_sha`, whose modules read the clips
    and pair the objects, the input hashes and the propagation rule. It lacks `pilot`, so the score tools
    refuse it as a score.

    Raises:
        ValueError, KeyError, FileNotFoundError, RuntimeError: The condition's seeds hold no usable rule
            (`seed_frames`), or a clip cannot be read, timed or measured. Nothing is returned for the others.
    """
    table = load_table(dataset, class_set)
    seeds, rule = seed_frames(tracks_root, tag, clips)
    rows, shas = [], {}
    for clip in clips:
        inputs = read_clip(data_root, tracks_root, tag, clip, table)
        rows.append(measure_clip(inputs, table, seeds[clip], frame_times(str(data_root), clip)))
        shas[clip] = dict(inputs.shas)
    return {
        "eval_code_sha": eval_code_sha(), "dataset": table.dataset, "class_set": table.class_set,
        "views": list(VIEWS), "track_dir_name": tag, "clips": list(clips), "propagation": rule,
        "input_shas": shas, "versions": versions(), "per_clip": rows,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Measure the condition the arguments name, and write its JSON."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, choices=sorted(CLASS_SETS))
    ap.add_argument("--class-set", default=None, help="original (default) or, for atlas120k, benchmark")
    ap.add_argument("--clips", required=True, help="the population: one clip name per line")
    ap.add_argument("--data-root", required=True, help="one directory per clip: manifest, depth, GT masks")
    ap.add_argument("--tracks-root", required=True, help="one directory per clip, one per condition inside")
    ap.add_argument("--tag", required=True, help="the condition's directory name under each clip")
    ap.add_argument("--out", required=True, help="the JSON to write")
    args = ap.parse_args(argv)
    clips = read_population(args.clips)
    write_scores(measure_condition(args.dataset, args.class_set, clips, args.data_root, args.tracks_root, args.tag),
                 args.out)
    print(f"{args.tag}: {len(clips)} clips measured, {args.out}")


if __name__ == "__main__":
    main()
