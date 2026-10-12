"""Measure how a condition keeps a region on each GT track over time: hold, IDF1, ID switches, fragmentation, re-entry.

A GT track follows one GT object over a clip's GT frames, in one view. The
tool builds the tracks in two ways: a class's whole region, and component
objects at each size of `MIN_PX`, linked from frame to frame. It reads the
clip and pairs the objects with the evaluator's own modules.
`docs/evaluation.md` defines the tracks and the measures ("The measures of
identity"). They are reference values and decide no claim. The tool refuses
a condition without a seed frame on a scored GT frame. The JSON records the
tool's hash beside `eval_code_sha`.

Usage:
    python -m evalkit.tools.track_metrics --dataset atlas120k --clips atlas120k_meta/clips.txt \\
        --data-root <dir of clips> --tracks-root <dir of predictions> --tag <condition> \\
        --out <condition>.identity.json
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
from evalkit.evaluate import SEED_INFO, check_table, read_population, rule_of_seed_info, versions, write_scores
from evalkit.inputs import ClipInputs, read_clip
from evalkit.keys import metric_key
from evalkit.objects import Objects, gt_objects, intersections, pair
from evalkit.scored import frame_is_excluded, scored_pixels, valid_depth
from evalkit.tools import component_scores
from evalkit.tools.component_scores import MIN_PX, component_objects, region_objects, tool_sha
from evalkit.tools.scores import PER_FRAME
from surgical_core.clip_time import frame_times

# The upper edges of the hold curve's bins, in seconds from the seed frame.
# A bin holds the samples above the edge before it, up to its own edge.
HOLD_BINS_SEC = (0.5, 1.0, 2.0, 4.0, 8.0, float("inf"))

# A piece joins a track of its class when its IoU with the track's last piece is at least this. The workbench's value.
LINK_IOU = 0.3
# A track takes a piece again after at most this many GT frames without one. The workbench's value.
BRIDGE = 3

# The measures a row holds once per view, as `measure/view`, in this order.
MEASURES = ("hold_mean", "hold_all_mean", "picked_share", "IDF1", "ID_switches", "fragmentations", "reentry")


@dataclass(frozen=True)
class FrameTracks:
    """One GT frame in one view: the GT tracks present, the region each hits, and every overlap's IoU.

    Attributes:
        time: The frame's time in seconds.
        tracks: The GT tracks present, one per GT object of the frame, in the objects' order.
        hits: Each track to the region the evaluator's pairing hits its object with. Read-only.
        n_regions: The frame's predicted objects.
        ious: Each overlapping (track, region) to their IoU over the view's scored pixels, whatever the region's
            size. Read-only.
    """

    time: float
    tracks: tuple[int, ...]
    hits: Mapping[int, int]
    n_regions: int
    ious: Mapping[tuple[int, int], float]


def tracks_on_frame(gt: Objects, track_of: Sequence[int], regions: np.ndarray, min_px: int,
                    time: float) -> FrameTracks:
    """Pair one frame's GT objects with its regions of at least `min_px` scored pixels, and name each by its track.

    Args:
        gt: The frame's GT objects in one view: whole classes from `gt_objects`, or pieces from
            `component_objects`.
        track_of: The track of each GT object, in the objects' order.
        regions: The (H, W) region ids, -1 for no region.
        min_px: The fewest scored pixels a region needs to be a predicted object.
        time: The frame's time in seconds.
    """
    every = region_objects(regions, gt.scored, 0)
    kept = region_objects(regions, gt.scored, min_px)
    inter = intersections(gt, every)
    ious = {(track_of[g], r): int(inter[g, p]) / (gt.areas[g] + every.areas[p] - int(inter[g, p]))
            for g in range(len(gt)) for p, r in enumerate(every.ids) if inter[g, p]}
    hits = {track_of[x.gt - 1]: kept.ids[x.pred - 1] for x in pair(gt, kept) if x.hit}
    return FrameTracks(time=float(time), tracks=tuple(track_of), hits=MappingProxyType(hits), n_regions=len(kept),
                       ious=MappingProxyType(ious))


def link_pieces(pieces: Sequence[Objects], link_iou: float = LINK_IOU, bridge: int = BRIDGE) -> list[tuple[int, ...]]:
    """Give each piece of each frame a track, linking it to a track of its class seen on a recent frame.

    On each frame, a track that has gone at most `bridge` frames without a piece is a candidate. Each piece and each
    candidate of its class with an IoU of at least `link_iou` form a pair, and the pairs are taken greedily, highest
    IoU first, each piece and each track once. A piece left without a pair starts a new track.

    Args:
        pieces: The frames' component objects in time order, from `component_objects`.

    Returns:
        Per frame, the track of each piece, in the pieces' order. Tracks are numbered from 0 as they start.
    """
    last: list[tuple[int, int, np.ndarray]] = []  # per track: its class, its last frame, its last piece's mask
    out = []
    for k, objs in enumerate(pieces):
        # The pairs of a piece and a recent track of its class, by IoU with the track's last piece.
        candidates = [t for t, (_, seen, _) in enumerate(last) if k - seen <= bridge + 1]
        pairs = []
        for i in range(len(objs)):
            mask = objs.mask(i + 1)
            for j, t in enumerate(candidates):
                cls, _, previous = last[t]
                if cls != objs.ids[i]:
                    continue
                union = int((mask | previous).sum())
                iou = int((mask & previous).sum()) / union if union else 0.0
                if iou >= link_iou:
                    pairs.append((iou, i, j))
        # Greedy, highest IoU first; equal IoUs keep the order of the pieces, then of the tracks.
        pairs.sort(key=lambda x: -x[0])
        track: list[int | None] = [None] * len(objs)
        taken: set[int] = set()
        for _, i, j in pairs:
            if track[i] is not None or j in taken:
                continue
            track[i] = candidates[j]
            taken.add(j)
        # Every piece left over starts a track; every piece updates its track's last piece.
        for i in range(len(objs)):
            if track[i] is None:
                track[i] = len(last)
                last.append((objs.ids[i], k, objs.mask(i + 1)))
            else:
                last[track[i]] = (objs.ids[i], k, objs.mask(i + 1))
        out.append(tuple(track))
    return out


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def hold(frames: Sequence[FrameTracks], seed: int) -> dict:
    """Follow every track present on the seed frame with the region picked for it there.

    The region picked for a track is the one the seed frame's pairing hits it with. On every other frame on which
    the track is present, the track gives one sample: the IoU of the picked region with the track's object, 0
    where the two do not overlap, and 0 for a track with no picked region.

    Args:
        frames: The clip's GT frames in one view, in time order.
        seed: The seed frame's place in `frames`.

    Returns:
        `hold_mean`, the mean over the picked tracks' samples; `hold_all_mean`, over every sample;
        `picked_share`, the share of the seed frame's tracks that have a picked region; each None when it
        averages nothing. `hold_curve`, the samples of `hold_mean` in `HOLD_BINS_SEC` by their distance in
        seconds from the seed frame. And the counts behind them.
    """
    # One sample per track present on the seed frame, on every other frame the track is present on.
    start = frames[seed]
    picked = dict(start.hits)
    samples: list[tuple[float, float, bool]] = []
    for k, f in enumerate(frames):
        if k == seed:
            continue
        for t in start.tracks:
            if t in f.tracks:
                iou = f.ious.get((t, picked[t]), 0.0) if t in picked else 0.0
                samples.append((abs(f.time - start.time), iou, t in picked))

    # The picked tracks' samples, binned by their distance in seconds from the seed frame.
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
        "picked_share": len(picked) / len(start.tracks) if start.tracks else None,
        "hold_curve": curve,
        "n_seed_tracks": len(start.tracks), "n_picked": len(picked),
        "n_hold_samples": len(of_picked), "n_hold_all_samples": len(samples),
    }


def id_true_positives(frames: Sequence[FrameTracks]) -> int:
    """Return the most frames on which a track is hit by its region, giving each track one region and each region one track."""
    together: dict[tuple[int, int], int] = {}
    for f in frames:
        for t, r in f.hits.items():
            together[(t, r)] = together.get((t, r), 0) + 1
    if not together:
        return 0
    # scipy is the `tools` extra; the module imports without it.
    from scipy.optimize import linear_sum_assignment

    tracks = sorted({t for t, _ in together})
    regions = sorted({r for _, r in together})
    counts = np.zeros((len(tracks), len(regions)), dtype=np.int64)
    for (t, r), n in together.items():
        counts[tracks.index(t), regions.index(r)] = n
    rows, cols = linear_sum_assignment(counts, maximize=True)
    return int(counts[rows, cols].sum())


def identity(frames: Sequence[FrameTracks]) -> dict:
    """Count each track's ID switches and fragmentations, and compute IDF1 over the clip.

    An ID switch is a hit by another region than the track's previous hit. A fragmentation is a frame on which the
    track is present and not hit, when its previous frame with the track was a hit. Both walk the frames on which
    the track is present, in time order.

    Returns:
        `IDF1` = 2 x `id_true_positives` / (GT objects + predicted objects, summed over the frames), None
        when no frame has a GT object; `ID_switches` and `fragmentations`, summed over the tracks; and the
        counts behind them.
    """
    # ID switches and fragmentations, track by track.
    tracks = sorted({t for f in frames for t in f.tracks})
    switches = fragmentations = 0
    for t in tracks:
        last_region, last_hit = None, False
        for f in frames:
            if t not in f.tracks:
                continue
            region = f.hits.get(t)
            if region is None:
                fragmentations += last_hit
            else:
                switches += last_region is not None and region != last_region
                last_region = region
            last_hit = region is not None

    # IDF1, over every frame's objects.
    n_gt = sum(len(f.tracks) for f in frames)
    n_pred = sum(f.n_regions for f in frames)
    tp = id_true_positives(frames)
    return {
        "IDF1": 2 * tp / (n_gt + n_pred) if n_gt else None,
        "ID_switches": switches, "fragmentations": fragmentations,
        "n_tracks": len(tracks), "n_gt_objects": n_gt, "n_pred_objects": n_pred, "id_true_positives": tp,
    }


def reentry(frames: Sequence[FrameTracks]) -> dict:
    """Find each track's gaps, and the share of them whose two ends are hit by the same region.

    A gap is a run of frames on which a track is absent, between two frames on which it is present. A gap
    whose two ends are not both hits is counted apart and left out of the share.

    Returns:
        `reentry`, that share, None when no gap has two hit ends; `n_gaps`, the gaps with two hit ends;
        `n_gaps_recovered`, those hit by one region at both ends; and `n_gaps_unmatched`, the others.
    """
    gaps = recovered = unmatched = 0
    for t in sorted({t for f in frames for t in f.tracks}):
        present = [k for k, f in enumerate(frames) if t in f.tracks]
        for a, b in zip(present, present[1:]):
            if b == a + 1:
                continue
            before, after = frames[a].hits.get(t), frames[b].hits.get(t)
            if before is None or after is None:
                unmatched += 1
                continue
            gaps += 1
            recovered += before == after
    return {"reentry": recovered / gaps if gaps else None, "n_gaps": gaps, "n_gaps_recovered": recovered,
            "n_gaps_unmatched": unmatched}


def view_measures(frames: Sequence[FrameTracks], seed: int, view: str) -> tuple[dict, list, dict]:
    """Return one view's measures as `measure/view`, its hold curve, and the counts behind them."""
    values = {**hold(frames, seed), **identity(frames), **reentry(frames)}
    keyed = {metric_key(m, view): values.pop(m) for m in MEASURES}
    curve = values.pop("hold_curve")
    return keyed, curve, values


def measure_clip(inputs: ClipInputs, table: ClassTable, seed: int, times: np.ndarray,
                 min_px: Sequence[int] = MIN_PX) -> dict:
    """Measure one clip in every view, from its seed frame, on whole-class tracks and component tracks.

    Args:
        inputs: The clip, as `read_clip` read it.
        table: The class table it was read with.
        seed: The index of the seed frame, as `seed_info.json` records it.
        times: Every frame's time in seconds, by index, from `surgical_core.clip_time`.
        min_px: The sizes of the component objects.

    Returns:
        The clip's name, `seed_frame`, the GT frames measured and excluded; on whole-class tracks each measure
        of `MEASURES` as `measure/view`, and by view the `hold_curve` and the counts behind the measures; and the
        same under `by_min_px`, per size, on component tracks.

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
    start = kept.index(seed)

    row: dict = {"clip": inputs.clip, "seed_frame": seed, "n_gt_frames": len(kept),
                 "n_excluded_frames": len(gt) - len(kept), "hold_curve": {}, "counts": {},
                 "by_min_px": {str(m): {"hold_curve": {}, "counts": {}} for m in min_px}}
    for view in VIEWS:
        scored = [scored_pixels(inputs.gt[i], table, view, valid[i]) for i in kept]

        # Whole-class tracks: each class is its own track.
        frames = []
        for s, i in zip(scored, kept):
            objs = gt_objects(s.classes, s.mask)
            frames.append(tracks_on_frame(objs, objs.ids, inputs.regions[i], 0, times[i]))
        keyed, row["hold_curve"][view], row["counts"][view] = view_measures(frames, start, view)
        row.update(keyed)

        # Component tracks: at each size, the pieces linked from frame to frame.
        for m in min_px:
            pieces = [component_objects(s.classes, s.mask, m) for s in scored]
            frames = [tracks_on_frame(p, track_of, inputs.regions[i], m, times[i])
                      for p, track_of, i in zip(pieces, link_pieces(pieces), kept)]
            block = row["by_min_px"][str(m)]
            keyed, block["hold_curve"][view], block["counts"][view] = view_measures(frames, start, view)
            block.update(keyed)
    return row


def seed_frames(tracks_root: str | Path, tag: str, clips: Sequence[str]) -> tuple[dict[str, int], str]:
    """Return each clip's seed frame and the condition's propagation rule, from every clip's `seed_info.json`.

    Raises:
        FileNotFoundError: A clip has no `seed_info.json`.
        ValueError: A record holds no rule (`rule_of_seed_info`), which refuses a seed from GT or from outside
            the tracking stage; the rule is `per_frame`, which has no seed frame and whose ids do not carry
            over; or two clips hold two rules.
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
                      tracks_root: str | Path, tag: str, min_px: Sequence[int] = MIN_PX) -> dict:
    """Measure one condition on every clip, and return its JSON as a dict.

    The JSON records what the evaluator records with a score: `eval_code_sha`, whose modules read the clips
    and pair the objects, the input hashes and the propagation rule. It also records the hashes of this file and
    of `component_scores`. It lacks `pilot`, so the score tools refuse it as a score.

    Raises:
        ValueError, KeyError, FileNotFoundError, RuntimeError: The condition's seeds hold no usable rule
            (`seed_frames`), or a clip cannot be read, timed or measured. Nothing is returned for the others.
    """
    table = load_table(dataset, class_set)
    check_table(table)
    seeds, rule = seed_frames(tracks_root, tag, clips)
    rows, shas = [], {}
    for clip in clips:
        inputs = read_clip(data_root, tracks_root, tag, clip, table)
        rows.append(measure_clip(inputs, table, seeds[clip], frame_times(str(data_root), clip), min_px))
        shas[clip] = dict(inputs.shas)
    return {
        "tool_sha": tool_sha(__file__, component_scores.__file__), "eval_code_sha": eval_code_sha(),
        "dataset": table.dataset, "class_set": table.class_set, "views": list(VIEWS), "min_px": list(min_px),
        "track_dir_name": tag, "clips": list(clips), "propagation": rule, "input_shas": shas,
        "versions": versions(), "per_clip": rows,
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
