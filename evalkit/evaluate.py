"""The evaluator's entry point: score one condition on every clip of a population, and write its score JSON.

For each clip it reads the inputs (`evalkit.inputs`), scores every GT frame
in every view (`evalkit.frame`), pools `time_IoU` over every frame with a
prediction, and averages over the frames (`evalkit.clip`). The JSON holds
what `docs/evaluation.md` records with every score, one row per clip, and
the per-frame values, with the condition's propagation rule as its tracker
recorded it. A clip that cannot be scored stops the run, and no JSON is
written: a population is scored whole or not at all.

Usage:
    python -m evalkit.evaluate --dataset cholecseg8k --clips clips.txt \\
        --data-root <dir of clips> --tracks-root <dir of predictions> \\
        --tag <condition> --out <condition>.json [--propagation <rule>] [--pilot]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import PIL

from evalkit.classes import CLASS_SETS, VIEWS, ClassTable, load_table
from evalkit.clip import ClipScores, ScoredFrame, summarize_clip
from evalkit.code_sha import eval_code_sha, hashed_files
from evalkit.frame import score_frame
from evalkit.inputs import ClipInputs, read_clip
from evalkit.keys import CLIP_METRICS, FRAME_METRICS, metric_key
from evalkit.pilot import PILOT_DOMAINS
from evalkit.pilot_clip import score_pilot_clip
from evalkit.scored import valid_depth
from evalkit.time_iou import time_iou

(TIME_KEY,) = CLIP_METRICS

# The propagation rules `docs/evaluation.md` defines.
PROPAGATION_RULES = ("both_ways_from_centre", "forward_from_first", "per_frame")

# The record a tracker or the per-frame stage writes beside a condition's labels.
SEED_INFO = "seed_info.json"

# What a pilot-mode JSON records for a condition that holds no rule. Pilot mode scores every condition the
# pilot evaluator scored, those seeded from GT among them, and its JSONs never enter a comparison.
NO_RULE = "neither"


def check_table(table: ClassTable) -> None:
    """Refuse a class table that is not one of the files `eval_code_sha` covers.

    A score records the package's sha, so a table read from elsewhere would
    give a score that looks comparable while its classes were not hashed.

    Raises:
        ValueError: The table was read from a file `hashed_files` does not list.
    """
    if Path(table.path).resolve() not in {p.resolve() for p in hashed_files()}:
        raise ValueError(f"the class table was read from {table.path}, which eval_code_sha does not cover; "
                         "a score is made with the package's own tables only")


def read_population(path: str | Path) -> list[str]:
    """The clips listed in `path`, one per line, in its order.

    Raises:
        ValueError: The file lists no clip, or one clip twice.
    """
    clips = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not clips:
        raise ValueError(f"{path} lists no clip")
    twice = sorted({c for c in clips if clips.count(c) > 1})
    if twice:
        raise ValueError(f"{path} lists {twice} more than once")
    return clips


def score_clip(inputs: ClipInputs, table: ClassTable) -> ClipScores:
    """Score every GT frame of one clip, pool `time_IoU`, and average."""
    # Every GT frame in time order, scored in every view or skipped as excluded.
    frames = [ScoredFrame(frame=inputs.numbers[i],
                          scores=score_frame(inputs.gt[i], inputs.regions[i], inputs.depth[i], table))
              for i in inputs.order if i in inputs.gt]
    # time_IoU over every frame with a prediction, GT or not, in time order.
    tracked = [i for i in inputs.order if i in inputs.regions]
    pooled = time_iou([inputs.regions[i] for i in tracked], [valid_depth(inputs.depth[i]) for i in tracked])
    return summarize_clip(frames, pooled)


def _number(value):
    return None if value is None else float(value)


def clip_row(clip: str, scores: ClipScores) -> dict:
    """One clip's row of the score JSON: the means, the counts behind them, and the per-frame values."""
    views = scores.views
    row: dict = {"clip": clip}
    row.update({metric_key(m, v): _number(views[v].means[m]) for v in VIEWS for m in FRAME_METRICS})
    row[TIME_KEY] = _number(scores.time_iou)
    row["n_frames"] = {metric_key(m, v): int(views[v].n_frames[m]) for v in VIEWS for m in FRAME_METRICS}
    row["n_scored_frames"] = scores.n_scored_frames
    row["n_excluded_frames"] = scores.n_excluded_frames
    row["pixels"] = {v: {k: int(n) for k, n in dataclasses.asdict(views[v].pixels).items()} for v in VIEWS}
    row["objects"] = {v: {"gt": int(views[v].n_gt_objects), "predicted": int(views[v].n_pred_objects),
                          "hits": int(views[v].n_hits), "inst_bf_hits": int(views[v].n_inst_bf_hits)}
                      for v in VIEWS}
    row["frames"] = [
        {"frame": f.frame, "excluded": f.scores.excluded,
         **{metric_key(m, v): _number(value) for v, s in f.scores.views.items() for m, value in s.metrics().items()}}
        for f in scores.frames
    ]
    return row


def versions() -> dict[str, str]:
    """The versions of Python and of the three libraries a hashed module may use."""
    return {"python": platform.python_version(), "numpy": np.__version__,
            "opencv": cv2.__version__, "pillow": PIL.__version__}


def rule_of_seed_info(info: dict) -> str | None:
    """Return the propagation rule a clip's `seed_info.json` records, or None when it holds none.

    The per-frame stage records `seed_source` `per_frame`. The tracker records `bidir`, the seed frame and the
    frames it labelled: `both_ways_from_centre` seeds on the centre of those frames and carries both ways;
    `forward_from_first` seeds on frame 0 and carries forward. A seed chosen anywhere else, from the GT for one,
    holds neither rule.
    """
    if info.get("seed_source") == "per_frame":
        return "per_frame"
    frames, seed = info.get("frames"), info.get("seed_frame")
    if not frames or seed is None or "bidir" not in info:
        return None
    if info["bidir"] and seed == frames[len(frames) // 2]:
        return "both_ways_from_centre"
    if not info["bidir"] and seed == 0 and frames[0] == 0:
        return "forward_from_first"
    return None


def condition_rule(tracks_root: str | Path, tag: str, clips: Sequence[str], stated: str | None = None,
                   *, pilot: bool = False) -> str:
    """Return the condition's propagation rule, as each clip's `seed_info.json` records it or as stated.

    Args:
        tracks_root: The directory holding one directory per clip.
        tag: The condition's directory name under each clip.
        clips: The population.
        stated: The rule the caller states, for a condition whose labels carry no `seed_info.json`.
        pilot: Pilot mode, which records `NO_RULE` where normal mode refuses for want of a rule.

    Raises:
        ValueError: A clip's record holds no rule; two clips hold two rules; the stated rule is not one or
            contradicts the records; or neither records nor a statement give one.
    """
    if stated is not None and stated not in PROPAGATION_RULES:
        raise ValueError(f"{stated!r} is no propagation rule; one of {PROPAGATION_RULES}")
    found = {}
    for clip in clips:
        path = Path(tracks_root) / clip / tag / SEED_INFO
        if path.is_file():
            rule = rule_of_seed_info(json.loads(path.read_text(encoding="utf-8")))
            if rule is None:
                if pilot:
                    return NO_RULE
                raise ValueError(f"{clip}: {tag}'s {SEED_INFO} holds no propagation rule; a seed off the "
                                 "centre and off the first frame is neither rule")
            found[clip] = rule
    rules = set(found.values()) | ({stated} if stated else set())
    if len(rules) > 1:
        raise ValueError(f"{tag}: the clips and the statement give {sorted(rules)}; a condition has one rule")
    if not rules:
        if pilot:
            return NO_RULE
        raise ValueError(f"{tag}: no clip has a {SEED_INFO}, so state the rule with --propagation")
    return rules.pop()


def score_condition(dataset: str, class_set: str | None, clips: Sequence[str],
                    data_root: str | Path, tracks_root: str | Path, tag: str,
                    propagation: str | None = None, *, pilot: bool = False) -> dict:
    """Score one condition on every clip, and return its score JSON as a dict.

    In pilot mode each clip is read and scored by the pilot evaluator's rules
    (`evalkit.pilot_clip`), in its four domains, for the check against it.

    Args:
        propagation: The condition's propagation rule, for labels that carry no `seed_info.json`.
        pilot: Score by the pilot evaluator's rules.

    Raises:
        ValueError, KeyError, FileNotFoundError: The condition's propagation rule is unknown
            (`condition_rule`), or a clip's inputs cannot be read or scored. Nothing is returned for the
            others.
    """
    table = load_table(dataset, class_set)
    check_table(table)
    rule = condition_rule(tracks_root, tag, clips, propagation, pilot=pilot)
    sha = eval_code_sha()
    rows, shas = [], {}
    for clip in clips:
        if pilot:
            row, shas[clip] = score_pilot_clip(data_root, tracks_root, tag, clip, table.dataset)
            rows.append(row)
            continue
        inputs = read_clip(data_root, tracks_root, tag, clip, table)
        rows.append(clip_row(clip, score_clip(inputs, table)))
        shas[clip] = dict(inputs.shas)
    return {
        "eval_code_sha": sha, "dataset": table.dataset, "class_set": table.class_set,
        "views": list(PILOT_DOMAINS if pilot else VIEWS), "pilot": pilot, "track_dir_name": tag,
        "clips": list(clips),
        "input_shas": shas, "versions": versions(), "propagation": rule, "per_clip": rows,
    }


def write_scores(summary: dict, out: str | Path) -> None:
    """Write the score JSON in one step, so that a run that fails leaves no partial file."""
    out = Path(out)
    tmp = out.with_name(out.name + ".partial")
    tmp.write_text(json.dumps(summary, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, out)


def main(argv: Sequence[str] | None = None) -> None:
    """Score the condition the arguments name, and write its score JSON."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, choices=sorted(CLASS_SETS))
    ap.add_argument("--class-set", default=None, help="original (default) or, for atlas120k, benchmark")
    ap.add_argument("--clips", required=True, help="the population: one clip name per line")
    ap.add_argument("--data-root", required=True, help="one directory per clip: manifest, depth, GT masks")
    ap.add_argument("--tracks-root", required=True, help="one directory per clip, one per condition inside")
    ap.add_argument("--tag", required=True, help="the condition's directory name under each clip")
    ap.add_argument("--out", required=True, help="the score JSON to write")
    ap.add_argument("--propagation", choices=PROPAGATION_RULES, default=None,
                    help="the condition's propagation rule, for labels that carry no seed_info.json")
    ap.add_argument("--pilot", action="store_true",
                    help="score by the pilot evaluator's rules, for the check against it")
    args = ap.parse_args(argv)
    clips = read_population(args.clips)
    if args.pilot and args.class_set not in (None, "original"):
        raise ValueError("pilot mode scores the original ids, as the pilot evaluator did")
    summary = score_condition(args.dataset, args.class_set, clips, args.data_root, args.tracks_root, args.tag,
                              args.propagation, pilot=args.pilot)
    write_scores(summary, args.out)
    print(f"{args.tag}: {len(clips)} clips scored, {args.out}")


if __name__ == "__main__":
    main()
