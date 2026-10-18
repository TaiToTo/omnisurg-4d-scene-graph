"""The evaluator's entry point: score one condition on every clip of a population, and write its score JSON.

For each clip it reads the inputs (`evalkit.inputs`), scores every GT frame
in every view (`evalkit.frame`), pools `time_IoU` over every frame with a
prediction, and averages over the frames (`evalkit.clip`). The JSON holds
what `docs/evaluation.md` records with every score, one row per clip, the
per-frame values, and the condition's propagation rule and seeds as its
tracker recorded them. With `--pilot`, each clip is read the same way and
scored by the pilot evaluator's rules instead (`evalkit.pilot_clip`). A clip
that cannot be scored stops the run, and no JSON is written: a population
is scored whole or not at all.

Usage:
    python -m evalkit.evaluate --dataset cholecseg8k --clips clips.txt \\
        --data-root <dir of clips> --tracks-root <dir of predictions> \\
        --tag <condition> --out <condition>.json [--propagation <rule>] \\
        [--seeds <dir of seeds>] [--seeds-gt-informed yes|no] [--pilot]
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
from evalkit.inputs import ClipInputs, read_clip, sha_of_files
from evalkit.keys import FRAME_METRICS, metric_key
from evalkit.pilot import PILOT_DOMAINS
from evalkit.pilot_clip import score_pilot_clip
from evalkit.scored import valid_depth
from evalkit.time_iou import time_iou

# The propagation rules `docs/evaluation.md` defines ("Propagation rule").
PROPAGATION_RULES = ("both_ways_from_centre", "forward_from_first", "per_frame")

# The record a tracker or the per-frame stage writes beside a condition's labels.
SEED_INFO = "seed_info.json"

# Where a condition's seeds came from, as each clip's record names it in `seed_source`:
# - `sam`: the tracking stage cut the seed frame itself;
# - `external`: the tracking stage read the seed's regions from files made outside it (`--seed-labels`);
# - `per_frame`: no seed, for a condition segmented frame by frame.
SEED_SOURCES = ("sam", "external", "per_frame")

# The sources whose record decides that no GT informed the seeds: the stage's own cut reads no GT, and a
# condition segmented frame by frame has no seed.
NOT_GT_INFORMED = frozenset({"sam", "per_frame"})


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
    """Read the population: the clips listed in `path`, one per line, in its order.

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
    """Build one clip's row of the score JSON: the means, the counts behind them, and the per-frame values."""
    views = scores.views
    row: dict = {"clip": clip}
    row.update({metric_key(m, v): _number(views[v].means[m]) for v in VIEWS for m in FRAME_METRICS})
    row["time_IoU"] = _number(scores.time_iou)
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
    """Return the versions of Python and of the three libraries a hashed module may use."""
    return {"python": platform.python_version(), "numpy": np.__version__,
            "opencv": cv2.__version__, "pillow": PIL.__version__}


def rule_of_seed_info(info: dict) -> str:
    """Return the propagation rule a clip's `seed_info.json` records.

    The per-frame stage records `seed_source` `per_frame`. The tracker records `seed_source` `sam` or
    `external`, `bidir`, the seed frame and the frames it labelled. A seed carried both ways must sit on the
    centre of the labelled frames, taken in time order; its rule is `both_ways_from_centre`. A seed carried
    forward must sit on frame 0; its rule is `forward_from_first`. Where the seed's regions came from does not
    change where it sits.

    Raises:
        ValueError: The record names no `seed_source`, or one outside `SEED_SOURCES`, the GT for one; the
            tracker's record is incomplete; its `bidir` is not true or false; its seed frame or a labelled frame
            is not a whole number; a labelled frame repeats; or the seed sits on neither the centre nor the
            first frame.
    """
    source = info.get("seed_source")
    if source is None:
        raise ValueError("the record names no seed_source")
    if source == "per_frame":
        return "per_frame"
    if source not in SEED_SOURCES:
        raise ValueError(f"the seed came from {source!r}, which is none of {list(SEED_SOURCES)}; "
                         "such a condition holds neither rule, wherever its seed sits")
    frames, seed, bidir = info.get("frames"), info.get("seed_frame"), info.get("bidir")
    if not isinstance(frames, list) or not frames or seed is None or bidir is None:
        raise ValueError("the tracker's record lacks the seed frame, bidir, or the list of frames it labelled")
    # A string such as "no" is truthy, so only a bool says which way the seed was carried.
    if not isinstance(bidir, bool):
        raise ValueError(f"bidir is {bidir!r}, not true or false")
    # A bool is an int in Python, so a seed frame of true would read as frame 1.
    if any(isinstance(f, bool) or not isinstance(f, int) for f in (seed, *frames)):
        raise ValueError(f"the seed frame {seed!r} and the labelled frames {frames!r} are not all whole numbers")
    # A frame listed twice moves the centre of the sorted list.
    if len(set(frames)) != len(frames):
        raise ValueError(f"the labelled frames repeat a frame: {frames}")
    frames = sorted(frames)
    if bidir:
        centre = frames[len(frames) // 2]
        if seed != centre:
            raise ValueError(f"the seed is on frame {seed}, carried both ways, "
                             f"and the centre of the labelled frames is {centre}")
        return "both_ways_from_centre"
    if seed != 0 or frames[0] != 0:
        raise ValueError(f"the seed is on frame {seed}, carried forward, and the labelled frames start at "
                         f"{frames[0]}; the rule starts on frame 0")
    return "forward_from_first"


def condition_rule(tracks_root: str | Path, tag: str, clips: Sequence[str], stated: str | None = None) -> str:
    """Return the condition's propagation rule, as every clip's `seed_info.json` records it or as stated.

    Args:
        tracks_root: The directory holding one directory per clip.
        tag: The condition's directory name under each clip.
        clips: The population.
        stated: The rule the caller states, for a condition whose labels carry no `seed_info.json`.

    Raises:
        ValueError: The stated rule is not one; a clip's record holds no rule (`rule_of_seed_info`); some
            clips have a record and some have none; two clips hold two rules; the statement contradicts
            the records; or neither records nor a statement give one.
    """
    if stated is not None and stated not in PROPAGATION_RULES:
        raise ValueError(f"{stated!r} is no propagation rule; one of {PROPAGATION_RULES}")
    found, without = {}, []
    for clip in clips:
        path = Path(tracks_root) / clip / tag / SEED_INFO
        if not path.is_file():
            without.append(clip)
            continue
        try:
            found[clip] = rule_of_seed_info(json.loads(path.read_text(encoding="utf-8")))
        except ValueError as e:
            raise ValueError(f"{clip}: {tag}'s {SEED_INFO} holds no propagation rule: {e}") from None
    # All or none: a record on some clips says nothing about the others.
    if found and without:
        raise ValueError(f"{tag}: {sorted(found)} have a {SEED_INFO} and {without} have none; "
                         "a condition records its rule on every clip or on none")
    rules = set(found.values()) | ({stated} if stated else set())
    if len(rules) > 1:
        raise ValueError(f"{tag}: the clips and the statement give {sorted(rules)}; a condition has one rule")
    if not rules:
        raise ValueError(f"{tag}: no clip has a {SEED_INFO}, so state the rule with --propagation")
    return rules.pop()


def seed_file(seeds: str, clip: str, seed_frame: int) -> Path:
    """Return the seed file the tracking stage read for one clip, as its `--seed-labels` places it.

    The clip's directory is `seeds` with `{clip}` replaced by the clip's name, or `<seeds>/<clip>` when `seeds`
    holds no `{clip}`. The file is `label_<seed frame>.npy` in it.
    """
    clip_dir = Path(seeds.replace("{clip}", clip)) if "{clip}" in seeds else Path(seeds) / clip
    return clip_dir / f"label_{seed_frame:04d}.npy"


def _seed_sha(seeds: str, clip: str, info: dict) -> str:
    """Return the sha256 of a clip's external seed file, once it holds the regions the tracker's record counted.

    The tracker keeps the seed's regions of `seed_min_area` pixels or more, and records their number as
    `n_seed_regions`. A file that holds another number is not the one it read.

    Raises:
        ValueError: The record lacks the seed frame, `seed_min_area` or `n_seed_regions`; the file is not an
            (H, W) integer map; or its regions are not the record's number.
        FileNotFoundError: The file is missing.
    """
    keys = ("seed_frame", "seed_min_area", "n_seed_regions")
    if any(not isinstance(info.get(k), int) or isinstance(info.get(k), bool) for k in keys):
        raise ValueError(f"{clip}: the tracker's record lacks one of {list(keys)}, so its seed file cannot be checked")
    path = seed_file(seeds, clip, info["seed_frame"])
    if not path.is_file():
        raise FileNotFoundError(f"{clip}: no seed file {path}")
    labels = np.load(path)
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"{clip}: {path} is not an (H, W) integer map, got {labels.dtype} of shape {labels.shape}")
    _, areas = np.unique(labels[labels >= 0], return_counts=True)
    kept = int((areas >= info["seed_min_area"]).sum())
    if kept != info["n_seed_regions"]:
        raise ValueError(f"{clip}: {path} holds {kept} regions of {info['seed_min_area']} pixels or more, and the "
                         f"tracker's record counts {info['n_seed_regions']}; it is not the seed the tracker read")
    return sha_of_files([path])


def condition_seed(tracks_root: str | Path, tag: str, clips: Sequence[str], rule: str, seeds: str | None = None,
                   gt_informed: bool | None = None) -> tuple[dict, dict[str, str]]:
    """Return how the condition's seeds were made, as its score records them, and each clip's seed file hash.

    The record holds `source`, the `seed_source` every clip's `seed_info.json` names, and `gt_informed`,
    whether the GT made the seeds or set one of their settings, such as K when K is the clip's number of GT
    objects. A source in `NOT_GT_INFORMED` decides it; for any other the caller states it. Labels without a
    record have the source None, or `per_frame` when their stated rule is. The files of `external` seeds are
    read from `seeds`, checked against each clip's record (`_seed_sha`) and hashed.

    Args:
        tracks_root: The directory holding one directory per clip.
        tag: The condition's directory name under each clip.
        clips: The population.
        rule: The condition's propagation rule, as `condition_rule` returned it.
        seeds: The directory of `external` seeds, as the tracking stage's `--seed-labels` took it.
        gt_informed: Whether the GT informed seeds whose source does not decide it.

    Returns:
        The record, and each clip's seed file sha256 for `external` seeds; no hash for any other source.

    Raises:
        ValueError: The clips' records name two sources; `gt_informed` is stated for a source that decides
            it, or not stated for one that does not; `seeds` is given for seeds that are not `external`, or
            missing for seeds that are; or `_seed_sha` refuses a clip's seed file.
        FileNotFoundError: A seed file is missing.
    """
    # The source every clip's record names, or the one the stated rule implies.
    infos = {}
    for clip in clips:
        path = Path(tracks_root) / clip / tag / SEED_INFO
        if path.is_file():
            infos[clip] = json.loads(path.read_text(encoding="utf-8"))
    sources = {str(info.get("seed_source")) for info in infos.values()}
    if len(sources) > 1:
        raise ValueError(f"{tag}: the clips' records name the seed sources {sorted(sources)}; "
                         "a condition's seeds are made one way")
    source = sources.pop() if sources else ("per_frame" if rule == "per_frame" else None)

    # Whether the GT informed the seeds: decided by the source, or stated.
    if source in NOT_GT_INFORMED:
        if gt_informed is not None:
            raise ValueError(f"{tag}: seed_source {source!r} already says that no GT informed the seeds; "
                             "drop --seeds-gt-informed")
        gt_informed = False
    elif not isinstance(gt_informed, bool):
        raise ValueError(f"{tag}: its seeds come from {source or 'no recorded source'}, so whether the GT made or "
                         "informed them is not recorded; state it with --seeds-gt-informed")

    # The external seeds' files, checked against each clip's record and hashed.
    record = {"source": source, "gt_informed": gt_informed}
    if source != "external":
        if seeds is not None:
            raise ValueError(f"{tag}: --seeds names seed files, but its seeds come from {source or 'no record'}")
        return record, {}
    if seeds is None:
        raise ValueError(f"{tag}: its seeds were read from files made outside the tracking stage; "
                         "give their directory with --seeds, as the stage's --seed-labels took it")
    return record, {clip: _seed_sha(seeds, clip, infos[clip]) for clip in clips}


def score_condition(dataset: str, class_set: str | None, clips: Sequence[str],
                    data_root: str | Path, tracks_root: str | Path, tag: str,
                    propagation: str | None = None, *, seeds: str | None = None,
                    seeds_gt_informed: bool | None = None, pilot: bool = False) -> dict:
    """Score one condition on every clip, and return its score JSON as a dict.

    In pilot mode each clip is read and scored by the pilot evaluator's rules
    (`evalkit.pilot_clip`), in its four domains, for the check against it.
    The propagation rule and the seeds are read as in normal mode.

    Args:
        propagation: The condition's propagation rule, for labels that carry no `seed_info.json`.
        seeds: The directory of seeds made outside the tracking stage (`condition_seed`).
        seeds_gt_informed: Whether the GT informed seeds whose source does not decide it.
        pilot: Score by the pilot evaluator's rules.

    Raises:
        ValueError, KeyError, FileNotFoundError: Pilot mode is asked for a class set other than `original`;
            the condition's propagation rule is unknown (`condition_rule`); its seeds are not known
            (`condition_seed`); or a clip's inputs cannot be read or scored. Nothing is returned for the
            others.
    """
    # The table, the rule and the seeds, each refused before any clip is scored.
    table = load_table(dataset, class_set)
    check_table(table)
    if pilot and table.class_set != "original":
        raise ValueError("pilot mode scores the original ids, as the pilot evaluator did")
    rule = condition_rule(tracks_root, tag, clips, propagation)
    seed, seed_shas = condition_seed(tracks_root, tag, clips, rule, seeds, seeds_gt_informed)
    sha = eval_code_sha()

    # Every clip, scored and fingerprinted.
    rows, shas = [], {}
    for clip in clips:
        if pilot:
            row, shas[clip] = score_pilot_clip(data_root, tracks_root, tag, clip, table)
            rows.append(row)
        else:
            inputs = read_clip(data_root, tracks_root, tag, clip, table)
            rows.append(clip_row(clip, score_clip(inputs, table)))
            shas[clip] = dict(inputs.shas)
        if clip in seed_shas:
            shas[clip]["seeds"] = seed_shas[clip]
    return {
        "eval_code_sha": sha, "dataset": table.dataset, "class_set": table.class_set,
        "views": list(PILOT_DOMAINS if pilot else VIEWS), "pilot": pilot, "track_dir_name": tag,
        "clips": list(clips),
        "input_shas": shas, "versions": versions(), "propagation": rule, "seed": seed, "per_clip": rows,
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
    ap.add_argument("--seeds", default=None,
                    help="the directory of seeds made outside the tracking stage, as its --seed-labels took it")
    ap.add_argument("--seeds-gt-informed", choices=("yes", "no"), default=None,
                    help="whether the GT made or informed seeds whose seed_source does not say, such as "
                         "external seeds merged to the clip's number of GT objects")
    ap.add_argument("--pilot", action="store_true",
                    help="score by the pilot evaluator's rules, for the check against it")
    args = ap.parse_args(argv)
    clips = read_population(args.clips)
    gt_informed = None if args.seeds_gt_informed is None else args.seeds_gt_informed == "yes"
    summary = score_condition(args.dataset, args.class_set, clips, args.data_root, args.tracks_root, args.tag,
                              args.propagation, seeds=args.seeds, seeds_gt_informed=gt_informed, pilot=args.pilot)
    write_scores(summary, args.out)
    print(f"{args.tag}: {len(clips)} clips scored, {args.out}")


if __name__ == "__main__":
    main()
