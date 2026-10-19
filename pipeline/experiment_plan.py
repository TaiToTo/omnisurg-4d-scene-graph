"""Read the experiment plan: the conditions to make on the GPU machine, and the pairs of conditions to compare.

`experiment_plan.json` lists each group of conditions to make by the parts
of their names (`evalkit.tools.condition_names`) and the stage it is made
in, and each comparison as pairs of names with the parts the pairs vary.
`check` refuses a pair that varies another part, a pair under two rules
that varies more than the rule, and a compared condition that the plan
does not make. `status` checks every clip's record of each condition to
make against the settings the name says. `commands` prints the
`pipeline.condition_population` command of each condition still to make.

Usage:
    python -m pipeline.experiment_plan check
    python -m pipeline.experiment_plan status --dataset atlas120k --clips atlas120k_meta/clips.txt --tracks-root <dir> \\
        [--stage 1]
    python -m pipeline.experiment_plan commands --dataset atlas120k --clips atlas120k_meta/clips.txt \\
        --input-dir <dir of clips> --tracks-root <dir> --sam-ckpt sam_vit_h_4b8939.pth --gpus 0 1 2 3 [--stage 1] [--all]
"""
from __future__ import annotations

import argparse
import json
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path

from evalkit.evaluate import SEED_INFO, read_population
from evalkit.tools.condition_names import DATASETS, READS_DEPTH, RULES, Parts, check_rules, parse
from pipeline.condition_population import check_settings, condition_dir, expected_record, read_record, stage_args

PLAN = Path(__file__).with_name("experiment_plan.json")

# The parts of a name, as `Parts` holds them; a comparison names the ones it varies.
PARTS = tuple(f.name for f in fields(Parts))
# Each depth model to the `depth_source` the stages take and record.
DEPTH_SOURCE = {"da3": "da3", "pi3x": "pi3"}
# The settings every name leaves out, as `condition_names` states them: the tracker's input, the seed regions'
# fewest pixels, and, in an input with edges, the edges' gain; the normals of `normal-edge` are not smoothed.
TRACK_BASE, SEED_MIN_AREA, EDGE_GAIN = "rgb", 400, 1.0
# The lists of a group that name what it makes; each must name something.
GROUP_LISTS = ("inputs", "depths", "points_per_side")


def conditions(plan: Mapping, dataset: str, stage: int | None = None) -> list[str]:
    """Return the names of the conditions the plan makes for `dataset`, group by group, in the plan's order.

    An input that reads depth is made on each depth model of its group; another input once.

    Args:
        stage: Only the groups of this stage; every group when None.

    Raises:
        ValueError: A group's stage is not a whole number from 1; a group's inputs, grids, or depth models under
            an input that reads depth, name nothing; its parts make no name (`condition_names.parse`); a name
            comes twice; or no group is of `stage`.
    """
    names, every, stages = [], [], set()
    for group in plan["make"]:
        if isinstance(group["stage"], bool) or not isinstance(group["stage"], int) or group["stage"] < 1:
            raise ValueError(f"{group['what']!r}: the stage {group['stage']!r} is not a whole number from 1")
        stages.add(group["stage"])
        reads_depth = any(i in READS_DEPTH for i in group["inputs"])
        empty = [k for k in GROUP_LISTS if not group[k] and (k != "depths" or reads_depth)]
        if empty:
            raise ValueError(f"{group['what']!r}: {empty} name nothing, so the group would make nothing of them")
        if dataset not in group["datasets"]:
            continue
        for inp in group["inputs"]:
            for depth in (group["depths"] if inp in READS_DEPTH else [None]):
                for pps in group["points_per_side"]:
                    name = Parts(inp, depth, pps, group["seed"], group["rule"]).name()
                    parse(name)
                    every.append(name)
                    if stage is None or group["stage"] == stage:
                        names.append(name)
    twice = sorted({n for n in every if every.count(n) > 1})
    if twice:
        raise ValueError(f"{dataset}: the plan makes {twice} more than once")
    if stage is not None and stage not in stages:
        raise ValueError(f"no group is of stage {stage}; the stages are {sorted(stages)}")
    return names


def check_plan(plan: Mapping) -> None:
    """Refuse a plan whose groups name an unknown dataset, or whose comparisons are not controlled.

    A comparison is controlled when each of its pairs varies exactly the parts it names, and a pair under two
    propagation rules varies the rule alone (`condition_names.check_rules`). Each compared condition must be
    one the plan makes for the dataset.

    Raises:
        ValueError: Every fault found, one per line.
    """
    faults = []
    for group in plan["make"] + plan["compare"]:
        faults += [f"{group['what']!r}: {d!r} is no dataset of {DATASETS}"
                   for d in group["datasets"] if d not in DATASETS]
    made = {d: set(conditions(plan, d)) for d in DATASETS}
    for comparison in plan["compare"]:
        unknown = sorted(set(comparison["varies"]) - set(PARTS))
        if unknown:
            faults.append(f"{comparison['what']!r}: {unknown} are no parts of a name; the parts are {PARTS}")
        for a, b in comparison["pairs"]:
            varied = {p for p in PARTS if getattr(parse(a), p) != getattr(parse(b), p)}
            if varied != set(comparison["varies"]):
                faults.append(f"{a} and {b} vary {sorted(varied)}, and the comparison {comparison['what']!r} "
                              f"varies {sorted(comparison['varies'])}")
            try:
                check_rules(parse(a), parse(b))
            except ValueError as e:
                faults.append(str(e))
            for dataset in comparison["datasets"]:
                faults += [f"{dataset}: the plan does not make {name}"
                           for name in (a, b) if name not in made.get(dataset, ())]
    if faults:
        raise ValueError("the plan is not controlled:\n" + "\n".join(faults))


def settings_of(name: str) -> tuple[str, argparse.Namespace]:
    """Return the stage that makes the named condition, and its settings as `condition_population` holds them.

    The settings hold every option the name says and every default a name leaves out, and no SAM weights.

    Raises:
        ValueError: The name breaks the rule of names; its seed is made outside the segmenting stages (GT, or
            merged regions); or it adds instruments by text, which is not ported.
    """
    parts = parse(name)
    if parts.seed not in (None, "sam"):
        raise ValueError(f"{name}: the seed {parts.seed!r} is made outside the stages; the plan makes SAM seeds only")
    if "instrument-text" in parts.settings:
        raise ValueError(f"{name}: instruments added by text prompts are not ported")
    s = argparse.Namespace(tag=name, sam_input=parts.input.replace("-", "_"),
                           depth_source=DEPTH_SOURCE[parts.depth or "da3"],
                           points_per_side=parts.points_per_side, seed_min_area=SEED_MIN_AREA,
                           keep_edge_ring="keepring" in parts.settings, sam_ckpt=None)
    if parts.rule == "perframe":
        s.edge_gain = EDGE_GAIN
        return "per_frame", s
    # The tracking stage's own default draws the edges at 0.85 on smoothed normals, so the name's are given.
    s.rule, s.track_base, s.seed_labels = RULES[parts.rule], TRACK_BASE, None
    s.seed_edge_gain, s.seed_no_smooth = EDGE_GAIN, True
    return "track", s


def label_dir(name: str) -> str:
    """Return the directory each clip's labels of the named condition go to, as `condition_population` names it."""
    return condition_dir(*settings_of(name))


def status(plan: Mapping, dataset: str, clips: Sequence[str], tracks_root: str | Path,
           stage: int | None = None) -> dict[str, dict]:
    """Return, for each condition the plan makes for `dataset` in `stage`, the clips made, missing, and made otherwise.

    A clip is made when its label directory holds a `seed_info.json` that records the settings the name says,
    every key of them (`condition_population.check_settings`). It is made otherwise when its record cannot be
    read or records other settings; the message says which.
    """
    out = {}
    for name in conditions(plan, dataset, stage):
        stage_name, s = settings_of(name)
        expected = expected_record(stage_name, s)
        made, missing, other = [], [], {}
        for clip in clips:
            lab_dir = Path(tracks_root) / clip / condition_dir(stage_name, s)
            if not (lab_dir / SEED_INFO).is_file():
                missing.append(clip)
                continue
            try:
                check_settings(stage_name, clip, read_record(lab_dir, clip), expected)
            except ValueError as e:
                other[clip] = str(e)
                continue
            made.append(clip)
        out[name] = {"made": made, "missing": missing, "other": other}
    return out


def command(name: str, input_dir: str, clips: str, tracks_root: str, sam_ckpt: str, gpus: Sequence[int]) -> str:
    """Return the `condition_population` command that makes the named condition on every clip."""
    stage, s = settings_of(name)
    s.sam_ckpt = sam_ckpt
    argv = ["python", "-m", "pipeline.condition_population", stage, "--input-dir", input_dir, "--clips", clips,
            "--tracks-root", tracks_root, "--tag", name, "--gpus", *(str(g) for g in gpus), *stage_args(stage, s)]
    return shlex.join(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Check the plan, report what is made, or print the commands of what is not."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("check", help="refuse comparisons that vary another part, or name a condition not made")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", required=True, choices=DATASETS)
    common.add_argument("--clips", required=True, help="the population: one clip name per line")
    common.add_argument("--tracks-root", required=True, help="one directory per clip, one per condition inside")
    common.add_argument("--stage", type=int, default=None, help="only the conditions of this stage")
    sub.add_parser("status", parents=[common], help="count the clips each condition is made on")
    cmd = sub.add_parser("commands", parents=[common], help="print the command of each condition still to make")
    cmd.add_argument("--input-dir", required=True, help="the directory that holds the clips")
    cmd.add_argument("--sam-ckpt", required=True, help="the SAM ViT-H weights, sam_vit_h_4b8939.pth")
    cmd.add_argument("--gpus", type=int, nargs="+", required=True, help="the GPUs to run on, one clip each")
    cmd.add_argument("--all", action="store_true", help="print every condition's command, made or not")
    args = ap.parse_args(argv)
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    check_plan(plan)
    if args.action == "check":
        print(f"{PLAN.name}: every comparison varies only the parts it names, and the plan makes every condition")
        return

    # What is made, per condition; a clip made otherwise stops both reports.
    clips = read_population(args.clips)
    try:
        report = status(plan, args.dataset, clips, args.tracks_root, args.stage)
    except ValueError as e:
        raise SystemExit(f"--stage: {e}") from e
    wrong = {name: r["other"] for name, r in report.items() if r["other"]}
    if wrong:
        raise SystemExit("labels under a planned name record other settings:\n" + "\n".join(
            f"  {name}: {said}" for name, other in wrong.items() for said in other.values()))
    if args.action == "status":
        for name, r in report.items():
            print(f"{name:45s} {len(r['made']):4d} made, {len(r['missing']):4d} missing")
        return
    for name, r in report.items():
        if r["missing"] or args.all:
            print(command(name, args.input_dir, args.clips, args.tracks_root, args.sam_ckpt, args.gpus))


if __name__ == "__main__":
    main()
