"""Read the experiment plan: the conditions to make on the GPU machine, and the pairs of conditions to compare.

`experiment_plan.json` lists each group of conditions to make by the parts
of their names (`evalkit.tools.condition_names`) and the stage it is made
in, and each comparison as pairs of names with the parts the pairs vary. `check` refuses a pair that
varies another part, and a compared condition that the plan does not make
and the table of workbench names does not hold. `status` reads every clip's
record of each condition to make, and names the condition the record
shows, which must be the plan's name. `commands` prints the
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
from types import SimpleNamespace

from evalkit.evaluate import SEED_INFO, read_population, rule_of_seed_info
from evalkit.tools.condition_names import DATASETS, DEPTH_MODELS, EDGE_INPUTS, READS_DEPTH, RULES, Parts, parse, read_table
from pipeline.condition_population import condition_dir

PLAN = Path(__file__).with_name("experiment_plan.json")

# The parts of a name, as `Parts` holds them; a comparison names the ones it varies.
PARTS = tuple(f.name for f in fields(Parts))
# Each depth model to the `depth_source` the stages take and record.
DEPTH_SOURCE = {"da3": "da3", "pi3x": "pi3"}
# The settings every name leaves out, as `condition_names` states them: the tracker's input, the seed regions'
# fewest pixels, and, in an input with edges, the edges' gain; the normals of `normal-edge` are not smoothed.
TRACK_BASE, SEED_MIN_AREA, EDGE_GAIN = "rgb", 400, 1.0


def conditions(plan: Mapping, dataset: str, stage: int | None = None) -> list[str]:
    """Return the names of the conditions the plan makes for `dataset`, group by group, in the plan's order.

    An input that reads depth is made on each depth model of its group; another input once.

    Args:
        stage: Only the groups of this stage; every group when None.

    Raises:
        ValueError: A group's stage is not a whole number from 1, its parts make no name
            (`condition_names.parse`), or a name comes twice.
    """
    names, every = [], []
    for group in plan["make"]:
        if isinstance(group["stage"], bool) or not isinstance(group["stage"], int) or group["stage"] < 1:
            raise ValueError(f"{group['what']!r}: the stage {group['stage']!r} is not a whole number from 1")
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
    return names


def check_plan(plan: Mapping, table: Mapping | None = None) -> None:
    """Refuse a plan whose groups name an unknown dataset, or whose comparisons are not controlled.

    A comparison is controlled when each of its pairs varies exactly the parts it names. Each compared condition
    must be one the plan makes for the dataset, or one `condition_names.tsv` names for it.

    Raises:
        ValueError: Every fault found, one per line.
    """
    table = read_table() if table is None else table
    named = {(dataset, row.name) for (dataset, _), row in table.items() if row.name}
    faults = []
    for group in plan["make"] + plan["compare"]:
        faults += [f"{group['what']!r}: {d!r} is no dataset of {DATASETS}" for d in group["datasets"] if d not in DATASETS]
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
            for dataset in comparison["datasets"]:
                faults += [f"{dataset}: {name} is neither made by the plan nor named in the table"
                           for name in (a, b) if name not in made.get(dataset, ()) and (dataset, name) not in named]
    if faults:
        raise ValueError("the plan is not controlled:\n" + "\n".join(faults))


def stage_options(name: str) -> tuple[str, list[str]]:
    """Return the stage that makes the named condition, and every option it takes from the name.

    Raises:
        ValueError: The name breaks the rule of names; its seed is made outside the segmenting stages (GT, or
            merged regions); or it adds instruments by text, which is not ported.
    """
    parts = parse(name)
    if parts.seed not in (None, "sam"):
        raise ValueError(f"{name}: the seed {parts.seed!r} is made outside the stages; the plan makes SAM seeds only")
    if "instrument-text" in parts.settings:
        raise ValueError(f"{name}: instruments added by text prompts are not ported")
    options = ["--sam-input", parts.input.replace("-", "_"), "--depth-source", DEPTH_SOURCE[parts.depth or "da3"],
               "--points-per-side", str(parts.points_per_side), "--seed-min-area", str(SEED_MIN_AREA)]
    options += ["--keep-edge-ring"] if "keepring" in parts.settings else []
    if parts.rule == "perframe":
        return "per_frame", options + ["--edge-gain", str(EDGE_GAIN)]
    # The tracking stage's own default draws the edges at 0.85 on smoothed normals, so the name's are given.
    return "track", options + ["--rule", RULES[parts.rule], "--track-base", TRACK_BASE,
                               "--seed-edge-gain", str(EDGE_GAIN), "--seed-no-smooth"]


def label_dir(name: str) -> str:
    """Return the directory each clip's labels of the named condition go to, as `condition_population` names it."""
    stage, _ = stage_options(name)
    return condition_dir(stage, SimpleNamespace(track_base=TRACK_BASE, tag=name))


def name_of_record(record: Mapping) -> str:
    """Return the name a clip's `seed_info.json` shows.

    Raises:
        ValueError: The record holds no propagation rule (`rule_of_seed_info`), or a setting off the defaults that
            a name leaves out: the tracker's input, the seed regions' fewest pixels, or, in an input with edges,
            the edges' gain and, in `normal-edge`, the smoothing.
    """
    # The rule and the seed, then the settings no name can say.
    rule = {v: k for k, v in RULES.items()}[rule_of_seed_info(record)]
    seed_input, inp = record["seed_input"], record["sam_input"].replace("_", "-")
    off = []
    if record.get("track_base", TRACK_BASE) != TRACK_BASE:
        off.append(f"the tracker's input {record['track_base']!r}")
    if record["seed_min_area"] != SEED_MIN_AREA:
        off.append(f"seed regions of {record['seed_min_area']} pixels or more")
    if inp in EDGE_INPUTS and seed_input["seed_edge_gain"] != EDGE_GAIN:
        off.append(f"edges at gain {seed_input['seed_edge_gain']}")
    if inp == "normal-edge" and seed_input["seed_smooth"]:
        off.append("smoothed normals")
    if off:
        raise ValueError(f"the record holds settings no name says: {', '.join(off)}")

    # The parts, joined and parsed back.
    settings = []
    if inp in EDGE_INPUTS and seed_input["edge_ring_masked"] is False:
        settings.append("keepring")
    if record.get("instrument_seed", "off") != "off":
        settings.append("instrument-text")
    depth = {v: k for k, v in DEPTH_SOURCE.items()}[record["depth_source"]] if inp in READS_DEPTH else None
    seed = None if rule == "perframe" else "sam"
    return parse(Parts(inp, depth, seed_input["points_per_side"], seed, rule, tuple(settings)).name()).name()


def status(plan: Mapping, dataset: str, clips: Sequence[str], tracks_root: str | Path,
           stage: int | None = None) -> dict[str, dict]:
    """Return, for each condition the plan makes for `dataset` in `stage`, the clips made, missing, and made as another one.

    A clip is made when its label directory holds a `seed_info.json` that shows the plan's name. It is made as
    another condition when its record shows another name, or none.
    """
    out = {}
    for name in conditions(plan, dataset, stage):
        made, missing, other = [], [], {}
        for clip in clips:
            path = Path(tracks_root) / clip / label_dir(name) / SEED_INFO
            if not path.is_file():
                missing.append(clip)
                continue
            try:
                shown = name_of_record(json.loads(path.read_text(encoding="utf-8")))
            except (ValueError, KeyError) as e:
                shown = f"no name: {e}"
            if shown == name:
                made.append(clip)
            else:
                other[clip] = shown
        out[name] = {"made": made, "missing": missing, "other": other}
    return out


def command(name: str, input_dir: str, clips: str, tracks_root: str, sam_ckpt: str, gpus: Sequence[int]) -> str:
    """Return the `condition_population` command that makes the named condition on every clip."""
    stage, options = stage_options(name)
    argv = ["python", "-m", "pipeline.condition_population", stage, "--input-dir", input_dir, "--clips", clips,
            "--tracks-root", tracks_root, "--tag", name, *options, "--sam-ckpt", sam_ckpt,
            "--gpus", *(str(g) for g in gpus)]
    return shlex.join(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Check the plan, report what is made, or print the commands of what is not."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="action", required=True)
    sub.add_parser("check", help="refuse comparisons that vary another part, or name an unknown condition")
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
        print(f"{PLAN.name}: every comparison varies only the parts it names")
        return

    # What is made, per condition; a clip made as another condition stops both reports.
    clips = read_population(args.clips)
    report = status(plan, args.dataset, clips, args.tracks_root, args.stage)
    wrong = {name: r["other"] for name, r in report.items() if r["other"]}
    if wrong:
        raise SystemExit("labels under a planned name show another condition:\n" + "\n".join(
            f"  {name}: {clip}: {shown}" for name, other in wrong.items() for clip, shown in other.items()))
    if args.action == "status":
        for name, r in report.items():
            print(f"{name:45s} {len(r['made']):4d} made, {len(r['missing']):4d} missing")
        return
    for name, r in report.items():
        if r["missing"] or args.all:
            print(command(name, args.input_dir, args.clips, args.tracks_root, args.sam_ckpt, args.gpus))


if __name__ == "__main__":
    main()
