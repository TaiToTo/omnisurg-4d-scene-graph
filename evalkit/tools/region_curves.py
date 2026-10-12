"""Lay each input's scores along its regions per frame: one curve per input, one point per grid.

The tool reads the conditions of a directory of JSONs whose names differ
only in the input and the grid, and fixes the other parts of the name. For
one key and one kind of object, it reads each condition's score and its
region count (`region_counts`). Each input's curve is its points ordered by
regions per frame. The tool reads every curve, linearly between points, at
each whole number of regions that every input reaches. These are
descriptions, with no interval and no mark. The conditions must be
comparable: one evaluator, dataset, class set and tool, one rule, which is
the one named, and one set of clips.

Usage:
    python -m evalkit.tools.region_curves --scores-dir <dir> --key F1_50/geometric \\
        [--min-px 300] [--rule centre] [--seed sam] [--depth da3] [--out curves.json]
"""
from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from evalkit.evaluate import write_scores
from evalkit.tools.condition_names import DEPTH_MODELS, READS_DEPTH, RULES, Parts, parse
from evalkit.tools.scores import check_comparable_table, load_scores, ruler

# What every component JSON of one set of curves must share: the evaluator and the objects it scored, the sizes,
# the rule, and the tool that scored them.
COMPONENT_FIELDS = ("eval_code_sha", "dataset", "class_set", "views", "min_px", "propagation", "tool_sha")
# What every region count must share: the tool that counted.
COUNT_FIELDS = ("tool_sha",)


def sweep(scores_dir: str | Path, rule: str, seed: str | None, depth: str) -> dict[str, list[str]]:
    """Return each input's conditions in `scores_dir` with the given rule, seed and depth model, by grid.

    A condition is one with a `<name>.regions.json` whose name has these parts and no setting off the defaults.

    Raises:
        ValueError: An input has fewer than two grids, or no input has any.
    """
    out: dict[str, list[str]] = {}
    for path in sorted(Path(scores_dir).glob("*.regions.json")):
        parts = parse(path.name[:-len(".regions.json")])
        wanted = Parts(parts.input, depth if parts.input in READS_DEPTH else None, parts.points_per_side, seed, rule)
        if parts == wanted:
            out.setdefault(parts.input, []).append(parts.name())
    if not out:
        raise ValueError(f"{scores_dir} holds no condition of the rule {rule}, the seed {seed} and the depth {depth}")
    thin = sorted(inp for inp, names in out.items() if len(names) < 2)
    if thin:
        raise ValueError(f"{thin} have one grid only, so no curve")
    return {inp: sorted(names, key=lambda n: parse(n).points_per_side) for inp, names in sorted(out.items())}


def read_json(scores_dir: Path, name: str, suffix: str) -> dict:
    """Read a condition's JSON.

    Raises:
        FileNotFoundError: The JSON is missing.
        ValueError: It holds a NaN, or names no label directory.
    """
    summary = load_scores(scores_dir / f"{name}{suffix}")
    if not summary.get("track_dir_name"):
        raise ValueError(f"{name}{suffix} names no label directory")
    return summary


def read_condition(scores_dir: Path, name: str, min_px: int | None) -> tuple[dict, dict]:
    """Read one condition's score, of the evaluator or of the component objects, and its region count.

    Raises:
        ValueError, FileNotFoundError: A JSON is missing or refused; the score is in pilot mode; the component
            JSON lacks the size; or the score and the count read other label directories.
    """
    regions = read_json(scores_dir, name, ".regions.json")
    if min_px is None:
        summary = read_json(scores_dir, name, ".json")
        if ruler(summary).pilot:
            raise ValueError(f"{name}.json is scored in pilot mode; the curves read the evaluator's scores")
    else:
        summary = read_json(scores_dir, name, ".components.json")
        if min_px not in summary["min_px"]:
            raise ValueError(f"{name}.components.json is scored at {summary['min_px']} px, not {min_px}")
    if summary["track_dir_name"] != regions["track_dir_name"]:
        raise ValueError(f"{name}: the score read {summary['track_dir_name']!r} and the region count "
                         f"{regions['track_dir_name']!r}")
    return summary, regions


def point(name: str, summary: Mapping, regions: Mapping, key: str, min_px: int | None) -> dict:
    """Return one condition's regions per frame and its key's mean over the clips that define it.

    Raises:
        ValueError, KeyError: The key is not in a row, or the score and the count hold other clips.
    """
    if min_px is None:
        values = {r["clip"]: r[key] for r in summary["per_clip"]}
    else:
        values = {r["clip"]: r["by_min_px"][str(min_px)][key] for r in summary["per_clip"]}
    if set(values) != set(regions["clips"]):
        raise ValueError(f"{name}: the score holds other clips than the region count")
    defined = [v for v in values.values() if v is not None]
    return {"condition": name, "points_per_side": parse(name).points_per_side,
            "regions_per_frame": regions["regions_per_frame"], "clips": sorted(values),
            "value": float(np.mean(defined)) if defined else None, "n_clips": len(defined)}


def check_alike(what: str, summaries: Mapping[str, Mapping], fields: Sequence[str]) -> None:
    """Refuse JSONs of one set of curves that differ in a field every one of them must share.

    Raises:
        ValueError: Two JSONs differ in one of `fields`, or one lacks it. The message names the field and both.
    """
    names = list(summaries)
    first = summaries[names[0]]
    for field in fields:
        for name in names:
            if field not in summaries[name]:
                raise ValueError(f"{name}: the {what} records no {field}, so it cannot be compared")
            if summaries[name][field] != first[field]:
                raise ValueError(f"two {what}s differ in {field}, so they are not one set of curves:\n"
                                 f"  {names[0]}: {first[field]!r}\n  {name}: {summaries[name][field]!r}")


def check_rule(what: str, recorded: str | None, rule: str) -> None:
    """Refuse a recorded propagation rule other than the one the curves are named by.

    Raises:
        ValueError: `recorded` is not `RULES[rule]`.
    """
    if recorded != RULES[rule]:
        raise ValueError(f"the {what}s record the propagation {recorded!r}, and the curves of the rule {rule!r} "
                         f"read {RULES[rule]!r}")


def check_conditions(summaries: Mapping[str, Mapping], regions: Mapping[str, Mapping], min_px: int | None,
                     rule: str) -> str:
    """Refuse conditions that are not comparable, and return the evaluator's `eval_code_sha` they share.

    The evaluator's scores pass `check_comparable_table`, which refuses two evaluators, datasets, class sets,
    views, modes, rules or populations. The component scores must share `COMPONENT_FIELDS`, and the region
    counts `COUNT_FIELDS`. The rule every score records must be the one the curves are named by.

    Raises:
        ValueError: Two conditions are not comparable, or record another rule than `rule`.
    """
    if min_px is None:
        shared = check_comparable_table(summaries)
        check_rule("score", shared["propagation"], rule)
        sha = shared["eval_code"]
    else:
        check_alike("component score", summaries, COMPONENT_FIELDS)
        check_rule("component score", next(iter(summaries.values()))["propagation"], rule)
        sha = next(iter(summaries.values()))["eval_code_sha"]
    check_alike("region count", regions, COUNT_FIELDS)
    return sha


def matched(curves: Mapping[str, Sequence[dict]]) -> list[dict]:
    """Read every input's curve at each whole number of regions that every curve reaches.

    Raises:
        ValueError: A curve has two points at one region count, or a point without a value.
    """
    xs, ys = {}, {}
    for inp, points in curves.items():
        ordered = sorted(points, key=lambda p: p["regions_per_frame"])
        x = [p["regions_per_frame"] for p in ordered]
        if any(b <= a for a, b in zip(x, x[1:])):
            raise ValueError(f"{inp}: two grids give one region count, so the curve is not a function of it")
        if any(p["value"] is None for p in ordered):
            raise ValueError(f"{inp}: a point has no value, so the curve has a hole")
        xs[inp], ys[inp] = x, [p["value"] for p in ordered]
    low, high = max(x[0] for x in xs.values()), min(x[-1] for x in xs.values())
    return [{"regions_per_frame": r, **{inp: float(np.interp(r, xs[inp], ys[inp])) for inp in curves}}
            for r in range(math.ceil(low), math.floor(high) + 1)]


def curves_of(scores_dir: str | Path, key: str, min_px: int | None, rule: str = "centre", seed: str | None = "sam",
              depth: str = "da3") -> dict:
    """Read every point of the conditions that differ in input and grid, and return the curves and the readings.

    Raises:
        ValueError: The conditions are not comparable (`check_conditions`), two hold other clips, or `sweep`,
            `read_condition`, `point` or `matched` refuses.
    """
    # Every condition's JSONs, then the checks across them, before any point is read.
    names = sweep(scores_dir, rule, seed, depth)
    read = {n: read_condition(Path(scores_dir), n, min_px) for inp in names for n in names[inp]}
    sha = check_conditions({n: s for n, (s, _) in read.items()}, {n: r for n, (_, r) in read.items()}, min_px, rule)

    # One point per condition, one curve per input, and the readings at matched counts.
    curves = {inp: [point(n, *read[n], key, min_px) for n in names[inp]] for inp in names}
    populations = {tuple(p["clips"]) for points in curves.values() for p in points}
    if len(populations) > 1:
        raise ValueError(f"the conditions hold {len(populations)} different sets of clips; a curve compares one")
    return {"key": key, "objects": "class" if min_px is None else f"components of {min_px} px",
            "eval_code_sha": sha, "rule": rule, "seed": seed, "depth": depth, "curves": curves,
            "matched": matched(curves)}


def main(argv: Sequence[str] | None = None) -> None:
    """Print each input's curve and the readings at matched counts, and write them if asked."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scores-dir", required=True,
                    help="<name>.json, <name>.components.json and <name>.regions.json of each condition")
    ap.add_argument("--key", required=True, help="a key as `metric/view`, such as F1_50/geometric")
    ap.add_argument("--min-px", type=int, default=None, help="read component objects of this size, not classes")
    ap.add_argument("--rule", default="centre", choices=sorted(RULES), help="the propagation rule the curves share")
    ap.add_argument("--seed", default="sam", help="the seed the curves share; none for frame by frame")
    ap.add_argument("--depth", default="da3", choices=DEPTH_MODELS, help="the depth model of the inputs that read it")
    ap.add_argument("--out", default=None, help="the JSON to write")
    args = ap.parse_args(argv)
    seed = None if args.seed == "none" else args.seed
    result = curves_of(args.scores_dir, args.key, args.min_px, args.rule, seed, args.depth)
    print(f"{args.key} / {result['objects']} / {args.rule} / seed {seed} / {args.depth}")
    for inp, points in result["curves"].items():
        print(f"  {inp:12s} " + "  ".join(f"pps{p['points_per_side']}: {p['regions_per_frame']:.2f} regions "
                                          f"{'-' if p['value'] is None else format(p['value'], '.4f')}" for p in points))
    print("  at matched regions per frame:" + (" none reached by every input" if not result["matched"] else ""))
    for row in result["matched"]:
        print(f"    {row['regions_per_frame']:3d}  " + "  ".join(f"{inp} {row[inp]:.4f}" for inp in result["curves"]))
    if args.out:
        write_scores(result, args.out)


if __name__ == "__main__":
    main()
