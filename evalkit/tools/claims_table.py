"""Print the table of the granularity result's claims from score JSONs, one column per dataset.

Each claim names a pair of conditions in each dataset, and a stage: both
propagated, or a per-frame condition against a propagated one. A cell holds
`cond − base` of one key, its video-level 95 % interval, the clip-level
Wilcoxon p and the mark of `paired_stats.verdict`, all from
`paired_stats.compare_pair`. A row gets ★★ only when both of its cells get
a star. A column holds one dataset's scores of one population. A condition
with no score JSON leaves its cell "not measured" and the command exits
non-zero, unless `--allow-unmeasured`. The table refuses a score JSON
without the key, and a pair whose recorded rules contradict its claim's stage.

Usage:
    python -m evalkit.tools.claims_table --key F1_50/geometric \\
        --scores atlas120k=/path/to/atlas120k/scores --scores cholecseg8k=/path/to/cholecseg8k/scores
    # the workbench's table, on the pilot evaluator's JSONs, as markdown
    python -m evalkit.tools.claims_table --key inst_F1_50_labeled --markdown --scores ...
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evalkit.tools.paired_stats import VERDICT_RULE, compare_pair, mark_of, sign_of_key
from evalkit.tools.scores import PER_FRAME, check_comparable, check_one_rule, load_scores, propagation_rule_of

# The datasets of the table, in the order of its columns, each with the names its score JSONs record: the
# evaluator's, and the pilot evaluator's.
DATASETS = ("atlas120k", "cholecseg8k")
RECORDED = {"atlas120k": ("atlas120k", "atlas"), "cholecseg8k": ("cholecseg8k", "cholec")}

# The stages a claim is measured at, each with whether its base and its
# compared condition are segmented frame by frame, the `per_frame` rule.
PROPAGATED = "propagated"
ACROSS = "per frame → propagated"
STAGES: Mapping[str, tuple[bool, bool]] = {PROPAGATED: (False, False), ACROSS: (True, False)}


@dataclass(frozen=True)
class Unmeasured:
    """Stand in for a cell whose pair lacks a score JSON.

    Attributes:
        missing: the tags of the pair without one.
    """

    missing: tuple[str, ...]


@dataclass(frozen=True)
class Claim:
    """One row of the table.

    Attributes:
        name: What the claim says.
        stage: One of `STAGES`.
        pairs: Each dataset's `(base, cond)` condition tags.
    """

    name: str
    stage: str
    pairs: Mapping[str, tuple[str, str]]


# The claims, in the table's order. The granularity result merges to a
# fixed K of 10, set before any merge was scored: `t5_k10` propagates a seed
# merged to 10 regions, `t5_floor` the same seed unmerged, and `<tag>_k10`
# is `<tag>` merged frame by frame by `kmerge`, whose output tag it is.
CLAIMS = (
    Claim("Merging fixes over-splitting (K=10)", PROPAGATED,
          {"atlas120k": ("t5_floor", "t5_k10"), "cholecseg8k": ("t5_floor", "t5_k10")}),
    Claim("Tracking links the regions (rgb, over-split)", ACROSS,
          {"atlas120k": ("op_rgb_perframe_pps24", "rgb_center18"),
           "cholecseg8k": ("ch_rgb_perframe_pps24", "ch_rgb_center")}),
    Claim("Tracking links the regions (normal_edge, over-split)", ACROSS,
          {"atlas120k": ("op_edge_perframe", "op_edge_center"),
           "cholecseg8k": ("ch_edge_perframe", "ch_edge_center")}),
    Claim("Tracking links the regions (rgb, both merged to K=10)", ACROSS,
          {"atlas120k": ("op_rgb_perframe_pps24_k10", "t5_k10"),
           "cholecseg8k": ("ch_rgb_perframe_pps24_k10", "t5_k10")}),
    Claim("Merging added to the operating point (K=10)", PROPAGATED,
          {"atlas120k": ("op_edge_center", "t5_k10"), "cholecseg8k": ("ch_edge_center", "t5_k10")}),
)


def read_column(scores_dir: str, tags: Sequence[str]) -> dict[str, dict | None]:
    """Read each tag's score JSON from one dataset's directory; a tag without one maps to None.

    Raises:
        ValueError: The directory does not exist, or holds the score JSON
            of none of `tags`. Every cell of the column would read "not
            measured", which says nothing about the claims.
    """
    if not os.path.isdir(scores_dir):
        raise ValueError(f"{scores_dir}: no such directory")
    paths = {tag: os.path.join(scores_dir, f"{tag}.json") for tag in tags}
    column = {tag: load_scores(p) if os.path.isfile(p) else None for tag, p in paths.items()}
    if all(s is None for s in column.values()):
        raise ValueError(f"{scores_dir}: holds the score JSON of none of the table's conditions {sorted(tags)}")
    return column


def check_columns(columns: Mapping[str, Mapping[str, dict | None]]) -> dict[str, dict]:
    """Refuse a column whose score JSONs are not its dataset's, or not comparable with one another.

    The two datasets share tags such as `t5_k10`, so a column given the other dataset's directory would print the
    other dataset's numbers under its name. A rgb pair shares no condition with the other pairs of its column, so
    checking each pair alone would let a column hold two populations.

    Returns:
        By column, the library versions that differ between two of its JSONs, by the two tags; reported, not
        refused.

    Raises:
        ValueError: A JSON records no dataset or another dataset than its column's, or two JSONs of a column are
            not comparable (`scores.check_comparable`).
    """
    differ: dict[str, dict] = {}
    for column, summaries in columns.items():
        present = {tag: s for tag, s in summaries.items() if s is not None}
        for tag, s in present.items():
            if s.get("dataset") is None:
                raise ValueError(f"{column}: {tag} records no dataset, so its column cannot be checked")
            if s["dataset"] not in RECORDED[column]:
                raise ValueError(f"{column}: {tag} records the dataset {s['dataset']!r}, not one of "
                                 f"{list(RECORDED[column])}")
        # Every two, so that a JSON that records no domain cannot pass between two that disagree.
        for ta, tb in itertools.combinations(sorted(present), 2):
            try:
                chk = check_comparable(present[ta], present[tb])
            except ValueError as e:
                raise ValueError(f"{column}: {ta} and {tb} cannot share a column: {e}") from e
            if "versions_differ" in chk:
                differ.setdefault(column, {})[f"{ta} and {tb}"] = chk["versions_differ"]
    return differ


def check_stage(claim: Claim, column: str, base: Mapping, cond: Mapping) -> None:
    """Refuse a pair whose propagation rules contradict the claim's stage.

    A claim measured on propagated regions and read as one about regions
    segmented frame by frame, or the reverse, is a different claim. A pair
    of pilot JSONs records no rule, so there is nothing to check.

    Raises:
        ValueError: The base or the compared condition is segmented frame
            by frame where the stage says propagated, or the reverse.
    """
    rules = (propagation_rule_of(base), propagation_rule_of(cond))
    if rules == (None, None):
        return
    if tuple(r == PER_FRAME for r in rules) != STAGES[claim.stage]:
        base_tag, cond_tag = claim.pairs[column]
        raise ValueError(f"the claim is measured {claim.stage!r}, but {base_tag} records the rule {rules[0]!r} "
                         f"and {cond_tag} the rule {rules[1]!r}")


def cell(claim: Claim, summaries: Mapping[str, dict | None], column: str, key: str) -> dict | Unmeasured:
    """Compute one cell: `paired_stats.compare_pair` on `key` for the claim's pair, or `Unmeasured`.

    Raises:
        ValueError: A JSON of the pair holds no value of `key` on any
            clip, the two are not comparable, or their rules contradict the
            claim's stage.
    """
    base_tag, cond_tag = claim.pairs[column]
    base, cond = summaries[base_tag], summaries[cond_tag]
    if base is None or cond is None:
        return Unmeasured(tuple(t for t, s in ((base_tag, base), (cond_tag, cond)) if s is None))
    # A condition scored without the key would otherwise read as not measured.
    for tag, s in ((base_tag, base), (cond_tag, cond)):
        if not any(key in row for row in s["per_clip"]):
            raise ValueError(f"{tag}'s score JSON holds no {key!r} on any clip")
    pair = compare_pair(base, cond, keys=[key])
    check_stage(claim, column, base, cond)
    return pair


def build(dirs: Mapping[str, str], key: str) -> tuple[list[tuple[Claim, list[dict | Unmeasured]]], str | None,
                                                       dict[str, dict]]:
    """Compute every cell of the table from each dataset's score directory.

    Returns:
        Each claim with its cells in the order of `DATASETS`; the table's
        propagation rule, None when every JSON is a pilot JSON; and, by
        column, the library versions that differ (`check_columns`).

    Raises:
        ValueError: A column is refused by `read_column` or
            `check_columns`, the table holds two propagation rules, `key`
            has no direction or is a reference value, or a cell is refused.
    """
    # Every condition each column names, read before anything is computed.
    columns = {ds: read_column(dirs[ds], sorted({t for c in CLAIMS for t in c.pairs[ds]})) for ds in DATASETS}
    loaded = {f"{ds}/{t}": s for ds, col in columns.items() for t, s in col.items() if s is not None}

    # The checks that hold across cells: one dataset and population per column, one rule per table, a key that
    # can be marked.
    differ = check_columns(columns)
    rule = check_one_rule(loaded)
    for name, s in loaded.items():
        try:
            sign = sign_of_key(s, key)
        except KeyError:
            raise ValueError(f"{key!r} is not a key with a direction in {name}'s score JSON") from None
        if sign == 0:
            raise ValueError(f"{key!r} is a reference value: no star marks it, so it goes in a table that carries "
                             "none")

    # Each cell, refused with its claim and its dataset named.
    rows = []
    for claim in CLAIMS:
        cells = []
        for ds in DATASETS:
            try:
                cells.append(cell(claim, columns[ds], ds, key))
            except ValueError as e:
                raise ValueError(f"{claim.name} ({ds}): {e}") from e
        rows.append((claim, cells))
    return rows, rule, differ


def star_in(pair: dict | Unmeasured, key: str) -> bool:
    """Return whether a cell gets a star, the mark `paired_stats` read; a cell not measured or not defined gets none."""
    r = None if isinstance(pair, Unmeasured) else pair["metrics"].get(key)
    return r is not None and mark_of(r) == "★"


def format_cell(pair: dict | Unmeasured, key: str) -> str:
    """Write one cell: the difference, the video interval, the clip-level p, the mark and the population."""
    if isinstance(pair, Unmeasured):
        return "not measured: no " + ", ".join(f"{t}.json" for t in pair.missing)
    r = pair["metrics"].get(key)
    if r is None:
        return "not defined on any common clip"
    ci = r["ci95_video"]
    interval = "[none]" if ci is None else f"[{ci[0]:+.3f},{ci[1]:+.3f}]"
    # The p is written to five places, so a p of 0.0 is one below 0.000005.
    p = "none" if r["wilcoxon_p"] is None else "<0.00001" if r["wilcoxon_p"] == 0 else r["wilcoxon_p"]
    size = f"{r['n_clips']} clips, {r['n_videos']} videos"
    # A population the key shrank is shown against the pair's, never silently.
    if (r["n_clips"], r["n_videos"]) != (pair["n_clips"], pair["n_videos"]):
        size = f"{r['n_clips']}/{pair['n_clips']} clips, {r['n_videos']}/{pair['n_videos']} videos"
    return f"{r['delta_mean']:+.4f} {interval} p={p}{mark_of(r)} ({size})"


def unmeasured(rows: list[tuple[Claim, list[dict | Unmeasured]]]) -> list[str]:
    """List the rows with a cell not measured, by their claims' names."""
    return [claim.name for claim, cells in rows if any(isinstance(p, Unmeasured) for p in cells)]


def render(rows: list[tuple[Claim, list[dict | Unmeasured]]], rule: str | None, key: str, markdown: bool,
           differ: Mapping[str, dict] | None = None) -> str:
    """Write the table, as text or as markdown, with its header and the counts of rows starred and not measured."""
    stage_note = ("the pilot evaluator's JSONs record no propagation rule, so each stage is as declared"
                  if rule is None else f"propagation rule {rule}; each stage is checked against the rules recorded")
    lines = [f"# Claims of the granularity result, on {key}",
             f"# ★ / ✗: {VERDICT_RULE}. ★★: a star in both datasets.",
             f"# {stage_note}"]
    lines += [f"# note: library versions differ in {column} between {tags}: {versions}"
              for column, pairs in (differ or {}).items() for tags, versions in pairs.items()]
    both = [all(star_in(p, key) for p in cells) for _, cells in rows]
    if markdown:
        lines += ["", f"| claim | stage | {' | '.join(DATASETS)} | both |", "|---" * (len(DATASETS) + 3) + "|"]
        for (claim, cells), b in zip(rows, both):
            lines.append(f"| {claim.name} | {claim.stage} | "
                         + " | ".join(format_cell(p, key) for p in cells) + f" | {'★★' if b else ''} |")
    else:
        for (claim, cells), b in zip(rows, both):
            lines += ["", f"{'★★' if b else '  '} [{claim.stage}] {claim.name}"]
            lines += [f"    {ds:12s} {format_cell(p, key)}" for ds, p in zip(DATASETS, cells)]
    lines += ["", f"→ {sum(both)} / {len(rows)} rows get a star in both datasets; "
                  f"{len(unmeasured(rows))} rows have a cell not measured"]
    return "\n".join(lines)


def parse_scores(values: Sequence[str]) -> dict[str, str]:
    """Read `--scores DATASET=DIR` into each dataset's directory.

    Raises:
        ValueError: A value is not `DATASET=DIR`, names a dataset the table
            has no column for, or names one twice; or a dataset is missing.
    """
    dirs: dict[str, str] = {}
    for value in values:
        dataset, sep, path = value.partition("=")
        if not (sep and dataset and path):
            raise ValueError(f"--scores takes DATASET=DIR, not {value!r}")
        if dataset not in DATASETS:
            raise ValueError(f"--scores names {dataset!r}; the table's datasets are {list(DATASETS)}")
        if dataset in dirs:
            raise ValueError(f"--scores names {dataset!r} twice")
        dirs[dataset] = path
    missing = [d for d in DATASETS if d not in dirs]
    if missing:
        raise ValueError(f"--scores names no directory for {missing}")
    return dirs


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", action="append", default=[], required=True,
                    help="DATASET=DIR, once per dataset: the directory of its score JSONs, one <tag>.json each.")
    ap.add_argument("--key", required=True,
                    help="The per-clip key every cell compares, e.g. F1_50/geometric, or inst_F1_50_labeled "
                         "on the pilot evaluator's JSONs.")
    ap.add_argument("--markdown", action="store_true", help="Print the table as markdown.")
    ap.add_argument("--allow-unmeasured", action="store_true",
                    help="Exit 0 though a cell has no score JSON; its row is printed as not measured.")
    args = ap.parse_args(argv)
    try:
        rows, rule, differ = build(parse_scores(args.scores), args.key)
    except (ValueError, OSError) as e:
        raise SystemExit(str(e)) from e
    print(render(rows, rule, args.key, args.markdown, differ))
    # A claim whose cell lacks a score JSON would leave the table without a word, as a row with no star.
    missing = unmeasured(rows)
    if missing and not args.allow_unmeasured:
        print(f"{len(missing)} row(s) have a cell not measured: {missing}; pass --allow-unmeasured to accept them",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
