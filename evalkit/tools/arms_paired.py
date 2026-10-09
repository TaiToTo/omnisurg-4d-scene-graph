"""Print one key's paired difference for several pairs of conditions, one line per pair.

Each line gives the mean of `cond − base` over the clips both define the
key on, its video-level bootstrap 95 % interval, how many clips moved up,
stayed and moved down, the clip-level Wilcoxon p for reference, and the
mark `paired_stats.verdict` gives the interval in the key's direction. A
reference value, such as `time_IoU`, is never marked.

Every two score JSONs of the table are checked with
`scores.check_comparable`. So one table holds one evaluator, dataset,
class set, set of views, mode, population and propagation rule; a
`per_frame` condition may be compared with a condition of either rule.

Usage:
    python -m evalkit.tools.arms_paired --eval-dir /path/to/scores \\
        --key F1_50/geometric --pairs rgb:rgb_edge,normal:normal_edge
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np

from evalkit.tools.compare_eval import DIRECTION, metrics_of
from evalkit.tools.paired_stats import VERDICT_RULE, boot_ci, load_json, verdict, video_of
from evalkit.tools.scores import check_comparable, check_one_rule, defined_clips, rows_of

# The width of the column that names a pair, as the workbench printed it.
NAME_WIDTH = 34


def parse_pairs(text: str) -> list[tuple[str, str]]:
    """Read `base:cond,base:cond` into `(base, cond)` tuples, in the order given.

    Raises:
        ValueError: An item is not `<base>:<cond>`, pairs a condition with
            itself, or no pair is given.
    """
    pairs = []
    for item in (p.strip() for p in text.split(",")):
        if not item:
            continue
        base, sep, cond = item.partition(":")
        if not (sep and base and cond) or ":" in cond:
            raise ValueError(f"{item!r}: a pair is <base>:<cond>")
        if base == cond:
            raise ValueError(f"{item!r}: a pair compares two conditions, not one with itself")
        pairs.append((base, cond))
    if not pairs:
        raise ValueError("no pair is given")
    return pairs


def check_table(summaries: Mapping[str, Mapping]) -> dict:
    """Check every two score JSONs of a table with `scores.check_comparable`, and say what they share.

    Checking each pair alone is not enough. Two pairs can each pass and
    still put two evaluators, two datasets or two propagation rules in one
    table. Checking each JSON against the first is not enough either. The
    check compares the pilot evaluator's domain and dataset only when both
    JSONs record them, so two JSONs that disagree each pass a check against
    a first JSON that records neither.

    Args:
        summaries: The score JSONs of the table, by tag; at least two.

    Returns:
        `clips`, sorted; `eval_code`; `propagation`, the table's rule, or
        None for pilot JSONs, which record none; and `versions_differ`,
        the library versions that differ between two JSONs, by the two
        tags, reported and not refused.

    Raises:
        ValueError: Two of the JSONs are not comparable.
    """
    shared, differ = None, {}
    for ta, tb in combinations(summaries, 2):
        try:
            shared = check_comparable(summaries[ta], summaries[tb])
        except ValueError as e:
            raise ValueError(f"{ta} and {tb}: {e}") from e
        if "versions_differ" in shared:
            differ[f"{ta} and {tb}"] = shared["versions_differ"]
    # Every two JSONs passed, so no two hold different rules besides
    # `per_frame`; this only reads which rule the table holds.
    return dict(clips=shared["clips"], eval_code=shared["eval_code"], propagation=check_one_rule(summaries),
                versions_differ=differ)


def pair_row(base: Mapping, cond: Mapping, clips: Sequence[str], key: str, sign: int) -> dict:
    """The paired difference of `key` between two conditions, over the clips both define it on.

    Args:
        base: The base condition's score JSON.
        cond: The compared condition's score JSON.
        clips: The table's clips, in the order the differences are taken.
        key: The key compared.
        sign: Which way `key` is better: +1, -1, or 0 for a reference value.

    Returns:
        `n_clips` and `n_videos`, the clips and videos the key is defined
        on in both; `delta`, the mean of `cond − base`; `ci95_video`, its
        video-level bootstrap interval, None below two videos; `up`, `same`
        and `down`, the clips on which `cond − base` is above, at or below
        zero; `p_clip`, the clip-level Wilcoxon p, None when no clip moved;
        and `mark`, from `paired_stats.verdict`.

    Raises:
        ValueError: `key` is defined on no clip of both conditions.
    """
    # scipy is the `tools` extra; the module imports without it.
    from scipy import stats

    # The differences, on the clips both define the key on.
    a, b = rows_of(base), rows_of(cond)
    kept = defined_clips(a, b, list(clips), key)
    if not kept:
        raise ValueError(f"{key} is defined on no clip of both conditions")
    d = np.array([b[c][key] - a[c][key] for c in kept], dtype=np.float64)
    videos = np.array([video_of(c) for c in kept])
    n_videos = len(np.unique(videos))

    # The interval and its mark; below two videos there is no interval to read.
    ci = boot_ci(d, videos) if n_videos >= 2 else None
    return dict(
        n_clips=len(kept), n_videos=n_videos, delta=float(d.mean()), ci95_video=ci,
        up=int((d > 0).sum()), same=int((d == 0).sum()), down=int((d < 0).sum()),
        p_clip=float(stats.wilcoxon(d, zero_method="wilcox").pvalue) if np.any(d != 0) else None,
        mark=verdict(ci, sign),
    )


def compare_pairs(summaries: Mapping[str, Mapping], pairs: Sequence[tuple[str, str]], key: str) -> dict:
    """The table: one row per pair, on one key, over a population every JSON shares.

    Args:
        summaries: Every condition's score JSON, by tag.
        pairs: `(base, cond)` tags, in the order the rows are printed.
        key: The key compared, as the JSONs spell it.

    Returns:
        `clips`, `n_videos`, `eval_code`, `propagation` and
        `versions_differ`, as `check_table` gives them; `key`; `sign`; and
        `rows`, a list of `(base, cond, row)` with `row` from `pair_row`.

    Raises:
        ValueError: The JSONs are not comparable, `key` is not a key they
            report, or it is defined on no clip of a pair.
    """
    # What every JSON of the table shares.
    table = check_table(summaries)
    clips = table["clips"]

    # The key and which way it is better; comparable JSONs report the same keys.
    signs = dict(metrics_of(next(iter(summaries.values()))))
    if key not in signs:
        raise ValueError(f"{key} is not a key these JSONs report; the keys are {list(signs)}")

    # One row per pair.
    rows = []
    for base, cond in pairs:
        try:
            rows.append((base, cond, pair_row(summaries[base], summaries[cond], clips, key, signs[key])))
        except ValueError as e:
            raise ValueError(f"{base}:{cond}: {e}") from e
    return dict(clips=clips, n_videos=len({video_of(c) for c in clips}), eval_code=table["eval_code"],
                propagation=table["propagation"], versions_differ=table["versions_differ"],
                key=key, sign=signs[key], rows=rows)


def format_row(base: str, cond: str, row: Mapping, n_clips: int, n_videos: int) -> str:
    """One printed line: the pair, the mean difference, the interval, the moves, p and the mark.

    The columns are the workbench's, so that a line can be compared with
    its line byte for byte. A row on fewer clips than the table says so.
    """
    ci = row["ci95_video"]
    interval = "none" if ci is None else f"[{ci[0]:+.4f},{ci[1]:+.4f}]"
    moves = f"{row['up']}-{row['same']}-{row['down']}"
    p = "none" if row["p_clip"] is None else f"{row['p_clip']:.4f}"
    line = f"{f'{cond} − {base}':<{NAME_WIDTH}}{row['delta']:+8.4f}{interval:>20}{moves:>12}{p:>9}"
    if row["mark"]:
        line += f" {row['mark']}"
    if (row["n_clips"], row["n_videos"]) != (n_clips, n_videos):
        line += f"   [{row['n_clips']}/{n_clips} clips, {row['n_videos']}/{n_videos} videos]"
    return line


def print_table(table: Mapping) -> None:
    """Print the table: what it was measured on, one line per pair, and the rule that marks a line."""
    n_clips, n_videos = len(table["clips"]), table["n_videos"]
    rule = "" if table["propagation"] is None else f", propagation={table['propagation']}"
    print(f"{n_clips} clips / {n_videos} videos, eval_code={table['eval_code']}{rule}")
    for tags, differ in table["versions_differ"].items():
        print(f"note: library versions differ between {tags}: {differ}")
    print(f"key {table['key']}: {DIRECTION[table['sign']]}; each line is cond − base, paired by clip;")
    print("the interval resamples videos, and p is the clip-level Wilcoxon test, for reference\n")
    print(f"{'pair':<{NAME_WIDTH}}{'Δ':>8}{'95% CI':>20}{'up-eq-down':>12}{'p':>9}")
    for base, cond, row in table["rows"]:
        print(format_row(base, cond, row, n_clips, n_videos))
    print(f"\n★ / ✗: {VERDICT_RULE}.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True, help="The directory of the score JSONs, one <tag>.json per condition.")
    ap.add_argument("--key", required=True,
                    help="The key compared, as the JSONs spell it: F1_50/geometric, or inst_F1_50 on a pilot JSON.")
    ap.add_argument("--pairs", required=True, help="<base>:<cond>, comma-separated; each line is cond − base.")
    args = ap.parse_args()

    # Every condition's JSON, read before any is compared; then the table.
    try:
        pairs = parse_pairs(args.pairs)
        tags = list(dict.fromkeys(tag for pair in pairs for tag in pair))
        table = compare_pairs({tag: load_json(tag, args.eval_dir) for tag in tags}, pairs, args.key)
    except (ValueError, OSError) as e:
        raise SystemExit(str(e)) from e
    print_table(table)


if __name__ == "__main__":
    main()
