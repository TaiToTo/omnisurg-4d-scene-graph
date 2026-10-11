"""Print one key's paired difference for several pairs of conditions, one line per pair.

Each line gives the mean of `cond − base` over the clips both define the
key on, its video-level bootstrap 95 % interval, how many clips moved up,
stayed and moved down, the clip-level Wilcoxon p for reference, and the
mark that `paired_stats.mark_of` reads from the interval in the key's
direction. A key of direction 0, such as `time_IoU`, is never marked.

The score JSONs of one table must pass `scores.check_comparable_table`.

Usage:
    python -m evalkit.tools.arms_paired --eval-dir /path/to/scores \\
        --key F1_50/geometric --pairs rgb:rgb_edge,normal:normal_edge
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence

import numpy as np

from evalkit.tools.compare_eval import DIRECTION
from evalkit.tools.paired_stats import MARK_RULE, boot_ci, load_json, mark_of, video_of
from evalkit.tools.scores import check_comparable_table, defined_clips, rows_of, sign_of_key

# The width of the column that names a pair, as the workbench printed it; a longer name widens it.
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


def pair_row(base: Mapping, cond: Mapping, clips: Sequence[str], key: str, sign: int) -> dict:
    """Compare `key` between two conditions, over the clips both define it on.

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
        and `mark`, from `paired_stats.mark_of`.

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
        mark=mark_of(ci, sign),
    )


def compare_pairs(summaries: Mapping[str, Mapping], pairs: Sequence[tuple[str, str]], key: str) -> dict:
    """Build the table: one row per pair, on one key, over a population every JSON shares.

    Args:
        summaries: Every condition's score JSON, by tag.
        pairs: `(base, cond)` tags, in the order the rows are printed.
        key: The key compared, as the JSONs spell it.

    Returns:
        What `scores.check_comparable_table` returns; `n_videos`; `key`;
        `sign`; and `rows`, a list of `(base, cond, row)` with `row` from
        `pair_row`.

    Raises:
        ValueError: `scores.check_comparable_table` refuses the JSONs;
            `key` has no direction on them (`scores.sign_of_key`); or it is
            defined on no clip of a pair.
    """
    # What every JSON of the table shares.
    table = check_comparable_table(summaries)
    clips = table["clips"]

    # The key and which way it is better; comparable JSONs report the same keys.
    sign = sign_of_key(next(iter(summaries.values())), key)

    # One row per pair.
    rows = []
    for base, cond in pairs:
        try:
            rows.append((base, cond, pair_row(summaries[base], summaries[cond], clips, key, sign)))
        except ValueError as e:
            raise ValueError(f"{base}:{cond}: {e}") from e
    return dict(table, n_videos=len({video_of(c) for c in clips}), key=key, sign=sign, rows=rows)


def format_row(base: str, cond: str, row: Mapping, n_clips: int, n_videos: int, width: int = NAME_WIDTH) -> str:
    """Write one line: the pair, the mean difference, the interval, the moves, p and the mark.

    The columns are the workbench's, so that a line can be compared with
    its line byte for byte. A row on fewer clips than the table says so.
    """
    ci = row["ci95_video"]
    interval = "none" if ci is None else f"[{ci[0]:+.4f},{ci[1]:+.4f}]"
    moves = f"{row['up']}-{row['same']}-{row['down']}"
    p = "none" if row["p_clip"] is None else f"{row['p_clip']:.4f}"
    line = f"{f'{cond} − {base}':<{width}}{row['delta']:+8.4f}{interval:>20}{moves:>12}{p:>9}"
    if row["mark"]:
        line += f" {row['mark']}"
    if (row["n_clips"], row["n_videos"]) != (n_clips, n_videos):
        line += f"   [{row['n_clips']}/{n_clips} clips, {row['n_videos']}/{n_videos} videos]"
    return line


def print_table(table: Mapping) -> None:
    """Print the table: what it was measured on, one line per pair, and the rule that marks a line."""
    n_clips, n_videos = len(table["clips"]), table["n_videos"]
    width = max([NAME_WIDTH] + [len(f"{cond} − {base}") + 1 for base, cond, _ in table["rows"]])
    rule = "" if table["propagation"] is None else f", propagation={table['propagation']}"
    print(f"{table['dataset']}: {n_clips} clips / {n_videos} videos, eval_code={table['eval_code']}{rule}")
    for (a, b), differ in table["versions_differ"].items():
        print(f"note: library versions differ between {a} and {b}: {differ}")
    print(f"key {table['key']}: {DIRECTION[table['sign']]}; each line is cond − base, paired by clip;")
    print("the interval resamples videos, and p is the clip-level Wilcoxon test, for reference\n")
    print(f"{'pair':<{width}}{'Δ':>8}{'95% CI':>20}{'up-eq-down':>12}{'p':>9}")
    for base, cond, row in table["rows"]:
        print(format_row(base, cond, row, n_clips, n_videos, width))
    print(f"\n★ / ✗: {MARK_RULE}.")


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
