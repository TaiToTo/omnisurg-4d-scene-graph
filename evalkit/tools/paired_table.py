"""Print the pilot's table of paired comparisons: blocks of rows under headings, a row per pair of conditions.

Each cell holds the mean of `cond − base` for one key, with the mark that
`paired_stats.mark_of` gives. A line under each row gives the first key's
interval and wins. Every two conditions of the table must pass
`scores.check_comparable`. The table is pasted as printed, never copied by hand.

Usage:
    python -m evalkit.tools.paired_table --eval-dir /path/to/scores
"""
from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence

import numpy as np

from evalkit.tools.compare_eval import DIRECTION
from evalkit.tools.paired_stats import MARK_RULE, boot_ci, mark_of, video_of
from evalkit.tools.scores import (
    check_comparable_table,
    defined_clips,
    is_pilot_json,
    load_scores,
    rows_of,
    sign_of_key,
    signs_of,
)

# The keys the table reports on a pilot JSON, in the workbench's order. F1
# is never read alone, so the boundary keys and the class map are reported with it.
PILOT_COLUMNS = ("inst_F1_50", "boundary_F", "boundary_R_raw", "GT_mIoU")

# The fewest clips a cell is computed on, as the workbench's table had it.
MIN_CLIPS = 5

# What a cell without a value prints.
NO_VALUE = "—"

# What follows a mark when the clips' signs disagree with it.
SIGN_NOTE = "(sign)"

# The pilot's table, the rows of the workbench's `summary97.py`: (heading, rows), each row (label, cond, base),
# whose difference is `cond − base`.
PILOT_BLOCKS = (
    ("Does propagation beat per-frame segmentation?", (
        ("normals+edges: propagated − per frame", "op_edge_center", "op_edge_perframe"),
        ("RGB: propagated − per frame (pps24)", "rgb_center18", "op_rgb_perframe_pps24"),
    )),
    ("Does merging the seed to K regions undo over-segmentation? (only the seeds differ)", (
        ("K=10 − no merge", "t5_k10", "t5_floor"),
        ("K=8 − no merge", "t5_k8", "t5_floor"),
        ("K=10 − K=8 (the choice of K)", "t5_k10", "t5_k8"),
    )),
    ("Merged RGB against unmerged normals+edges (only the RGB seed is merged)", (
        ("K=10 − normals+edges", "t5_k10", "op_edge_center"),
        ("K=8 − normals+edges", "t5_k8", "op_edge_center"),
    )),
    ("Geometry as the seed's input (each at its own number of regions)", (
        ("normals+edges − RGB", "op_edge_center", "rgb_center18"),
        ("normals − RGB", "op_normal_center", "rgb_center18"),
        ("depth − RGB", "depth_center18", "rgb_center18"),
    )),
    ("The 3D edges drawn into RGB (propagated)", (
        ("RGB+edges − RGB", "rgbedge_center18", "rgb_center18"),
        ("RGB+edges − normals+edges", "rgbedge_center18", "op_edge_center"),
        ("RGB+edges − K=10", "rgbedge_center18", "t5_k10"),
    )),
    ("Depth from Pi3X in place of DA3 (per frame, pps8)", (
        ("normals+edges: Pi3X − DA3", "op_edge_pf_pps8_pi3x", "op_edge_pf_pps8"),
        ("normals: Pi3X − DA3", "op_normal_pf_pps8_pi3x", "op_normal_pf_pps8"),
    )),
    ("Points per side of the prompt grid (RGB, per frame)", (
        ("pps4 − pps8", "op_rgb_perframe_pps4", "op_rgb_perframe_pps8"),
        ("pps8 − pps24", "op_rgb_perframe_pps8", "op_rgb_perframe_pps24"),
        ("pps12 − pps24", "op_rgb_perframe_pps12", "op_rgb_perframe_pps24"),
    )),
)


def tags_of(blocks: Sequence) -> list[str]:
    """List the conditions a table names, each once, in the order the table first names them."""
    seen = {}
    for _, rows in blocks:
        for _, cond, base in rows:
            seen.setdefault(cond, None)
            seen.setdefault(base, None)
    return list(seen)


def load_conditions(blocks: Sequence, eval_dir: str) -> dict[str, dict]:
    """Read the score JSON of every condition a table names, by tag.

    Raises:
        ValueError: A condition has no score JSON, or its JSON holds a NaN.
    """
    out = {}
    for tag in tags_of(blocks):
        path = os.path.join(eval_dir, f"{tag}.json")
        try:
            out[tag] = load_scores(path)
        except OSError as e:
            raise ValueError(f"{tag}: no score JSON to read ({e})") from e
    return out


def columns_of(summary: Mapping, keys: Sequence[str] = ()) -> list[tuple[str, int]]:
    """List the keys a table reports, each with the way it is better: `keys`, or `PILOT_COLUMNS` on a pilot JSON.

    Raises:
        ValueError: No keys are given for an evaluator JSON, whose table
            keys are not settled; a key has no direction on the JSON
            (`scores.sign_of_key`); or a key is a reference value, which no
            star marks and so goes in a table that carries none.
    """
    pilot = is_pilot_json(summary)
    markable = [k for k, sign in signs_of(summary).items() if sign != 0]
    if not keys:
        if not pilot:
            raise ValueError(f"the table's keys on the evaluator's scores are not settled; name them with --keys, "
                             f"from {markable}")
        keys = PILOT_COLUMNS
    out = []
    for key in keys:
        sign = sign_of_key(summary, key)
        if sign == 0:
            raise ValueError(f"{key} is a reference value: no star marks it, so it goes in a table that carries none")
        out.append((key, sign))
    return out


def check_table(tags: Sequence[str], summaries: Mapping[str, Mapping], columns: Sequence[tuple[str, int]]) -> dict:
    """Check that a table's conditions can share one table, and return what they share.

    Args:
        tags: The table's conditions, in its order, two at least; each has a score JSON in `summaries`.
        summaries: The score JSONs, by tag.
        columns: The keys the table reports, with their directions.

    Returns:
        What `scores.check_comparable_table` returns, and `regions`,
        whether the table has a column of regions, which only pilot JSONs
        record.

    Raises:
        ValueError: `scores.check_comparable_table` refuses the conditions;
            a key is in no row of a condition; or a pilot JSON records no
            `n_regions_mean`.
    """
    shared = check_comparable_table({t: summaries[t] for t in tags})
    # A key in no row is a wrong key or a wrong JSON, not a key undefined on every clip.
    for tag in tags:
        absent = [k for k, _ in columns if not any(k in r for r in summaries[tag]["per_clip"])]
        if absent:
            raise ValueError(f"{tag}: {absent} in no row")
    regions = is_pilot_json(summaries[tags[0]])
    if regions:
        unrecorded = [t for t in tags if not isinstance(summaries[t].get("n_regions_mean"), (int, float))]
        if unrecorded:
            raise ValueError(f"{unrecorded}: a pilot JSON records n_regions_mean, and these do not")
    return dict(shared, regions=regions)


def cell_of(base: Mapping, cond: Mapping, clips: list[str], key: str, sign: int) -> dict:
    """Compare one key of two conditions: the mean difference, its interval and mark, and the clips' signs.

    `mean`, `ci` and `mark` are None when fewer than `MIN_CLIPS` clips or
    fewer than two videos define the key: one video has no interval, and
    the point it would give reads as a mark.

    Args:
        base: The base condition's rows, by clip.
        cond: The compared condition's rows, by clip.
        clips: The population.
        key: The key compared.
        sign: Which way the key is better.
    """
    ks = defined_clips(base, cond, clips, key)
    videos = np.array([video_of(c) for c in ks])
    out = dict(n_clips=len(ks), n_videos=int(len(np.unique(videos))), mean=None, ci=None, mark=None)
    if out["n_clips"] < MIN_CLIPS or out["n_videos"] < 2:
        return out
    d = (np.array([cond[c][key] for c in ks], dtype=np.float64)
         - np.array([base[c][key] for c in ks], dtype=np.float64))
    ci = boot_ci(d, videos)
    out.update(mean=float(d.mean()), ci=ci, mark=mark_of(ci, sign),
               wins=int((d * sign > 0).sum()), losses=int((d * sign < 0).sum()))
    return out


def mark_text(cell: Mapping) -> str:
    """Write the cell's mark, with `SIGN_NOTE` when as many clips or more moved against the mark as with it."""
    if cell["mark"] == "★" and cell["losses"] >= cell["wins"]:
        return "★" + SIGN_NOTE
    if cell["mark"] == "✗" and cell["wins"] >= cell["losses"]:
        return "✗" + SIGN_NOTE
    return cell["mark"] or " "


def table_lines(blocks: Sequence, summaries: Mapping[str, Mapping], keys: Sequence[str] = ()) -> list[str]:
    """Return the lines of a table, from the score JSONs of the conditions it names.

    Raises:
        ValueError: A row compares a condition with itself, a condition has
            no score JSON, the table's conditions cannot share a table
            (`check_table`), or its keys cannot be reported (`columns_of`).
    """
    # The conditions, two to a row, each with a score JSON. A row of one condition
    # differs by zero everywhere, and `check_table` checks a condition only against another.
    same = [label for _, rows in blocks for label, cond, base in rows if cond == base]
    if same:
        raise ValueError(f"rows {same} compare a condition with itself")
    tags = tags_of(blocks)
    missing = [t for t in tags if t not in summaries]
    if missing:
        raise ValueError(f"no score JSON for {missing}")

    # The keys and their directions, read from the first condition; `check_table` makes every other share its ruler.
    columns = columns_of(summaries[tags[0]], keys)
    shared = check_table(tags, summaries, columns)
    clips = shared["clips"]
    n_videos = len({video_of(c) for c in clips})

    # The header: what the table was measured with, on what, and how a cell is read.
    rule = "" if shared["propagation"] is None else f" / propagation {shared['propagation']}"
    lines = [f"{shared['dataset']}: {len(clips)} clips / {n_videos} videos / "
             f"eval_code_sha {shared['eval_code'][:8]}{rule}",
             f"★ better, ✗ worse, each in its key's direction: {MARK_RULE}",
             f"{SIGN_NOTE} after a mark: as many clips or more moved against the mark as with it",
             f"{NO_VALUE}: fewer than {MIN_CLIPS} clips or 2 videos define the key",
             "directions: " + ", ".join(f"{k} {DIRECTION[s]}" for k, s in columns)]
    if not shared["regions"]:
        lines.append("regions: the evaluator records no n_regions_mean, only objects.predicted, per view and summed "
                     "over the scored frames; the table has no column of it")
    for (a, b), differ in shared["versions_differ"].items():
        lines.append(f"note: library versions differ between {a} and {b}: {differ}")
    lines.append("")

    # Each block: its heading, the keys, and each row's cells, its interval line and any shrunken population.
    label_w = max(28, *(len(label) for _, rows in blocks for label, _, _ in rows))
    widths = [max(17, len(k) + 1) for k, _ in columns]
    head = " " * (label_w + 2) + "".join(f"{k:>{w}s}" for (k, _), w in zip(columns, widths))
    head += "   regions" if shared["regions"] else ""
    for heading, rows in blocks:
        lines += ["", "=" * len(head), heading, head]
        for label, cond, base in rows:
            ra, rb = rows_of(summaries[base]), rows_of(summaries[cond])
            cells = [cell_of(ra, rb, clips, k, s) for k, s in columns]
            line = f"  {label:{label_w}s}"
            for cell, w in zip(cells, widths):
                text = NO_VALUE if cell["mean"] is None else f"{cell['mean']:+9.4f}{mark_text(cell):>8s}"
                line += f"{text:>{w}s}"
            if shared["regions"]:
                line += f"   {summaries[cond]['n_regions_mean']:.2f}/{summaries[base]['n_regions_mean']:.2f}"
            lines.append(line)
            lead = cells[0]
            if lead["mean"] is not None:
                lines.append(f"{'':{label_w + 2}s}{columns[0][0]} CI [{lead['ci'][0]:+.4f},{lead['ci'][1]:+.4f}] "
                             f"wins-losses {lead['wins']}-{lead['losses']} / {lead['n_videos']} videos")
            # A mean over fewer clips than the population is said, never averaged silently.
            for (k, _), cell in zip(columns, cells):
                if cell["n_clips"] < len(clips):
                    lines.append(f"{'':{label_w + 2}s}[{k}: {cell['n_clips']}/{len(clips)} clips, "
                                 f"{cell['n_videos']}/{n_videos} videos]")
    return lines


def main(argv: Sequence[str] | None = None) -> None:
    """Print the pilot's table from the score JSONs in the directory the arguments name."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True, help="The directory of the score JSONs, one <tag>.json per condition.")
    ap.add_argument("--keys", default="",
                    help=f"The keys to report, comma-separated: {','.join(PILOT_COLUMNS)} on pilot JSONs by default, "
                         "and required on the evaluator's.")
    args = ap.parse_args(argv)
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    try:
        lines = table_lines(PILOT_BLOCKS, load_conditions(PILOT_BLOCKS, args.eval_dir), keys)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    print("\n".join(lines))


if __name__ == "__main__":
    main()
