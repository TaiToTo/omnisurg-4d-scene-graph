"""Print the tables of two analyses from the JSONs those analyses wrote, computing nothing new.

- each geometry input against rgb, merged to one K (`fair_merge_recheck --mode input_track`);
- identity over time (`summarize_track12`), with no mark: its measures are reference values.

Every mark is the one `paired_stats.mark_of` gives on the stored
video-level interval, in the key's direction. Each table refuses a JSON
that is not the one it prints, rather than printing it under the wrong
heading. The tables are pasted as printed, never copied by hand.

Usage:
    python -m evalkit.tools.print_status_tables --input-track e1_input_track.json --identity e2_track12.json
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from evalkit.tools.paired_stats import MARK_RULE, mark_of, sign_only_note
from evalkit.tools.scores import PILOT_SIGNS

SEPARATOR = "=" * 100

# The geometry inputs and the K each was merged to, in the order of the table's rows.
INPUTS = ("edge", "normal", "depth")
KS = (0, 6, 8, 10, 12)
# The keys of each table and their headings, in the order of its columns.
INPUT_TRACK_KEYS = (("full", "F1@.5"), ("labeled", "F1 labeled"), ("boundary_F", "boundary F"),
                    ("boundary_R_raw", "boundary R raw"), ("GT_mIoU", "mIoU"))
# The identity keys, their headings and their directions: +1 higher is better, -1 lower is better.
IDENTITY_KEYS = (("hold_mean", "hold", +1), ("idf1", "IDF1", +1), ("idsw_per_track", "IDsw/track", -1),
                 ("frag_per_track", "Frag/track", -1))
# The pilot key each column holds. The analysis scored with the pilot
# evaluator's functions but names a key by its domain alone; each column is
# read in its pilot key's direction.
PILOT_KEY_OF = {"full": "inst_F1_50", "labeled": "inst_F1_50_labeled", "boundary_F": "boundary_F",
                "boundary_R_raw": "boundary_R_raw", "GT_mIoU": "GT_mIoU"}


def load_analysis(path: str) -> dict:
    """Read one analysis JSON.

    Raises:
        ValueError: The file holds a `NaN` or an infinity, which would print as a value and give no mark.
    """
    def refuse(token: str):
        raise ValueError(f"{path}: holds {token}, which is not a JSON number")

    with open(path, encoding="utf-8") as f:
        return json.load(f, parse_constant=refuse)


def interval(ci: Sequence[float] | None, where: str) -> list[float]:
    """Return a stored video-level interval, refusing a missing one.

    Raises:
        ValueError: The interval is `null`, so the row has no difference to print.
    """
    if ci is None:
        raise ValueError(f"{where}: the video-level interval is null")
    return list(ci)


def mark_text(ci: Sequence[float], key: str, n_videos: int) -> str:
    """Return the mark `paired_stats.mark_of` gives on a stored interval in `key`'s direction, or a space for none.

    Args:
        ci: The stored video-level 95 % interval of the difference.
        key: The analysis's name for the key, a key of `PILOT_KEY_OF`.
        n_videos: The videos the interval resampled.
    """
    return mark_of(ci, PILOT_SIGNS[PILOT_KEY_OF[key]], n_videos=n_videos) or " "


def input_track_lines(d: dict) -> list[str]:
    """Return the table of each geometry input against rgb, merged to one K, then at rgb's matched region count.

    Raises:
        ValueError: The JSON is of another mode of the analysis, its pairs are
            not the table's rows, its matched rows are not its pairs, or an
            interval is `null`.
    """
    # Another mode writes the same arm names from other labels, so the mode is checked, not inferred.
    if d["mode"] != "input_track":
        raise ValueError(f"the analysis ran with --mode {d['mode']}, not input_track")
    want = {f"in_{i}_k{k}" for i in INPUTS for k in KS}
    if set(d["pairs"]) != want:
        raise ValueError(f"the pairs are not the table's rows: not printed {sorted(set(d['pairs']) - want)}, "
                         f"missing {sorted(want - set(d['pairs']))}")
    if set(d["matched"]) != want:
        raise ValueError(f"the matched rows are not the pairs: not printed {sorted(set(d['matched']) - want)}, "
                         f"missing {sorted(want - set(d['matched']))}")
    base = d["base_name"]
    lines = [f"Inputs merged to one K, fair_merge_recheck --mode input_track / dataset {d['dataset']} / "
             f"stride {d['stride']} / {d['n_clips']} clips / {d['n_videos']} videos / {d['n_frames']} frames / "
             f"eval_code_sha {d['eval_code_sha'][:8]} / base {base}",
             f"Δ = arm − {base}, both merged to the same K. Mark: {MARK_RULE}"]
    # Every interval resamples the analysis's videos, so one line says when all of them give no mark.
    if sign_only_note(d["n_videos"]):
        lines.append(f"every interval is on {d['n_videos']} videos, {sign_only_note(d['n_videos'])}")
    lines += ["", f"{'arm':18s}{'regions':>8s} " + " ".join(f"{h:>27s}" for _, h in INPUT_TRACK_KEYS)]

    # One row per input and K: the regions of the two arms, then each key's difference, mark and interval.
    arms = d["arms"]
    for inp in INPUTS:
        for k in KS:
            name = f"in_{inp}_k{k}"
            row = f"{name:18s}{arms[name]['n']:4.1f}/{arms[f'in_rgb_k{k}']['n']:4.1f}"
            for key, _ in INPUT_TRACK_KEYS:
                v = d["pairs"][name][key]
                ci = interval(v["ci95_video"], f"{name} {key}")
                row += f" {v['delta']:+.4f} {mark_text(ci, key, d['n_videos'])} [{ci[0]:+.4f},{ci[1]:+.4f}]"
            lines.append(row)
        lines.append("")
    lines += ["k0 is compared with rgb pps24 unmerged, so the region counts differ. "
              "k6 to k12 merge both arms to the same K.", ""]

    # The unmerged geometry arms against rgb's curve over K, read at the arm's region count.
    lines.append("matched: rgb's curve over K, read at the geometry arm's region count. Δ = geometry − rgb matched")
    for name in (f"in_{i}_k0" for i in INPUTS):
        m = d["matched"][name]
        row = f"{name:14s} reached {m['n_reached']:.2f} / clamped {m['n_clamped']:3d} clips |"
        for key, _ in INPUT_TRACK_KEYS:
            ci = interval(m[key]["ci95_video"], f"{name} {key}")
            row += f" {m[key]['delta']:+.4f} {mark_text(ci, key, d['n_videos'])}"
        bp = m["by_procedure"]
        pos_f1 = sum(1 for p in bp.values() if p["delta"]["full"] > 0)
        pos_bf = sum(1 for p in bp.values() if p["delta"]["boundary_F"] > 0)
        lines.append(row + f" | procedures with a positive Δ, of {len(bp)}: F1 {pos_f1} / boundary F {pos_bf}")
    return lines


def identity_lines(d: dict) -> list[str]:
    """Return the table of identity over time, with no mark: every value in it is a reference value.

    The analysis stores the clips with a positive and with a negative
    difference. The table prints them as wins and losses in the key's
    direction, as `paired_table` does, so that a loss reads as worse.

    Raises:
        ValueError: The conditions were measured against different GT
            tracks, which the table's one header cannot say, or an interval
            is `null`.
    """
    c = d["conditions"]
    tracks = {(v["n_gt_tracks"], v["gt_inst_per_frame"]) for v in c.values()}
    if len(tracks) != 1:
        raise ValueError(f"the conditions were measured against different GT tracks: {sorted(tracks)}")
    (n_tracks, per_frame), = tracks
    lines = [f"Identity over time, summarize_track12 / dataset {d['dataset']} / n_boot {d['n_boot']} / "
             f"{n_tracks} GT tracks / {per_frame} instances per frame / min_area {d['params']['min_area']} px / "
             f"primary {d['primary']}",
             "pair base:cond, Δ = cond − base. Hold and IDF1: higher is better. IDsw/track and Frag/track: lower "
             "is better. wins-losses: clips better-worse in the key's direction. "
             "No mark: no measure over time carries a star.", "",
             f"{'pair':40s} " + " ".join(f"{h:>33s}" for _, h, _ in IDENTITY_KEYS)]

    # One row per pair: each key's mean difference, interval, and clips won and lost in the key's direction.
    for pair, v in d["pairs"].items():
        row = f"{pair:40s}"
        for key, _, sign in IDENTITY_KEYS:
            x = v["metrics"][key]
            ci = interval(x["ci95_video"], f"{pair} {key}")
            wins, losses = (x["wins"], x["losses"]) if sign > 0 else (x["losses"], x["wins"])
            row += f" {x['delta_mean']:+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] {wins:3d}-{losses:3d}"
        lines.append(row + f"  / {v['n_clips']} clips {v['n_videos']} videos")

    # One row per condition: its measures, as reference values.
    lines += ["", f"{'condition':24s} {'hold':>7s} {'IDF1':>7s} {'IDsw':>7s} {'Frag':>7s} {'re-entry':>8s} {'gaps':>6s}"]
    for name, v in c.items():
        lines.append(f"{name:24s} {v['hold_mean']:7.4f} {v['idf1']:7.4f} {v['idsw_per_track']:7.4f} "
                     f"{v['frag_per_track']:7.4f} {v['re_recovery']:8.4f} {v['re_n_gaps']:6d}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-track", default="", help="The JSON `fair_merge_recheck --mode input_track` wrote.")
    ap.add_argument("--identity", default="", help="The JSON `summarize_track12` wrote.")
    args = ap.parse_args()
    if not (args.input_track or args.identity):
        raise SystemExit("name at least one analysis JSON")

    # Every table, built before any is printed, so that a refused JSON prints nothing.
    sections = []
    for path, build in ((args.input_track, input_track_lines), (args.identity, identity_lines)):
        if path:
            try:
                sections.append(build(load_analysis(path)))
            except (KeyError, ValueError, OSError) as e:
                raise SystemExit(f"{path}: {e!r}") from e

    print(f"\n\n{SEPARATOR}\n".join("\n".join(lines) for lines in sections))


if __name__ == "__main__":
    main()
