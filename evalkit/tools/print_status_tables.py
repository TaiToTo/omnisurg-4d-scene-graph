"""Print the tables of four analyses from the JSONs those analyses wrote, computing nothing new.

The four are: each geometry input against rgb, merged to one K
(`fair_merge_recheck --mode input_track`); identity over time
(`summarize_track12`); conditions merged to one K by `kmerge --pairs`;
and a classic watershed, normal against rgb (`classic_seg`). The tables
are pasted as printed, never copied by hand. Every mark is the one
`paired_stats.mark_of` gives on the stored video-level interval, read in
the key's direction, and a mark a JSON stores must equal it. The measures
of identity are reference values and carry no mark.

Usage:
    python -m evalkit.tools.print_status_tables --input-track e1_input_track.json \\
        --identity e2_track12.json --kmerge-pairs kmerge_pps_ci.json --classic classic_seg__*.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence

from evalkit.tools.paired_stats import MARK_RULE, mark_of
from evalkit.tools.scores import PILOT_SIGNS

SEPARATOR = "=" * 100

# The geometry inputs and the K each was merged to, in the order of the table's rows.
INPUTS = ("edge", "normal", "depth")
KS = (0, 6, 8, 10, 12)
# The keys of each table and their headings, in the order of its columns.
INPUT_TRACK_KEYS = (("full", "F1@.5"), ("labeled", "F1 labeled"), ("boundary_F", "boundary F"),
                    ("boundary_R_raw", "boundary R raw"), ("GT_mIoU", "mIoU"))
CLASSIC_KEYS = INPUT_TRACK_KEYS[1:]
IDENTITY_KEYS = (("hold_mean", "hold"), ("idf1", "IDF1"), ("idsw_per_track", "IDsw/track"),
                 ("frag_per_track", "Frag/track"))
# The pilot key each column holds. The analyses scored with the pilot
# evaluator's functions but name a key by its domain alone, and `kmerge
# --pairs` names its F1 at 0.5 `f1`; each column is read in its pilot key's direction.
PILOT_KEY_OF = {"full": "inst_F1_50", "labeled": "inst_F1_50_labeled", "boundary_F": "boundary_F",
                "boundary_R_raw": "boundary_R_raw", "GT_mIoU": "GT_mIoU", "f1": "inst_F1_50"}


def load_analysis(path: str) -> dict:
    """Read one analysis JSON.

    Raises:
        ValueError: The file holds a `NaN` or an infinity, which would print as a value and give no mark.
    """
    def refuse(token: str):
        raise ValueError(f"{path}: holds {token}, which is not a JSON number")

    with open(path, encoding="utf-8") as f:
        return json.load(f, parse_constant=refuse)


def mark_text(ci: Sequence[float], key: str, stored: str | None = None, withheld: bool = False) -> str:
    """Return the mark `paired_stats.mark_of` gives on a stored interval in `key`'s direction, or two spaces for none.

    Args:
        ci: The stored video-level 95 % interval of the difference.
        key: The analysis's name for the key, a key of `PILOT_KEY_OF`.
        stored: The mark the JSON stores beside the interval, if it stores one.
        withheld: The analysis withheld the mark on this row, so no mark is given.

    Raises:
        ValueError: `stored` is not the mark this gives. A JSON whose mark
            another rule decided is refused rather than printed under this one.
    """
    m = "" if withheld else mark_of(ci, PILOT_SIGNS[PILOT_KEY_OF[key]])
    if stored is not None and stored != m:
        raise ValueError(f"{key} {list(ci)}: the JSON stores the mark {stored!r}, "
                         f"and paired_stats.mark_of gives {m!r}")
    return m or "  "


def input_track_lines(d: dict) -> list[str]:
    """Return the table of each geometry input against rgb, merged to one K, then at rgb's matched region count.

    Raises:
        ValueError: The JSON's pairs are not the table's rows, or a row at
            the matched region count is one the analysis found broken.
    """
    want = {f"in_{i}_k{k}" for i in INPUTS for k in KS}
    if set(d["pairs"]) != want:
        raise ValueError(f"the pairs are not the table's rows: not printed {sorted(set(d['pairs']) - want)}, "
                         f"missing {sorted(want - set(d['pairs']))}")
    base = d["base_name"]
    lines = [f"Inputs merged to one K, fair_merge_recheck --mode input_track / dataset {d['dataset']} / "
             f"stride {d['stride']} / {d['n_clips']} clips / {d['n_videos']} videos / {d['n_frames']} frames / "
             f"eval_code_sha {d['eval_code_sha'][:8]} / base {base}",
             f"Δ = arm − {base}, both merged to the same K. Mark: {MARK_RULE}", "",
             f"{'arm':18s}{'regions':>8s} " + " ".join(f"{h:>26s}" for _, h in INPUT_TRACK_KEYS)]

    # One row per input and K: the regions of the two arms, then each key's difference, mark and interval.
    arms = d["arms"]
    for inp in INPUTS:
        for k in KS:
            name = f"in_{inp}_k{k}"
            row = f"{name:18s}{arms[name]['n']:4.1f}/{arms[f'in_rgb_k{k}']['n']:4.1f}"
            for key, _ in INPUT_TRACK_KEYS:
                v = d["pairs"][name][key]
                ci = v["ci95_video"]
                row += f" {v['delta']:+.4f} {mark_text(ci, key)} [{ci[0]:+.3f},{ci[1]:+.3f}]"
            lines.append(row)
        lines.append("")
    lines += ["k0 is compared with rgb pps24 unmerged, so the region counts differ. "
              "k6 to k12 merge both arms to the same K.", ""]

    # The unmerged geometry arms against rgb's curve over K, read at the arm's region count.
    lines.append("matched: rgb's curve over K, read at the geometry arm's region count. Δ = geometry − rgb matched")
    for name in (f"in_{i}_k0" for i in INPUTS):
        m = d["matched"][name]
        # A region count past twice the K asked means the merge did not control granularity.
        if m["broken"]:
            raise ValueError(f"{name}: the analysis found the matched row broken, so it is not a comparison")
        row = f"{name:14s} reached {m['n_reached']:.2f} / clamped {m['n_clamped']:3d} clips |"
        for key, _ in INPUT_TRACK_KEYS:
            row += f" {m[key]['delta']:+.4f} {mark_text(m[key]['ci95_video'], key)}"
        bp = m["by_procedure"]
        pos_f1 = sum(1 for p in bp.values() if p["delta"]["full"] > 0)
        pos_bf = sum(1 for p in bp.values() if p["delta"]["boundary_F"] > 0)
        lines.append(row + f" | procedures with a positive Δ, of {len(bp)}: F1 {pos_f1} / boundary F {pos_bf}")
    return lines


def identity_lines(d: dict) -> list[str]:
    """Return the table of identity over time, with no mark: every value in it is a reference value.

    Raises:
        ValueError: The conditions were measured against different GT
            tracks, which the table's one header cannot say.
    """
    c = d["conditions"]
    tracks = {(v["n_gt_tracks"], v["gt_inst_per_frame"]) for v in c.values()}
    if len(tracks) != 1:
        raise ValueError(f"the conditions were measured against different GT tracks: {sorted(tracks)}")
    (n_tracks, per_frame), = tracks
    lines = [f"Identity over time, summarize_track12 / dataset {d['dataset']} / n_boot {d['n_boot']} / "
             f"{n_tracks} GT tracks / {per_frame} instances per frame / min_area {d['params']['min_area']} px",
             "Δ = cond − base. Hold and IDF1: higher is better. IDsw/track and Frag/track: lower is better. "
             "No mark: no measure over time carries a star.", "",
             f"{'pair':40s} " + " ".join(f"{h:>28s}" for _, h in IDENTITY_KEYS)]
    for pair, v in d["pairs"].items():
        row = f"{pair:40s}"
        for key, _ in IDENTITY_KEYS:
            x = v["metrics"][key]
            ci = x["ci95_video"]
            row += f" {x['delta_mean']:+.4f} [{ci[0]:+.3f},{ci[1]:+.3f}] {x['wins']:3d}-{x['losses']:3d}"
        lines.append(row + f"  / {v['n_clips']} clips {v['n_videos']} videos")
    lines += ["", f"{'condition':24s} {'hold':>7s} {'IDF1':>7s} {'IDsw':>7s} {'Frag':>7s} {'re-entry':>8s} {'gaps':>6s}"]
    for name, v in c.items():
        lines.append(f"{name:24s} {v['hold_mean']:7.4f} {v['idf1']:7.4f} {v['idsw_per_track']:7.4f} "
                     f"{v['frag_per_track']:7.4f} {v['re_recovery']:8.4f} {v['re_n_gaps']:6d}")
    return lines


def kmerge_pairs_lines(d: dict) -> list[str]:
    """Return the pairs `kmerge --pairs` compared, each at one K or at the matched region count.

    Raises:
        ValueError: A stored mark is not the one `paired_stats.mark_of` gives on its interval.
    """
    lines = [f"Merged pairs, kmerge --pairs / dataset {d['dataset']} / stride {d['stride']} / K {d['k_list']} / "
             f"mark: {MARK_RULE}",
             "Δ = a − b: the per-frame stage's F1, both merged to the same K"]
    for p in d["pairs"]:
        ci = p["ci95_video"]
        lines.append(f"  {p['a']} − {p['b']} @ {p['at']:8s} {p['delta']:+.4f} {mark_text(ci, 'f1', p['verdict'])} "
                     f"[{ci[0]:+.4f},{ci[1]:+.4f}] wins-losses {p['win']}-{p['lose']} / "
                     f"{p['n_clips']} clips {p['n_videos']} videos")
    return lines


def classic_lines(docs: Sequence[dict]) -> list[str]:
    """Return the table of the classic watershed, normal against rgb, one row per dataset, markers, K and smoothing.

    Raises:
        ValueError: The JSONs differ in stride or in whether instruments
            were dropped, which the table's one header states; or a stored
            mark is not the one `paired_stats.mark_of` gives on its interval.
    """
    setting = {(d["stride"], d["drop_instruments"]) for d in docs}
    if len(setting) != 1:
        raise ValueError(f"the JSONs differ in stride or in dropping instruments: {sorted(setting)}")
    (stride, dropped), = setting
    lines = [f"Classic segmentation, classic_seg: marker-controlled watershed, Δ = normal − rgb, "
             f"instruments {'dropped' if dropped else 'kept'}, stride {stride}",
             f"{'data':8s} {'markers':7s} {'K':>2s} {'σ':>4s} {'clips':>5s} {'videos':>6s} {'regions n/r':>12s} "
             + " ".join(f"{h:>22s}" for _, h in CLASSIC_KEYS) + "  degenerate"]
    for d in docs:
        for sig, r in d["by_sigma"].items():
            a = r["arms"]
            row = (f"{d['dataset']:8s} {d['markers']:7s} {d['k']:2d} {float(sig):4.0f} {r['n_clips']:5d} "
                   f"{r['n_videos']:6d} {a['normal']['n']:5.2f}/{a['rgb']['n']:5.2f} ")
            for key, _ in CLASSIC_KEYS:
                v = r["pair"][key]
                ci = v["ci95_video"]
                # A row whose markers fell short of K holds no mark: one arm was
                # not cut into K regions, so the comparison is not the one asked.
                m = mark_text(ci, key, v["verdict"], withheld=r["degenerate"])
                row += f" {v['delta']:+.4f} {m} [{ci[0]:+.3f},{ci[1]:+.3f}]"
            lines.append(row + ("  degenerate" if r["degenerate"] else ""))
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-track", default="", help="The JSON `fair_merge_recheck --mode input_track` wrote.")
    ap.add_argument("--identity", default="", help="The JSON `summarize_track12` wrote.")
    ap.add_argument("--kmerge-pairs", default="", help="The JSON `kmerge --pairs` wrote.")
    ap.add_argument("--classic", nargs="+", default=[],
                    help="The JSONs `classic_seg` wrote, one per dataset, markers and K; printed in sorted order.")
    args = ap.parse_args()
    if not (args.input_track or args.identity or args.kmerge_pairs or args.classic):
        raise SystemExit("name at least one analysis JSON")

    # Every table, built before any is printed, so that a refused JSON prints nothing.
    sections = []
    for path, build in ((args.input_track, input_track_lines), (args.identity, identity_lines),
                        (args.kmerge_pairs, kmerge_pairs_lines)):
        if path:
            try:
                sections.append(build(load_analysis(path)))
            except (KeyError, ValueError, OSError) as e:
                raise SystemExit(f"{path}: {e!r}") from e
    if args.classic:
        paths = sorted(args.classic)
        try:
            sections.append(classic_lines([load_analysis(p) for p in paths]))
        except (KeyError, ValueError, OSError) as e:
            raise SystemExit(f"{', '.join(os.path.basename(p) for p in paths)}: {e!r}") from e

    print(f"\n\n{SEPARATOR}\n".join("\n".join(lines) for lines in sections))


if __name__ == "__main__":
    main()
