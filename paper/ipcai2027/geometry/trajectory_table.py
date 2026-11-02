"""Print the relative ATE table: each model and baseline on StereoMIS and D4D, and the paired differences.

The rows come through `scores.read_rows`, one dataset at a time. On StereoMIS each condition's mean is over the
clips kept, with an interval over sequences; the pairs are `stereomis_table.compare`'s, with their marks. The
static floor scores 1 on every clip, so a condition's interval is 1 plus the interval of its difference from the
floor, as the paper's table takes it. The means with the excluded clips are printed too. D4D's clips come from
four specimens, too few for an interval, so its rows are means and medians, read for their sign.

Usage:
    python paper/ipcai2027/geometry/trajectory_table.py --stereomis controls.json methods.json mapanything_rows.json \\
        --d4d d4d_rows.json --exclude P2_6__clip_0007 P2_7__clip_0001 --out trajectory.json
"""

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np

from stereomis_summary import kept_clips, write_json
from trajectory_eval.tools.scores import read_rows
from trajectory_eval.tools.stereomis_table import check_population, compare

BASE = "floor_static"
MODELS = ("pi3x", "da3", "mapanything")
# The conditions of each dataset's table, and the StereoMIS pairs, `cond - base`.
STEREOMIS = (*MODELS, "floor_line")
D4D = ("pi3x", "da3", "floor_line", "floor_static")
PAIRS = (("da3", "pi3x"), ("pi3x", "mapanything"), ("floor_line", "pi3x"), ("floor_line", "da3"))


def quiet_compare(rows: list[dict], base: str, cond: str) -> dict:
    """Return `stereomis_table.compare`'s pair without the line it prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return compare(rows, base, cond)


def stereomis(rows: list[dict], exclude: list[str]) -> dict:
    """Return each condition's mean, median and interval on the kept clips, the pairs, and the means on all clips.

    Raises:
        ValueError: `check_population` refuses the rows, `exclude` names a clip the rows do not hold, or the
            static floor is not 1 on every clip.
    """
    # Refuse rows of other clips, and a static floor that is not 1, before anything is computed.
    check_population(rows, BASE)
    if any(r["ate_rel"] != 1.0 for r in rows if r["cond"] == BASE):
        raise ValueError(f"{BASE} is not 1 on every clip, so a condition's interval cannot be read from its difference")
    kept = set(kept_clips(sorted({r["clip"] for r in rows}), exclude))
    sel = [r for r in rows if r["clip"] in kept]

    # Each condition's mean and median, and its interval, read from its difference from the static floor.
    out = dict(n_clips=len(kept), n_sequences=len({r["seq"] for r in sel}), excluded=sorted(exclude), conditions={},
               differences={}, with_excluded={})
    for cond in STEREOMIS:
        p = quiet_compare(sel, BASE, cond)
        values = [r["ate_rel"] for r in sel if r["cond"] == cond]
        out["conditions"][cond] = dict(mean=round(float(np.mean(values)), 4),
                                       ci95=[round(1 + x, 4) for x in p["ci95_seq"]],
                                       median=round(float(np.median(values)), 4))
    out["conditions"][BASE] = dict(mean=1.0)

    # The pairs, and the models' means with the excluded clips.
    for base, cond in PAIRS:
        p = quiet_compare(sel, base, cond)
        out["differences"][f"{cond} - {base}"] = dict(delta=p["delta"], ci95=p["ci95_seq"], mark=p["mark"],
                                                      n_clips=p["n"], n_sequences=p["n_seq"])
    out["with_excluded"] = dict(n_clips=len({r["clip"] for r in rows}), **{
        m: round(float(np.mean([r["ate_rel"] for r in rows if r["cond"] == m])), 4) for m in MODELS})
    return out


def d4d(rows: list[dict]) -> dict:
    """Return each condition's mean and median, Pi3X less DA3, and the clips each model scores below the line."""
    check_population(rows, BASE)
    by = {c: {r["clip"]: r["ate_rel"] for r in rows if r["cond"] == c} for c in D4D}
    clips = sorted(by[BASE])
    return dict(n_clips=len(clips), n_specimens=len({r["seq"] for r in rows}),
                conditions={c: dict(mean=round(float(np.mean(list(v.values()))), 4),
                                    median=round(float(np.median(list(v.values()))), 4)) for c, v in by.items()},
                pi3x_minus_da3=round(float(np.mean([by["pi3x"][c] - by["da3"][c] for c in clips])), 4),
                below_line={m: sum(by[m][c] < by["floor_line"][c] for c in clips) for m in ("pi3x", "da3")})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stereomis", nargs="+", required=True,
                    help="The StereoMIS score files: `score_stereomis --controls`, `--methods da3 pi3x`, and "
                         "`mapanything_trajectories.py`.")
    ap.add_argument("--d4d", nargs="+", required=True, help="The D4D score file of `score_d4d score`.")
    ap.add_argument("--exclude", nargs="*", default=[], help="The clips `excluded_clips.py` prints.")
    ap.add_argument("--out", required=True, help="The JSON file the table is written to.")
    args = ap.parse_args()
    try:
        result = dict(stereomis=stereomis(read_rows([Path(p) for p in args.stereomis]), args.exclude),
                      d4d=d4d(read_rows([Path(p) for p in args.d4d])))
    except ValueError as e:
        raise SystemExit(f"ValueError: {e}")

    # Print the table.
    s, d = result["stereomis"], result["d4d"]
    print(f"relative ATE: StereoMIS {s['n_clips']} clips / {s['n_sequences']} sequences; "
          f"D4D {d['n_clips']} clips / {d['n_specimens']} specimens")
    for cond in (*STEREOMIS, BASE):
        row = s["conditions"][cond]
        ci = f" [{row['ci95'][0]:.3f}, {row['ci95'][1]:.3f}]" if "ci95" in row else ""
        far = d["conditions"].get(cond)
        print(f"  {cond:12s} {row['mean']:.3f}{ci:18s}" + (f" D4D {far['mean']:.3f} (median {far['median']:.3f})"
                                                           if far else ""))
    for pair, row in s["differences"].items():
        print(f"  {pair:22s} {row['delta']:+.3f} [{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}] {row['mark']}")
    print(f"  D4D pi3x - da3 {d['pi3x_minus_da3']:+.3f} (sign); below the line: pi3x {d['below_line']['pi3x']}, "
          f"da3 {d['below_line']['da3']} of {d['n_clips']}")
    w = s["with_excluded"]
    print(f"  with the excluded clips ({w['n_clips']} clips): pi3x {w['pi3x']:.3f}, da3 {w['da3']:.3f}")
    write_json(args.out, result)


if __name__ == "__main__":
    main()
