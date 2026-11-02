"""Print the D4D AbsRel table: each answer's mean, the differences, and the share of the gap to the half scan.

It reads `d4d_depth.py`'s rows and summarizes all sides, the moving-camera sides and the static ones. A mean is
over the sides scored. Every interval resamples the sessions 10 000 times, one draw shared by every statistic of
a stratum, so that a ratio of two means has an interval too. The mark is `paired_stats.mark_of`, with each
session counted as a video. A model with AbsRel A closes the share (A_flat - A) / (A_flat - A_half) of the gap
between the flat plane and the half scan.

Usage:
    python paper/ipcai2027/geometry/d4d_depth_table.py --sides d4d_sides.json --out d4d_depth.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from evalkit.tools.paired_stats import N_BOOT, mark_of, round_keeping_sign
from stereomis_summary import write_json

MODELS = ("production", "da3_whole", "pi3x_whole", "mapanything_whole")
# The paired differences, `a - b`; less is better.
PAIRS = (*((m, "flat") for m in MODELS), ("da3_whole", "production"), ("pi3x_whole", "da3_whole"))
STRATA = {"all": lambda r: True, "moving": lambda r: r["moved_camera"], "static": lambda r: not r["moved_camera"]}
# The seed of the session draws.
SEED = 0


def stratum_table(rows: list[dict]) -> dict:
    """Return the means, the differences and the shares of the gap closed, with intervals over sessions."""
    # Draw the sessions once; a draw's mean is a weighted sum of the sessions' sums.
    sessions = sorted({r["session"] for r in rows})
    g = np.array([sessions.index(r["session"]) for r in rows])
    draws = np.random.default_rng(SEED).integers(0, len(sessions), (N_BOOT, len(sessions)))
    weights = np.stack([np.bincount(d, minlength=len(sessions)) for d in draws]).astype(float)
    n = np.bincount(g, minlength=len(sessions)).astype(float)
    value = {a: np.array([r["absrel"][a] for r in rows]) for a in rows[0]["absrel"]}
    boot = {a: (weights @ np.bincount(g, v, len(sessions))) / (weights @ n) for a, v in value.items()}

    def ci(samples: np.ndarray) -> list[float]:
        return [round_keeping_sign(x) for x in np.percentile(samples, [2.5, 97.5])]

    out = dict(n_sides=len(rows), n_clips=len({r["key"] for r in rows}), n_sessions=len(sessions),
               absrel={a: dict(mean=round(float(v.mean()), 4), ci95=ci(boot[a])) for a, v in value.items()},
               differences={}, gap_closed={})
    for a, b in PAIRS:
        interval = ci(boot[a] - boot[b])
        out["differences"][f"{a} - {b}"] = dict(delta=round(float(value[a].mean() - value[b].mean()), 4),
                                                ci95=interval, mark=mark_of(interval, -1, n_videos=len(sessions)),
                                                sides_lower=int((value[a] < value[b]).sum()))
    flat, half = value["flat"].mean(), value["half_scan"].mean()
    for m in MODELS:
        share = (boot["flat"] - boot[m]) / (boot["flat"] - boot["half_scan"])
        out["gap_closed"][m] = dict(share=round(float((flat - value[m].mean()) / (flat - half)), 4), ci95=ci(share))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sides", required=True, help="The rows `d4d_depth.py` wrote.")
    ap.add_argument("--out", required=True, help="The JSON file the table is written to.")
    args = ap.parse_args()
    sides = [r for r in json.loads(Path(args.sides).read_text())["sides"] if "absrel" in r]
    result = {name: stratum_table([r for r in sides if keep(r)]) for name, keep in STRATA.items()}

    # Print the table.
    for name, t in result.items():
        print(f"{name}: {t['n_sides']} sides, {t['n_clips']} clips, {t['n_sessions']} sessions")
        for a, row in t["absrel"].items():
            share = t["gap_closed"].get(a)
            print(f"  {a:18s} {row['mean']:.4f} [{row['ci95'][0]:.4f}, {row['ci95'][1]:.4f}]"
                  + (f"  gap closed {share['share']:.2f} [{share['ci95'][0]:.2f}, {share['ci95'][1]:.2f}]"
                     if share else ""))
        for pair, row in t["differences"].items():
            print(f"  {pair:30s} {row['delta']:+.4f} [{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}] {row['mark']}"
                  f"  ({row['sides_lower']} sides lower)")
    write_json(args.out, result)


if __name__ == "__main__":
    main()
