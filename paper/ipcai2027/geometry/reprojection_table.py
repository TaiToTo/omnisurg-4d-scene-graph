"""Print the reprojection consistency table on StereoMIS: mean IoU, differences, and how it follows the ATE per clip.

It reads `reprojection.py`'s rows and the depth stages' trajectory rows, on the clips kept. The means and the
paired differences have intervals over sequences (`stereomis_summary`). Per clip, the IoU names the lower-ATE
model when the model with the higher IoU of Pi3X and DA3 has the lower relative ATE. That share, and Spearman's
rho between a model's IoU and its relative ATE across clips, have intervals over 10 000 resamples of the clips.
The criterion fixed before any score was read asks rho at most -0.5 for both models, with an interval excluding
zero.

Usage:
    python paper/ipcai2027/geometry/reprojection_table.py --rows reprojection_rows.json --trajectories methods.json \\
        --exclude P2_6__clip_0007 P2_7__clip_0001 --out reprojection.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from evalkit.tools.paired_stats import N_BOOT
from stereomis_summary import diff_row, interval, kept_clips, write_json
from trajectory_eval.tools.scores import read_rows

MODELS = ("pi3x", "da3", "mapanything")
PAIRS = (("pi3x_own", "da3_own"), ("da3_own", "mapanything_own"))
# The seed of the clip resamples.
SEED = 0
RHO_MAX = -0.5


def clip_bootstrap(n: int, statistic) -> list[float]:
    """Return the 95 % interval of `statistic(indices)` over `N_BOOT` resamples of `n` clips."""
    rng = np.random.default_rng(SEED)
    draws = np.array([statistic(rng.integers(0, n, n)) for _ in range(N_BOOT)])
    return [round(float(x), 4) for x in np.nanpercentile(draws, [2.5, 97.5])]


def table(rows: list[dict], ate: dict, clips: list[str]) -> dict:
    """Return the means, the differences, the clips on which the IoU names the lower-ATE model, and rho."""
    iou = {r["clip"]: r["iou"] for r in rows}
    value = {k: [iou[c][k] for c in clips] for k in iou[clips[0]]}
    out = dict(n_clips=len(clips), n_sequences=len({c.split("__")[0] for c in clips}),
               mean_iou={k: round(float(np.mean(v)), 4) for k, v in value.items()},
               own_minus_still={m: diff_row(value[f"{m}_own"], value[f"{m}_still"], clips, +1) for m in MODELS},
               between_models={f"{a} - {b}": diff_row(value[a], value[b], clips, +1) for a, b in PAIRS})

    # The clips on which the higher IoU of Pi3X and DA3 names the lower-ATE model.
    same = np.array([(iou[c]["pi3x_own"] > iou[c]["da3_own"]) == (ate[c]["pi3x"] < ate[c]["da3"]) for c in clips])
    out["lower_ate_named"] = dict(clips=int(same.sum()), share=round(float(same.mean()), 4),
                                  ci95=clip_bootstrap(len(clips), lambda i: same[i].mean()),
                                  always_pi3x=int(sum(ate[c]["pi3x"] < ate[c]["da3"] for c in clips)),
                                  chance=len(clips) / 2)

    # Spearman's rho between each model's IoU and its relative ATE, across clips.
    out["rho_with_ate"] = {}
    for m in ("da3", "pi3x"):
        x, y = np.array(value[f"{m}_own"]), np.array([ate[c][m] for c in clips])
        out["rho_with_ate"][m] = dict(rho=round(float(spearmanr(x, y).statistic), 4),
                                      ci95=clip_bootstrap(len(clips), lambda i: spearmanr(x[i], y[i]).statistic))
    out["criterion_met"] = all(r["rho"] <= RHO_MAX and r["ci95"][1] < 0 for r in out["rho_with_ate"].values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True, help="The rows `reprojection.py` wrote.")
    ap.add_argument("--trajectories", nargs="+", required=True,
                    help="The score file of `score_stereomis --methods da3 pi3x`.")
    ap.add_argument("--exclude", nargs="*", default=[], help="The clips `excluded_clips.py` prints.")
    ap.add_argument("--out", required=True, help="The JSON file the table is written to.")
    args = ap.parse_args()

    # Read the IoU rows and the ATE of each clip, and keep the clips not excluded.
    rows = json.loads(Path(args.rows).read_text())["clips"]
    ate = {}
    for r in read_rows([Path(p) for p in args.trajectories]):
        ate.setdefault(r["clip"], {})[r["cond"]] = r["ate_rel"]
    names = [r["clip"] for r in rows]
    lacking = sorted(c for c in names if not {"da3", "pi3x"} <= set(ate.get(c, {})))
    if lacking:
        raise SystemExit(f"no trajectory row of DA3 or Pi3X for {lacking}")
    try:
        clips = kept_clips(names, args.exclude)
    except ValueError as e:
        raise SystemExit(str(e))
    result = dict(excluded=sorted(args.exclude), **table(rows, ate, clips))

    # Print the table.
    print(f"reprojection consistency, next frame: {result['n_clips']} clips, {result['n_sequences']} sequences")
    print("  mean IoU " + ", ".join(f"{k} {v:.3f}" for k, v in result["mean_iou"].items()))
    for m, row in result["own_minus_still"].items():
        print(f"  {m} own - still   {interval(row, 'delta')}")
    for pair, row in result["between_models"].items():
        print(f"  {pair:24s} {interval(row, 'delta')}")
    n = result["lower_ate_named"]
    print(f"  lower-ATE model named on {n['clips']} of {result['n_clips']} ({n['share']:.2f} [{n['ci95'][0]:.2f}, "
          f"{n['ci95'][1]:.2f}]); always Pi3X {n['always_pi3x']}; chance {n['chance']}")
    for m, r in result["rho_with_ate"].items():
        print(f"  rho {m} {r['rho']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]")
    print(f"  criterion (rho <= {RHO_MAX} for both, interval below zero) met: {result['criterion_met']}")
    write_json(args.out, result)


if __name__ == "__main__":
    main()
