"""Print the StereoMIS table: each condition's camera trajectory scores, and each condition against a base.

The rows are read through `scores.read_rows`, so they come from one version
of the scorer. Each condition must hold the base's clips. A pair is the
paired difference `cond - base` on `ate_rel`, where less is better; its
interval is the bootstrap that `paired_stats.boot_ci` draws over sequences,
the unit a clip is not independent of, and `paired_stats.mark_of` gives its
mark. The clip-level Wilcoxon p is printed beside it. The table and the
pairs are printed for every clip, then for each stratum; a last table gives
each sequence's mean difference from the base.

Usage:
    python -m trajectory_eval.tools.stereomis_table controls.json methods.json [methods_masked.json] \\
        [--base floor_static] [--out summary.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np

from evalkit.tools.paired_stats import N_BOOT, SEED, SEED_SCHEME, boot_ci, mark_of, round_keeping_sign
from trajectory_eval.tools.scores import read_rows

# The key that decides the result. Less is better.
KEY = "ate_rel"
SIGN = -1
# The strata a clip is given by its true trajectory's radius, as `surgical_core.stereomis.stratum_of` names them.
STRATA = ("moving", "slow")
# The shape of every clip, printed with each table, since a score depends on it.
CLIP_SHAPE = "112 frames 0.2 s apart, 22.4 s, not overlapping"


def check_population(rows: list[dict], base: str) -> None:
    """Refuse a condition scored on other clips than the base, so every mean and every pair has one population.

    Raises:
        ValueError: the base has no row, or a condition's clips differ from the base's. The message names each
            such condition with the counts.
    """
    clips = {}
    for r in rows:
        clips.setdefault(r["cond"], set()).add(r["clip"])
    if base not in clips:
        raise ValueError(f"no row of the base {base!r}; the conditions are {sorted(clips)}")
    other = [f"{c} ({len(v - clips[base])} clips the base lacks, {len(clips[base] - v)} it lacks)"
             for c, v in sorted(clips.items()) if v != clips[base]]
    if other:
        raise ValueError(f"conditions scored on other clips than the base {base!r}: {other}")


def table(rows: list[dict], stratum: str = "") -> None:
    """Print each condition's count, sequences, mean and median `ate_rel`, and the medians of the other measures."""
    sel = [r for r in rows if not stratum or r["stratum"] == stratum]
    print(f"\n===== conditions{f' ({stratum} only)' if stratum else ''}, {CLIP_SHAPE} =====")
    print(f"  {'condition':14s} {'n':>3s} {'seqs':>4s} {KEY + ' mean':>12s} {'median':>8s} "
          f"{'RPE trans':>9s} {'RPE rot°':>9s} {'scale ratio':>11s}")
    for c in sorted({r["cond"] for r in sel}):
        v = [r for r in sel if r["cond"] == c]
        a = np.array([r[KEY] for r in v])
        rt = np.array([r["rpe_trans_rel"] for r in v], float)
        rr = np.array([r["rpe_rot_deg"] for r in v], float)
        sr = np.array([r["scale_ratio"] for r in v if r["scale_ratio"] is not None], float)
        print(f"  {c:14s} {len(v):3d} {len({r['seq'] for r in v}):4d} {a.mean():12.4f} {np.median(a):8.4f} "
              f"{np.nanmedian(rt):9.3f} {np.nanmedian(rr):9.3f} "
              f"{np.nanmedian(sr) if len(sr) else float('nan'):11.3f}")


def compare(rows: list[dict], base: str, cond: str, stratum: str = "") -> dict:
    """Return and print the paired difference `cond - base` over the clips of `stratum`, or of every stratum.

    Raises:
        ValueError: `boot_ci` refuses the pair: its clips come from fewer than two sequences.
    """
    from scipy import stats

    sel = [r for r in rows if not stratum or r["stratum"] == stratum]
    a = {r["clip"]: r for r in sel if r["cond"] == base}
    b = {r["clip"]: r for r in sel if r["cond"] == cond}
    clips = sorted(a)
    d = np.array([b[c][KEY] - a[c][KEY] for c in clips])
    seqs = np.array([a[c]["seq"] for c in clips])
    lo, hi = boot_ci(d, seqs)
    w = stats.wilcoxon(d) if np.any(d != 0) else None
    ci = [round_keeping_sign(lo), round_keeping_sign(hi)]
    mark = mark_of(ci, SIGN)
    p = None if w is None else round(float(w.pvalue), 5)
    print(f"  {mark or ' '} {cond} - {base}: delta {d.mean():+.4f}  median {np.median(d):+.4f}  "
          f"95%CI(seq) [{lo:+.4f}, {hi:+.4f}]  better-worse {(d < 0).sum()}-{(d > 0).sum()}  "
          f"n={len(clips)} / {len(set(seqs))} seqs  p={'-' if p is None else p}")
    return dict(base=base, cond=cond, key=KEY, stratum=stratum, n=len(clips), n_seq=int(len(set(seqs))),
                delta=round(float(d.mean()), 4), median=round(float(np.median(d)), 4), ci95_seq=ci,
                better=int((d < 0).sum()), worse=int((d > 0).sum()), wilcoxon_p=p, mark=mark)


def per_seq(rows: list[dict], base: str) -> None:
    """Print each sequence's mean difference from the base, for each condition; a mean hides a sequence that loses."""
    conds = [c for c in sorted({r["cond"] for r in rows}) if c != base]
    print(f"\n===== {KEY} by sequence (difference from {base}; negative is better) =====")
    print(f"  {'seq':7s} {'n':>2s} " + "".join(f"{c:>16s}" for c in conds))
    for s in sorted({r["seq"] for r in rows}):
        a = {r["clip"]: r[KEY] for r in rows if r["cond"] == base and r["seq"] == s}
        cells = []
        for c in conds:
            b = {r["clip"]: r[KEY] for r in rows if r["cond"] == c and r["seq"] == s}
            cells.append(f"{np.mean([b[x] - a[x] for x in sorted(a)]):+16.4f}")
        print(f"  {s:7s} {len(a):2d} " + "".join(cells))


def summarize(rows: list[dict], base: str) -> dict:
    """Print the tables and the pairs, and return the pairs with the bootstrap's settings.

    Raises:
        ValueError: `check_population` refuses the rows, or `compare` a pair.
    """
    check_population(rows, base)
    out = {"clip_shape": CLIP_SHAPE, "n_boot": N_BOOT, "seed": SEED, "seed_scheme": SEED_SCHEME, "pairs": []}
    others = sorted({r["cond"] for r in rows} - {base})
    table(rows)
    print(f"\n===== pairs with {base} (negative is better; ★ better, ✗ worse: the 95% CI does not straddle 0) =====")
    out["pairs"] += [compare(rows, base, c) for c in others]
    for st in STRATA:
        if any(r["stratum"] == st for r in rows):
            table(rows, stratum=st)
            print(f"  -- pairs within {st} --")
            out["pairs"] += [compare(rows, base, c, stratum=st) for c in others]
    per_seq(rows, base)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="The score files `score_stereomis` wrote.")
    ap.add_argument("--base", default="floor_static", help="The condition every other one is compared with.")
    ap.add_argument("--out", help="A JSON file to write the pairs to.")
    args = ap.parse_args()
    try:
        rows = read_rows([Path(p) for p in args.inputs])
        print(f"read {[Path(p).name for p in args.inputs]}, {len(rows)} rows")
        out = summarize(rows, args.base)
    except ValueError as e:
        raise SystemExit(f"ValueError: {e}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False)
        print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
