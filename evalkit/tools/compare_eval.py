"""Two conditions' score JSONs side by side, refused unless measured alike on the same clips.

Every judgement between conditions takes this form. *base* is the
condition compared against and *cond* the one compared; a difference is
always `cond - base`. Two things break a conclusion without showing in the
numbers, and both stop the comparison rather than warn
(`scores.check_comparable`). One is a different *ruler*, what a score was
measured with: the evaluator's `eval_code_sha`, the dataset, the class
set, the views and the mode (`scores.Ruler`), and the GT and depth maps
each clip read, where the JSONs record them. A different ruler's
difference would be read as the methods'. The other is a different set
of clips, since means over different clips say nothing, and a condition
that lost its hardest clips wins without doing anything. `--allow-subset`
compares on the clips both hold instead, and the summary's `population`
then says `intersection` where it otherwise says `identical`.

A *key* is one column of a JSON's per-clip rows. The evaluator writes
`metric/view` (`F1_50/geometric`), and `time_IoU`, one value per clip,
with no view. A *pilot JSON*, one the pilot evaluator wrote, is told
apart by holding none of the fields the evaluator records
(`scores.is_pilot_json`), and keeps the pilot evaluator's spellings:
`inst_F1_50`, `GT_mIoU`, some with a domain after an underscore
(`inst_F1_50_tissue`). A JSON the evaluator wrote in pilot mode is not a
pilot JSON, and spells its keys as the evaluator does.

The clips can shrink per key. A key is None on a clip where its metric
is undefined: `SQ` and `inst_BF` on a clip with no hit, `F1_50` in a view
with no GT object, and each JSON's own mean leaves such clips out
independently. Subtracting those means compares two sets of clips, so
the mean is taken again over the clips where both conditions define the
key (`scores.defined_clips`, the same alignment `paired_stats` uses).
Measured once, this changed a sign: one `inst_BF` difference was -0.0235
between the JSONs' means and +0.0029 over the 16 clips both defined. The
aligned count is printed and written whenever it is smaller.

This is a look, not a verdict. The table prints each key's *direction*,
the way it is better: higher, lower, or neither for a reference value
(`scores.SIGNS`, `scores.PILOT_SIGNS`). It prints nothing that reads as
better or worse: whether a difference is distinguishable from zero is
decided by `paired_stats.verdict` alone, on the video-level bootstrap.
One key, `--key`, is followed further, into the per-clip list, the chart
and *wins*: the clips on which it moved the better way, among those where
both conditions define it. It is meant to be the key that decides the
question asked. The default is `F1_50/geometric` on the evaluator's
JSONs, one of the two keys that decide how much of the labelled
structure the regions hold, and the workbench's `inst_F1_50` on a pilot
JSON.

Usage:
    python -m evalkit.tools.compare_eval --base single5 --cond cons5 --dir /path/to/scores
    # boundary_R_raw, which decides what input puts the regions' boundaries
    # on the GT's class boundaries, in the tissue view
    python -m evalkit.tools.compare_eval ... --key boundary_R_raw/tissue
    # knowingly, on clip sets that differ (the summary records population: intersection)
    python -m evalkit.tools.compare_eval ... --allow-subset
    # knowingly, on JSONs that record no eval_code_sha (the summary records
    # eval_code: legacy-unverified)
    python -m evalkit.tools.compare_eval ... --allow-legacy-code
"""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence

from evalkit.tools.scores import (
    PILOT_SIGNS,
    check_comparable,
    defined_clips,
    is_pilot_json,
    load_scores,
    metric_key,
    metric_keys,
    rows_of,
    sign_of,
)

# The default key of the per-clip list, the chart and `wins`. On the
# evaluator's JSONs it is `F1_50` in the geometric view, one of the two keys
# that decide how much of the labelled structure the regions hold; on a pilot
# JSON it is the workbench's. A key that decides another question is asked
# for with `--key`.
PILOT_KEY = "inst_F1_50"
DEFAULT_KEY = ("F1_50", "geometric")

# What the table says beside each metric, from its sign alone.
DIRECTION = {+1: "higher is better", -1: "lower is better", 0: "reference, never marked"}


def metrics_of(summary: dict) -> Sequence[tuple[str, int]]:
    """The keys to report and their directions: the pilot's list on a pilot JSON, the evaluator's own otherwise."""
    if is_pilot_json(summary):
        return tuple(PILOT_SIGNS.items())
    return tuple((k, sign_of(k)) for k in metric_keys(summary))


def primary_key(summary: dict, key: str = "") -> tuple[str, int]:
    """The key the per-clip list, the chart and `wins` follow, and which way it is better.

    Args:
        summary: Either condition's JSON; the two are comparable, so they
            report the same keys.
        key: The key asked for; the default when empty.

    Raises:
        ValueError: `key` is not one the JSON reports, or is a reference
            value, which has no direction to count a win in; or the
            default is asked of an evaluator JSON without the geometric
            view, where it lives.
    """
    signs = dict(metrics_of(summary))
    if not key:
        if is_pilot_json(summary):
            key = PILOT_KEY
        elif DEFAULT_KEY[1] not in summary["views"]:
            raise ValueError(f"the JSON holds no {DEFAULT_KEY[1]} view, where the default key lives: "
                             f"{summary['views']}; name one with --key")
        else:
            key = metric_key(*DEFAULT_KEY)
    if key not in signs:
        raise ValueError(f"{key} is not a key this JSON reports; the keys are {list(signs)}")
    # The pilot's list names every key the pilot evaluator ever wrote, and
    # the evaluator's is built from the views; a key missing from every row
    # is told apart from one that is None on every clip.
    if not any(key in r for r in summary["per_clip"]):
        raise ValueError(f"{key} is in no row of this JSON; the keys its rows hold are "
                         f"{sorted({k for r in summary['per_clip'] for k in r if k in signs})}")
    if signs[key] == 0:
        raise ValueError(f"{key} is a reference value with no direction, so no clip can win on it")
    return key, signs[key]


def load(out_dir: str, tag: str) -> dict:
    """The score JSON of one condition."""
    return load_scores(os.path.join(out_dir, f"{tag}.json"))


def clip_mean(rows: dict, clips: list[str], key: str) -> float:
    """The mean of `key` over `clips` alone, which the JSON's own mean is not when the population is a subset."""
    return sum(rows[c][key] for c in clips) / len(clips)


def compare(a: dict, b: dict, allow_subset: bool = False, allow_legacy_code: bool = False, key: str = "") -> dict:
    """The comparison of two score JSONs: every metric's aligned means and the per-clip wins on one key.

    Args:
        a: The base condition's JSON.
        b: The compared condition's JSON.
        allow_subset: Compare on the common clips when the populations differ.
        allow_legacy_code: Let through JSONs that record no `eval_code_sha`.
        key: The key `wins` counts on; `primary_key`'s default when empty.

    Returns:
        `n_clips`, `population`, `eval_code`, `clips`, `wins` (clips on
        which `key` moved the better way, among those where both define
        it), `metrics` (key to `base`, `cond`, `delta`, `n_clips`, over
        the clips both define; a key that no clip has in both JSONs is
        left out, including one only a single JSON holds),
        `versions_differ` when the check reports it, and `key` and `sign`
        (its direction, +1 when higher is better and -1 when lower) unless
        they are the workbench's own on a pilot JSON: that summary stays
        byte for byte what the workbench wrote.

    Raises:
        ValueError: The two are not comparable, the key is not one to
            count wins on, or it is defined on no common clip.
    """
    # The clips compared, from the check that the two share a ruler; and the key followed.
    chk = check_comparable(a, b, allow_subset=allow_subset, allow_legacy_code=allow_legacy_code)
    clips = chk["clips"]
    ia, ib = rows_of(a), rows_of(b)
    key, sign = primary_key(a, key)

    # Every key's two means and their difference, over the clips both define it on.
    deltas = {}
    for k, _sign in metrics_of(a):
        # The JSONs' own means are not used: each left out its own None
        # clips. And a key's presence is read from the rows, not the
        # summary, which once listed fewer keys than were measured.
        ks = defined_clips(ia, ib, clips, k)
        if not ks:
            continue
        va, vb = clip_mean(ia, ks, k), clip_mean(ib, ks, k)
        deltas[k] = dict(base=round(va, 4), cond=round(vb, 4), delta=round(vb - va, 4), n_clips=len(ks))

    # The wins on the key followed, from its per-clip differences.
    # The key is aligned like the others: a None cannot be a win or a loss.
    # Its count is in `metrics[key]["n_clips"]`; with no clip left there is
    # no per-clip comparison, which is said rather than written as zero wins.
    primary = defined_clips(ia, ib, clips, key)
    if not primary:
        raise ValueError(f"{key} is defined on no common clip, so there is no per-clip comparison to make")
    wins = sum(1 for c in primary if (ib[c][key] - ia[c][key]) * sign > 0)

    # The summary, from the check, the means and the wins.
    out = dict(n_clips=len(clips), population=chk["population"], eval_code=chk["eval_code"],
               clips=clips, wins=wins, metrics=deltas)
    if "versions_differ" in chk:
        out["versions_differ"] = chk["versions_differ"]
    if not (is_pilot_json(a) and key == PILOT_KEY):
        out["key"], out["sign"] = key, sign
    return out


def plot(clips: list[str], ia: dict, ib: dict, tags: tuple[str, str], key: str, sign: int, path: str) -> None:
    """The per-clip change on the left, the two paired distributions on the right.

    A mean alone reads as "uniformly better"; the per-clip bars say whether
    it is every clip or one big win. The bars are coloured by the way the
    key is better, so a fall in a lower-is-better key is green.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = [ib[c][key] - ia[c][key] for c in clips]
    order = sorted(range(len(clips)), key=lambda i: d[i] * sign)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 1]})

    ax1.barh(range(len(order)), [d[i] for i in order],
             color=["#2ca02c" if d[i] * sign > 0 else "#d62728" for i in order])
    ax1.set_yticks(range(len(order)))
    ax1.set_yticklabels([clips[i].replace("__", " ")[:38] for i in order], fontsize=7)
    ax1.axvline(0, color="0.3", lw=1)
    mean_d = sum(d) / len(d)
    ax1.axvline(mean_d, color="#1f77b4", lw=1.2, ls="--", label=f"mean {mean_d:+.3f}")
    ax1.set_xlabel(f"Δ {key}  ({tags[1]} − {tags[0]})")
    ax1.set_title(f"per-clip change ({sum(1 for x in d if x * sign > 0)}/{len(d)} improved)")
    ax1.legend(fontsize=8)
    ax1.grid(axis="x", alpha=0.3)

    for k, (tag, src) in enumerate(zip(tags, (ia, ib))):
        vals = [src[c][key] for c in clips]
        ax2.scatter([k] * len(vals), vals, s=28, alpha=0.75, color="#1f77b4" if k == 0 else "#ff7f0e")
        ax2.hlines(sum(vals) / len(vals), k - 0.22, k + 0.22, color="0.2", lw=2)
    for c in clips:
        ax2.plot([0, 1], [ia[c][key], ib[c][key]], color="0.7", lw=0.6, zorder=0)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(tags, fontsize=8)
    ax2.set_xlim(-0.4, 1.4)
    ax2.set_ylabel(key)
    ax2.set_title("paired per-clip\n(bar = mean)")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="The base condition's tag.")
    ap.add_argument("--cond", required=True, help="The compared condition's tag.")
    ap.add_argument("--dir", required=True, help="The directory of the score JSONs, one <tag>.json per condition.")
    ap.add_argument("--key", default="",
                    help="The key of the per-clip list, the chart and wins, meant to be the one that decides "
                         f"the question asked (default {PILOT_KEY} on a pilot JSON, {metric_key(*DEFAULT_KEY)} on the evaluator's).")
    ap.add_argument("--allow-subset", action="store_true",
                    help="Compare on the clips both hold when the clip sets differ "
                         "(the summary records population: intersection).")
    ap.add_argument("--allow-legacy-code", action="store_true",
                    help="Let through JSONs that record no eval_code_sha "
                         "(the summary records eval_code: legacy-unverified).")
    ap.add_argument("--plot", action="store_true", help="Draw the per-clip chart into <dir>/../figs/.")
    ap.add_argument("--out_json", default="", help="Write the comparison summary to this JSON.")
    args = ap.parse_args()

    # The comparison, from the two JSONs; and the key followed, which the
    # summary leaves out when it is the workbench's default on a pilot JSON.
    try:
        a, b = load(args.dir, args.base), load(args.dir, args.cond)
        res = compare(a, b, allow_subset=args.allow_subset, allow_legacy_code=args.allow_legacy_code, key=args.key)
        key, sign = primary_key(a, args.key)
    except (ValueError, OSError) as e:
        raise SystemExit(f"{args.base} vs {args.cond}: {e}") from e
    clips = res["clips"]
    ia, ib = rows_of(a), rows_of(b)

    # The table: every key's aligned means, its difference and its direction.
    print(f"=== {args.cond} vs {args.base} ({len(clips)} clips, "
          f"population={res['population']}, eval_code={res['eval_code']}) ===")
    if "versions_differ" in res:
        print(f"note: library versions differ: {res['versions_differ']}")
    metrics = metrics_of(a)
    w = max(len("metric"), *(len(k) for k, _ in metrics))
    print(f"{'metric':{w}s} {args.base:>10s} {args.cond:>10s} {'Δ':>9s}  direction")
    for k, s in metrics:
        r = res["metrics"].get(k)
        if r is None:
            print(f"{k:{w}s} {'—':>10s} {'—':>10s} {'—':>9s}  {DIRECTION[s]}   (not in this JSON / defined on no common clip)")
            continue
        note = "" if r["n_clips"] == len(clips) else f"   [{r['n_clips']}/{len(clips)} clips]"
        print(f"{k:{w}s} {r['base']:10.4f} {r['cond']:10.4f} {r['delta']:+9.4f}  {DIRECTION[s]}{note}")

    # The per-clip list on the key followed, and its wins.
    # Per clip: every clip a little, or one clip a lot, mean different things.
    # Only the clips on which both conditions define the key are listed, the
    # worst move first; `compare` has already refused the case where none is.
    primary = defined_clips(ia, ib, clips, key)
    print(f"\n{'clip':46s} {key + ' base':>18s} {'cond':>8s} {'Δ':>8s}")
    for c in sorted(primary, key=lambda c: (ib[c][key] - ia[c][key]) * sign):
        print(f"{c:46s} {ia[c][key]:18.3f} {ib[c][key]:8.3f} {ib[c][key] - ia[c][key]:+8.3f}")
    left_out = "" if len(primary) == len(clips) else f"   [{key} undefined on {len(clips) - len(primary)} clips]"
    moved = "rose" if sign > 0 else "fell"
    print(f"\nclips on which {key} {moved}: {res['wins']}/{len(primary)}{left_out}")

    # The summary, written when asked. A NaN is refused as `paired_stats`
    # refuses it: `json` would write a bare `NaN`, which is not JSON, and a
    # reader would take it for a value.
    summary = dict(base=args.base, cond=args.cond, **res)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        try:
            text = json.dumps(summary, indent=1, ensure_ascii=False, allow_nan=False)
        except ValueError as e:
            raise SystemExit(f"{args.out_json}: not written, a score holds a NaN ({e})") from e
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"→ {args.out_json}")

    # The chart, drawn when asked, over the clips of the per-clip list.
    if args.plot:
        fig_dir = os.path.join(os.path.dirname(os.path.abspath(args.dir)), "figs")
        os.makedirs(fig_dir, exist_ok=True)
        p = os.path.join(fig_dir, f"delta__{args.base}_vs_{args.cond}.png")
        plot(primary, ia, ib, (args.base, args.cond), key, sign, p)
        print(f"→ {p}")


if __name__ == "__main__":
    main()
