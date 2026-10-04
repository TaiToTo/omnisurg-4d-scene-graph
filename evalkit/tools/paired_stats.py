"""Paired statistics between two conditions, and the one rule that gives a claim a star.

The decision rests on the paired differences. Tracking is deterministic,
so a difference between two conditions carries no run noise, and the
spread between clips cancels when the same clips are subtracted; two SDs
laid side by side throw that away, and F1 is a bounded ratio whose SD is
not a standard error. A bootstrap 95 % interval of the mean difference
decides, with a Wilcoxon p-value reported beside it.

The bootstrap resamples videos, not clips: a video's clips are not
independent, and resampling them gives an interval too narrow. Below two
videos there is no interval, only a point that would read as a verdict;
`boot_ci` refuses, and the row says None.

The population is the one `check_comparable` compared, aligned per metric
by `scores.defined_clips` as `compare_eval` does, since `SQ` and `inst_BF`
are None on a clip with no hit. A shrunken population is printed as
`[16/18 clips, 6/7 videos]`, never averaged silently.

Each row records `cond - base`, the way the metric moved, and the metric's
`sign`; `verdict` reads the two together, so a metric where less is better
(`VI_split`) or a reference value never marked (`time_IoU`) is not
oriented by hand.

Usage:
    python -m evalkit.tools.paired_stats --eval-dir /path/to/scores \\
        --pairs base:cond,base:other --out /path/to/paired.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from types import MappingProxyType

import numpy as np

from evalkit.tools.scores import (
    check_comparable,
    defined_clips,
    is_pilot_json,
    load_scores,
    metric_keys,
    rows_of,
    sign_of,
)

# The keys reported on a pilot evaluator's JSON, in the order the workbench
# reported them. A key the JSON does not hold is skipped and said so. The
# evaluator's JSONs name their keys themselves (`scores.metric_keys`).
PILOT_KEYS = ("inst_F1_50", "inst_F1_75", "inst_F1_avg", "PQ", "SQ", "inst_BF",
              "boundary_F", "boundary_R_raw", "boundary_P_raw", "GT_mIoU", "underseg_error")
# Which way each pilot key is better, as `scores.SIGNS` says for the
# evaluator's keys. `underseg_error` is the share of GT that a prediction
# swallowed into a neighbour, so less is better; every other key is a score.
PILOT_SIGNS: Mapping[str, int] = MappingProxyType({k: -1 if k == "underseg_error" else +1 for k in PILOT_KEYS})
N_BOOT = 10000
# The global seed. The draw is not decided by it alone: `_boot_rng` builds
# the generator from `[SEED, blake2b(differences, groups)]`, so that the same
# input gives the same interval whatever was computed before it. A JSON that
# records `seed` alone cannot identify the draw, so `SEED_SCHEME` is written
# beside it.
SEED = 0
SEED_SCHEME = "default_rng([SEED, blake2b(d, groups)]), independent of call order"


def keys_of(summary: dict) -> Sequence[str]:
    """The keys to report on a JSON: the pilot's list on a pilot JSON, the evaluator's own otherwise."""
    return PILOT_KEYS if is_pilot_json(summary) else metric_keys(summary)


def sign_of_key(summary: dict, key: str) -> int:
    """Which way `key` is better on this JSON: the pilot's table on a pilot JSON, `scores.sign_of` otherwise."""
    return PILOT_SIGNS[key] if is_pilot_json(summary) else sign_of(key)


def load_json(tag: str, eval_dir: str) -> dict:
    """The score JSON of one condition, whole, as `check_comparable` needs it."""
    return load_scores(os.path.join(eval_dir, f"{tag}.json"))


def video_of(clip: str) -> str:
    """The video a clip was cut from: the unit the bootstrap resamples.

    Clips of one video are not independent, so the video is the resampling
    unit. This is the one definition; a second copy that is fixed alone
    would split the unit silently, with intervals still being printed.

    ATLAS-120k: `gastric_surgery__4FHGGFZsPzw__gt_0004` is from
    `gastric_surgery__4FHGGFZsPzw`. CholecSeg8k: `VID26_s15_1855_crop` is
    from `VID26`.
    """
    if "__" in clip:
        return "__".join(clip.split("__")[:-1])
    return clip.split("_")[0]


def wilcoxon_video(d: np.ndarray, groups: np.ndarray) -> float | None:
    """The Wilcoxon p-value on the per-video mean differences.

    Clip pairs within a video are not independent, so the clip-level test
    overstates the evidence. With five videos or fewer the smallest
    attainable p is 2^-(n-1) and the value says nothing, so it is None.
    """
    from scipy import stats

    uniq = np.unique(groups)
    if len(uniq) < 6:
        return None
    dv = np.array([d[groups == g].mean() for g in uniq])
    if not np.any(dv != 0):
        return None
    return float(stats.wilcoxon(dv, zero_method="wilcox").pvalue)


def _boot_rng(d: np.ndarray, groups: np.ndarray | None, seed: int) -> np.random.Generator:
    """A generator decided by the data, so that the interval does not depend on call order.

    One generator shared across pairs and metrics advances with every call,
    and the interval of a pair then changed with the order of `--pairs` and
    with how many metrics were skipped before it. Measured: endpoints moved
    by up to 0.0007, and of 5,008 intervals in one analysis 122 had an
    endpoint within 0.0015 of zero, so a star could come and go with the
    order alone. Seeding from the content of the input removes that.
    """
    h = hashlib.blake2b(np.ascontiguousarray(d, dtype=np.float64).tobytes(), digest_size=8)
    if groups is not None:
        h.update(b"|")
        h.update("\x00".join(map(str, groups)).encode())
    return np.random.default_rng([seed, int.from_bytes(h.digest(), "big")])


def boot_ci(d: np.ndarray, groups: np.ndarray | None = None, rng=None,
            *, seed: int = SEED) -> tuple[float, float]:
    """The bootstrap 95 % interval of the mean difference, resampled by `groups` when given.

    The generator is made inside, from the data. The third argument is what
    the old interface took; passing it raises rather than being ignored,
    because an ignored seed reads as the seed that was used.

    Args:
        d: The paired differences.
        groups: The resampling unit of each difference; None resamples
            the differences themselves.
        rng: Refused.
        seed: The global seed, to redraw everything at once.

    Raises:
        TypeError: `rng` was passed.
        ValueError: `groups` does not name one unit per difference, or
            there are fewer than two units to resample. With one unit
            every resample is the same one, and the point that comes out
            is not an interval but reads as a verdict.
    """
    if rng is not None:
        raise TypeError(
            "boot_ci no longer takes a generator: the interval depended on call order. "
            "Call boot_ci(d, groups); pass seed= only to redraw everything."
        )
    d = np.asarray(d, dtype=np.float64)
    if groups is not None:
        # A list compared with one group is a single False, not a mask, and
        # the interval came out (nan, nan): no mark, and no error.
        groups = np.asarray(groups)
        if groups.shape != d.shape:
            raise ValueError(f"groups names {groups.shape} units for {d.shape} differences")
    units = len(d) if groups is None else len(np.unique(groups))
    if units < 2:
        what = "difference" if groups is None else "video"
        raise ValueError(
            f"a bootstrap over {units} {what} has no interval: every resample is the same one, "
            "and the point it gives would read as a verdict"
        )
    rng = _boot_rng(d, groups, seed)
    if groups is None:
        idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
        means = d[idx].mean(1)
    else:
        uniq = np.unique(groups)
        by = [d[groups == g] for g in uniq]
        means = np.empty(N_BOOT)
        for b in range(N_BOOT):
            pick = rng.integers(0, len(uniq), size=len(uniq))
            means[b] = np.concatenate([by[i] for i in pick]).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# The verdict. One definition, here; every script that prints a star
# borrows `verdict` rather than deciding one.
#
# The decision is the video-level bootstrap 95 % interval not straddling
# zero, read in the metric's direction, and nothing else. The clip-level
# interval and the p-values are printed beside it as reference and never
# decide.
#
# Why p is not part of it: the unit. ATLAS-120k's clips come from a few
# videos, and one video can supply a third of them, so a Wilcoxon over clip
# pairs inflates the evidence. The video-level p is conservative but
# undefined below six videos, which some populations have; the interval is
# defined on any population of two videos or more. The claim about 3D
# predicates was decided by the interval alone already, so one rule covers
# every claim.
#
# The change was not small: of 4,883 comparisons that had both the video
# interval and a p-value, 340 (7.0 %) changed mark under this rule, every
# one from no mark to a star or a cross. When the rule changes, run every
# script that prints a mark and compare before and after; scanning saved
# analyses misses the tools that compute an interval on the spot.
VERDICT_RULE = "the video-level bootstrap 95 % CI does not straddle zero (p is reported, never decisive)"


def verdict(ci95_video: tuple[float, float] | list[float] | None, sign: int = +1) -> str:
    """A star (better), a cross (worse) or nothing (not distinguishable), from the video-level interval.

    Args:
        ci95_video: `[lo, hi]` of the video-level bootstrap of `cond - base`;
            None when there is none.
        sign: Which way the metric is better, as `scores.SIGNS` spells it:
            +1 larger, -1 smaller, 0 a reference value that is never marked.

    Returns:
        `"★"`, `"✗"` or `""`.

    Raises:
        ValueError: `sign` is not +1, -1 or 0.
    """
    if sign not in (1, -1, 0):
        raise ValueError(f"sign is +1, -1 or 0, not {sign!r}")
    if sign == 0 or ci95_video is None or ci95_video[0] is None or ci95_video[1] is None:
        return ""
    lo, hi = float(ci95_video[0]), float(ci95_video[1])
    if sign < 0:
        lo, hi = -hi, -lo
    if lo > 0:
        return "★"
    if hi < 0:
        return "✗"
    return ""


def is_star(ci95_video, sign: int = +1) -> bool:
    """Whether `verdict` gives a star, for a caller that wants the truth value alone."""
    return verdict(ci95_video, sign) == "★"


def _sd(x: np.ndarray) -> float | None:
    """The sample SD, or None on one value, where `std(ddof=1)` would write NaN into the JSON."""
    return None if len(x) < 2 else round(float(x.std(ddof=1)), 4)


def _stats_of(va: np.ndarray, vb: np.ndarray, kvids: np.ndarray, sign: int) -> dict:
    """The row of one key: the counts, the two distributions, the paired difference and its intervals.

    A quantity that has no value on this few clips or videos is None, never
    a NaN or a point: the SDs on one clip, the clip interval on one clip,
    the video interval and the video-level p below their minimum.
    """
    from scipy import stats

    d = vb - va
    n_videos = int(len(np.unique(kvids)))
    w = stats.wilcoxon(d, zero_method="wilcox") if np.any(d != 0) else None
    ci_clip = None if len(d) < 2 else [round(x, 4) for x in boot_ci(d, None)]
    ci_video = None if n_videos < 2 else [round(x, 4) for x in boot_ci(d, kvids)]
    pv = wilcoxon_video(d, kvids)
    return dict(
        # A shrunken population is never averaged silently: the counts stay.
        n_clips=len(va), n_videos=n_videos, sign=sign,
        base_mean=round(float(va.mean()), 4), base_sd=_sd(va),
        cond_mean=round(float(vb.mean()), 4), cond_sd=_sd(vb),
        delta_mean=round(float(d.mean()), 4), delta_sd=_sd(d),
        delta_median=round(float(np.median(d)), 4),
        wins=int((d > 0).sum()), losses=int((d < 0).sum()), ties=int((d == 0).sum()),
        wilcoxon_p=None if w is None else round(float(w.pvalue), 5),
        wilcoxon_p_video=None if pv is None else round(pv, 5),
        ci95_clip=ci_clip,
        ci95_video=ci_video,
    )


def compare_pair(ja: dict, jb: dict, drop: Sequence[str] = (), allow_legacy_code: bool = False) -> dict:
    """The paired statistics of every key between two conditions' score JSONs.

    Args:
        ja: The base condition's JSON.
        jb: The compared condition's JSON.
        drop: Videos whose clips are left out, to measure again without
            them. A subset has fewer videos, so its interval is read for
            its sign only.
        allow_legacy_code: Let through JSONs that record no `eval_code_sha`.

    Returns:
        `n_clips`, `n_videos`, `dropped_videos`, `eval_code`, `population`,
        `metrics` (key to its statistics; a key absent from the JSONs or
        defined on no common clip is left out), and `versions_differ` when
        the check reports it.

    Raises:
        ValueError: The two JSONs are not comparable, `drop` names a video
            the scores do not have, or `drop` leaves no clip.
    """
    # The ruler and the domain are checked, not only the population; the
    # check also settles that the populations are equal (no subset here).
    chk = check_comparable(ja, jb, allow_legacy_code=allow_legacy_code)
    a, b = rows_of(ja), rows_of(jb)
    # The subset is taken from the clips the check compared, and after it:
    # taking it first would hide a mismatch, and taking it from the rows
    # would be a second population beside the one checked.
    drop = set(drop)
    unknown = drop - {video_of(c) for c in chk["clips"]}
    if unknown:
        raise ValueError(f"--drop-video names no video of these scores: {sorted(unknown)}")
    clips = [c for c in chk["clips"] if video_of(c) not in drop]
    if not clips:
        raise ValueError(f"--drop-video left no clip ({sorted(drop)})")
    vids = [video_of(c) for c in clips]
    res = {}
    for k in keys_of(ja):
        ks = defined_clips(a, b, clips, k)
        if not ks:
            continue
        kvids = np.array([video_of(c) for c in ks])
        va = np.array([a[c][k] for c in ks], dtype=np.float64)
        vb = np.array([b[c][k] for c in ks], dtype=np.float64)
        res[k] = _stats_of(va, vb, kvids, sign_of_key(ja, k))
    out = dict(n_clips=len(clips), n_videos=int(len(set(vids))), dropped_videos=sorted(drop),
               # What it was measured with stays with the result.
               eval_code=chk["eval_code"], population=chk["population"], metrics=res)
    if "versions_differ" in chk:
        out["versions_differ"] = chk["versions_differ"]
    return out


def _num(x: float | None, spec: str) -> str:
    return "none" if x is None else format(x, spec)


def _interval(ci: list[float] | None) -> str:
    return "none" if ci is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"


def _print_pair(base: str, cond: str, pair: dict, keys: Sequence[str]) -> None:
    print(f"\n===== {cond} vs {base}  ({pair['n_clips']} clips / {pair['n_videos']} videos) =====")
    if "versions_differ" in pair:
        print(f"  note: library versions differ: {pair['versions_differ']}")
    for k in keys:
        r = pair["metrics"].get(k)
        if r is None:
            print(f"  {k}: (not in this JSON / defined on no common clip)")
            continue
        note = ""
        if (r["n_clips"], r["n_videos"]) != (pair["n_clips"], pair["n_videos"]):
            note = f"   [{r['n_clips']}/{pair['n_clips']} clips, {r['n_videos']}/{pair['n_videos']} videos]"
        print(f"  {k}{note}")
        print(f"    per condition (SD over clips): {base} {r['base_mean']:.4f}±{_num(r['base_sd'], '.4f')}"
              f"   {cond} {r['cond_mean']:.4f}±{_num(r['cond_sd'], '.4f')}")
        print(f"    paired difference: mean {r['delta_mean']:+.4f} / median {r['delta_median']:+.4f}"
              f" / SD {_num(r['delta_sd'], '.4f')} / wins-losses-ties {r['wins']}-{r['losses']}-{r['ties']}")
        print(f"    95% CI  clips resampled {_interval(r['ci95_clip'])}"
              f"   videos resampled {_interval(r['ci95_video'])}"
              f"   Wilcoxon p={r['wilcoxon_p']} (per video p={r['wilcoxon_p_video']})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True, help="The directory of the score JSONs, one <tag>.json per condition.")
    ap.add_argument("--pairs", required=True, help="<base>:<cond>, comma-separated.")
    ap.add_argument("--drop-video", default="",
                    help="Leave these videos' clips out and measure again (comma-separated). "
                         "A subset has fewer videos, so read its interval for its sign only.")
    ap.add_argument("--allow-legacy-code", action="store_true",
                    help="Let through JSONs that record no eval_code_sha (refused by default).")
    ap.add_argument("--out", default="", help="Write the statistics of every pair to this JSON.")
    args = ap.parse_args()

    drop = {v.strip() for v in args.drop_video.split(",") if v.strip()}
    out = {}
    for pair in [p.strip() for p in args.pairs.split(",") if p.strip()]:
        try:
            base, sep, cond = pair.partition(":")
            if not (sep and base and cond) or ":" in cond:
                raise ValueError("--pairs takes <base>:<cond>")
            ja, jb = load_json(base, args.eval_dir), load_json(cond, args.eval_dir)
            out[pair] = compare_pair(ja, jb, drop, allow_legacy_code=args.allow_legacy_code)
        except (ValueError, OSError) as e:
            raise SystemExit(f"{pair}: {e}") from e
        _print_pair(base, cond, out[pair], keys_of(ja))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"n_boot": N_BOOT, "seed": SEED, "seed_scheme": SEED_SCHEME, "pairs": out}, f,
                      indent=1, allow_nan=False)
        print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
