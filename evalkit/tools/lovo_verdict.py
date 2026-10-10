"""Name the videos a verdict rests on: leave each video out in turn, and take the verdict again.

For each pair of conditions and each key, the verdict is `paired_stats.verdict`
on the video-level bootstrap of `cond - base`, read in the key's direction.
A video whose removal changes the mark holds the verdict: the claim then
says something about that video, not about the population.

This is not a test. A subset has fewer videos, so its interval is read for
its sign only. Leaving a video out moves the mean and changes the interval's
width, and either can change the mark. So each row records the mean without
the video and the ratio of widths, and the printout says neither is the cause.

Usage:
    python -m evalkit.tools.lovo_verdict --eval-dir /path/to/scores \\
        --pairs base:cond,base:other --keys inst_F1_50,SQ,boundary_F --out /path/to/lovo.json
"""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence

import numpy as np

from evalkit.tools.paired_stats import (
    N_BOOT,
    SEED,
    SEED_SCHEME,
    VERDICT_RULE,
    boot_ci,
    load_json,
    sign_of_key,
    verdict,
    video_of,
)
from evalkit.tools.scores import check_comparable, check_one_rule, defined_clips, rows_of

# A video is left out only while this many videos remain, so a key on no more videos than this is refused. On
# two videos the bootstrap has four distinct resamples, and its interval is the range of the two video means: a
# mark there says only that both means have one sign.
MIN_VIDEOS_AFTER_DROP = 3
# An interval taken on fewer videos than this is printed with a warning that it is thin.
THIN_VIDEOS = 5


def _r(x: float) -> float:
    """Round to four decimals, keeping a value that would round to zero at two significant digits.

    A mark follows the sign of an interval's end. An end of -2.4e-05
    written as -0.0 cannot be read back to its mark.
    """
    r = round(float(x), 4)
    return r if r != 0.0 or float(x) == 0.0 else float(f"{float(x):.2g}")


def _f(x: float) -> str:
    """Print a difference with four decimals, or in exponent form where four decimals would show zero."""
    return f"{x:+.4f}" if abs(x) >= 5e-5 or x == 0 else f"{x:+.2e}"


def _ci(ci: Sequence[float]) -> str:
    return f"[{_f(ci[0])}, {_f(ci[1])}]"


def direction(summary: dict, key: str) -> int:
    """Return which way `key` is better on this JSON, refusing a key no mark can be read on.

    Raises:
        ValueError: The key has no direction in the table `paired_stats`
            reads, or its direction is 0. A key of direction 0, such as the
            reference value `time_IoU` or the count `n_regions_mean`, is
            never marked, so no video can hold its mark.
    """
    try:
        sign = sign_of_key(summary, key)
    except KeyError:
        raise ValueError(f"{key}: no direction is known for this key, so no mark can be read on it") from None
    if sign == 0:
        raise ValueError(f"{key} has direction 0 and is never marked, so no video can hold its mark")
    return sign


def lovo(d: np.ndarray, vids: np.ndarray, sign: int) -> dict:
    """Take the verdict on all the videos together, then again with each video left out.

    Args:
        d: The paired differences `cond - base`, one per clip.
        vids: The video of each clip.
        sign: Which way the key is better: +1 higher, -1 lower.

    Returns:
        `n_clips`, `n_videos` and `sign`; `delta` and `ci95_video`, as
        `cond - base` and not turned by `sign`; `verdict`, read in the
        key's direction; `n_flips` and `flips`, the videos whose removal
        changes the mark; and `per_video`, one row per video left out,
        with `video`, `n_clips` (its clips), `n_videos_wo` (the videos
        left), `own_delta` (its clips' mean difference), `delta_wo`,
        `ci95_wo`, `ci_width_ratio` (the interval's width without the
        video over its width with every video, None when that is zero)
        and `verdict_wo`. A video is left out only while
        `MIN_VIDEOS_AFTER_DROP` videos remain.

    Raises:
        ValueError: `sign` is not +1 or -1; the clips come from fewer than
            two videos, where the bootstrap has no interval; or they come
            from no more than `MIN_VIDEOS_AFTER_DROP` videos, where no
            video can be left out.
    """
    if sign not in (1, -1):
        raise ValueError(f"sign is +1 or -1, not {sign!r}: a reference value is never marked")
    d = np.asarray(d, dtype=np.float64)
    vids = np.asarray(vids)
    full_ci = list(boot_ci(d, vids))
    full = verdict(full_ci, sign)
    n_videos = len(set(vids.tolist()))
    if n_videos <= MIN_VIDEOS_AFTER_DROP:
        raise ValueError(f"{n_videos} videos: leaving one out would leave fewer than {MIN_VIDEOS_AFTER_DROP}, "
                         "so no video can be left out and none can be named")
    per = []
    for v in sorted(set(vids.tolist())):
        m = vids != v
        n_wo = len(set(vids[m].tolist()))
        if n_wo < MIN_VIDEOS_AFTER_DROP:
            continue
        ci = list(boot_ci(d[m], vids[m]))
        per.append(dict(
            video=v, n_clips=int((~m).sum()), n_videos_wo=n_wo,
            own_delta=_r(d[~m].mean()), delta_wo=_r(d[m].mean()),
            ci95_wo=[_r(ci[0]), _r(ci[1])],
            ci_width_ratio=_r((ci[1] - ci[0]) / (full_ci[1] - full_ci[0])) if full_ci[1] != full_ci[0] else None,
            verdict_wo=verdict(ci, sign)))
    flips = [p["video"] for p in per if p["verdict_wo"] != full]
    return dict(n_clips=int(len(d)), n_videos=int(n_videos),
                # `delta` is `cond - base` and `verdict` is read in the key's
                # direction, so the direction stays with them.
                sign=int(sign), delta=_r(d.mean()), ci95_video=[_r(full_ci[0]), _r(full_ci[1])],
                verdict=full, n_flips=len(flips), flips=flips, per_video=per)


def lovo_pair(ja: dict, jb: dict, keys: Sequence[str]) -> dict:
    """Take `lovo` on every key of one pair, over the clips `check_comparable` compared.

    Args:
        ja: The base condition's score JSON.
        jb: The compared condition's score JSON.
        keys: The keys to take the verdict on.

    Returns:
        Key to `lovo`'s result. A key defined on no common clip is left out.

    Raises:
        ValueError: The two JSONs are not comparable, a key has no
            direction to read a mark in, a key is in no row of either JSON,
            or a key's clips come from too few videos (`lovo`).
    """
    chk = check_comparable(ja, jb)
    a, b = rows_of(ja), rows_of(jb)
    res = {}
    for k in keys:
        sign = direction(ja, k)
        # A key in no row is a wrong key or a wrong JSON, not a key undefined on every clip.
        for tag, rows in ((ja.get("tag", "base"), a), (jb.get("tag", "cond"), b)):
            if not any(k in r for r in rows.values()):
                raise ValueError(f"{k} is in no row of {tag}")
        # Per key, the clips on which both conditions define it: `SQ` is None on a clip with no hit.
        ks = defined_clips(a, b, chk["clips"], k)
        if not ks:
            continue
        vids = np.array([video_of(c) for c in ks])
        d = np.array([b[c][k] for c in ks], dtype=np.float64) - np.array([a[c][k] for c in ks], dtype=np.float64)
        try:
            res[k] = lovo(d, vids, sign)
        except ValueError as e:
            raise ValueError(f"{k}: {e}") from e
    return res


def _thin(n_videos: int) -> str:
    return f"  <- an interval on {n_videos} videos is thin" if n_videos < THIN_VIDEOS else ""


def _print_key(k: str, r: dict) -> None:
    lower = " (lower is better)" if r["sign"] < 0 else ""
    print(f"  {k}{lower}  Δ={_f(r['delta'])} {_ci(r['ci95_video'])} {r['verdict'] or 'no mark'}"
          f" ({r['n_clips']} clips / {r['n_videos']} videos){_thin(r['n_videos'])}")
    if not r["n_flips"]:
        print("    no single video left out changes the verdict")
        return
    print(f"    {r['n_flips']} video(s) change the verdict when left out:")
    for p in r["per_video"]:
        if p["verdict_wo"] == r["verdict"]:
            continue
        # Both the mean and the width move; which of them changed the mark is not read from either alone.
        w = p["ci_width_ratio"]
        width = "" if w is None else f", the interval {w:.2f} times as wide"
        print(f"      without {p['video']} ({p['n_clips']} clips, its own Δ={_f(p['own_delta'])}):"
              f" Δ={_f(p['delta_wo'])} {_ci(p['ci95_wo'])} {p['verdict_wo'] or 'no mark'}"
              f" ({p['n_videos_wo']} videos, the mean moved {_f(p['delta_wo'] - r['delta'])}{width})"
              f"{_thin(p['n_videos_wo'])}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True, help="The directory of the score JSONs, one <tag>.json per condition.")
    ap.add_argument("--pairs", required=True, help="<base>:<cond>, comma-separated.")
    ap.add_argument("--keys", required=True,
                    help="The keys to take the verdict on, comma-separated; each must have a direction.")
    ap.add_argument("--out", default="", help="Write the verdicts of every pair to this JSON.")
    args = ap.parse_args()

    # Every pair's two JSONs, read before any is compared.
    pairs, loaded = [], {}
    for pair in [p.strip() for p in args.pairs.split(",") if p.strip()]:
        try:
            base, sep, cond = pair.partition(":")
            if not (sep and base and cond) or ":" in cond:
                raise ValueError("--pairs takes <base>:<cond>")
            for tag in (base, cond):
                if tag not in loaded:
                    loaded[tag] = load_json(tag, args.eval_dir)
        except (ValueError, OSError) as e:
            raise SystemExit(f"{pair}: {e}") from e
        pairs.append((pair, base, cond))

    # One rule for the whole run, because all its pairs go into one JSON.
    try:
        rule = check_one_rule(loaded)
    except ValueError as e:
        raise SystemExit(f"--pairs: {e}") from e

    # Every key's direction, on every JSON, before any bootstrap runs.
    keys = [k.strip() for k in args.keys.split(",") if k.strip()]
    if not keys:
        raise SystemExit("--keys names no key")
    try:
        for summary in loaded.values():
            for k in keys:
                direction(summary, k)
    except ValueError as e:
        raise SystemExit(f"--keys: {e}") from e

    # The verdicts of each pair, printed as they come.
    out = {"verdict_rule": VERDICT_RULE, "seed_scheme": SEED_SCHEME, "min_videos_after_drop": MIN_VIDEOS_AFTER_DROP}
    # A pilot JSON records no rule, and the workbench wrote no evaluator or bootstrap, so none is written.
    if rule is not None:
        out.update(propagation=rule, n_boot=N_BOOT, seed=SEED, eval_code={})
    out["pairs"] = {}
    for pair, base, cond in pairs:
        try:
            res = lovo_pair(loaded[base], loaded[cond], keys)
        except ValueError as e:
            raise SystemExit(f"{pair}: {e}") from e
        print(f"\n===== {cond} vs {base} =====")
        for k in keys:
            if k in res:
                _print_key(k, res[k])
            else:
                print(f"  {k}: (defined on no clip of both conditions)")
        out["pairs"][pair] = res
        if rule is not None:
            out["eval_code"][pair] = check_comparable(loaded[base], loaded[cond])["eval_code"]

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False, allow_nan=False)
        print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
