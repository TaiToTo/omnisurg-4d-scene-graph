"""Select the D4D sides that the measurements score, from a census of the sides' inputs.

`d4d_meta/README.md` defines a side, the census and the rule. The rule reads no score. It keeps each
side or leaves it out for one reason. The step writes `population.json`, which lists the kept sides and
counts the sides of each reason, and `clips.txt`, which lists the clips with a kept side. It refuses:

- a list of clips that is empty or holds a clip twice;
- a census that lacks a clip of `--clips`, such as a census cut short;
- a census that holds a clip `--clips` does not hold, holds a clip twice, or holds them in another order;
- a side that carries an `error`, which the census recorded when it failed to measure the side;
- a side with a point cloud and no `active`, whose tissue motion is unknown.

Usage:
    python -m pipeline.select_d4d_population --census /path/to/census.json \\
        --clips d4d_meta/census_clips.txt --out d4d_meta
"""

import argparse
import json
from collections import Counter
from pathlib import Path

SIDES = ("start", "end")
KEEP = "keep"
# The least `gt_blk_frac` a kept side may have. It leaves out no side of the census; `d4d_meta/README.md`
# gives the least.
MIN_GT_BLOCK_FRAC = 0.20
# A side whose tissue moved is left out. Its point cloud holds one shape, and the frames hold another.
EXCLUDE_ACTIVE = True
# The longest time allowed between a side's scan and the frame nearest to it, in seconds.
MAX_GT_FRAME_GAP_S = 10.0


def read_clips(path: str) -> list[str]:
    """Read the clips a complete census holds, one `<specimen>/<session>/<clip>` per line.

    Raises:
        ValueError: the list is empty or holds a clip twice.
    """
    clips = Path(path).read_text(encoding="utf-8").split()
    if not clips:
        raise ValueError(f"{path} lists no clip")
    twice = sorted(k for k, n in Counter(clips).items() if n > 1)
    if twice:
        raise ValueError(f"{path} lists a clip twice: {twice}")
    return clips


def check_census(census: list[dict], clips: list[str]) -> None:
    """Refuse a census that does not hold each of `clips` once and in order, or holds a side the rule cannot classify.

    Raises:
        ValueError: a clip of `clips` is missing, a clip is not in `clips` or appears twice, the clips are
            not in the order of `clips`, a side carries an `error`, or a side has a point cloud and no
            `active`.
    """
    keys = [c["key"] for c in census]
    held = set(keys)
    missing = [k for k in clips if k not in held]
    if missing:
        raise ValueError(f"the census lacks {len(missing)} of the {len(clips)} clips it must hold, the first "
                         f"being {missing[0]}; a population selected from it would cover part of D4D")
    extra = sorted(held - set(clips))
    twice = sorted(k for k, n in Counter(keys).items() if n > 1)
    if extra or twice:
        raise ValueError(f"the census holds clips the list does not: {extra}, or holds a clip twice: {twice}")
    if keys != clips:
        first = next(i for i, (a, b) in enumerate(zip(keys, clips)) if a != b)
        raise ValueError(f"the census holds its clips in another order than the list, first at line {first + 1}: "
                         f"{keys[first]}, where the list has {clips[first]}")
    for c in census:
        for side in SIDES:
            s = c[side]
            if "error" in s:
                raise ValueError(f"{c['key']} {side}: the census failed to measure the side: {s['error']}")
            if s.get("present") and "active" not in s:
                raise ValueError(f"{c['key']} {side}: the side has a point cloud and no `active`, so whether "
                                 "the tissue moved is unknown")


def classify(side: dict) -> str:
    """Return `KEEP` for a side the measurements score, or the reason it is left out."""
    if not side.get("present"):
        return "no_gt"
    if side["gt_blk_frac"] < MIN_GT_BLOCK_FRAC:
        return "gt_not_visible"
    if EXCLUDE_ACTIVE and side["active"]:
        return "tissue_moving"
    if abs(side["frame_minus_gt_s"]) > MAX_GT_FRAME_GAP_S:
        return "gt_stale"
    return KEEP


def select(census: list[dict]) -> dict:
    """Classify every side of `census` and return the population: the kept sides, and the counts of each reason."""
    reasons: dict[str, int] = {}
    kept = []
    for c in census:
        for side in SIDES:
            reason = classify(c[side])
            reasons[reason] = reasons.get(reason, 0) + 1
            if reason == KEEP:
                kept.append({"key": c["key"], "side": side, "moved_camera": c["moved_camera"],
                             "specimen": c["specimen"]})
    keys = {k["key"] for k in kept}
    moved = {k["key"] for k in kept if k["moved_camera"]}
    static = {k["key"] for k in kept if not k["moved_camera"]}
    return {
        "n_clips_total": len(census),
        "n_sides_total": len(SIDES) * len(census),
        "exclusions": reasons,
        "n_sides_kept": len(kept),
        "n_clips_kept": len(keys),
        "n_clips_moved": len(moved),
        "n_clips_static": len(static),
        "n_sessions_kept": len({"/".join(k["key"].split("/")[:2]) for k in kept}),
        "n_specimens_kept": len({k["specimen"] for k in kept}),
        "thresholds": {"min_gt_block_frac": MIN_GT_BLOCK_FRAC, "exclude_active": EXCLUDE_ACTIVE,
                       "max_gt_frame_gap_s": MAX_GT_FRAME_GAP_S},
        "sides": kept,
    }


def kept_clips(population: dict) -> list[str]:
    """Return the clips with at least one kept side, sorted."""
    return sorted({s["key"] for s in population["sides"]})


def main(argv: list[str] | None = None) -> None:
    """Read the census and the list of clips, select the sides, and write `population.json` and `clips.txt`."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--census", required=True, help="the census of every clip, as JSON")
    ap.add_argument("--clips", required=True, help="the clips a complete census holds, one per line")
    ap.add_argument("--out", required=True, help="the directory to write population.json and clips.txt into")
    args = ap.parse_args(argv)

    census = json.loads(Path(args.census).read_text(encoding="utf-8"))
    try:
        check_census(census, read_clips(args.clips))
    except ValueError as e:
        raise SystemExit(str(e)) from e
    population = select(census)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "population.json").write_text(json.dumps(population, indent=1), encoding="utf-8")
    (out / "clips.txt").write_text("\n".join(kept_clips(population)) + "\n", encoding="utf-8")

    print(f"{population['n_clips_total']} clips, {population['n_sides_total']} sides")
    for reason, n in sorted(population["exclusions"].items(), key=lambda kv: -kv[1]):
        print(f"  {reason:16s} {n:4d}")
    print(f"kept {population['n_sides_kept']} sides of {population['n_clips_kept']} clips: "
          f"{population['n_clips_moved']} with a moving camera, {population['n_clips_static']} with a still one")


if __name__ == "__main__":
    main()
