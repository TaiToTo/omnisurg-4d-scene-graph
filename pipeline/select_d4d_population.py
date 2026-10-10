"""Select the D4D sides that the measurements score, from a census of the sides' inputs.

`d4d_meta/README.md` defines a side, the census and the rule. The rule reads no score. It keeps each
side or leaves it out for one reason. The step writes `population.json`, which lists the kept sides and
counts the sides of each reason, and `clips.txt`, which lists the clips with a kept side. It refuses:

- a list of clips that is empty, holds a clip twice, or holds a line that is not `<specimen>/<session>/<clip>`;
- a census that does not hold each clip of `--clips` once and in its order, such as a census cut short;
- a side that carries an `error`, which the census recorded when it failed to measure the side;
- a field the rule reads that is missing or holds a value of the wrong kind, such as a side with a point
  cloud and no `active`, whose tissue motion is unknown, or a `NaN`, which no threshold leaves out.

Usage:
    python -m pipeline.select_d4d_population --census /path/to/census.json \\
        --clips d4d_meta/census_clips.txt --out d4d_meta
"""

import argparse
import json
import math
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
        ValueError: the list is empty, holds a clip twice, or holds a line that is not three names joined by `/`.
    """
    clips = Path(path).read_text(encoding="utf-8").split()
    if not clips:
        raise ValueError(f"{path} lists no clip")
    # A kept side's specimen is read from the first name of its key.
    malformed = [k for k in clips if len(k.split("/")) != 3 or "" in k.split("/")]
    if malformed:
        raise ValueError(f"{path} lists a clip that is not `<specimen>/<session>/<clip>`: {malformed[0]}")
    twice = sorted(k for k, n in Counter(clips).items() if n > 1)
    if twice:
        raise ValueError(f"{path} lists a clip twice: {twice}")
    return clips


def check_census(census: list[dict], clips: list[str]) -> None:
    """Refuse a census that does not hold each of `clips` once and in order, or holds a side the rule cannot classify.

    Raises:
        ValueError: a clip of `clips` is missing, a clip is not in `clips` or appears twice, the clips are
            not in the order of `clips`, a side carries an `error`, or a field the rule reads is missing or
            holds a value of the wrong kind.
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
    # Each field the rule reads is there and holds a value of its kind.
    for c in census:
        check_bool(c, "moved_camera", c["key"])
        for side in SIDES:
            s = c[side]
            where = f"{c['key']} {side}"
            if "error" in s:
                raise ValueError(f"{where}: the census failed to measure the side: {s['error']}")
            check_bool(s, "present", where)
            if not s["present"]:
                continue
            if "active" not in s:
                raise ValueError(f"{where}: the side has a point cloud and no `active`, so whether the tissue "
                                 "moved is unknown")
            check_bool(s, "active", where)
            for name in ("gt_blk_frac", "frame_minus_gt_s"):
                check_finite(s, name, where)


def check_bool(record: dict, name: str, where: str) -> None:
    """Refuse a record whose field `name` is missing or is neither true nor false.

    Raises:
        ValueError: the field is missing, or holds another value, such as the string `"false"`, which reads as true.
    """
    if name not in record:
        raise ValueError(f"{where}: no `{name}`")
    if not isinstance(record[name], bool):
        raise ValueError(f"{where}: `{name}` is {record[name]!r}, not true or false")


def check_finite(record: dict, name: str, where: str) -> None:
    """Refuse a record whose field `name` is missing or is not a finite number.

    Python's `json` reads `NaN` and `Infinity`, and a `NaN` compares false with every threshold, so a side
    holding one would be kept.

    Raises:
        ValueError: the field is missing, or holds a value that is not a finite number, such as `NaN`, a
            string, or true or false.
    """
    if name not in record:
        raise ValueError(f"{where}: no `{name}`")
    v = record[name]
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise ValueError(f"{where}: `{name}` is {v!r}, not a finite number")


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
                             "specimen": c["key"].split("/")[0]})
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
