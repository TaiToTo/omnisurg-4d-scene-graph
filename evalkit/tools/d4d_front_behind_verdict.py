"""Compare an answer's front/behind scores with a floor's on the same pairs, and give the verdict.

The tool reads two outputs of `evalkit.tools.d4d_front_behind` on the same regions: an answer, such as `mono`,
and a floor, such as `area`. The floor's direction is taken once, over all sides: below 0.5 it is read flipped.
For sessions, clips and specimens, the tool resamples the units `N_BOOT` times and gives the difference of the
shares answered right with its 95 % interval. The verdict is `A` when the answer is ahead and no interval
contains zero, `C` when it is behind and none does, `B` when one does, and `undecided` when the units' signs
differ. The tool also gives the difference for moving and static cameras, and the answer's share by depth gap.
It refuses outputs of other regions, other answer pixels, other code, or other pairs.

Usage:
    python -m evalkit.tools.d4d_front_behind_verdict --answer mono.json --floor area.json \\
        --population d4d_meta/population.json [--out verdict.json]
"""

import argparse
import json
from pathlib import Path

import numpy as np

UNITS = ("session", "clip", "specimen")
N_BOOT = 2000
# The least depth gaps of a pair, in mm, at which the answer's share is given.
GAP_STEPS = (0.0, 2.0, 5.0, 10.0, 20.0)
VERDICTS = {"A": "the answer is ahead of the floor, and no interval contains zero",
            "B": "indistinguishable: an interval contains zero",
            "C": "the answer is behind the floor, and no interval contains zero",
            "undecided": "the differences over sessions, clips and specimens differ in sign"}


def unit_of(rec: dict, unit: str) -> str:
    """Return the clip, session or specimen of a side's record."""
    if unit == "clip":
        return rec["key"]
    if unit == "session":
        return "/".join(rec["key"].split("/")[:2])
    return rec["key"].split("/")[0]


def share(recs: list) -> float:
    """Return the share of the records' pairs answered right, pooled; NaN with no pair."""
    n = sum(r["n_pairs"] for r in recs)
    return sum(r["n_correct"] for r in recs) / n if n else np.nan


def check_comparable(answer: dict, floor: dict) -> None:
    """Refuse two outputs that were not scored alike.

    Raises:
        ValueError: the regions differ, the answer pixels differ while neither answer is `area`, or the code
            hashes differ.
    """
    if answer["regions"] != floor["regions"]:
        raise ValueError(f"the outputs were scored on other regions: {answer['regions']} and {floor['regions']}; "
                         "a floor is measured on the answer's regions")
    # `area` reads no depth, so its answer pixels mean nothing.
    if "area" not in (answer["answer"], floor["answer"]) and answer["answer_pixels"] != floor["answer_pixels"]:
        raise ValueError(f"the outputs took the answer's median over other pixels: {answer['answer_pixels']} and "
                         f"{floor['answer_pixels']}")
    if answer["front_behind_code_sha"] != floor["front_behind_code_sha"]:
        raise ValueError(f"the outputs were scored by other code: {answer['front_behind_code_sha'][:12]} and "
                         f"{floor['front_behind_code_sha'][:12]}")


def check_pairs(answer: list, floor: list) -> None:
    """Refuse two outputs that did not score the same pairs on the same sides.

    The answer leaves out a region whose depth is too sparse and the area floor does not, so the same side can
    hold other pairs.

    Raises:
        ValueError: a side holds pairs in one output only, or other pairs.
    """
    a = {(r["key"], r["side"]): r["pairs"] for r in answer if r["n_pairs"]}
    f = {(r["key"], r["side"]): r["pairs"] for r in floor if r["n_pairs"]}
    if set(a) != set(f):
        raise ValueError(f"sides with pairs in the answer only: {sorted(set(a) - set(f))[:5]}; in the floor only: "
                         f"{sorted(set(f) - set(a))[:5]}")
    other = sorted(k for k in a if a[k] != f[k])
    if other:
        raise ValueError(f"{len(other)} side(s) hold other pairs in the two outputs, the first being {other[0]}")


def paired_delta(answer: list, floor: list, unit: str, flip_floor: bool, seed: int = 0) -> dict:
    """Return the answer's share less the floor's, with its interval, resampling `unit`s together in both."""
    ag, fg = {}, {}
    for recs, groups in ((answer, ag), (floor, fg)):
        for r in recs:
            if r["n_pairs"]:
                groups.setdefault(unit_of(r, unit), []).append(r)
    keys = sorted(ag)

    def shares(sel):
        sa = share([r for k in sel for r in ag[k]])
        sf = share([r for k in sel for r in fg[k]])
        return sa, 1.0 - sf if flip_floor else sf

    sa, sf = shares(keys)
    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(N_BOOT):
        a, f = shares([keys[i] for i in rng.randint(0, len(keys), len(keys))])
        deltas.append(a - f)
    # Count the units the answer wins and loses; the verdict reads the interval.
    per_unit = [np.subtract(*shares([k])) for k in keys]
    return dict(unit=unit, n_units=len(keys), acc_answer=round(sa, 4), acc_floor=round(sf, 4),
                delta=round(sa - sf, 4), lo=round(float(np.nanpercentile(deltas, 2.5)), 4),
                hi=round(float(np.nanpercentile(deltas, 97.5)), 4),
                wins=int(sum(d > 0 for d in per_unit)), losses=int(sum(d < 0 for d in per_unit)))


def gap_curve(recs: list) -> list:
    """Return the answer's share on the pairs whose depth gap is at least each of `GAP_STEPS`; None with no pair."""
    gap = np.concatenate([np.asarray(r["gap_mm"]) for r in recs])
    ok = np.concatenate([np.asarray(r["ok"]) for r in recs])
    return [dict(gap_mm=g, n_pairs=int((gap >= g).sum()),
                 acc=round(float(ok[gap >= g].mean()), 4) if (gap >= g).any() else None) for g in GAP_STEPS]


def verdict(main: dict) -> str:
    """Return the verdict of the units' differences: `A`, `B`, `C` or `undecided`, as `VERDICTS` says."""
    if len({np.sign(main[u]["delta"]) for u in main}) > 1:
        return "undecided"
    if any(main[u]["lo"] <= 0 <= main[u]["hi"] for u in main):
        return "B"
    return "A" if main["session"]["delta"] > 0 else "C"


def compare(answer: dict, floor: dict, moved: set) -> dict:
    """Return the verdict and its numbers for two outputs, `moved` holding the clips whose camera moves."""
    # Refuse outputs not scored alike, then take the floor's direction once.
    check_comparable(answer, floor)
    arecs = [r for r in answer["per_frame"] if r["n_pairs"]]
    frecs = [r for r in floor["per_frame"] if r["n_pairs"]]
    check_pairs(arecs, frecs)
    flip = bool(share(frecs) < 0.5)

    # Compare on every unit, then on the clips whose camera moves and those whose camera is static.
    main = {u: paired_delta(arecs, frecs, u, flip) for u in UNITS}
    strata = {}
    for label, is_moved in (("moved", True), ("static", False)):
        a = [r for r in arecs if (r["key"] in moved) == is_moved]
        f = [r for r in frecs if (r["key"] in moved) == is_moved]
        strata[label] = paired_delta(a, f, "clip", flip, seed=1) if a else None
    return dict(flip_floor=flip, main=main, strata=strata, gap_curve_answer=gap_curve(arecs), verdict=verdict(main))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--answer", required=True, help="the output of d4d_front_behind for the answer")
    ap.add_argument("--floor", required=True, help="the output of d4d_front_behind for the floor")
    ap.add_argument("--population", required=True, help="the population file, which says whose camera moves")
    ap.add_argument("--out", help="the JSON file to write")
    args = ap.parse_args(argv)

    answer, floor = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (args.answer, args.floor))
    population = json.loads(Path(args.population).read_text(encoding="utf-8"))
    try:
        out = dict(answer=args.answer, floor=args.floor, **compare(
            answer, floor, {s["key"] for s in population["sides"] if s["moved_camera"]}))
    except ValueError as e:
        raise SystemExit(f"refused: {e}")

    print(f"floor {floor['answer']}{', read flipped' if out['flip_floor'] else ''}")
    for unit, d in out["main"].items():
        print(f"  {unit:9s} n={d['n_units']:3d}  answer={d['acc_answer']:.4f} floor={d['acc_floor']:.4f}  "
              f"delta={d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]  wins {d['wins']}-{d['losses']}")
    for label, d in out["strata"].items():
        print(f"  {label:9s} " + (f"n={d['n_units']:3d}  delta={d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"
                                  if d else "no clip"))
    for row in out["gap_curve_answer"]:
        print(f"  gap >= {row['gap_mm']:4.1f} mm: n={row['n_pairs']:6d}  acc={row['acc']}")
    print(f"verdict: {out['verdict']} ({VERDICTS[out['verdict']]})")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
