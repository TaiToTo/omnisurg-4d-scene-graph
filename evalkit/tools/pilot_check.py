"""The check against the pilot evaluator: pilot mode must write the pilot evaluator's numbers.

This is the verification of `docs/evaluation.md`, "Checked against the
pilot evaluator", and not a comparison under `scores.check_comparable`:
the two shas differ by construction, so that check would refuse the pair.
Only the keys the two evaluators share are compared, the metrics table
names them, and on every clip of every condition the values written must
be equal. There is no tolerance to loosen: whatever width were allowed
here is the width a fault in the port could pass through.

A pilot evaluator's JSON spells its keys as it did, with a suffix per
domain (`inst_F1_50`, `inst_F1_50_labeled`, ...). The evaluator in pilot
mode scores the same four domains in place of views, so its keys are the
view spelling over those domains (`F1_50/full`, `F1_50/labeled`, ...).
The class map, the boundary metrics and the variation of information the
pilot computed on the `full` domain alone; `time_IoU` is one value per
clip. `SHARED` is that correspondence.

The keys the pilot wrote and the evaluator does not (`inst_F1_75`, `PQ`,
`GT_mDice`, the tolerances, the counts) are not compared: the metrics
table says what each one is replaced by, and that replacement is the
point of the new evaluator, not a fault. A shared key that the evaluator's
JSON lacks on a clip the pilot scored is a difference, and so is a None
where the pilot wrote 0: pilot mode writes the pilot's zeros.

The driver that scores a condition in pilot mode is not here; this module
diffs what it wrote against the pilot's JSON of the same condition, by
tag, and the run happens where the pilot evaluator and its scores are.

Usage:
    python -m evalkit.tools.pilot_check --pilot-dir /path/to/pilot/eval_v2 \\
        --eval-dir /path/to/pilot_mode/scores
"""
from __future__ import annotations

import argparse
import glob
import math
import os
from collections.abc import Mapping

from evalkit.tools.scores import (
    PILOT_DOMAINS,
    PILOT_EVAL_CODE_SHA,
    clips_of,
    is_pilot_json,
    load_scores,
    metric_key,
    rows_of,
    ruler,
)


def _pilot_suffix(domain: str) -> str:
    return "" if domain == "full" else f"_{domain}"


def _shared() -> tuple[tuple[str, str], ...]:
    pairs = []
    # The instance metrics, in every domain.
    for domain in PILOT_DOMAINS:
        suffix = _pilot_suffix(domain)
        for pilot, ours in (("inst_F1_50", "F1_50"), ("SQ", "SQ"), ("inst_BF", "inst_BF")):
            pairs.append((f"{pilot}{suffix}", metric_key(ours, domain)))
    # The class map, the boundaries and VI, on the full domain alone.
    for pilot, ours in (("GT_mIoU", "mIoU"), ("boundary_F", "boundary_F"), ("boundary_R_raw", "boundary_R_raw"),
                        ("VI_split", "VI_split"), ("VI_merge", "VI_merge")):
        pairs.append((pilot, metric_key(ours, "full")))
    pairs.append(("time_IoU", "time_IoU"))
    return tuple(pairs)


# `(pilot key, evaluator key)` for every key the two evaluators share.
SHARED = _shared()


def _equal(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
            return True
        return a == b
    return a == b


def diff_shared(pilot: Mapping, ours: Mapping) -> list[str]:
    """The differences between a pilot evaluator's JSON and a pilot-mode JSON of the same condition.

    Args:
        pilot: The pilot evaluator's score JSON.
        ours: The evaluator's score JSON of the same condition, in pilot mode.

    Returns:
        One line per difference: a clip only one side scored, a shared key
        the evaluator's row lacks, or a value that is not equal. Empty when
        every shared value is equal on every clip.

    Raises:
        ValueError: `pilot` is not the pilot evaluator's (another sha, or
            the evaluator's fields), or `ours` is not a pilot-mode JSON.
    """
    if not is_pilot_json(pilot) or pilot.get("eval_code_sha") != PILOT_EVAL_CODE_SHA:
        raise ValueError(
            f"the first JSON is not the pilot evaluator's (sha {str(pilot.get('eval_code_sha'))[:16]}); "
            f"the check compares against {PILOT_EVAL_CODE_SHA[:16]}"
        )
    r = ruler(ours)
    if is_pilot_json(ours) or not r.pilot:
        raise ValueError("the second JSON is not a pilot-mode score of this evaluator; only pilot mode is checked")
    if tuple(r.views) != PILOT_DOMAINS:
        raise ValueError(f"a pilot-mode JSON scores the pilot's domains {list(PILOT_DOMAINS)}, not {list(r.views)}")

    out = []
    a, b = rows_of(pilot), rows_of(ours)
    ca, cb = set(clips_of(pilot)), set(clips_of(ours))
    for c in sorted(ca ^ cb):
        out.append(f"{c}: scored by {'the pilot evaluator' if c in ca else 'the evaluator'} only")
    for c in sorted(ca & cb):
        ra, rb = a[c], b[c]
        for pk, ok in SHARED:
            if pk not in ra:
                # The pilot's first JSONs lack some keys (VI, inst_BF); a
                # key the pilot did not write is not compared.
                continue
            if ok not in rb:
                out.append(f"{c}.{ok}: missing; the pilot wrote {pk}={ra[pk]!r}")
            elif not _equal(ra[pk], rb[ok]):
                out.append(f"{c}.{ok}: {rb[ok]!r} != pilot {pk}={ra[pk]!r}")
    return out


def check_dirs(pilot_dir: str, eval_dir: str) -> dict[str, list[str]]:
    """Diff every condition of `pilot_dir` against the JSON of the same tag in `eval_dir`.

    Returns:
        Tag to its differences; a tag with none maps to an empty list. A
        tag the evaluator's directory lacks is a difference of its own.

    Raises:
        ValueError: `pilot_dir` holds no JSON, or a pair cannot be checked.
    """
    tags = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(pilot_dir, "*.json")))
    if not tags:
        raise ValueError(f"no score JSON in {pilot_dir}")
    out = {}
    for tag in tags:
        theirs = os.path.join(eval_dir, f"{tag}.json")
        if not os.path.exists(theirs):
            out[tag] = [f"{tag}: not scored in pilot mode ({theirs} is missing)"]
            continue
        out[tag] = diff_shared(load_scores(os.path.join(pilot_dir, f"{tag}.json")), load_scores(theirs))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot-dir", required=True, help="The pilot evaluator's score JSONs, one <tag>.json per condition.")
    ap.add_argument("--eval-dir", required=True, help="This evaluator's pilot-mode JSONs of the same conditions, by tag.")
    args = ap.parse_args()
    try:
        result = check_dirs(args.pilot_dir, args.eval_dir)
    except ValueError as e:
        raise SystemExit(str(e))
    bad = {t: d for t, d in result.items() if d}
    for tag in sorted(result):
        d = result[tag]
        print(f"  {tag:32s} {'diff 0' if not d else f'!! {len(d)} differences'}")
    if bad:
        print(f"\n!! {len(bad)} of {len(result)} conditions differ")
        for tag, d in sorted(bad.items()):
            for line in d[:10]:
                print(f"    {line}")
            if len(d) > 10:
                print(f"    ... {len(d) - 10} more")
        raise SystemExit(1)
    print(f"\nevery shared key equal on all {len(result)} conditions")


if __name__ == "__main__":
    main()
