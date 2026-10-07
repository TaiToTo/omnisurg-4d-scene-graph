"""Pilot mode against the pilot evaluator: every key the two share, equal on every clip.

The verification of `docs/evaluation.md`, "Checked against the pilot
evaluator". For each condition, matched by tag, the pilot evaluator's
JSON and this evaluator's pilot-mode JSON (`evalkit.evaluate --pilot`)
must hold the same value, None included, for every pair of keys in
`SHARED`, and the same number of frames behind each, `COUNTS`, with no
tolerance. The result names the evaluator's sha, the one the freeze
records. The check runs where the pilot evaluator and its scores are.

Usage:
    python -m evalkit.tools.pilot_check --pilot-dir /path/to/the/pilot/scores \\
        --eval-dir /path/to/the/pilot_mode/scores
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
    check_rows,
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
    # The class map and the boundaries, on the full domain alone.
    for pilot, ours in (("GT_mIoU", "mIoU"), ("boundary_F", "boundary_F"), ("boundary_R_raw", "boundary_R_raw")):
        pairs.append((pilot, metric_key(ours, "full")))
    # VI, which the pilot computed on the valid pixels whose GT is not
    # background: the `labeled` domain's mask.
    for key in ("VI_split", "VI_merge"):
        pairs.append((key, metric_key(key, "labeled")))
    pairs.append(("time_IoU", "time_IoU"))
    return tuple(pairs)


# `(pilot key, evaluator key)` for every key the two evaluators share. A key the
# evaluator replaced (`inst_F1_75`, `PQ`, the counts) is not compared: the replacement is the point.
SHARED = _shared()


def _counts() -> tuple[tuple[str, str], ...]:
    pairs = []
    for domain in PILOT_DOMAINS:
        suffix = _pilot_suffix(domain)
        for pilot, ours in (("n_inst_frames", "F1_50"), ("n_SQ_frames", "SQ"), ("n_BF_frames", "inst_BF")):
            pairs.append((f"{pilot}{suffix}", metric_key(ours, domain)))
    for ours in ("mIoU", "boundary_F", "boundary_R_raw"):
        pairs.append(("n_gt_frames", metric_key(ours, "full")))
    for ours in ("VI_split", "VI_merge"):
        pairs.append(("n_vi_frames", metric_key(ours, "labeled")))
    return tuple(pairs)


# `(pilot count, evaluator key)`: the frames behind each shared key, which the
# evaluator's row holds under `n_frames`. A clip mean rounded to four decimals
# can absorb a frame left out or added; its count cannot.
COUNTS = _counts()


# No tolerance: whatever width were allowed here is the width a fault in the
# port could pass through.
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
        or count the evaluator's row lacks, or a value or count that is not
        equal. Empty when every one is equal on every clip.

    Raises:
        ValueError: `pilot` is not the pilot evaluator's (another sha, or
            the evaluator's fields); `ours` is not a pilot-mode JSON over
            the pilot's domains; a JSON's `clips` and `per_clip` disagree
            or name a clip twice; neither holds a clip; a pilot row removes
            an `extra_ignore`, which pilot mode scores with none; or a
            pilot row lacks a shared key or a count.
    """
    # The two JSONs, refused unless one is the pilot evaluator's and the other pilot mode's.
    # Not `scores.check_comparable`: the two shas differ by construction, and it would refuse the pair.
    if not is_pilot_json(pilot) or pilot.get("eval_code_sha") != PILOT_EVAL_CODE_SHA:
        raise ValueError(
            "the JSON given as the pilot evaluator's is not its "
            f"(sha {str(pilot.get('eval_code_sha'))[:16]}); the check compares against {PILOT_EVAL_CODE_SHA[:16]}"
        )
    r = ruler(ours)
    if is_pilot_json(ours) or not r.pilot:
        raise ValueError("the JSON given as the evaluator's is not a pilot-mode score of this evaluator; "
                         "only pilot mode is checked")
    if tuple(r.views) != PILOT_DOMAINS:
        raise ValueError(f"a pilot-mode JSON scores the pilot's domains {list(PILOT_DOMAINS)}, not {list(r.views)}")

    # The populations, each whole in itself, and the clips only one side scored.
    check_rows(pilot, "the pilot evaluator's JSON")
    check_rows(ours, "the pilot-mode JSON")
    out = []
    a, b = rows_of(pilot), rows_of(ours)
    ca, cb = set(clips_of(pilot)), set(clips_of(ours))
    if not ca and not cb:
        raise ValueError("neither JSON holds a clip, so there is nothing to check")
    for c in sorted(ca ^ cb):
        out.append(f"{c}: scored by {'the pilot evaluator' if c in ca else 'the evaluator'} only")

    # The shared keys, clip by clip, on the clips both scored.
    for c in sorted(ca & cb):
        ra, rb = a[c], b[c]
        # Pilot mode scores with no `extra_ignore`; a pilot score that
        # removed one was measured on another domain and cannot be matched.
        if ra.get("extra_ignore"):
            raise ValueError(f"{c}: the pilot evaluator's row removed extra_ignore={ra['extra_ignore']}; "
                             "pilot mode scores with none, so this condition cannot be checked")
        for pk, ok in SHARED:
            if pk not in ra:
                raise ValueError(f"{c}: the pilot evaluator's row lacks {pk}; the pilot evaluator writes every "
                                 "shared key on every clip, so this JSON is not whole")
            if ok not in rb:
                out.append(f"{c}.{ok}: missing; the pilot wrote {pk}={ra[pk]!r}")
            elif not _equal(ra[pk], rb[ok]):
                out.append(f"{c}.{ok}: {rb[ok]!r} != pilot {pk}={ra[pk]!r}")
        counts = rb.get("n_frames", {})
        for pk, ok in COUNTS:
            if pk not in ra:
                raise ValueError(f"{c}: the pilot evaluator's row lacks the count {pk}, so the frames behind "
                                 f"{ok} cannot be checked")
            if ok not in counts:
                out.append(f"{c}.n_frames[{ok}]: missing; the pilot wrote {pk}={ra[pk]!r}")
            elif counts[ok] != ra[pk]:
                out.append(f"{c}.n_frames[{ok}]: {counts[ok]!r} != pilot {pk}={ra[pk]!r}")
    return out


def _read(path: str, role: str) -> dict:
    """`load_scores`, with a file that will not open or parse refused by its role and path."""
    try:
        return load_scores(path)
    except (OSError, ValueError) as e:
        raise ValueError(f"{role} {path} cannot be read: {e}") from e


def check_dirs(pilot_dir: str, eval_dir: str) -> tuple[str, dict[str, list[str]]]:
    """Diff every condition of `pilot_dir` against the JSON of the same tag in `eval_dir`.

    Returns:
        The evaluator's `eval_code_sha`, read from the pilot-mode JSONs,
        and tag to its differences; a tag with none maps to an empty list.
        A tag the evaluator's directory lacks is a difference of its own.

    Raises:
        ValueError: `pilot_dir` holds no JSON; `eval_dir` holds none of
            the tags; the pilot-mode JSONs were made by more than one
            evaluator; a JSON of a pair cannot be read; or a pair cannot
            be checked.
    """
    tags = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(pilot_dir, "*.json")))
    if not tags:
        raise ValueError(f"no score JSON in {pilot_dir}")
    out, shas = {}, {}
    for tag in tags:
        theirs = os.path.join(eval_dir, f"{tag}.json")
        if not os.path.exists(theirs):
            out[tag] = [f"{tag}: not scored in pilot mode ({theirs} is missing)"]
            continue
        try:
            ours = _read(theirs, "the pilot-mode JSON")
            out[tag] = diff_shared(_read(os.path.join(pilot_dir, f"{tag}.json"), "the pilot evaluator's JSON"), ours)
        except ValueError as e:
            raise ValueError(f"{tag}: {e}") from e
        shas.setdefault(ruler(ours).eval_code_sha, []).append(tag)
    # The evaluator's sha: one, or the result belongs to no evaluator.
    if not shas:
        raise ValueError(f"none of the {len(tags)} conditions is scored in pilot mode under {eval_dir}")
    if len(shas) > 1:
        raise ValueError(
            f"the pilot-mode JSONs were made by {len(shas)} evaluators, and the check is of one:\n"
            + "\n".join(f"  {str(s)[:16]}: {v}" for s, v in sorted(shas.items(), key=lambda x: str(x[0])))
        )
    (sha,) = shas
    if sha is None:
        raise ValueError("the pilot-mode JSONs record no eval_code_sha, so what passed the check cannot be named")
    return sha, out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot-dir", required=True, help="The pilot evaluator's score JSONs, one <tag>.json per condition.")
    ap.add_argument("--eval-dir", required=True, help="This evaluator's pilot-mode JSONs of the same conditions, by tag.")
    args = ap.parse_args()
    try:
        sha, result = check_dirs(args.pilot_dir, args.eval_dir)
    except ValueError as e:
        raise SystemExit(str(e))
    print(f"pilot mode of evaluator {sha[:16]} against the pilot evaluator {PILOT_EVAL_CODE_SHA[:16]}")
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
    print(f"\nevery shared key equal on all {len(result)} conditions: evaluator {sha[:16]} reproduces the pilot evaluator")


if __name__ == "__main__":
    main()
