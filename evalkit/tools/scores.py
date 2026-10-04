"""One reader for every score JSON: its clips, its rows, its keys, and the ruler it was measured with.

Every tool reads scores through this module, so the rule of
`docs/evaluation.md` ("Recorded with every score") is checked in one
place. `check_comparable` refuses a pair unless everything that must
match does: `eval_code_sha`, dataset, class set, views, mode, the clip
population, and the GT and depth each clip was scored against. Library
versions are the one difference reported instead of refused.

Two kinds of JSON arrive. The evaluator's carry every field in
`EVALUATOR_FIELDS`. The pilot evaluator's predate those fields and carry
none; they are read with the pilot's own settings filled in (class set
`original`, its four domains as views, its rules as the mode). A JSON
with some fields but not all is neither one, and is refused. A pilot
JSON also carries its run-time domain — `tissue_ignore` for the run,
`extra_ignore` clip by clip — which decided what its `tissue` metrics
removed; two pilot JSONs must agree on that too.

The key layout is fixed here and nowhere else. The evaluator writes one
key per metric per view, `metric_key(metric, view)`, and `time_IoU` once
per clip. A pilot JSON keeps the pilot evaluator's own spellings
(`inst_F1_50`, `GT_mIoU`, ...), and each tool lists the ones it reports.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

# The pilot evaluator's sha. Its JSONs are the ones with none of the fields
# below, and this is the value their `eval_code_sha` holds.
PILOT_EVAL_CODE_SHA = "1f8a813a5be31dd204fe4130a1799053411c41166f19821f80ca30d3a943fc58"

# What the evaluator records with every score beyond what the pilot did.
# All or none: a JSON with a part of them is refused.
EVALUATOR_FIELDS = ("class_set", "views", "pilot", "input_shas", "versions")

# The pilot evaluator's four domains, which its JSONs score in place of views.
PILOT_DOMAINS = ("full", "labeled", "tissue", "labeled_tissue")

# The per-frame metrics of `docs/evaluation.md`, one key per view, in the
# table's order; and the one metric that is one value per clip.
FRAME_METRICS = ("F1_50", "SQ", "inst_BF", "mIoU", "boundary_F", "boundary_R_raw",
                 "VI_split", "VI_merge", "unlabelled_share")
CLIP_METRICS = ("time_IoU",)

# Which way is better: +1 larger, -1 smaller, 0 a reference value that is
# reported and never marked. The variation of information counts bits of
# disagreement, so less is better; `time_IoU` and `unlabelled_share` are
# reference values by the specification and get no mark either way.
SIGNS: Mapping[str, int] = MappingProxyType({
    "F1_50": +1, "SQ": +1, "inst_BF": +1, "mIoU": +1, "boundary_F": +1,
    "boundary_R_raw": +1, "VI_split": -1, "VI_merge": -1,
    "unlabelled_share": 0, "time_IoU": 0,
})

# The pilot evaluator's own keys, in the order the workbench printed them
# (detection first, then mask quality and contours, since F1 at 0.5 stays
# perfect while a boundary is tens of pixels off), with which way each is
# better. `underseg_error` is the share of GT a prediction swallowed into a
# neighbour and `overseg_mean` counts the fragments a GT object was split
# into, so less is better. `time_IoU` is a reference value by the
# specification and `n_regions_mean` a count read for its direction alone,
# so neither is marked. Every tool that reports a pilot key takes its
# direction from here; a second table would drift from this one.
PILOT_SIGNS: Mapping[str, int] = MappingProxyType({
    "inst_F1_50": +1, "inst_F1_75": +1, "inst_F1_avg": +1, "PQ": +1, "SQ": +1, "inst_BF": +1,
    "GT_mIoU": +1, "boundary_F": +1, "boundary_R_raw": +1, "boundary_P_raw": +1,
    "time_IoU": 0, "underseg_error": -1, "overseg_mean": -1, "n_regions_mean": 0,
    "inst_F1_50_labeled": +1, "inst_F1_50_tissue": +1, "inst_F1_50_labeled_tissue": +1,
})

# The entry of `input_shas` that is the condition's own. Every other entry
# names an input two comparable scores must have read alike.
PREDICTION_INPUT = "predictions"


def metric_key(metric: str, view: str) -> str:
    """The per-clip key of one metric in one view, as the evaluator writes it."""
    return f"{metric}/{view}"


def split_key(key: str) -> tuple[str, str | None]:
    """A key back into its metric and its view; the view is None for a clip-level key or a pilot key."""
    metric, sep, view = key.partition("/")
    return (metric, view) if sep else (key, None)


def sign_of(key: str) -> int:
    """Which way `key` is better, for a key the evaluator writes.

    Raises:
        KeyError: `key` is not one the evaluator writes.
    """
    metric, _ = split_key(key)
    return SIGNS[metric]


@dataclass(frozen=True)
class Ruler:
    """What a score was measured with: everything that must match for two scores to be compared.

    Attributes:
        eval_code_sha: The evaluator's sha, or None in a JSON that records none.
        dataset: The dataset's name as the JSON spells it.
        pilot: Whether the score was made under the pilot evaluator's rules,
            by the pilot evaluator itself or by this one in pilot mode.
        class_set: The class set scored; `original` for a pilot JSON.
        views: The views the JSON holds; the pilot's domains for a pilot JSON.
        domain: The pilot evaluator's run-level domain, `(dataset,
            tissue_ignore)`, or None when the JSON records none, or when it
            is the evaluator's. The per-clip `extra_ignore` is domain too,
            and is checked clip by clip on the clips compared.
    """

    eval_code_sha: str | None
    dataset: str | None
    pilot: bool
    class_set: str
    views: tuple[str, ...]
    domain: tuple | None


def load_scores(path: str | Path) -> dict:
    """Read one score JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_pilot_json(summary: Mapping) -> bool:
    """Whether `summary` is a pilot evaluator's JSON: one with none of `EVALUATOR_FIELDS`.

    Raises:
        ValueError: The JSON has some of the fields but not all.
    """
    present = [f for f in EVALUATOR_FIELDS if f in summary]
    if present and len(present) != len(EVALUATOR_FIELDS):
        missing = [f for f in EVALUATOR_FIELDS if f not in summary]
        raise ValueError(
            f"a score JSON records {present} but not {missing}; the evaluator writes all of "
            f"{list(EVALUATOR_FIELDS)}, and the pilot evaluator none, so this one is neither"
        )
    return not present


def clips_of(summary: Mapping) -> list[str]:
    """The clips a JSON scored, from `clips`, or from `per_clip` in a JSON written before `clips` existed.

    An empty `clips` is a claim, not a missing field: it says nothing was
    scored, and does not fall back on whatever `per_clip` holds.
    """
    if "clips" in summary:
        return list(summary["clips"])
    return [r["clip"] for r in summary["per_clip"]]


def rows_of(summary: Mapping) -> dict[str, dict]:
    """The per-clip rows by clip name."""
    return {r["clip"]: r for r in summary["per_clip"]}


def _check_rows(summary: Mapping, side: str) -> None:
    """Refuse a JSON whose `clips` and `per_clip` name different clips, or name one twice.

    The tools take the population from `clips_of` and the values from
    `rows_of`. Where the two disagree, a pair is checked on one set of
    clips and averaged over another with nothing failing; a clip named
    twice in `per_clip` keeps its last row and loses the other silently.
    """
    rows = [r["clip"] for r in summary["per_clip"]]
    twice = sorted({c for c in rows if rows.count(c) > 1})
    if twice:
        raise ValueError(f"{side}: per_clip holds more than one row for {twice}")
    claimed = set(clips_of(summary))
    if claimed != set(rows):
        raise ValueError(
            f"{side}: the JSON's clips and its per_clip rows name different clips\n"
            f"  in clips only: {sorted(claimed - set(rows))}\n"
            f"  in per_clip only: {sorted(set(rows) - claimed)}\n"
            "  The population is read from one and the values from the other, so they must agree"
        )


def _pilot_domain(summary: Mapping) -> tuple | None:
    # The pilot's first JSONs recorded no `eval_version` and no domain; the
    # pilot's check compared domains only when both sides had one.
    if summary.get("eval_version") is None:
        return None
    return (summary.get("dataset"), tuple(sorted(summary.get("tissue_ignore") or ())))


def ruler(summary: Mapping) -> Ruler:
    """The ruler a JSON was measured with, the pilot evaluator's where the JSON records none.

    Raises:
        ValueError: The JSON has some of the evaluator's fields but not all.
    """
    if is_pilot_json(summary):
        return Ruler(
            eval_code_sha=summary.get("eval_code_sha"), dataset=summary.get("dataset"),
            pilot=True, class_set="original", views=PILOT_DOMAINS, domain=_pilot_domain(summary),
        )
    return Ruler(
        eval_code_sha=summary.get("eval_code_sha"), dataset=summary.get("dataset"),
        pilot=bool(summary["pilot"]), class_set=str(summary["class_set"]),
        views=tuple(summary["views"]), domain=None,
    )


def metric_keys(summary: Mapping) -> list[str]:
    """The per-clip keys that hold the evaluator's metrics, in the table's order, view by view.

    Raises:
        ValueError: `summary` is a pilot JSON, whose keys each tool lists itself.
    """
    if is_pilot_json(summary):
        raise ValueError("a pilot JSON spells its keys as the pilot evaluator did; the tool lists them")
    keys = [metric_key(m, v) for v in summary["views"] for m in FRAME_METRICS]
    return keys + list(CLIP_METRICS)


def defined_clips(a: Mapping[str, Mapping], b: Mapping[str, Mapping], clips: list[str], key: str) -> list[str]:
    """The clips on which `key` is defined in both conditions, in the order of `clips`.

    `SQ` and `inst_BF` are None on a clip with no hit, and each JSON's own
    mean leaves such clips out independently. Subtracting those means
    compares two populations; the mean is taken again over this list, and
    its length is reported wherever it is shorter than `clips`.
    """
    return [c for c in clips if a[c].get(key) is not None and b[c].get(key) is not None]


def _check_extra_ignores(a: Mapping, b: Mapping, clips: list[str]) -> None:
    rows_a, rows_b = rows_of(a), rows_of(b)
    for clip in clips:
        ea = tuple(sorted(rows_a[clip].get("extra_ignore") or ()))
        eb = tuple(sorted(rows_b[clip].get("extra_ignore") or ()))
        if ea != eb:
            raise ValueError(
                f"{clip}: the two scores removed a different extra_ignore ({list(ea)} vs {list(eb)}); "
                "what `tissue` removed on one clip must be one set of classes"
            )


def _check_inputs(a: Mapping, b: Mapping, clips: list[str]) -> None:
    sa, sb = a["input_shas"], b["input_shas"]
    for clip in clips:
        ia, ib = sa.get(clip), sb.get(clip)
        if ia is None or ib is None:
            raise ValueError(f"{clip}: a score records no input shas for it, so what it read cannot be checked")
        names = (set(ia) | set(ib)) - {PREDICTION_INPUT}
        if not names:
            raise ValueError(
                f"{clip}: the scores name no input beyond the predictions, so whether they "
                "read one GT and one depth map cannot be checked"
            )
        for name in sorted(names):
            if ia.get(name) != ib.get(name):
                raise ValueError(
                    f"{clip}: the two scores read a different {name} "
                    f"({str(ia.get(name))[:16]} vs {str(ib.get(name))[:16]}); "
                    "the scores of one clip must come from one GT and one depth map"
                )


def check_comparable(
    a: Mapping, b: Mapping, allow_subset: bool = False, allow_legacy_code: bool = False,
) -> dict:
    """Check that two score JSONs can be compared, and say on which clips.

    Both checks stop the caller rather than warn: a mix that goes unnoticed
    is the failure this exists to prevent. The ATLAS production set was once
    a mix of two tracking configurations, and a new condition run on it was
    being compared against that mix.

    Args:
        a: The base condition's score JSON.
        b: The compared condition's score JSON.
        allow_subset: Compare on the common clips when the populations differ.
        allow_legacy_code: Let through a JSON that records no `eval_code_sha`.

    Returns:
        `clips`, sorted; `population`, `"identical"` or `"intersection"`;
        `eval_code`, the sha's first 16 characters or `"legacy-unverified"`;
        and, only when the two evaluator JSONs record different library
        versions, `versions_differ`, name to the pair of values.

    Raises:
        ValueError: The shas differ or one is missing; one JSON is the pilot
            evaluator's and the other the evaluator's; the mode, class set,
            views or dataset differ; the pilot domains or a compared clip's
            `extra_ignore` differ; a JSON's `clips` and `per_clip` name
            different clips; the clips differ, no clip is common, or
            neither JSON holds one; or the two read different inputs.
    """
    ra, rb = ruler(a), ruler(b)
    sa, sb = ra.eval_code_sha, rb.eval_code_sha
    ta, tb = a.get("eval_code_tag"), b.get("eval_code_tag")
    if sa is None or sb is None:
        if not allow_legacy_code:
            raise ValueError(
                "a score JSON records no eval_code_sha, so what it was measured with cannot be\n"
                f"  recovered and it cannot be compared: base={sa!r} cond={sb!r}\n"
                "  Score it again, or pass --allow-legacy-code knowingly\n"
                "  (the summary then records eval_code: legacy-unverified)"
            )
        code = "legacy-unverified"
    elif sa != sb:
        raise ValueError(
            "two conditions scored by different evaluators cannot be compared: the difference\n"
            "  between the methods would carry the difference between the evaluators.\n"
            f"  base: tag={ta!r} sha={sa[:16]}\n"
            f"  cond: tag={tb!r} sha={sb[:16]}\n"
            "  Score one of them again with the other's evaluator"
        )
    else:
        code = sa[:16]

    pilot_a, pilot_b = is_pilot_json(a), is_pilot_json(b)
    if pilot_a != pilot_b:
        raise ValueError(
            "one JSON is the pilot evaluator's and the other the evaluator's; the two are never\n"
            "  compared, the check against the pilot evaluator is a verification of its own"
        )
    # The pilot's domain was set at run time under one sha, so the sha alone
    # did not say what `tissue` removed; the pilot's own check compared it
    # whenever both sides recorded one.
    if ra.domain is not None and rb.domain is not None and ra.domain != rb.domain:
        raise ValueError(
            "two conditions scored on different domains cannot be compared (their `tissue`\n"
            f"  metrics remove different classes).\n  base: {ra.domain}\n  cond: {rb.domain}\n"
            "  Score them again with the same --dataset"
        )
    # The dataset is part of the domain, but the pilot's first JSONs record
    # no domain; the dataset they record is still compared, since a score of
    # one dataset set beside a score of another says nothing whatever else
    # matches.
    if ra.dataset is not None and rb.dataset is not None and ra.dataset != rb.dataset:
        raise ValueError(
            f"two conditions scored on different datasets cannot be compared: "
            f"base={ra.dataset!r} cond={rb.dataset!r}"
        )
    settings_a = (ra.dataset, ra.pilot, ra.class_set, ra.views)
    settings_b = (rb.dataset, rb.pilot, rb.class_set, rb.views)
    if not pilot_a and settings_a != settings_b:
        raise ValueError(
            "the same eval_code_sha is not enough: pilot mode and the normal mode share it, and\n"
            "  so do the class sets and the views. These differ:\n"
            f"  base: dataset={ra.dataset!r} pilot={ra.pilot} class_set={ra.class_set!r} views={list(ra.views)}\n"
            f"  cond: dataset={rb.dataset!r} pilot={rb.pilot} class_set={rb.class_set!r} views={list(rb.views)}"
        )

    _check_rows(a, "base")
    _check_rows(b, "cond")
    ca, cb = clips_of(a), clips_of(b)
    if sorted(ca) == sorted(cb):
        if not ca:
            raise ValueError("neither score holds a clip, so there is nothing to compare")
        clips, population = sorted(ca), "identical"
    else:
        only_a, only_b = sorted(set(ca) - set(cb)), sorted(set(cb) - set(ca))
        if not allow_subset:
            raise ValueError(
                "the clip sets differ, so the two cannot be compared (means over different\n"
                f"  populations say nothing).\n  base only: {only_a}\n  cond only: {only_b}\n"
                "  Score them on the same --clips, or pass --allow-subset knowingly"
            )
        clips = sorted(set(ca) & set(cb))
        if not clips:
            raise ValueError(
                f"no clip is common, so --allow-subset cannot help: base={len(ca)} cond={len(cb)}"
            )
        population = "intersection"

    out = dict(clips=clips, population=population, eval_code=code)
    # The per-clip extra_ignore is run-time domain too, and comparing its
    # values as a set would lose which clip each belongs to: the same values
    # on the other clips are a different domain. Checked clip by clip, on
    # the compared clips only, and only when both sides record a domain, as
    # the run-level check is.
    if pilot_a and ra.domain is not None and rb.domain is not None:
        _check_extra_ignores(a, b, clips)
    if not pilot_a:
        _check_inputs(a, b, clips)
        va, vb = a["versions"] or {}, b["versions"] or {}
        differ = {k: (va.get(k), vb.get(k)) for k in sorted(set(va) | set(vb)) if va.get(k) != vb.get(k)}
        if differ:
            out["versions_differ"] = differ
    return out
