"""What ran under which condition, on which clips, counted from the files on disk.

Reads the provenance the pipeline writes beside each clip's labels
(`seed_info.json`) and the score JSONs, prints a table per root, and
exits non-zero on a problem: a tag whose clips were made under more than
one condition; a condition missing clips or labels the others have; a
score whose labels are gone, or with `--require-scored` labels nobody
scored; and scores in one directory that `scores.check_comparable` would
not compare with one another. Terms as in `docs/evaluation.md`.

Usage:
    python -m evalkit.tools.condition_inventory \\
        --root '/data/tracks/seed:/data/scores:seed stage' \\
        --root '/data/tracks/propagated::propagation stage'
    python -m evalkit.tools.condition_inventory --root '...' --matrix
"""
from __future__ import annotations

import argparse
import collections
import datetime
import glob
import json
import os
import sys
from collections.abc import Mapping, Sequence

from evalkit.tools.scores import check_comparable, check_one_rule, clips_of, load_scores, ruler

# The provenance file the pipeline writes beside a clip's labels.
PROV_NAME = "seed_info.json"

# The provenance fields that tell one condition from another. `clip` and
# `tag` say where the labels are, not what made them; of `frames`, which is
# that clip's own frame numbers, only the kind is kept. `bidir` says
# whether the tracker ran both ways. A tag tracked both ways on some clips
# and forward on others mixes two propagation rules. `seed_labels` names
# seeds made outside the tracking stage, which is all that tells two merges
# of one condition to different K apart.
PROV_KEYS = ("sam_input", "depth_source", "seed_source", "track_base",
             "seed_min_area", "seed_topk", "point_grids", "bidir", "seed_labels")
SEED_KEYS = ("points_per_side", "seed_edge_gain", "seed_smooth",
             "edge_ring_masked", "seed_sam_kwargs", "produced_by")

# A label directory that holds pictures, never labels of a condition.
VIZ_DIR = "viz"

# What a cell of the matrix says about a condition. Letters, not symbols: a symbol
# beside a condition's name reads as a mark, and only `paired_stats.mark_of` gives a mark.
SCORED, LABELS_ONLY, PROVENANCE_ONLY, NOT_RUN = "S", "L", "P", "-"


def parse_root(spec: str) -> tuple[str, str, str]:
    """One `--root` into `(label root, score root, title)`.

    The form is `<label root>[:<score root>[:<title>]]`. An empty score
    root says that this stage's labels are not scored in a directory of
    their own, as the propagation stage's are not. `:` separates, so a path
    with a `:` in it cannot be given; fix the path rather than the
    separator.

    Raises:
        ValueError: The label root is empty, or there are more than three fields.
    """
    parts = spec.split(":")
    if len(parts) > 3:
        raise ValueError(f"--root takes at most <label root>:<score root>:<title>: {spec!r}")
    track = parts[0].strip()
    if not track:
        raise ValueError(f"--root has an empty label root: {spec!r}")
    ev = parts[1].strip() if len(parts) > 1 else ""
    title = parts[2].strip() if len(parts) > 2 else track
    return track, ev, title


def check_root_exists(track_root: str, eval_root: str = "") -> None:
    """Refuse a label root, or a score root that was given, that is not a directory.

    A root that does not exist reads as zero labels, or zero scores, no
    check fires, and the tool ends with "no problem": the gate passes
    silently on a typo.

    Raises:
        ValueError: `track_root`, or a non-empty `eval_root`, is not a directory.
    """
    if not os.path.isdir(track_root):
        raise ValueError(f"the --root label directory does not exist (a typo?): {track_root}")
    if eval_root and not os.path.isdir(eval_root):
        raise ValueError(f"the --root score directory does not exist (a typo?): {eval_root}")


def frames_kind(v) -> str:
    """The kind of a provenance `frames` value: `all`, `gt`, `list` or `None`."""
    if isinstance(v, list):
        return "list"
    return str(v)


def condition_key(record) -> str:
    """Return what tells a clip's condition from another's: the fields of `PROV_KEYS` and `SEED_KEYS`, as JSON.

    Raises:
        ValueError: `record`, or its `seed_input`, is not a JSON object.
    """
    if not isinstance(record, Mapping):
        raise ValueError(f"a JSON {type(record).__name__}, not an object")
    si = record.get("seed_input") or {}
    if not isinstance(si, Mapping):
        raise ValueError(f"seed_input is a JSON {type(si).__name__}, not an object")
    key = {**{k: record.get(k) for k in PROV_KEYS}, **{k: si.get(k) for k in SEED_KEYS},
           "frames": frames_kind(record.get("frames"))}
    # The tracker writes "" when it read no seed labels, and its versions before `seed_labels` wrote no key.
    # Neither names seeds, so the two are one condition.
    key["seed_labels"] = key["seed_labels"] or None
    return json.dumps(key, sort_keys=True, ensure_ascii=False)


def read_conditions(track_root: str) -> dict:
    """The provenance under `track_root`, by tag.

    Returns:
        `{tag: {"keys": {condition key as JSON: [clip, ...]}, "notes": {reason:
        [clip, ...]}, "labels": {clip: label count}, "mtime": [...]}}`. `keys`
        holds only provenance that could be read; a clip whose provenance is
        missing or broken goes under `notes`, so that a reader of `keys` can
        always parse them. `labels` holds every clip that has a label
        directory, whatever its provenance.
    """
    out = collections.defaultdict(lambda: {"keys": collections.defaultdict(list),
                                           "notes": collections.defaultdict(list),
                                           "labels": {}, "mtime": []})
    # The clips with a provenance file, by what it says. Only the provenance is
    # read, never the script that launched the run: a script is edited later.
    for p in glob.glob(os.path.join(track_root, "*", "*", PROV_NAME)):
        lab_dir = os.path.dirname(p)
        tag = os.path.basename(lab_dir)
        clip = os.path.basename(os.path.dirname(lab_dir))
        # Counted before the provenance is read, so that a clip with a broken
        # one is counted once, here, and not again below as one without.
        out[tag]["labels"][clip] = len(glob.glob(os.path.join(lab_dir, "label_*.npy")))
        out[tag]["mtime"].append(os.path.getmtime(p))
        try:
            with open(p, encoding="utf-8") as f:
                key = condition_key(json.load(f))
        except (OSError, ValueError) as e:
            # A broken provenance is reported, not hidden.
            out[tag]["notes"][f"unreadable provenance: {e}"].append(clip)
            continue
        out[tag]["keys"][key].append(clip)
    # The clips with labels and no provenance file, per clip: a tag can hold
    # both kinds, and skipping it would hide these from the mixed-condition check.
    for d in glob.glob(os.path.join(track_root, "*", "*", "")):
        tag = os.path.basename(d.rstrip("/"))
        clip = os.path.basename(os.path.dirname(d.rstrip("/")))
        if tag == VIZ_DIR or not glob.glob(os.path.join(d, "label_*.npy")):
            continue
        if clip not in out[tag]["labels"]:
            out[tag]["labels"][clip] = len(glob.glob(os.path.join(d, "label_*.npy")))
            out[tag]["notes"]["no provenance (what made these labels is unknown)"].append(clip)
    return out


def read_evals(eval_root: str) -> dict:
    """Every score JSON under `eval_root`, by its tag, with the label directory it names.

    Returns:
        `{tag: {"dir": track_dir_name, "summary": the JSON}}`; empty when
        `eval_root` is empty.

    Raises:
        ValueError: A JSON there is not a score JSON, records some of the
            evaluator's fields but not all, or does not name the directory
            it scored. A directory of scores must hold nothing a reader
            cannot place, and a score that does not say what it scored
            cannot be matched to its labels.
    """
    if not eval_root:
        return {}
    out = {}
    for p in sorted(glob.glob(os.path.join(eval_root, "*.json"))):
        try:
            d = load_scores(p)
        except (OSError, ValueError) as e:
            raise ValueError(f"{p}: not a readable JSON: {e}") from e
        if not isinstance(d, dict) or "per_clip" not in d:
            raise ValueError(f"{p}: not a score JSON (no per_clip); a score directory holds scores only")
        ruler(d)
        if not d.get("track_dir_name"):
            raise ValueError(f"{p}: records no track_dir_name, so which labels it scored is unknown "
                             "and it cannot be matched to them")
        out[os.path.basename(p)[:-5]] = dict(dir=d["track_dir_name"], summary=d)
    return out


def clips_of_condition(c: dict) -> set[str]:
    """The clips a tag covers: those with readable provenance and those without, in one set."""
    return set(c["labels"])


def comparable_groups(evals: dict) -> tuple[list[list[str]], list[tuple[str, str, str]]]:
    """Group the score tags so that every two in a group are comparable, by `scores.check_comparable`.

    Each tag joins the first group whose every member it is comparable
    with, and starts one when there is none. Every member, not the first
    alone: the check is not transitive. A pilot JSON that records no domain
    is comparable with one of either domain, and a `per_frame` condition
    with one of either propagation rule.

    Returns:
        The groups, each a sorted list of tags, largest first; and, per tag
        that started a group after the first, the tag, the first member of
        the first group it is not comparable with, and the check's message.
    """
    groups: list[list[str]] = []
    reasons: list[tuple[str, str, str]] = []
    for tag in sorted(evals):
        for g in groups:
            refused = _refusal(evals, g, tag)
            if refused is None:
                g.append(tag)
                break
            if g is groups[0]:
                first = refused
        else:
            if groups:
                reasons.append((tag, *first))
            groups.append([tag])
    return sorted(groups, key=lambda g: (-len(g), g)), reasons


def _refusal(evals: dict, group: list[str], tag: str) -> tuple[str, str] | None:
    """The first member of `group` that `tag` is not comparable with, and the check's message; None when it joins."""
    for member in group:
        try:
            check_comparable(evals[member]["summary"], evals[tag]["summary"])
        except ValueError as e:
            return member, str(e)
    return None


def describe_group(summaries: Mapping[str, dict]) -> str:
    """Describe in one line what a group's scores were measured with, under which rule, and on how many clips.

    Args:
        summaries: The group's score JSONs, by tag.
    """
    # The members are comparable with one another, so the first one's ruler
    # and clips stand for the group.
    first = next(iter(summaries.values()))
    r, rule = ruler(first), check_one_rule(summaries)
    # A pilot JSON records no rule, so none is printed.
    return (f"sha={str(r.eval_code_sha)[:8]} pilot={r.pilot} class_set={r.class_set} "
            f"views={list(r.views)} dataset={r.dataset} n_clips={len(clips_of(first))}"
            + (f" propagation={rule}" if rule is not None else ""))


def report(track_root: str, evals: dict, title: str, scored_dirs: set[str]) -> tuple[list[str], dict]:
    """Count one root and print its table.

    Args:
        track_root: The label root.
        evals: This root's scores, as `read_evals` gives them.
        title: The heading.
        scored_dirs: The label directories the scores of every root name,
            so that labels scored from another root read as scored.

    Returns:
        The problems found, and `{tag: {clip: label count}}`. Scores are
        matched to labels in `inventory`, across roots.
    """
    # The conditions of this root, from disk.
    conds = read_conditions(track_root)
    problems = []
    if not conds:
        print(f"\n### {title}: no labels ({track_root})")
        return problems, {}

    # The full population: the clip set most tags cover, the larger on a tie.
    sets = collections.Counter(tuple(sorted(clips_of_condition(c))) for c in conds.values())
    full = set(max(sets, key=lambda k: (sets[k], len(k), k)))

    # The table, one line per tag, and the problems of each tag.
    print(f"\n### {title} ({track_root})")
    print(f"{'tag':34s}{'clip':>5s}{'label':>7s}  {'input':<12s}{'pps':>4s} "
          f"{'depth':<6s}{'seed':<10s}{'base':<12s}{'scored':<7s}date")
    print("-" * 128)
    for tag in sorted(conds):
        c = conds[tag]
        clips = clips_of_condition(c)
        nclip, nlab = len(clips), sum(c["labels"].values())
        ts = c["mtime"] or [0]
        day = lambda t: datetime.datetime.fromtimestamp(t).strftime("%m-%d")
        dt = day(min(ts)) if day(min(ts)) == day(max(ts)) else f"{day(min(ts))}..{day(max(ts))}"
        scored = "scored" if tag in scored_dirs else "no"
        notes = c.get("notes") or {}
        if len(c["keys"]) > 1:
            problems.append(f"{title}/{tag}: {len(c['keys'])} conditions are mixed under one tag")
            print(f"{tag:34s}{nclip:5d}{nlab:7d}  !! {len(c['keys'])} conditions mixed !!   {scored:<7s}{dt}")
            for k, cl in sorted(c["keys"].items()):
                print(f"      [{len(cl):3d} clips] {k}")
            for note, cl in sorted(notes.items()):
                print(f"      [{len(cl):3d} clips] !! {note}")
            continue
        if not c["keys"]:
            # No readable provenance at all: said so, and on to the next tag,
            # so that the scores are still checked and the exit code set.
            for note, cl in sorted(notes.items()):
                problems.append(f"{title}/{tag}: {note} ({len(cl)} clips); "
                                "the condition cannot be verified, so it enters no comparison")
            print(f"{tag:34s}{nclip:5d}{nlab:7d}  !! no readable provenance !!      {scored:<7s}{dt}")
            continue
        if notes:
            n_note = sum(len(v) for v in notes.values())
            problems.append(f"{title}/{tag}: {n_note} clips have no provenance (mixed with clips that have)")
        k = json.loads(next(iter(c["keys"])))
        print(f"{tag:34s}{nclip:5d}{nlab:7d}  {str(k['sam_input']):<12s}"
              f"{str(k['points_per_side']):>4s} {str(k['depth_source']):<6s}"
              f"{str(k['seed_source']):<10s}{str(k['track_base']):<12s}{scored:<7s}{dt}")
        if clips != full:
            missing, extra = sorted(full - clips), sorted(clips - full)
            problems.append(f"{title}/{tag}: {nclip} clips where the others have {len(full)}"
                            f"{' (missing ' + str(missing) + ')' if missing else ''}"
                            f"{' (extra ' + str(extra) + ')' if extra else ''}; "
                            "a run that died half-way is not a condition")
        if nlab == 0:
            problems.append(f"{title}/{tag}: provenance but no label; a run that was started and stopped")

    # The groups of scores comparable with one another, and the problem when there is more than one.
    if evals:
        groups, reasons = comparable_groups(evals)
        print(f"\n  comparable groups: {len(groups)}")
        for g in groups:
            print(f"    {describe_group({t: evals[t]['summary'] for t in g})} → {len(g)} conditions: {g}")
        for tag, member, why in reasons:
            print(f"    !! {tag} vs {member}: " + why.replace("\n", "\n       "))
        if len(groups) > 1:
            problems.append(f"{title}: the scores fall into {len(groups)} groups that cannot be compared "
                            "with one another; compare conditions within a group only")
    return problems, {t: c["labels"] for t, c in conds.items()}


def matrix(track_root: str, eval_root: str, title: str = "", scored_dirs: set[str] | None = None) -> None:
    """The input by points-per-side table of the per-frame conditions; a hole is a condition not yet run.

    Reads one root only. With an empty `eval_root` the stage is not scored
    here, so no cell is marked scored, and that is not an omission.

    Args:
        track_root: The label root.
        eval_root: Its score root, read when `scored_dirs` is not given.
        title: The heading.
        scored_dirs: The label directories the scores of every root name.
    """
    # The cells, from the conditions seeded per frame.
    conds = read_conditions(track_root)
    if scored_dirs is None:
        scored_dirs = {e["dir"] for e in read_evals(eval_root).values()}
    cell = collections.defaultdict(list)
    for tag, c in conds.items():
        if len(c["keys"]) != 1:
            continue
        k = json.loads(next(iter(c["keys"])))
        if k["seed_source"] != "per_frame":
            continue
        nlab = sum(c["labels"].values())
        state = SCORED if (tag in scored_dirs and nlab) else (LABELS_ONLY if nlab else PROVENANCE_ONLY)
        cell[(k["sam_input"], k["points_per_side"], k["depth_source"])].append(
            (state, tag, sum(len(v) for v in c["keys"].values()), nlab))
    # The axes: a provenance field can be None, so the inputs and the sources
    # sort by their spelling, and a None points-per-side is the last column.
    inputs = sorted({a for a, _, _ in cell}, key=str)
    ppss = sorted({b for _, b, _ in cell}, key=lambda p: (p is None, p or 0))
    srcs = sorted({c for _, _, c in cell}, key=str)

    # The table, one per depth source.
    print(f"\n### input by pps ({title or track_root}"
          f"{'' if eval_root else ', not scored in this directory'}"
          f"; per-frame conditions; {SCORED} scored, {LABELS_ONLY} labels only, "
          f"{PROVENANCE_ONLY} provenance only, {NOT_RUN} not run)")
    for src in srcs:
        print(f"\n  depth = {src}")
        print(f"    {'input':<14s}" + "".join(f"{'pps' + str(p):>10s}" for p in ppss))
        for inp in inputs:
            row = [str(inp).ljust(14)]
            for p in ppss:
                # Every condition in the cell, one letter each, in the legend's order.
                states = {s for s, *_ in cell.get((inp, p, src), ())}
                letters = "/".join(s for s in (SCORED, LABELS_ONLY, PROVENANCE_ONLY) if s in states)
                row.append((letters or NOT_RUN).rjust(10))
            print("    " + "".join(row))
    print(f"\n  A '{NOT_RUN}' is a condition to run next; match its name against the table above.")


def inventory(roots: Sequence[tuple[str, str, str]], require_scored: bool = False,
              draw_matrix: bool = False) -> list[str]:
    """Count every root, match scores to labels across them, and return the problems.

    Args:
        roots: `(label root, score root, title)` triples, as `parse_root` gives them.
        require_scored: Count a label directory nobody scored as a problem,
            as it is once every condition is meant to be scored.
        draw_matrix: Print the input by pps table of the first root.

    Raises:
        ValueError: A label root or a score root does not exist, or a
            score directory holds something that is not a score JSON.
    """
    # The roots, refused before anything is counted when one is missing.
    for tr, ev, _ in roots:
        check_root_exists(tr, ev)

    # The scores of every root, read once: a score can name labels under another root.
    evals = {ev: read_evals(ev) for _, ev, _ in roots if ev}
    all_evals = {k: e["dir"] for evs in evals.values() for k, e in evs.items()}
    scored_dirs = set(all_evals.values())

    # Each root's table and problems, and its label counts per clip and tag.
    problems, seen_tags, cover = [], set(), collections.defaultdict(dict)
    for tr, ev, title in roots:
        probs, labels = report(tr, evals.get(ev, {}), title, scored_dirs)
        problems += probs
        for t, per_clip in labels.items():
            seen_tags.add(t)
            # A tag with no label at all is already reported by `report`;
            # counting its zeros here would report every clip of it again.
            if not any(per_clip.values()):
                continue
            for clip, n in per_clip.items():
                cover[clip][f"{title}/{t}"] = n

    # Scores whose labels are gone, matched by directory name across roots.
    orphan = sorted(t for t, d in all_evals.items() if d not in seen_tags)
    if orphan:
        problems.append(f"{len(orphan)} scores have no labels: {orphan}")
    # Labels nobody scored.
    unscored = sorted(t for t in seen_tags if t not in scored_dirs and t != VIZ_DIR)
    if unscored:
        print(f"\n### {len(unscored)} label directories nobody scored: {unscored}")
        if require_scored:
            problems.append(f"{len(unscored)} conditions are not scored: {unscored}")

    # One clip, the same number of labels under every condition. A zero counts:
    # provenance with no label under one condition, where the others have
    # labels, is a run that died on that clip, and nothing else reports it.
    ragged = []
    for clip, per_tag in sorted(cover.items()):
        ns = set(per_tag.values())
        if len(ns) > 1:
            worst = sorted(per_tag.items(), key=lambda x: x[1])
            ragged.append(f"{clip}: {min(ns)}..{max(ns)} labels (fewest {worst[0][0]} / most {worst[-1][0]})")
    if ragged:
        problems.append(f"{len(ragged)} clips are covered differently by the conditions: "
                        + "; ".join(ragged[:5]) + (" ..." if len(ragged) > 5 else ""))
    print(f"\n### clip coverage: {len(cover)} clips, {len(ragged)} covered differently across conditions")

    # The matrix of the first root, when asked.
    if draw_matrix:
        matrix(*roots[0], scored_dirs=scored_dirs)
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # No default root: a default would count another tree, silently, the day a directory is renamed.
    ap.add_argument("--root", action="append", default=[], metavar="SPEC", required=True,
                    help="A directory to count, as `<label root>[:<score root>[:<title>]]`. "
                         "May be given several times. A root that does not exist stops the run.")
    ap.add_argument("--require-scored", action="store_true",
                    help="Count labels nobody scored as a problem (once every condition is to be scored).")
    ap.add_argument("--matrix", action="store_true", help="Print the input by pps table of the first root too.")
    args = ap.parse_args()
    try:
        roots = tuple(parse_root(r) for r in args.root)
        problems = inventory(roots, require_scored=args.require_scored, draw_matrix=args.matrix)
    except ValueError as e:
        raise SystemExit(str(e))

    print("\n" + "=" * 70)
    if problems:
        print(f"{len(problems)} problems:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("no problem: no mixed condition, no missing clip, every score comparable with the others")


if __name__ == "__main__":
    main()
