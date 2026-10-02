"""What ran, under which condition, on how many clips: counted from the files on disk, never judged.

A record of runs kept by hand cannot be checked from outside: a run left
out or written down wrong looks like any other line. This tool goes the
other way. It reads only the provenance file the pipeline writes beside
each clip's labels (`seed_info.json`, from `run_per_frame_seg` and
`track_sam3`) and the score JSONs, and tables what is on disk.

Nothing is guessed. What the provenance does not say is written as None,
and the shell scripts that launched a run are never read: a script is
edited later, the provenance is written with the labels.

Four things are checked, and any of them failing sets the exit code:

1. A condition is not mixed across clips. Labels under one tag made with
   different arguments would measure a difference in arguments as a
   difference in inputs.
2. Every condition covers the same clips, with the same number of labels
   per clip. A run that died half-way is not counted as a condition.
3. Scores and labels correspond: a score whose labels are gone, and, when
   asked, labels nobody scored.
4. The ruler is one. The scores in a directory were made by one
   evaluator, in one mode, on one class set, one set of views, one
   dataset and one population; a directory where that is not so must not
   be compared within.

The directories are always arguments. A default would make the tool count
another tree, silently, the day a directory is renamed.

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
from collections.abc import Sequence

from evalkit.tools.scores import load_scores, ruler

# The provenance file the pipeline writes beside a clip's labels.
PROV_NAME = "seed_info.json"

# The provenance fields that make a condition what it is. `clip` and `tag`
# are where the labels are, not what made them, so they are not keys.
# `frames` differs per clip by nature (it is that clip's frame numbers), so
# only its kind is kept, or every clip would be a condition of its own.
PROV_KEYS = ("sam_input", "depth_source", "seed_source", "track_base",
             "seed_min_area", "seed_topk", "point_grids")
SEED_KEYS = ("points_per_side", "seed_edge_gain", "seed_smooth",
             "edge_ring_masked", "seed_sam_kwargs", "produced_by")

# A label directory that holds pictures, never labels of a condition.
VIZ_DIR = "viz"


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


def check_root_exists(track_root: str) -> None:
    """Refuse a label root that is not a directory.

    A root that does not exist reads as zero labels, no check fires, and
    the tool ends with "no problem": the gate passes silently on a typo.

    Raises:
        ValueError: `track_root` is not a directory.
    """
    if not os.path.isdir(track_root):
        raise ValueError(f"the --root label directory does not exist (a typo?): {track_root}")


def frames_kind(v) -> str:
    """The kind of a provenance `frames` value: `all`, `gt`, `list` or `None`."""
    if isinstance(v, list):
        return "list"
    return str(v)


def read_conditions(track_root: str) -> dict:
    """The provenance under `track_root`, by tag.

    Returns:
        `{tag: {"keys": {condition key as JSON: [clip, ...]}, "notes": {reason:
        [clip, ...]}, "labels": {clip: label count}, "mtime": [...]}}`. `keys`
        holds only provenance that could be read; a clip whose provenance is
        missing or broken goes under `notes`, so that a reader of `keys` can
        always parse them.
    """
    out = collections.defaultdict(lambda: {"keys": collections.defaultdict(list),
                                           "notes": collections.defaultdict(list),
                                           "labels": {}, "mtime": []})
    for p in glob.glob(os.path.join(track_root, "*", "*", PROV_NAME)):
        lab_dir = os.path.dirname(p)
        tag = os.path.basename(lab_dir)
        clip = os.path.basename(os.path.dirname(lab_dir))
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError) as e:
            # A broken provenance is reported, not hidden.
            out[tag]["notes"][f"unreadable provenance: {e}"].append(clip)
            continue
        si = d.get("seed_input") or {}
        key = json.dumps({**{k: d.get(k) for k in PROV_KEYS},
                          **{k: si.get(k) for k in SEED_KEYS},
                          "frames": frames_kind(d.get("frames"))},
                         sort_keys=True, ensure_ascii=False)
        out[tag]["keys"][key].append(clip)
        # Per clip: a clip's label count is its length, so it varies within
        # a tag by nature. What must agree is the count of one clip across
        # tags, which `inventory` checks.
        out[tag]["labels"][clip] = len(glob.glob(os.path.join(lab_dir, "label_*.npy")))
        out[tag]["mtime"].append(os.path.getmtime(p))
    # Labels without provenance, per clip: a tag can hold clips with and
    # without it, and skipping the tag would hide the latter from check 1.
    for d in glob.glob(os.path.join(track_root, "*", "*", "")):
        tag = os.path.basename(d.rstrip("/"))
        clip = os.path.basename(os.path.dirname(d.rstrip("/")))
        if tag == VIZ_DIR or not glob.glob(os.path.join(d, "label_*.npy")):
            continue
        if clip not in out[tag]["labels"]:
            out[tag]["notes"]["no provenance (what made these labels is unknown)"].append(clip)
    return out


def read_evals(eval_root: str) -> dict:
    """The ruler and the population of every score JSON under `eval_root`.

    Returns:
        `{tag: {"dir": track_dir_name, "sha": sha[:8], "tag": eval_code_tag,
        "ver": eval_version, "ds": dataset, "pilot": ..., "class_set": ...,
        "views": ..., "n_clips": ..., "miss": ..., "fail": ..., "ignore":
        tissue_ignore}}`; empty when `eval_root` is empty.

    Raises:
        ValueError: A JSON there is not a score JSON, or records some of
            the evaluator's fields but not all. A directory of scores
            must hold nothing a reader cannot place.
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
        r = ruler(d)
        out[os.path.basename(p)[:-5]] = dict(
            dir=d.get("track_dir_name"),
            sha=str(r.eval_code_sha)[:8], tag=d.get("eval_code_tag"), ver=d.get("eval_version"),
            ds=r.dataset, pilot=r.pilot, class_set=r.class_set, views=r.views,
            n_clips=d.get("n_clips"), miss=d.get("n_missing"), fail=d.get("n_failed"),
            ignore=tuple(d.get("tissue_ignore") or ()))
    return out


def evals_by_dir(evals: dict) -> dict:
    """Score tags by the label directory they read, from `track_dir_name`, never from the tag's name.

    The two names differ (`t12_normal` reads `track_normal_t12_gtseed`), and
    one directory can be scored under several tags, so the value is a list.
    A JSON without `track_dir_name` is left out.
    """
    out = collections.defaultdict(list)
    for t, e in evals.items():
        if e.get("dir"):
            out[e["dir"]].append(t)
    return out


def n_clips_of(c: dict) -> int:
    """The clips a tag covers: those with readable provenance and those without, counted in one place."""
    return (sum(len(v) for v in c["keys"].values())
            + sum(len(v) for v in (c.get("notes") or {}).values()))


def _stick_of(e: dict) -> tuple:
    return (e["sha"], e["tag"], e["ver"], e["ds"], e["pilot"], e["class_set"], e["views"],
            e["n_clips"], e["miss"], e["fail"], e["ignore"])


def report(track_root: str, eval_root: str, title: str) -> tuple[list[str], dict]:
    """Count one root and print its table.

    Returns:
        The problems found, and `{tag: {clip: label count}}`. Scores are
        not matched to labels here: a score can read a directory under
        another root, so `inventory` matches them across roots.
    """
    conds = read_conditions(track_root)
    evals = read_evals(eval_root)
    by_dir = evals_by_dir(evals)
    problems = []
    if not conds:
        print(f"\n### {title}: no labels ({track_root})")
        return problems, {}
    n_clips = collections.Counter(n_clips_of(c) for c in conds.values())
    # On a tie the larger count is the full one; `most_common` would break
    # the tie by insertion order, which is `glob`'s, and change between runs.
    full = max(n_clips, key=lambda k: (n_clips[k], k)) if n_clips else 0
    print(f"\n### {title} ({track_root})")
    print(f"{'tag':34s}{'clip':>5s}{'label':>7s}  {'input':<12s}{'pps':>4s} "
          f"{'depth':<6s}{'seed':<10s}{'base':<12s}{'scored':<7s}date")
    print("-" * 128)
    for tag in sorted(conds):
        c = conds[tag]
        nclip = n_clips_of(c)
        nlab = sum(c["labels"].values())
        ts = c["mtime"] or [0]
        day = lambda t: datetime.datetime.fromtimestamp(t).strftime("%m-%d")
        dt = day(min(ts)) if day(min(ts)) == day(max(ts)) else f"{day(min(ts))}..{day(max(ts))}"
        mark = "scored" if by_dir.get(tag) else "—"
        notes = c.get("notes") or {}
        if len(c["keys"]) > 1:
            problems.append(f"{title}/{tag}: {len(c['keys'])} conditions are mixed under one tag")
            print(f"{tag:34s}{nclip:5d}{nlab:7d}  !! {len(c['keys'])} conditions mixed !!   {mark:<7s}{dt}")
            for k, cl in sorted(c["keys"].items()):
                print(f"      [{len(cl):3d} clips] {k}")
            for note, cl in sorted(notes.items()):
                print(f"      [{len(cl):3d} clips] !! {note}")
            continue
        if not c["keys"]:
            # No readable provenance at all: said so, and on to the next
            # tag, so that check 4 is still reached and the exit code set.
            for note, cl in sorted(notes.items()):
                problems.append(f"{title}/{tag}: {note} ({len(cl)} clips); "
                                "the condition cannot be verified, so it enters no comparison")
            print(f"{tag:34s}{nclip:5d}{nlab:7d}  !! no readable provenance !!      {mark:<7s}{dt}")
            continue
        if notes:
            n_note = sum(len(v) for v in notes.values())
            problems.append(f"{title}/{tag}: {n_note} clips have no provenance (mixed with clips that have)")
        k = json.loads(next(iter(c["keys"])))
        print(f"{tag:34s}{nclip:5d}{nlab:7d}  {str(k['sam_input']):<12s}"
              f"{str(k['points_per_side']):>4s} {str(k['depth_source']):<6s}"
              f"{str(k['seed_source']):<10s}{str(k['track_base']):<12s}{mark:<7s}{dt}")
        if nclip != full:
            problems.append(f"{title}/{tag}: {nclip} clips where the others have {full}; "
                            "a run that died half-way is not a condition")
        if nlab == 0:
            problems.append(f"{title}/{tag}: provenance but no label; a run that was started and stopped")

    if evals:
        sticks = collections.defaultdict(list)
        for t, e in evals.items():
            sticks[_stick_of(e)].append(t)
        print(f"\n  rulers: {len(sticks)}")
        for k, v in sorted(sticks.items(), key=lambda x: -len(x[1])):
            print(f"    sha={k[0]} tag={k[1]} ver={k[2]} ds={k[3]} pilot={k[4]} class_set={k[5]} "
                  f"views={list(k[6])} n_clips={k[7]} miss={k[8]} fail={k[9]} ignore={k[10]} → {len(v)} conditions")
        if len(sticks) > 1:
            problems.append(f"{title}: the scores were made with {len(sticks)} rulers; "
                            "conditions must not be compared within this directory")
    return problems, {t: c["labels"] for t, c in conds.items()}


def matrix(track_root: str, eval_root: str, title: str = "") -> None:
    """The input × points-per-side table of the per-frame arms; a hole is a condition not yet run.

    Reads one root only. With an empty `eval_root` the stage is not scored
    here, so no cell is marked scored, and that is not an omission.
    """
    conds = read_conditions(track_root)
    by_dir = evals_by_dir(read_evals(eval_root))
    cell = collections.defaultdict(list)
    for tag, c in conds.items():
        if len(c["keys"]) != 1:
            continue
        k = json.loads(next(iter(c["keys"])))
        if k["seed_source"] != "per_frame":
            continue
        nlab = sum(c["labels"].values())
        state = "○" if (by_dir.get(tag) and nlab) else ("△" if nlab else "×")
        cell[(k["sam_input"], k["points_per_side"], k["depth_source"])].append(
            (state, tag, sum(len(v) for v in c["keys"].values()), nlab))
    inputs = sorted({a for a, _, _ in cell})
    ppss = sorted({b for _, b, _ in cell if b is not None})
    srcs = sorted({c for _, _, c in cell})
    print(f"\n### input × pps ({title or track_root}"
          f"{'' if eval_root else ', not scored in this directory'}"
          "; per-frame arms; ○ scored, △ labels only, × provenance only)")
    for src in srcs:
        print(f"\n  depth = {src}")
        print(f"    {'input':<14s}" + "".join(f"{'pps' + str(p):>10s}" for p in ppss))
        for inp in inputs:
            row = [inp.ljust(14)]
            for p in ppss:
                v = cell.get((inp, p, src))
                row.append((v[0][0] if v else "—").rjust(10))
            print("    " + "".join(row))
    print("\n  A '—' is a condition to run next; match its name against the table above.")


def inventory(roots: Sequence[tuple[str, str, str]], require_scored: bool = False,
              draw_matrix: bool = False) -> list[str]:
    """Count every root, match scores to labels across them, and return the problems.

    Args:
        roots: `(label root, score root, title)` triples, as `parse_root` gives them.
        require_scored: Count a label directory nobody scored as a problem,
            as it is once every condition is meant to be scored.
        draw_matrix: Print the input × pps table of the first root.

    Raises:
        ValueError: A label root does not exist, or a score directory
            holds something that is not a score JSON.
    """
    for tr, _, _ in roots:
        check_root_exists(tr)
    problems, seen_tags, cover = [], set(), collections.defaultdict(dict)
    all_evals = {}
    for tr, ev, title in roots:
        probs, labels = report(tr, ev, title)
        problems += probs
        for t, per_clip in labels.items():
            seen_tags.add(t)
            for clip, n in per_clip.items():
                cover[clip][f"{title}/{t}"] = n
        for k, e in read_evals(ev).items():
            all_evals[k] = e.get("dir")

    # Scores whose labels are gone, matched by directory name across roots.
    orphan = sorted(t for t, d in all_evals.items() if d and d not in seen_tags)
    if orphan:
        problems.append(f"{len(orphan)} scores have no labels: {orphan}")
    # Labels nobody scored.
    scored_dirs = {d for d in all_evals.values() if d}
    unscored = sorted(t for t in seen_tags if t not in scored_dirs and t != VIZ_DIR)
    if unscored:
        print(f"\n### {len(unscored)} label directories nobody scored: {unscored}")
        if require_scored:
            problems.append(f"{len(unscored)} conditions are not scored: {unscored}")

    # One clip, the same number of labels under every condition. A clip's
    # length varies between clips; across conditions it must not.
    ragged = []
    for clip, per_tag in sorted(cover.items()):
        ns = {n for n in per_tag.values() if n}
        if len(ns) > 1:
            worst = sorted(per_tag.items(), key=lambda x: x[1])
            ragged.append(f"{clip}: {min(ns)}..{max(ns)} labels (fewest {worst[0][0]} / most {worst[-1][0]})")
    if ragged:
        problems.append(f"{len(ragged)} clips are covered differently by the conditions: "
                        + "; ".join(ragged[:5]) + (" ..." if len(ragged) > 5 else ""))
    print(f"\n### clip coverage: {len(cover)} clips, {len(ragged)} covered differently across conditions")

    if draw_matrix:
        matrix(*roots[0])
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", default=[], metavar="SPEC", required=True,
                    help="A directory to count, as `<label root>[:<score root>[:<title>]]`. "
                         "May be given several times. A root that does not exist stops the run.")
    ap.add_argument("--require-scored", action="store_true",
                    help="Count labels nobody scored as a problem (once every condition is to be scored).")
    ap.add_argument("--matrix", action="store_true", help="Print the input × pps table of the first root too.")
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
    print("no problem: no mixed condition, no missing clip, no split ruler")


if __name__ == "__main__":
    main()
