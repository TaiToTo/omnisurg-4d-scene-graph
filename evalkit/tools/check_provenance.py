"""Check that every clip of a condition was made with the settings the condition states.

The tool reads each clip's `seed_info.json`, the record that the tracking stage and the per-frame stage write
beside its labels, in this repository and in the workbench alike. It prints each combination of the settings that
`condition_inventory.condition_key` tells conditions apart by, with its clips, and shows the seed source, the
segmenter input and the tracker input of each; a record without one of these three holds `(no such key)`.
It exits non-zero on a clip without a record, a record that cannot be read, a seed source, a tracker input, a
segmenter input or seed labels other than the ones stated, and records that hold more than one combination. The
edge ring of the tracker's input is in no record, so it is not compared.

Usage:
    python -m evalkit.tools.check_provenance --tracks-root /path/to/tracks --tag op_edge_center \\
        --clips atlas120k_meta/clips.txt --seed-source sam --sam-input normal_edge [--track-base rgb]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from evalkit.evaluate import SEED_INFO, read_population
from evalkit.tools.condition_inventory import condition_key

# The values `seed_source` takes: the segmenter of the tracking stage; seeds made outside the stage
# (`--seed-labels`); the GT masks of the workbench's `t12_*` conditions and of its controls `t12_paste`,
# `none_ceil` and `none_floor`; and the per-frame stage.
SEED_SOURCES = ("sam", "external", "gt", "per_frame")
# The settings the report shows for each combination.
KEYS = ("seed_source", "sam_input", "track_base")
# The seed sources whose seed the segmenter cut from the `sam_input` image; under the others it is recorded only.
CUT_FROM_SAM_INPUT = ("sam", "per_frame")
# How many clips a line of the report names, after their count.
SHOWN = 5


class Absent:
    """Stand in for a key a record lacks, so that it differs from every value, None included."""

    def __repr__(self) -> str:
        return "(no such key)"

    __str__ = __repr__


ABSENT = Absent()


@dataclass
class Provenance:
    """Hold what the records of one condition's clips say.

    Attributes:
        label_dir: the label directory's name, `track_<track_base>_<tag>`.
        clips: the population, in its order.
        settings: each combination the records hold, the values of `KEYS` followed by
            `condition_key`, with the clips that hold it.
        missing: the clips whose label directory has no `seed_info.json`.
        unreadable: each clip whose record, or its `seed_input`, is not a JSON object, or whose setting of `KEYS`
            is neither a string nor null, with why.
        mismatched: (clip, key, recorded, stated) for each setting other than the one stated.
    """

    label_dir: str
    clips: list[str]
    settings: dict[tuple, list[str]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    unreadable: dict[str, str] = field(default_factory=dict)
    mismatched: list[tuple[str, str, object, object]] = field(default_factory=list)

    def problems(self) -> list[str]:
        """Return one line per problem, empty when every clip holds the settings stated."""
        out = []
        if self.missing:
            out.append(f"{len(self.missing)} clip(s) without {SEED_INFO}: {self.missing[:SHOWN]}")
        if self.unreadable:
            out.append(f"{len(self.unreadable)} clip(s) whose {SEED_INFO} cannot be read: "
                       + "; ".join(f"{c} ({why})" for c, why in list(self.unreadable.items())[:SHOWN]))
        if len(self.settings) > 1:
            out.append(f"the clips hold {len(self.settings)} combinations of the settings that tell one condition "
                       "from another, so the population mixes conditions")
        if self.mismatched:
            out.append(f"{len(self.mismatched)} setting(s) other than stated: "
                       + "; ".join(f"{c}: {k} {got!r}, not {want!r}" for c, k, got, want in self.mismatched[:SHOWN]))
        return out


def read_provenance(tracks_root: str | Path, label_dir: str, clips: Sequence[str], seed_source: str,
                    track_base: str, sam_input: str | None = None, seed_labels: str | None = None) -> Provenance:
    """Read every clip's record of the condition, and compare it with the settings stated.

    Args:
        tracks_root: the directory holding one directory per clip.
        label_dir: the label directory's name under each clip.
        clips: the population.
        seed_source: the `seed_source` every record must hold, one of `SEED_SOURCES`.
        track_base: the `track_base` every record must hold.
        sam_input: the `sam_input` every record must hold, stated under the seed sources of `CUT_FROM_SAM_INPUT`
            and only under them.
        seed_labels: the `seed_labels` every record must hold, which names seeds made outside the stage.

    Raises:
        ValueError: `seed_source` is not one of `SEED_SOURCES`; `seed_labels` is not given with the seed source
            `external`, whose seeds it alone tells apart, or is given with another; `sam_input` is not given with
            a seed source whose seed the segmenter cut from it, or is given with another; or `tracks_root` is not
            a directory. A root that does not exist would read as every clip without a record.
    """
    if seed_source not in SEED_SOURCES:
        raise ValueError(f"{seed_source!r} is no seed source; one of {SEED_SOURCES}")
    if (seed_source == "external") != (seed_labels is not None):
        raise ValueError("seed labels are stated with the seed source 'external', and only with it: they alone "
                         "tell its seeds apart")
    if sam_input is None and seed_source in CUT_FROM_SAM_INPUT:
        raise ValueError(f"the seed source {seed_source!r} needs sam_input stated: the segmenter cut the seed "
                         "from it")
    if sam_input is not None and seed_source not in CUT_FROM_SAM_INPUT:
        raise ValueError(f"under the seed source {seed_source!r} no segmenter cut the seed from sam_input, which is "
                         "recorded only, so it is not compared")
    if not Path(tracks_root).is_dir():
        raise ValueError(f"no such directory: {tracks_root}")
    stated = dict(seed_source=seed_source, track_base=track_base, sam_input=sam_input, seed_labels=seed_labels)
    out = Provenance(label_dir, list(clips))
    for clip in clips:
        path = Path(tracks_root) / clip / label_dir / SEED_INFO
        if not path.is_file():
            out.missing.append(clip)
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            out.unreadable[clip] = f"not JSON: {e}"
            continue
        if not isinstance(record, dict):
            out.unreadable[clip] = f"a JSON {type(record).__name__}, not an object"
            continue
        values = {k: record.get(k, ABSENT) for k in KEYS}
        odd = [f"{k} {v!r}" for k, v in values.items() if not (v is ABSENT or v is None or isinstance(v, str))]
        if odd:
            out.unreadable[clip] = f"holds {', '.join(odd)}, not a string"
            continue
        try:
            condition = condition_key(record)
        except ValueError as e:
            out.unreadable[clip] = str(e)
            continue
        out.settings.setdefault((*values.values(), condition), []).append(clip)
        for key, want in stated.items():
            got = record.get(key, ABSENT)
            if want is not None and got != want:
                out.mismatched.append((clip, key, got, want))
    return out


def report(prov: Provenance, seed_source: str, track_base: str) -> list[str]:
    """Return the report's lines: the population, each combination of settings with its clips, and the problems.

    When the clips hold more than one combination, each line also names the settings in which they differ.
    """
    lines = [f"{prov.label_dir}: {len(prov.clips)} clips, stated seed_source {seed_source}, track_base {track_base}"]
    fields = {combo: json.loads(combo[-1]) for combo in prov.settings}
    differ = sorted({k for f in fields.values() for k in f
                     if len({json.dumps(g.get(k), sort_keys=True) for g in fields.values()}) > 1} - set(KEYS))
    for combo, clips in sorted(prov.settings.items(), key=lambda kv: (-len(kv[1]), repr(kv[0]))):
        shown = ", ".join(f"{k} {v}" for k, v in zip(KEYS, combo))
        shown += "".join(f", {k} {fields[combo].get(k)!r}" for k in differ)
        lines.append(f"  {shown}: {len(clips)} clip(s)")
    lines += [f"  !! {p}" for p in prov.problems()]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks-root", required=True, help="The directory holding one directory per clip.")
    ap.add_argument("--tag", required=True, help="The condition's tag; its labels are in track_<track-base>_<tag>.")
    ap.add_argument("--track-base", default="rgb",
                    help="The tracker's input the condition states; its labels are in track_<track-base>_<tag>.")
    ap.add_argument("--clips", required=True, help="The population: one clip name per line.")
    ap.add_argument("--seed-source", required=True, choices=SEED_SOURCES,
                    help="Where the condition's seeds come from, as every record must say.")
    ap.add_argument("--sam-input",
                    help="The segmenter's input the condition states; needed with --seed-source sam and per_frame, "
                         "refused with the others.")
    ap.add_argument("--seed-labels",
                    help="The seed labels the condition states, as the records hold them: the path the tracking "
                         "stage was given, compared as written; needed with --seed-source external.")
    args = ap.parse_args()
    try:
        prov = read_provenance(args.tracks_root, f"track_{args.track_base}_{args.tag}", read_population(args.clips),
                               args.seed_source, args.track_base, args.sam_input, args.seed_labels)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"{type(e).__name__}: {e}")
    print("\n".join(report(prov, args.seed_source, args.track_base)))
    if prov.problems():
        sys.exit(1)


if __name__ == "__main__":
    main()
