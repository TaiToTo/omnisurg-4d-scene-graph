"""Check that every clip of a condition was made with the seed source and the tracker input the condition states.

The tool reads each clip's `seed_info.json`, the record that the tracking
stage and the per-frame stage write beside its labels, in this repository
and in the workbench alike. It prints each combination of seed source,
segmenter input and tracker input that the records hold, with its clips; a
record without one of these keys holds `(no such key)` in its place. It
exits non-zero on a clip without a record, a record that cannot be read, a
seed source or a tracker input other than the one stated, and records that
hold more than one combination.

Usage:
    python -m evalkit.tools.check_provenance --tracks-root /path/to/tracks --tag op_edge_center \\
        --clips atlas120k_meta/clips.txt --seed-source sam [--track-base rgb]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from evalkit.evaluate import SEED_INFO, read_population

# The values `seed_source` takes: the tracker's own segmentation, seeds made outside it, the GT masks of the
# workbench's `t12_gtseed`, and the per-frame stage.
SEED_SOURCES = ("sam", "external", "gt", "per_frame")
# The settings a record is read for. Their combination is what one condition holds on every clip.
KEYS = ("seed_source", "sam_input", "track_base")
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
        condition: the label directory's name, `track_<track_base>_<tag>`.
        clips: the population, in its order.
        settings: each combination of `KEYS` the records hold, with the clips that hold it.
        missing: the clips whose label directory has no `seed_info.json`.
        unreadable: each clip whose record is not a JSON object, with why.
        mismatched: (clip, key, recorded, stated) for each setting other than the one stated.
    """

    condition: str
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
            out.append(f"the clips hold {len(self.settings)} combinations of {', '.join(KEYS)}, "
                       "so the population mixes conditions")
        if self.mismatched:
            out.append(f"{len(self.mismatched)} setting(s) other than stated: "
                       + "; ".join(f"{c}: {k} {got!r}, not {want!r}" for c, k, got, want in self.mismatched[:SHOWN]))
        return out


def read_provenance(tracks_root: str | Path, condition: str, clips: Sequence[str], seed_source: str,
                    track_base: str) -> Provenance:
    """Read every clip's record of the condition, and compare it with the seed source and tracker input stated.

    Args:
        tracks_root: the directory holding one directory per clip.
        condition: the label directory's name under each clip.
        clips: the population.
        seed_source: the `seed_source` every record must hold, one of `SEED_SOURCES`.
        track_base: the `track_base` every record must hold.

    Raises:
        ValueError: `seed_source` is not one of `SEED_SOURCES`, or `tracks_root` is not a directory. A root
            that does not exist would read as every clip without a record.
    """
    if seed_source not in SEED_SOURCES:
        raise ValueError(f"{seed_source!r} is no seed source; one of {SEED_SOURCES}")
    if not Path(tracks_root).is_dir():
        raise ValueError(f"no such directory: {tracks_root}")
    out = Provenance(condition, list(clips))
    for clip in clips:
        path = Path(tracks_root) / clip / condition / SEED_INFO
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
        out.settings.setdefault(tuple(values.values()), []).append(clip)
        for key, want in (("seed_source", seed_source), ("track_base", track_base)):
            if values[key] != want:
                out.mismatched.append((clip, key, values[key], want))
    return out


def report(prov: Provenance, seed_source: str, track_base: str) -> list[str]:
    """Return the report's lines: the population, each combination of settings with its clips, and the problems."""
    lines = [f"{prov.condition}: {len(prov.clips)} clips, stated seed_source {seed_source}, track_base {track_base}"]
    for values, clips in sorted(prov.settings.items(), key=lambda kv: (-len(kv[1]), repr(kv[0]))):
        lines.append("  " + ", ".join(f"{k} {v}" for k, v in zip(KEYS, values)) + f": {len(clips)} clip(s)")
    lines += [f"  !! {p}" for p in prov.problems()]
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tracks-root", required=True, help="The directory holding one directory per clip.")
    ap.add_argument("--tag", required=True, help="The condition's name.")
    ap.add_argument("--track-base", default="rgb",
                    help="The tracker's input the condition states; its labels are in track_<track-base>_<tag>.")
    ap.add_argument("--clips", required=True, help="The population: one clip name per line.")
    ap.add_argument("--seed-source", required=True, choices=SEED_SOURCES,
                    help="Where the condition's seeds come from, as every record must say.")
    args = ap.parse_args()
    try:
        prov = read_provenance(args.tracks_root, f"track_{args.track_base}_{args.tag}",
                               read_population(args.clips), args.seed_source, args.track_base)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"{type(e).__name__}: {e}")
    print("\n".join(report(prov, args.seed_source, args.track_base)))
    if prov.problems():
        sys.exit(1)


if __name__ == "__main__":
    main()
