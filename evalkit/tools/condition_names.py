"""Parse a condition's name into its parts, and read the table that maps each workbench condition to its name.

A name lists, separated by `_`: the input; the depth model, for an input
that reads depth; the points per side of the segmenter's grid; the seed;
the propagation rule; then each setting off its default. A part joins its
words with `-`. The dataset is not in the name. `condition_names.tsv`
holds one row per dataset and workbench name: the name, or why the
condition's records show none.
"""
from __future__ import annotations

import csv
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

# The inputs: the image the segmenter cut the regions from, or, under a seed from GT, the tracker's input.
INPUTS = ("depth", "normal", "normal-edge", "rgb", "rgb-edge")
# The inputs drawn from depth, whose name says the depth model.
READS_DEPTH = ("depth", "normal", "normal-edge", "rgb-edge")
# The inputs that burn the geometric edges in, and so have an edge ring to mask or keep.
EDGE_INPUTS = ("normal-edge", "rgb-edge")
# The depth models: Depth Anything 3, and Pi3X. `seed_info.json` records them as the `depth_source` `da3` and `pi3`.
DEPTH_MODELS = ("da3", "pi3x")
# The seeds: `sam`, the tracking stage's segmenter on the seed frame; `gt`, the GT mask of the seed frame;
# `unmerged`, a per-frame condition's regions of the seed frame; `kmerge<K>`, those regions merged to K by
# `kmerge`; `kmerge-gt`, merged to the clip's median count of GT regions.
SEED = re.compile(r"sam|gt|unmerged|kmerge(?:[1-9][0-9]*|-gt)")
GRID = re.compile(r"pps([1-9][0-9]*)")
# The propagation rules, each with the rule a score records.
RULES = {"centre": "both_ways_from_centre", "forward": "forward_from_first", "perframe": "per_frame"}
# The settings off their defaults, in the order a name lists them: `keepring`, the edge ring along the image border
# and around invalid depth kept; `instrument-text`, instruments added to the seed by SAM 3's text prompts. The
# defaults are the settings every named condition shares, and a name leaves them out: the edge ring masked, no
# instruments added, the tracker's input `rgb` under a seed not from GT, seed regions of 400 pixels or more, and,
# in an input with edges, the edges drawn at gain 1.0, on normals not smoothed in `normal-edge`. The gain and the
# smoothing change no other input, so a record of 0.85 with smoothing, the tracking stage's own default, is no
# setting off the default there.
SETTINGS = ("keepring", "instrument-text")

TABLE = Path(__file__).with_name("condition_names.tsv")
COLUMNS = ("dataset", "workbench", "name", "unresolved", "record")
DATASETS = ("atlas120k", "cholecseg8k")
# What a row was read from: the clips' `seed_info.json`; with the run logs, whose commands give no depth option, so
# the tracker read the depth each clip's manifest records as DA3's; with the manifest of the merged seeds, which
# names the per-frame condition merged and K; or the run logs alone, for runs made before the edge ring was masked
# and before `seed_info.json` was written.
RECORDS = ("seed_info.json", "seed_info.json, run log", "seed_info.json, seed manifest", "run log")


@dataclass(frozen=True)
class Parts:
    """The parts of a name; a part the name leaves out is None, or empty for the settings."""

    input: str
    depth: str | None
    points_per_side: int | None
    seed: str | None
    rule: str
    settings: tuple[str, ...] = ()

    def name(self) -> str:
        """Join the parts into the name."""
        grid = None if self.points_per_side is None else f"pps{self.points_per_side}"
        parts = (self.input, self.depth, grid, self.seed, self.rule, *self.settings)
        return "_".join(p for p in parts if p is not None)


def parse(name: str) -> Parts:
    """Split a name into its parts, refusing one that breaks the rule of names.

    Raises:
        ValueError: A part is unknown or out of place; the depth model is
            missing for an input that reads depth, or given for one that
            does not; the seed is missing under a propagated rule, or given
            under `perframe`; the grid is missing where a segmenter cut the
            regions, or given under a seed from GT; or a setting is unknown,
            repeated, out of order or off its input.
    """
    words = name.split("_")
    if "" in words:
        raise ValueError(f"{name!r}: an empty part")

    # The input, then the depth model if the input reads depth.
    inp = words.pop(0)
    if inp not in INPUTS:
        raise ValueError(f"{name!r}: {inp!r} is no input; one of {INPUTS}")
    depth = words.pop(0) if words and words[0] in DEPTH_MODELS else None
    if depth is None and inp in READS_DEPTH:
        raise ValueError(f"{name!r}: the input {inp!r} reads depth, so the name needs a depth model")
    if depth is not None and inp not in READS_DEPTH:
        raise ValueError(f"{name!r}: the input {inp!r} reads no depth, so the name takes no depth model")

    # The grid, the seed and the propagation rule.
    grid = GRID.fullmatch(words[0]) if words else None
    pps = int(words.pop(0)[3:]) if grid else None
    seed = words.pop(0) if words and SEED.fullmatch(words[0]) else None
    rule = words.pop(0) if words else None
    # A grid or a merge that the patterns refused would otherwise be reported as a rule.
    if rule is not None and rule.startswith(("pps", "kmerge")):
        raise ValueError(f"{name!r}: {rule!r} is no grid and no seed; pps and kmerge take a whole number from 1")
    if rule not in RULES:
        raise ValueError(f"{name!r}: {rule!r} is no propagation rule; one of {tuple(RULES)}")
    if (seed is None) != (rule == "perframe"):
        raise ValueError(f"{name!r}: a seed is named under a propagated rule, and only there")
    if (pps is None) != (seed == "gt"):
        raise ValueError(f"{name!r}: the grid is named wherever a segmenter cut the regions, and only there")

    # The settings off their defaults, each once, in their order.
    unknown = [w for w in words if w not in SETTINGS]
    if unknown:
        raise ValueError(f"{name!r}: {unknown} are no settings; the settings are {SETTINGS}")
    if words != sorted(set(words), key=SETTINGS.index):
        raise ValueError(f"{name!r}: the settings go once each, in the order {SETTINGS}")
    if "keepring" in words and inp not in EDGE_INPUTS:
        raise ValueError(f"{name!r}: only an input with edges, {EDGE_INPUTS}, has a ring to keep")
    if "instrument-text" in words and seed != "sam":
        raise ValueError(f"{name!r}: only the tracking stage's segmenter adds instruments to its seed")
    return Parts(inp, depth, pps, seed, rule, tuple(words))


@dataclass(frozen=True)
class Row:
    """One row of the table: a workbench condition, and its name or why it has none."""

    dataset: str
    workbench: str
    name: str
    unresolved: str
    record: str


def read_table(path: str | Path = TABLE) -> dict[tuple[str, str], Row]:
    """Read the table, by dataset and workbench name.

    Raises:
        ValueError: The header is not `COLUMNS`; a row has another number
            of fields, a dataset outside `DATASETS` or a record outside
            `RECORDS`; a row holds both a name and a reason, or neither;
            `parse` refuses a name; a workbench name comes twice in one
            dataset; or two workbench names of one dataset share a name.
    """
    with open(path, encoding="utf-8", newline="") as f:
        lines = list(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    if not lines or tuple(lines[0]) != COLUMNS:
        raise ValueError(f"{path}: the header is not {COLUMNS}")
    out: dict[tuple[str, str], Row] = {}
    named: dict[tuple[str, str], str] = {}
    for n, fields in enumerate(lines[1:], start=2):
        if len(fields) != len(COLUMNS):
            raise ValueError(f"{path}:{n}: {len(fields)} fields, not {len(COLUMNS)}")
        row = Row(*fields)
        where = f"{path}:{n} ({row.dataset} {row.workbench})"
        if row.dataset not in DATASETS:
            raise ValueError(f"{where}: no dataset of {DATASETS}")
        if row.record not in RECORDS:
            raise ValueError(f"{where}: {row.record!r} is no record of {RECORDS}")
        if bool(row.name) == bool(row.unresolved):
            raise ValueError(f"{where}: a row holds a name or why it has none, not both and not neither")
        if row.name:
            parse(row.name)
            if (row.dataset, row.name) in named:
                raise ValueError(f"{where}: {row.name} also names {named[row.dataset, row.name]}")
            named[row.dataset, row.name] = row.workbench
        if (row.dataset, row.workbench) in out:
            raise ValueError(f"{where}: the workbench name comes twice")
        out[row.dataset, row.workbench] = row
    return out


def name_of(dataset: str, workbench: str, table: Mapping[tuple[str, str], Row] | None = None) -> str:
    """Return the name of a workbench condition of `dataset`.

    Raises:
        ValueError: The table holds no such condition, or its records show
            no name.
    """
    row = (read_table() if table is None else table).get((dataset, workbench))
    if row is None:
        raise ValueError(f"{workbench!r} is no workbench condition of {dataset} in the table")
    if not row.name:
        raise ValueError(f"{dataset} {workbench}: its records show no name: {row.unresolved}")
    return row.name
