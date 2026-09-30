"""Class tables: what each dataset's classes are, and which of them a view scores.

Each dataset has one table, a JSON file under `class_tables/`, giving every
class its id, name and type, and for CholecSeg8k the colour its masks use.
`docs/evaluation.md` gives the types and the views, and the reasons behind
each class's type; this module only reads the files and refuses the ones it
cannot trust.

The tables are hashed into `eval_code_sha` with the evaluator's code, so a
changed type is a new evaluator. `table_paths()` lists exactly the files that
`load_table` can read, and nothing else: a stray file in the directory raises
instead of changing the sha on one machine only.

Everything here fails closed: a colour or id the table does not know raises,
and so does a table with a field missing, a type it does not define, a key
given twice, or two classes sharing an id, a name or a colour.
"""
from __future__ import annotations

import enum
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

TABLE_DIR = Path(__file__).resolve().parent / "class_tables"

Colour = tuple[int, int, int]


class ClassType(enum.StrEnum):
    """What kind of thing a class is, which decides whether and where it is scored.

    The pipeline splits a frame into regions from depth and shape alone; it
    never learns a class. So whether a region "should" have found a class
    depends on what kind of class it is, and the types make that explicit
    instead of leaving it to a per-experiment ignore list. Every class of every
    dataset has exactly one type, given in its table, and the reasons for each
    class's type are in `docs/evaluation.md`.

    Three types are never scored; they say how a pixel or a frame leaves the
    evaluation, and the evaluator counts every pixel and frame it removes:

    - `ignored`: not part of the scene at all. CholecSeg8k's black surround
      outside the endoscope's circle, and the white line its annotation tool
      draws between regions. Removing them is not a judgement about the
      pipeline, so no metric sees them.
    - `background`: inside the scene, but the dataset labelled it as nothing.
      This is unlabelled anatomy, not empty space, so a region the pipeline
      places there is neither an object found nor a false positive. It is
      removed like `ignored`, and kept as a separate type so that the counts
      say how much of a frame the dataset left unlabelled.
    - `excluded`: a marker the annotators put on a whole frame to take it out
      of evaluation (ATLAS-120k's `Excluded frames`). The frame is skipped
      before anything is scored, and counted.

    Five types are scored. They differ in which of the views (`VIEWS`, below)
    include them, and that is the point: the primary metrics are taken on the
    tissue that geometry alone could separate, so that a condition is not
    rewarded or punished for classes it could never tell apart:

    - `tool`: an instrument. Scored, but left out of the tissue views because
      instruments are not what the scene-graph is about.
    - `tissue`: anatomy whose extent depth and shape can separate. The
      geometric view is exactly this type.
    - `appearance`: tissue told apart only by colour or texture, such as
      blood lying on the liver. Geometry cannot find its edge.
    - `expert`: tissue whose boundary is anatomical convention, such as which
      vessel is which or where one stretch of a duct ends. Neither shape nor
      colour shows it without anatomical knowledge.
    - `backdrop`: a surface the scene sits against, nearly background but
      labelled (abdominal wall, diaphragm).
    """

    IGNORED = "ignored"
    BACKGROUND = "background"
    EXCLUDED = "excluded"
    TOOL = "tool"
    TISSUE = "tissue"
    APPEARANCE = "appearance"
    EXPERT = "expert"
    BACKDROP = "backdrop"


# The three views: which types a metric is computed over. Every metric is
# computed once per view, the same way; a view removes the pixels of the types
# it leaves out from the GT *and* the prediction, so a region on a removed
# pixel is neither an object nor a false positive there. The paper's claims
# are judged in the geometric view, chosen before any score was seen.
#   all       every scored type: the whole labelled scene
#   tissue    the same without tools: the anatomy
#   geometric only `tissue`: what depth and shape alone could separate
# `ignored` and `background` are in no view; `excluded` skips the frame
# before any view is taken.
_ALL = frozenset({
    ClassType.TOOL, ClassType.TISSUE, ClassType.APPEARANCE,
    ClassType.EXPERT, ClassType.BACKDROP,
})
VIEWS: Mapping[str, frozenset[ClassType]] = MappingProxyType({
    "all": _ALL,
    "tissue": _ALL - {ClassType.TOOL},
    "geometric": frozenset({ClassType.TISSUE}),
})

# How each dataset's GT masks encode a class: CholecSeg8k's masks are colour
# images, and a colour means a class only through the table; ATLAS-120k's
# masks hold the class id itself. This is the list of datasets the evaluator
# knows: a dataset not here has no table format, and the tables of exactly
# these datasets are what `table_paths()` hashes into `eval_code_sha`.
_MASK_ENCODING: Mapping[str, str] = MappingProxyType({"cholecseg8k": "colour"})


@dataclass(frozen=True)
class ClassEntry:
    """One class of a dataset.

    Attributes:
        id: The value a GT id map holds for this class.
        name: The dataset's name for it.
        type: Its type, which decides the views that score it.
        colour: The RGB triplet its masks use, for a dataset whose masks are
            colour images; None when the masks carry the id itself.
    """

    id: int
    name: str
    type: ClassType
    colour: Colour | None = None


@dataclass(frozen=True)
class ClassTable:
    """A dataset's classes, as one file under `class_tables/` defines them.

    Attributes:
        dataset: The dataset's name, as the file spells it.
        entries: Every class, keyed by id.
        colour_to_id: The mask colour of every class, for a dataset whose
            masks are colour images; empty when they carry the id itself. Read
            it through `id_of_colour`, which raises on an unknown colour.
        path: The file the table was read from, for `eval_code_sha`.
    """

    dataset: str
    entries: Mapping[int, ClassEntry]
    colour_to_id: Mapping[Colour, int]
    path: Path

    # The four readers below are the only way the evaluator asks the table
    # anything, and each raises on what the table does not know. That is
    # deliberate: the pilot evaluator answered "background" to an unknown
    # colour and silently lost a class, and a silent default anywhere here
    # would change scores without changing `eval_code_sha`.

    def type_of(self, class_id: int) -> ClassType:
        """The type of `class_id`; raises when the table does not have it."""
        try:
            return self.entries[class_id].type
        except KeyError:
            raise KeyError(
                f"{self.dataset}: class id {class_id} is not in the class table"
            ) from None

    def ids_of_type(self, *types: ClassType) -> frozenset[int]:
        """The ids whose type is one of `types`; at least one type is required."""
        if not types:
            raise TypeError("ids_of_type needs at least one type; no type means no ids, not all")
        return frozenset(e.id for e in self.entries.values() if e.type in types)

    def ids_in_view(self, view: str) -> frozenset[int]:
        """The ids a view scores; raises on a view name `VIEWS` does not have."""
        try:
            types = VIEWS[view]
        except KeyError:
            raise KeyError(f"unknown view {view!r}; views are {sorted(VIEWS)}") from None
        return self.ids_of_type(*types)

    def id_of_colour(self, colour: Colour) -> int:
        """The id a mask colour stands for; raises on a colour not in the table.

        This is the one place a CholecSeg8k mask meets the table, and it never
        maps an unknown colour to background: the pilot evaluator did, and lost
        a class that way.

        TODO(mask loading): four CholecSeg8k colour masks are RGBA (video18
        frame 1216, video35 frame 858, video37 frames 865 and 926), alpha 255
        throughout. The loader has to check that alpha is 255 and drop it;
        a 4-tuple here raises, which is safe but unhelpful.
        """
        if not self.colour_to_id:
            raise TypeError(f"{self.dataset}: this table's masks carry ids, not colours")
        try:
            return self.colour_to_id[colour]
        except KeyError:
            raise KeyError(
                f"{self.dataset}: colour {colour} is not in the class table"
            ) from None


# Reading and checking the JSON. The checks are strict on purpose: the file is
# hashed into `eval_code_sha`, so anything the loader would tolerate (a typo
# in a type, a field it ignores, a key given twice) is a way for two files
# with different shas to mean the same thing, or one file to mean two things.

_ENTRY_FIELDS = {"id", "name", "type"}


def _is_int(v) -> bool:
    # bool is a subclass of int, so `true` would pass as 1 without this.
    return isinstance(v, int) and not isinstance(v, bool)


def _entry(dataset: str, raw: dict, with_colour: bool) -> ClassEntry:
    """Build one entry, refusing a field missing, a stray one or a bad value."""
    fields = _ENTRY_FIELDS | ({"colour"} if with_colour else set())
    if set(raw) != fields:
        raise ValueError(
            f"{dataset}: class {raw.get('id')!r} has fields {sorted(raw)}, "
            f"expected {sorted(fields)}"
        )
    if not _is_int(raw["id"]) or raw["id"] < 0:
        raise ValueError(f"{dataset}: class id {raw['id']!r} is not a non-negative integer")
    if not isinstance(raw["name"], str) or not raw["name"].strip():
        raise ValueError(f"{dataset}: class {raw['id']} has name {raw['name']!r}, expected a non-empty string")
    try:
        class_type = ClassType(raw["type"])
    except ValueError:
        raise ValueError(
            f"{dataset}: class {raw['id']} has type {raw['type']!r}; "
            f"types are {[t.value for t in ClassType]}"
        ) from None
    colour = None
    if with_colour:
        c = raw["colour"]
        if not isinstance(c, list) or len(c) != 3 or not all(_is_int(v) and 0 <= v <= 255 for v in c):
            raise ValueError(f"{dataset}: class {raw['id']} has colour {c!r}, expected three ints in 0..255")
        colour = (c[0], c[1], c[2])
    return ClassEntry(id=raw["id"], name=raw["name"], type=class_type, colour=colour)


def _unique(dataset: str, what: str, values: list) -> None:
    seen = set()
    for v in values:
        if v in seen:
            raise ValueError(f"{dataset}: {what} {v!r} is given to two classes")
        seen.add(v)


def _refuse_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    """`json.load` keeps the last of two equal keys; a class typed twice would
    silently take the second type, so the file is refused instead."""
    out: dict = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"key {k!r} is given twice in one object")
        out[k] = v
    return out


def load_table(dataset: str, path: Path | None = None) -> ClassTable:
    """Read a dataset's class table.

    Args:
        dataset: `cholecseg8k`; the file is `class_tables/<dataset>.json`.
        path: Read this file instead. For tests that plant a fault; the
            evaluator never passes it.

    Returns:
        The table, validated: every field present once, every type known, and
        ids, names (ignoring case) and colours unique. The file's `dataset`
        field must match.

    Raises:
        ValueError: The file does not describe a table this module accepts.
    """
    path = TABLE_DIR / f"{dataset}.json" if path is None else Path(path)
    with open(path, encoding="utf-8") as f:
        try:
            raw = json.load(f, object_pairs_hook=_refuse_duplicate_keys)
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from None

    # The file's shape: exactly the two top-level keys, naming the dataset it
    # was asked for, of a dataset this module has a format for.
    if set(raw) != {"dataset", "classes"}:
        raise ValueError(f"{path}: top-level keys are {sorted(raw)}, expected ['classes', 'dataset']")
    if raw["dataset"] != dataset:
        raise ValueError(f"{path}: file says dataset {raw['dataset']!r}, asked for {dataset!r}")
    if dataset not in _MASK_ENCODING:
        raise ValueError(f"{path}: no table format is defined for dataset {dataset!r}")
    # Each class on its own, then the classes against each other: two classes
    # with one id, one name or one colour would make a mask ambiguous. Names
    # are compared ignoring case because "Liver" and "liver" are one typo, not
    # two classes.
    with_colour = _MASK_ENCODING[dataset] == "colour"
    entries = [_entry(dataset, r, with_colour=with_colour) for r in raw["classes"]]
    if not entries:
        raise ValueError(f"{path}: the table has no classes")
    _unique(dataset, "id", [e.id for e in entries])
    _unique(dataset, "name", [e.name.casefold() for e in entries])
    if with_colour:
        _unique(dataset, "colour", [e.colour for e in entries])
    return ClassTable(
        dataset=dataset,
        entries=MappingProxyType({e.id: e for e in entries}),
        colour_to_id=MappingProxyType({e.colour: e.id for e in entries} if with_colour else {}),
        path=path,
    )


def table_paths() -> list[Path]:
    """The table files to hash into `eval_code_sha`: one per dataset in
    `_MASK_ENCODING`, and the directory may hold nothing else.

    Raises:
        FileNotFoundError: A dataset's table is missing.
        ValueError: The directory holds a file this module would not read.
    """
    # The hashed set is defined by the code, not by what the directory
    # happens to hold: a file left behind on one machine would give that
    # machine a different sha for the same evaluator.
    expected = {TABLE_DIR / f"{dataset}.json" for dataset in _MASK_ENCODING}
    present = {p for p in TABLE_DIR.iterdir() if p.name != "__pycache__"}
    missing = expected - present
    if missing:
        raise FileNotFoundError(f"class table missing: {sorted(str(p) for p in missing)}")
    stray = present - expected
    if stray:
        raise ValueError(
            f"class_tables/ holds files no dataset reads, which would change "
            f"eval_code_sha here only: {sorted(p.name for p in stray)}"
        )
    return sorted(expected)
