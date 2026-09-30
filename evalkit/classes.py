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
    """What kind of thing a class is; decides whether and in which views it is scored.

    The pipeline cuts regions from depth and shape alone and never learns a
    class, so whether it could have found a class depends on the class's kind.
    Each class has one type, given in its table; the reasons per class are in
    `docs/evaluation.md`. Every pixel or frame a type removes is counted.
    """

    # Never scored. Removed from the GT and the prediction alike, so a region
    # there is neither an object nor a false positive.
    IGNORED = "ignored"        # not part of the scene: the black surround, the annotation tool's line
    BACKGROUND = "background"  # in the scene but labelled as nothing: unlabelled anatomy, not empty space
    EXCLUDED = "excluded"      # an annotators' marker that skips the whole frame

    # Scored. The views below differ only in which of these they include.
    TOOL = "tool"              # an instrument; in the `all` view only
    TISSUE = "tissue"          # anatomy that depth and shape can separate; the geometric view
    APPEARANCE = "appearance"  # told apart only by colour or texture (blood on the liver)
    EXPERT = "expert"          # boundary set by anatomical convention (which vessel, where a duct ends)
    BACKDROP = "backdrop"      # a surface the scene sits against (abdominal wall, diaphragm)


# The views: the types a metric is computed over. Every metric is computed
# once per view; a view removes the other types from the GT and the
# prediction alike. The paper's claims are judged in the geometric view, so a
# condition is not scored on classes geometry could never tell apart.
_ALL = frozenset({
    ClassType.TOOL, ClassType.TISSUE, ClassType.APPEARANCE,
    ClassType.EXPERT, ClassType.BACKDROP,
})
VIEWS: Mapping[str, frozenset[ClassType]] = MappingProxyType({
    "all": _ALL,
    "tissue": _ALL - {ClassType.TOOL},
    "geometric": frozenset({ClassType.TISSUE}),
})

# How each dataset's masks encode a class: a colour, read through the table
# (CholecSeg8k), or the id itself (ATLAS-120k). Also the list of datasets the
# evaluator knows: exactly their tables are hashed into `eval_code_sha`.
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

    # Every reader raises on what the table does not know. The pilot evaluator
    # answered "background" to an unknown colour and silently lost a class.

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


# Reading and checking the JSON. Strict on purpose: the file is hashed into
# `eval_code_sha`, so a typo, a stray field or a key given twice must not be
# read as something else silently.

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

    # The file's shape.
    if set(raw) != {"dataset", "classes"}:
        raise ValueError(f"{path}: top-level keys are {sorted(raw)}, expected ['classes', 'dataset']")
    if raw["dataset"] != dataset:
        raise ValueError(f"{path}: file says dataset {raw['dataset']!r}, asked for {dataset!r}")
    if dataset not in _MASK_ENCODING:
        raise ValueError(f"{path}: no table format is defined for dataset {dataset!r}")
    # Each class on its own, then against each other: a shared id, name or
    # colour would make a mask ambiguous. Names are compared ignoring case.
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
    # The hashed set is fixed by the code, not by what the directory holds:
    # a file left behind on one machine would give it a different sha.
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
