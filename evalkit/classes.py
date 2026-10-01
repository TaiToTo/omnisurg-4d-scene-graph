"""Class tables: what each dataset's classes are, and which of them a view scores.

Each dataset has one table, a JSON file under `class_tables/`, listing the
classes its masks hold, the dataset's own labels: each with its id, name,
type and the colour that stands for it. Claims are judged on these. ATLAS-120k's
file also lists the 30 classes its benchmark scores, and the one each original
id merges into, so that the same file yields a second table for comparison
with the benchmark. `docs/evaluation.md` gives the types, the views and the
mapping, and the reasons behind each type; this module only reads the files
and refuses the ones it cannot trust.

A GT mask meets its table in one place, `ClassTable.mask_ids`, and both
datasets pass through it the same way: a mask that stores ids is read by its
values, one that stores colours through the table.

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

import numpy as np
from PIL import Image

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
    TOOL = "tool"              # an instrument, or another object that is not anatomy; in the `all` view only
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

# How each dataset's masks may store a pixel's label: as the id itself, in a
# palette or greyscale image, or as the id's colour in an RGB image. Most
# ATLAS-120k masks are palette images, but 34 of its clips store colours.
# CholecSeg8k's single-channel masks hold the watershed's own codes, not the
# table's ids, so only its colour masks are read. Also the list of datasets
# the evaluator knows: exactly their tables are hashed into `eval_code_sha`.
_MASK_ENCODINGS: Mapping[str, frozenset[str]] = MappingProxyType({
    "cholecseg8k": frozenset({"colour"}),
    "atlas120k": frozenset({"id", "colour"}),
})

# The datasets whose file also lists a benchmark's classes.
_WITH_BENCHMARK = frozenset({"atlas120k"})

# The class sets each dataset is scored with. `original` is the dataset's own
# labels, the ids its masks hold, and the one claims are judged on: how
# another group merged classes for training is not this evaluator's to
# inherit. `benchmark` is ATLAS-120k's 30 classes, scored for comparison with
# its benchmark and never given a star.
# TODO(pilot mode): pilot mode scores ATLAS-120k's original ids by the pilot
# evaluator's rules, where Tools/camera is the only type there is. The
# `original` set types every id, and makes Catheter and Non anatomical
# structures tools too. Pilot mode needs a typing of its own and must not read
# these types.
CLASS_SETS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "cholecseg8k": ("original",),
    "atlas120k": ("original", "benchmark"),
})


@dataclass(frozen=True)
class ClassEntry:
    """One class of a class set.

    Attributes:
        id: The value a GT id map holds for this class.
        name: The dataset's name for it.
        type: Its type, which decides the views that score it.
        colour: The RGB triplet that stands for it in a mask stored as
            colours. None for a benchmark class, which a mask reaches only
            through the mapping, and for a marker without a colour of its own
            (ATLAS-120k's `Excluded frames`, whose colour is Background's).
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
        class_set: Which of the dataset's class sets this is (`CLASS_SETS`).
        entries: Every class of the set, keyed by id.
        encodings: How the dataset's masks may store a label: `id`,
            `colour`, or both.
        colour_to_mask_id: The mask id each colour stands for. Read it
            through `mask_id_of_colour`, which raises on an unknown colour.
        mask_id_to_class: The id a mask holds, to the class it belongs to in
            this set. The identity for `original`; for `benchmark`, ATLAS-120k's
            47 ids onto the 30 classes. Read it through `class_of`.
        excluded_mask_ids: The mask ids that take a whole frame out. They are
            checked on the mask's own ids, before the mapping: in `benchmark`
            the mapping turns `Excluded frames` into background.
        path: The file the table was read from, for `eval_code_sha`.
    """

    dataset: str
    class_set: str
    entries: Mapping[int, ClassEntry]
    encodings: frozenset[str]
    colour_to_mask_id: Mapping[Colour, int]
    mask_id_to_class: Mapping[int, int]
    excluded_mask_ids: frozenset[int]
    path: Path

    # The readers below are the only way the evaluator asks the table
    # anything, and each raises on what the table does not know. That is
    # deliberate: the pilot evaluator answered "background" to an unknown
    # colour and silently lost a class, and a silent default anywhere here
    # would change scores without changing `eval_code_sha`.

    def class_of(self, mask_id: int) -> int:
        """The class a mask id belongs to; raises on an id no mask may hold."""
        try:
            return self.mask_id_to_class[mask_id]
        except KeyError:
            raise KeyError(
                f"{self.dataset}: mask id {mask_id} is not in the class table"
            ) from None

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

    def mask_id_of_colour(self, colour: Colour) -> int:
        """The mask id a colour stands for; raises on a colour not in the table."""
        try:
            return self.colour_to_mask_id[colour]
        except KeyError:
            raise KeyError(
                f"{self.dataset}: colour {colour} is not in the class table"
            ) from None

    def mask_ids(self, image: Image.Image) -> np.ndarray:
        """The mask id of every pixel of a GT mask, read through the table.

        A palette or greyscale image holds the ids themselves and is read by
        its values, never through its palette: the palettes embedded in
        ATLAS-120k's masks disagree with one another, and some give several
        ids one colour. An RGB image holds colours, read through the table.
        The mask comes as a PIL image rather than an array because an array
        does not say whether its channels are RGB or BGR, and ATLAS-120k's
        colours include pairs that swap under that mistake, Artery and Vein
        among them.

        Args:
            image: The mask as PIL opened it.

        Returns:
            An (H, W) int32 array of mask ids, which `classes_of` takes to
            the classes of this set.

        Raises:
            ValueError: The image stores its labels in a way this dataset's
                masks do not; an RGBA mask is not opaque; or a colour mask is
                the background colour alone, which a colour cannot tell from
                a frame the excluded marker takes out.
            KeyError: A pixel holds an id or a colour the table does not have.
        """
        mode = image.mode
        if mode in ("P", "L"):
            self._require("id", mode)
            ids = np.asarray(image).astype(np.int32)
            for v in np.unique(ids).tolist():
                self.class_of(v)
            return ids
        if mode in ("RGB", "RGBA"):
            self._require("colour", mode)
            rgb = np.asarray(image)
            if mode == "RGBA":
                if (rgb[..., 3] != 255).any():
                    raise ValueError(f"{self.dataset}: an RGBA mask has pixels that are not opaque")
                rgb = rgb[..., :3]
            packed = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2]
            colours, inverse = np.unique(packed.ravel(), return_inverse=True)
            lut = np.array(
                [self.mask_id_of_colour((c >> 16, (c >> 8) & 255, c & 255)) for c in colours.tolist()],
                dtype=np.int32,
            )
            # A marker without a colour of its own would fill its frame with
            # the background colour, so a colour mask of that colour alone is
            # the one frame it could hide in, and that frame is not scored.
            marker_hidden = self.excluded_mask_ids - set(self.colour_to_mask_id.values())
            if marker_hidden and len(lut) == 1 and self.type_of(self.class_of(int(lut[0]))) is ClassType.BACKGROUND:
                raise ValueError(
                    f"{self.dataset}: a colour mask of background alone cannot be told from "
                    f"a frame that mask id {sorted(marker_hidden)} excludes"
                )
            return lut[inverse].reshape(packed.shape)
        raise ValueError(f"{self.dataset}: a mask of mode {mode!r} is neither ids nor RGB colours")

    def classes_of(self, mask_ids: np.ndarray) -> np.ndarray:
        """The class of every pixel of this set, from the ids `mask_ids` read."""
        lut = np.full(max(self.mask_id_to_class) + 1, -1, dtype=np.int32)
        for mask_id, class_id in self.mask_id_to_class.items():
            lut[mask_id] = class_id
        for v in np.unique(mask_ids).tolist():
            self.class_of(v)
        return lut[mask_ids]

    def _require(self, encoding: str, mode: str) -> None:
        if encoding not in self.encodings:
            raise ValueError(
                f"{self.dataset}: a mask of mode {mode!r} stores {encoding}s, and this "
                f"dataset's masks store only {sorted(self.encodings)}"
            )


# Reading and checking the JSON. Strict on purpose: the file is hashed into
# `eval_code_sha`, so a typo, a stray field or a key given twice must not be
# read as something else silently.

_CLASS_FIELDS = {"id", "name", "colour", "type"}
_BENCHMARK_FIELDS = {"id", "name", "type"}


def _is_int(v) -> bool:
    # bool is a subclass of int, so `true` would pass as 1 without this.
    return isinstance(v, int) and not isinstance(v, bool)


def _fields(dataset: str, raw: dict, expected: set[str]) -> None:
    if set(raw) != expected:
        raise ValueError(
            f"{dataset}: class {raw.get('id')!r} has fields {sorted(raw)}, "
            f"expected {sorted(expected)}"
        )


def _check_id_and_name(dataset: str, raw: dict) -> None:
    if not _is_int(raw["id"]) or raw["id"] < 0:
        raise ValueError(f"{dataset}: class id {raw['id']!r} is not a non-negative integer")
    if not isinstance(raw["name"], str) or not raw["name"].strip():
        raise ValueError(f"{dataset}: class {raw['id']} has name {raw['name']!r}, expected a non-empty string")


def _type(dataset: str, raw: dict) -> ClassType:
    try:
        return ClassType(raw["type"])
    except ValueError:
        raise ValueError(
            f"{dataset}: class {raw['id']} has type {raw['type']!r}; "
            f"types are {[t.value for t in ClassType]}"
        ) from None


def _colour(dataset: str, raw: dict, class_type: ClassType) -> Colour | None:
    """The class's colour. Only an `excluded` marker may lack one, because the
    one ATLAS-120k has shares Background's."""
    c = raw["colour"]
    if c is None:
        if class_type is not ClassType.EXCLUDED:
            raise ValueError(f"{dataset}: class {raw['id']} has no colour; only an excluded marker may lack one")
        return None
    if not isinstance(c, list) or len(c) != 3 or not all(_is_int(v) and 0 <= v <= 255 for v in c):
        raise ValueError(f"{dataset}: class {raw['id']} has colour {c!r}, expected three ints in 0..255")
    return (c[0], c[1], c[2])


def _entry(dataset: str, raw: dict, extra: set[str]) -> ClassEntry:
    """One class a mask holds, refusing a field missing, a stray one or a bad value."""
    _fields(dataset, raw, _CLASS_FIELDS | extra)
    _check_id_and_name(dataset, raw)
    class_type = _type(dataset, raw)
    return ClassEntry(id=raw["id"], name=raw["name"], type=class_type, colour=_colour(dataset, raw, class_type))


def _benchmark_entry(dataset: str, raw: dict) -> ClassEntry:
    """One benchmark class. It has no colour: no mask holds it directly."""
    _fields(dataset, raw, _BENCHMARK_FIELDS)
    _check_id_and_name(dataset, raw)
    return ClassEntry(id=raw["id"], name=raw["name"], type=_type(dataset, raw))


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


def _benchmark_map(
    dataset: str, raws: list[dict], entries: list[ClassEntry], benchmark: Mapping[int, ClassEntry],
) -> dict[int, int]:
    """Each original id to the benchmark class it merges into.

    Every benchmark class but background must be reached, or no mask could
    hold it. A benchmark class reached by one id alone is that id under
    another name, so the two must have the same type; a difference there is a
    typo, not a judgment.
    """
    to_class = {}
    for raw in raws:
        b = raw["benchmark_class"]
        if not _is_int(b) or b not in benchmark:
            raise ValueError(f"{dataset}: class {raw['id']} merges into benchmark class {b!r}, which is not in the table")
        to_class[raw["id"]] = b
    unreached = sorted(
        {c.id for c in benchmark.values() if c.type is not ClassType.BACKGROUND} - set(to_class.values())
    )
    if unreached:
        raise ValueError(f"{dataset}: no class merges into benchmark class {unreached}, so no mask can hold it")
    members: dict[int, list[ClassEntry]] = {}
    for e in entries:
        members.setdefault(to_class[e.id], []).append(e)
    for b, es in members.items():
        if len(es) == 1 and es[0].type is not benchmark[b].type:
            raise ValueError(
                f"{dataset}: class {es[0].id} is benchmark class {b} alone, but types it "
                f"{es[0].type.value!r} where the benchmark has {benchmark[b].type.value!r}"
            )
    return to_class


def _objects(path: Path, raw: dict, key: str) -> list[dict]:
    items = raw[key]
    if not isinstance(items, list) or not items or not all(isinstance(i, dict) for i in items):
        raise ValueError(f"{path}: {key!r} is not a non-empty list of objects")
    return items


def _read(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        try:
            return json.load(f, object_pairs_hook=_refuse_duplicate_keys)
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from None


def load_table(dataset: str, class_set: str | None = None, path: Path | None = None) -> ClassTable:
    """Read a dataset's class table.

    Args:
        dataset: `cholecseg8k` or `atlas120k`; the file is
            `class_tables/<dataset>.json`.
        class_set: One of `CLASS_SETS[dataset]`; None takes `original`, the
            one claims are judged on.
        path: Read this file instead. For tests that plant a fault; the
            evaluator never passes it.

    Returns:
        The table, validated: every field present once, every type known, and
        ids, names (ignoring case) and colours unique. The file's `dataset`
        field must match.

    Raises:
        ValueError: The file does not describe a table this module accepts,
            or `class_set` is not one of the dataset's.
    """
    # Checked before the file is opened, so an unknown dataset fails the same
    # way whether or not a file exists for it.
    if dataset not in _MASK_ENCODINGS:
        raise ValueError(f"no table format is defined for dataset {dataset!r}")
    if class_set is None:
        class_set = CLASS_SETS[dataset][0]
    if class_set not in CLASS_SETS[dataset]:
        raise ValueError(f"{dataset}: class set {class_set!r} is not one of {CLASS_SETS[dataset]}")
    path = TABLE_DIR / f"{dataset}.json" if path is None else Path(path)
    raw = _read(path)

    # The file's shape. Only ATLAS-120k has the benchmark's list.
    with_benchmark = dataset in _WITH_BENCHMARK
    keys = {"dataset", "classes"} | ({"benchmark_classes"} if with_benchmark else set())
    if set(raw) != keys:
        raise ValueError(f"{path}: top-level keys are {sorted(raw)}, expected {sorted(keys)}")
    if raw["dataset"] != dataset:
        raise ValueError(f"{path}: file says dataset {raw['dataset']!r}, asked for {dataset!r}")

    # The classes the masks hold, each on its own, then against each other: a
    # shared id, name or colour would make a mask ambiguous. Names are
    # compared ignoring case.
    raw_classes = _objects(path, raw, "classes")
    entries = [_entry(dataset, r, {"benchmark_class"} if with_benchmark else set()) for r in raw_classes]
    _unique(dataset, "id", [e.id for e in entries])
    _unique(dataset, "name", [e.name.casefold() for e in entries])
    _unique(dataset, "colour", [e.colour for e in entries if e.colour is not None])
    classes = MappingProxyType({e.id: e for e in entries})
    to_class = {e.id: e.id for e in entries}

    # ATLAS-120k's benchmark: its 30 classes, and the one each original id
    # merges into. Excluded frames are found on the original ids in both
    # sets, since the merge hides the marker.
    if with_benchmark:
        benchmark_entries = [_benchmark_entry(dataset, r) for r in _objects(path, raw, "benchmark_classes")]
        _unique(dataset, "benchmark id", [e.id for e in benchmark_entries])
        _unique(dataset, "benchmark name", [e.name.casefold() for e in benchmark_entries])
        benchmark = MappingProxyType({e.id: e for e in benchmark_entries})
        benchmark_map = _benchmark_map(dataset, raw_classes, entries, benchmark)
        if class_set == "benchmark":
            classes, to_class = benchmark, benchmark_map

    return ClassTable(
        dataset=dataset,
        class_set=class_set,
        entries=classes,
        encodings=_MASK_ENCODINGS[dataset],
        colour_to_mask_id=MappingProxyType({e.colour: e.id for e in entries if e.colour is not None}),
        mask_id_to_class=MappingProxyType(to_class),
        excluded_mask_ids=frozenset(e.id for e in entries if e.type is ClassType.EXCLUDED),
        path=path,
    )


def table_paths() -> list[Path]:
    """The table files to hash into `eval_code_sha`: one per dataset in
    `_MASK_ENCODINGS`, and the directory may hold nothing else.

    Raises:
        FileNotFoundError: A dataset's table is missing.
        ValueError: The directory holds a file this module would not read.
    """
    # The hashed set is fixed by the code, not by what the directory holds:
    # a file left behind on one machine would give it a different sha.
    expected = {TABLE_DIR / f"{dataset}.json" for dataset in _MASK_ENCODINGS}
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
