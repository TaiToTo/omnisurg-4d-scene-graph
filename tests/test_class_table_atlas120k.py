"""ATLAS-120k's table: the 47 ids its masks hold, and the 30 classes its benchmark scores."""
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from evalkit.classes import CLASS_SETS, ClassType, load_table

EVALUATION_MD = Path(__file__).resolve().parent.parent / "docs" / "evaluation.md"


def _table_rows(title: str, n_cells: int) -> list[list[str]]:
    """The rows of the table under `### <title>` in docs/evaluation.md whose
    first cell is a number, so that the JSON is checked against the document
    and not against a third copy typed by hand."""
    text = EVALUATION_MD.read_text(encoding="utf-8")
    section = text.split(f"### {title}\n", 1)[1].split("\n### ", 1)[0]
    rows = []
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == n_cells and cells[0].isdigit():
            rows.append(cells)
    return rows


def _type_cell(cell: str) -> ClassType:
    return ClassType(re.match(r"[a-z]+", cell).group())


def original_ids_of_the_specification() -> dict[int, tuple[str, str, ClassType]]:
    """id: (label, benchmark class name, type), from the document's table."""
    rows = _table_rows("ATLAS-120k: the original ids and their types", 5)
    out = {int(r[0]): (r[1], r[2], _type_cell(r[4])) for r in rows}
    assert len(out) == len(rows) == 47, "the original ids' table was not found in docs/evaluation.md"
    return out


def benchmark_classes_of_the_specification() -> dict[int, tuple[str, list[str], ClassType]]:
    """id: (name, original labels merged into it, type), from the document's table."""
    rows = _table_rows("ATLAS-120k: the benchmark's 30 classes", 4)
    out = {int(r[0]): (r[1], [n.strip() for n in r[2].split(",")], _type_cell(r[3])) for r in rows}
    assert len(out) == len(rows) == 30, "the benchmark classes' table was not found in docs/evaluation.md"
    return out


@pytest.fixture
def original():
    return load_table("atlas120k", "original")


@pytest.fixture
def benchmark():
    return load_table("atlas120k", "benchmark")


def test_claims_are_judged_on_the_original_ids_by_default():
    assert CLASS_SETS["atlas120k"] == ("original", "benchmark")
    assert load_table("atlas120k").class_set == "original"


def test_the_original_ids_match_the_specification(original, benchmark):
    spec = original_ids_of_the_specification()
    assert set(original.entries) == set(spec) == set(range(47))
    by_name = {e.name: e.id for e in benchmark.entries.values()}
    for i, (label, benchmark_name, class_type) in spec.items():
        e = original.entries[i]
        assert (e.name, e.type) == (label, class_type), i
        assert benchmark.class_of(i) == by_name[benchmark_name], i


def test_the_benchmark_classes_match_the_specification(original, benchmark):
    spec = benchmark_classes_of_the_specification()
    assert set(benchmark.entries) == set(spec) == set(range(30))
    for cid, (name, members, class_type) in spec.items():
        e = benchmark.entries[cid]
        assert (e.name, e.type) == (name, class_type), cid
        merged = {original.entries[i].name for i in range(47) if benchmark.class_of(i) == cid}
        assert merged == set(members), cid


def test_the_specification_s_two_tables_give_the_same_mapping():
    ids = original_ids_of_the_specification()
    classes = benchmark_classes_of_the_specification()
    for name, members, _ in classes.values():
        assert {label for label, b, _ in ids.values() if b == name} == set(members), name


def test_ids_in_no_mask_take_their_benchmark_class_s_type(original, benchmark):
    # Hepatic vein, Thoracic duct and Nerves occur in no mask of the release.
    for i in (15, 38, 39):
        assert original.type_of(i) is benchmark.type_of(benchmark.class_of(i)), i


def test_the_ids_the_benchmark_drops_are_scored_in_the_original_set(original, benchmark):
    dropped = {i for i in range(47) if benchmark.class_of(i) == 0} - {0, 42}
    assert dropped == {32, 40, 43, 44, 45, 46}
    assert dropped <= original.ids_in_view("all")
    assert not {benchmark.class_of(i) for i in dropped} & benchmark.ids_in_view("all")


def test_excluded_frames_is_seen_on_the_original_ids(original, benchmark):
    # In the benchmark set the mapping turns id 42 into background, which would hide it.
    assert benchmark.class_of(42) == 0
    assert original.excluded_mask_ids == benchmark.excluded_mask_ids == {42}


def test_a_mask_id_outside_the_47_raises(original, benchmark):
    for table in (original, benchmark):
        with pytest.raises(KeyError, match="mask id 47"):
            table.class_of(47)


# Every colour found in the release's 4,381 colour masks, with the id it stands
# for. 441 of those masks are also stored as palette masks in the adjacent clip
# of the same video, and both read as the same ids.
COLOURS_IN_THE_COLOUR_MASKS = {
    (0, 0, 0): 0, (255, 255, 255): 1, (255, 0, 0): 3, (0, 200, 100): 6,
    (200, 150, 100): 7, (250, 150, 100): 8, (255, 200, 100): 9, (150, 100, 50): 12,
    (0, 255, 255): 13, (0, 200, 255): 14, (255, 150, 50): 16, (255, 220, 200): 17,
    (200, 100, 200): 18, (255, 0, 150): 23, (255, 100, 200): 24, (200, 100, 255): 25,
    (150, 0, 100): 26, (200, 0, 150): 29, (255, 150, 255): 31, (50, 50, 50): 41,
}


def _palette_mask(ids: np.ndarray, palette: list[int]) -> Image.Image:
    image = Image.frombytes("P", (ids.shape[1], ids.shape[0]), ids.astype(np.uint8).tobytes())
    image.putpalette(palette)
    return image


def _colour_mask(table, ids: np.ndarray) -> Image.Image:
    rgb = np.zeros(ids.shape + (3,), dtype=np.uint8)
    by_id = {i: c for c, i in table.colour_to_mask_id.items()}
    for i in np.unique(ids).tolist():
        rgb[ids == i] = by_id[i]
    return Image.fromarray(rgb)


def test_the_colours_are_the_ones_the_colour_masks_use(original):
    for colour, mask_id in COLOURS_IN_THE_COLOUR_MASKS.items():
        assert original.mask_id_of_colour(colour) == mask_id, colour
    assert len(original.colour_to_mask_id) == 46


def test_excluded_frames_has_no_colour_of_its_own(original, benchmark):
    # Its colour in the dataset's palette is (0, 0, 0), Background's.
    assert original.entries[42].colour is None
    assert 42 not in benchmark.colour_to_mask_id.values()


def test_a_mask_reads_the_same_stored_as_ids_or_as_colours(original, benchmark):
    ids = np.array([[0, 1, 2, 3], [12, 41, 46, 10], [33, 37, 20, 0]])
    for table in (original, benchmark):
        as_ids = table.mask_ids(Image.fromarray(ids.astype(np.uint8)))
        as_colours = table.mask_ids(_colour_mask(table, ids))
        assert (as_ids == ids).all() and (as_colours == ids).all()
    assert (original.classes_of(ids) == ids).all()
    assert (benchmark.classes_of(ids) == np.vectorize(benchmark.class_of)(ids)).all()


def test_a_palette_mask_is_read_by_its_index_not_its_palette(original):
    # Some palettes in the release give Ligated plexus Liver's colour and ids
    # 43 to 46 black, so reading through the palette would merge them.
    palette = [0] * (3 * 256)
    for c, i in COLOURS_IN_THE_COLOUR_MASKS.items():
        palette[3 * i:3 * i + 3] = c
    palette[3 * 28:3 * 28 + 3] = (150, 100, 50)
    ids = np.array([[0, 12, 28], [43, 44, 46]])
    assert (original.mask_ids(_palette_mask(ids, palette)) == ids).all()


def test_a_colour_mask_of_background_alone_is_refused(original, benchmark):
    # An excluded frame could hide there; stored as ids the same frame is read.
    black = Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8))
    for table in (original, benchmark):
        with pytest.raises(ValueError, match="background alone"):
            table.mask_ids(black)
    assert (original.mask_ids(Image.fromarray(np.zeros((2, 3), dtype=np.uint8))) == 0).all()


def test_a_colour_mask_of_one_class_the_benchmark_drops_is_read_in_both_sets(original, benchmark):
    # The benchmark maps Kidney to background, but no marker can hide in Kidney's
    # colour, so the mask reads the same whichever class set reads it.
    kidney = _colour_mask(original, np.full((2, 3), 32))
    for table in (original, benchmark):
        assert (table.mask_ids(kidney) == 32).all()
    assert original.background_mask_ids == benchmark.background_mask_ids == {0}


def test_an_unknown_id_or_colour_in_a_mask_raises(original):
    with pytest.raises(KeyError, match="mask id 47"):
        original.mask_ids(Image.fromarray(np.array([[0, 47]], dtype=np.uint8)))
    rgb = np.zeros((1, 2, 3), dtype=np.uint8)
    rgb[0, 1] = (252, 186, 3)  # Pancreas in some embedded palettes, in no colour mask
    with pytest.raises(KeyError, match=r"\(252, 186, 3\)"):
        original.mask_ids(Image.fromarray(rgb))


# Faults planted in a copy of the real table, which the loader has to refuse.
# The mapping to the benchmark's classes is checked above against the
# document, and both were typed here; against its source, ATLAS-bench's
# `datasets/class_mapping.py`, it was checked by hand at commit e286a584, all
# 47 ids agreeing. A script for that check is an open question of the port.

def _plant(tmp_path: Path, edit) -> Path:
    raw = json.loads(Path(load_table("atlas120k").path).read_text(encoding="utf-8"))
    edit(raw)
    out = tmp_path / "atlas120k.json"
    out.write_text(json.dumps(raw), encoding="utf-8")
    return out


def _class(raw, cid):
    return next(c for c in raw["classes"] if c["id"] == cid)


def _benchmark_class(raw, cid):
    return next(c for c in raw["benchmark_classes"] if c["id"] == cid)


@pytest.mark.parametrize("fault, message", [
    (lambda r: _class(r, 32).pop("type"), "fields"),
    (lambda r: _class(r, 32).update(type="tisue"), "type 'tisue'"),
    (lambda r: _class(r, 12).pop("benchmark_class"), "fields"),
    (lambda r: _class(r, 12).pop("colour"), "fields"),
    (lambda r: _class(r, 12).update(note="x"), "fields"),
    (lambda r: _class(r, 12).update(id=13), "id 13 is given to two"),
    (lambda r: _class(r, 14).update(name="liver"), "name 'liver' is given to two"),
    (lambda r: _class(r, 12).update(benchmark_class=30), "benchmark class 30, which is not"),
    (lambda r: _class(r, 12).update(benchmark_class=True), "benchmark class True"),
    (lambda r: _class(r, 12).update(benchmark_class=10.0), "benchmark class 10.0"),
    (lambda r: _class(r, 12).update(benchmark_class=[10]), "benchmark class \\[10\\]"),
    (lambda r: _class(r, 12).update(colour=None), "only an excluded marker"),
    (lambda r: _class(r, 12).update(colour=[0, 0, 255]), "colour .* is given to two"),
    (lambda r: _class(r, 42).update(colour=[0, 0, 0]), "colour .* is given to two"),
    (lambda r: _class(r, 33).update(benchmark_class=28), "no class merges into benchmark class \\[26\\]"),
    # Liver is benchmark class 10 alone, so a different type is a typo.
    (lambda r: _class(r, 12).update(type="appearance"), "benchmark class 10 alone"),
    (lambda r: _benchmark_class(r, 10).update(type="appearance"), "benchmark class 10 alone"),
    (lambda r: _benchmark_class(r, 12).update(colour=[1, 2, 3]), "fields"),
    (lambda r: _benchmark_class(r, 12).update(id=13), "benchmark id 13 is given to two"),
    (lambda r: r.pop("benchmark_classes"), "top-level keys"),
    (lambda r: r.update(classes=[]), "'classes' is not a non-empty list"),
    (lambda r: r.update(benchmark_classes={"a": 1}), "'benchmark_classes' is not a non-empty list"),
])
def test_a_planted_fault_is_refused(tmp_path, fault, message):
    path = _plant(tmp_path, fault)
    with pytest.raises(ValueError, match=message):
        load_table("atlas120k", path=path)


def test_the_planted_copy_loads_when_nothing_is_planted(tmp_path):
    path = _plant(tmp_path, lambda r: None)
    assert load_table("atlas120k", path=path).entries.keys() == set(range(47))
    assert load_table("atlas120k", "benchmark", path=path).entries.keys() == set(range(30))
