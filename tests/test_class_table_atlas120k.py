"""ATLAS-120k's table: the 30 classes its benchmark scores, and the 47 ids its masks hold."""
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from evalkit.classes import CLASS_SETS, ClassType, load_table

EVALUATION_MD = Path(__file__).resolve().parent.parent / "docs" / "evaluation.md"

# The merges docs/evaluation.md lists under "ATLAS-120k: the 30 classes", by
# name. Every other original id maps to the class of its own name.
# TODO(port): this checks the table against the document, both typed here.
# Check it against the source, ATLAS-bench's `datasets/class_mapping.py`, with
# a script that takes that file's path; the check was made by hand at commit
# e286a584, and all 47 ids agreed.
MERGED_INTO = {
    "Aorta": "Artery",
    "Vena cava": "Vein", "Hepatic vein": "Vein", "V azygos": "Vein",
    "Cystic duct": "Bile/lymph duct", "Ductus choledochus": "Bile/lymph duct",
    "Ductus hepaticus": "Bile/lymph duct", "Thoracic duct": "Bile/lymph duct",
    "Omentum": "Fat", "Mesenterium": "Fat",
    "Nerves": "Nerve",
    "Catheter": "Non anatomical", "Non anatomical structures": "Non anatomical",
    # The "(major)" classes keep their id under a shorter name.
    "Vein (major)": "Vein", "Artery (major)": "Artery", "Nerve (major)": "Nerve",
}
DROPPED_TO_BACKGROUND = {
    "Kidney": "tissue", "Ureter": "expert", "Excluded frames": "excluded",
    "Mesocolon": "tissue", "Adrenal gland": "tissue", "Pancreas": "expert",
    "Duodenum": "appearance",
}


def types_of_the_specification() -> dict[int, tuple[str, str]]:
    """The 30 classes and their types, read from the "ATLAS-120k: types" table
    of docs/evaluation.md. One row there covers ids 16 to 21."""
    text = EVALUATION_MD.read_text(encoding="utf-8")
    section = text.split("### ATLAS-120k: types", 1)[1].split("\n### ", 1)[0]
    rows = {}
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 4 or not re.fullmatch(r"\d+(–\d+)?", cells[0]):
            continue
        ids = [int(v) for v in cells[0].split("–")]
        ids = range(ids[0], ids[-1] + 1)
        names = [n.strip() for n in cells[1].split(",")] if len(ids) > 1 else [cells[1]]
        assert len(names) == len(ids), line
        class_type = re.match(r"[a-z]+", cells[3]).group()
        for cid, name in zip(ids, names):
            rows[cid] = (name, class_type)
    assert len(rows) == 30, "the ATLAS-120k types table was not found in docs/evaluation.md"
    return rows


@pytest.fixture
def atlas30():
    return load_table("atlas120k", "atlas30")


@pytest.fixture
def atlas47():
    return load_table("atlas120k", "atlas47")


def test_the_30_classes_match_the_specification(atlas30):
    spec = types_of_the_specification()
    assert set(atlas30.entries) == set(spec)
    for cid, (name, class_type) in spec.items():
        e = atlas30.entries[cid]
        assert (e.name, e.type) == (name, ClassType(class_type)), cid


def test_the_default_class_set_is_the_benchmark_s_30():
    assert CLASS_SETS["atlas120k"][0] == "atlas30"
    assert load_table("atlas120k").class_set == "atlas30"


def test_the_47_ids_map_as_the_specification_says(atlas30, atlas47):
    assert len(atlas47.entries) == 47 and set(atlas47.entries) == set(range(47))
    by_name = {e.name: e.id for e in atlas30.entries.values()}
    for e in atlas47.entries.values():
        if e.name in DROPPED_TO_BACKGROUND:
            assert atlas30.class_of(e.id) == 0, e
        else:
            assert atlas30.class_of(e.id) == by_name[MERGED_INTO.get(e.name, e.name)], e
    assert [n for n, c in DROPPED_TO_BACKGROUND.items()] == [
        atlas47.entries[i].name for i in (32, 40, 42, 43, 44, 45, 46)]


def test_a_47_id_takes_the_type_of_the_class_it_merges_into(atlas30, atlas47):
    # Except the seven the mapping drops, which carry their own verdicts.
    for e in atlas47.entries.values():
        if e.name in DROPPED_TO_BACKGROUND:
            assert e.type == ClassType(DROPPED_TO_BACKGROUND[e.name]), e
        else:
            assert e.type == atlas30.type_of(atlas30.class_of(e.id)), e
    assert atlas47.entries[0].type is ClassType.BACKGROUND


def test_the_two_class_sets_differ_in_pixels_only_by_the_dropped_ids(atlas30, atlas47):
    # A merged id is scored on the same pixels in both sets. The six anatomical
    # ids the mapping drops are background in atlas30 and scored in atlas47 by
    # their own types, so there atlas47 scores more pixels.
    scored_only_in_atlas47 = {
        "all": {32, 40, 43, 44, 45, 46},
        "tissue": {32, 40, 43, 44, 45, 46},
        "geometric": {32, 43, 44},
    }
    for view, extra in scored_only_in_atlas47.items():
        via30 = {i for i in range(47) if atlas30.class_of(i) in atlas30.ids_in_view(view)}
        via47 = set(atlas47.ids_in_view(view))
        assert via47 - via30 == extra, view
        assert via30 <= via47, view


def test_excluded_frames_is_seen_on_the_original_ids(atlas30, atlas47):
    # In atlas30 the mapping turns id 42 into background, which would hide it.
    assert atlas30.class_of(42) == 0
    assert atlas30.excluded_mask_ids == atlas47.excluded_mask_ids == {42}


def test_a_mask_id_outside_the_47_raises(atlas30):
    with pytest.raises(KeyError, match="mask id 47"):
        atlas30.class_of(47)


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


def test_the_colours_are_the_ones_the_colour_masks_use(atlas30):
    for colour, mask_id in COLOURS_IN_THE_COLOUR_MASKS.items():
        assert atlas30.mask_id_of_colour(colour) == mask_id, colour
    assert len(atlas30.colour_to_mask_id) == 46


def test_excluded_frames_has_no_colour_of_its_own(atlas30, atlas47):
    # Its colour in the dataset's palette is (0, 0, 0), Background's.
    assert atlas47.entries[42].colour is None
    assert 42 not in atlas30.colour_to_mask_id.values()


def test_a_mask_reads_the_same_stored_as_ids_or_as_colours(atlas30, atlas47):
    ids = np.array([[0, 1, 2, 3], [12, 41, 46, 10], [33, 37, 20, 0]])
    for table in (atlas30, atlas47):
        as_ids = table.mask_ids(Image.fromarray(ids.astype(np.uint8)))
        as_colours = table.mask_ids(_colour_mask(table, ids))
        assert (as_ids == ids).all() and (as_colours == ids).all()
    assert (atlas30.classes_of(ids) == np.vectorize(atlas30.class_of)(ids)).all()
    assert (atlas47.classes_of(ids) == ids).all()


def test_a_palette_mask_is_read_by_its_index_not_its_palette(atlas30):
    # Some palettes in the release give Ligated plexus Liver's colour and ids
    # 43 to 46 black, so reading through the palette would merge them.
    palette = [0] * (3 * 256)
    for c, i in COLOURS_IN_THE_COLOUR_MASKS.items():
        palette[3 * i:3 * i + 3] = c
    palette[3 * 28:3 * 28 + 3] = (150, 100, 50)
    ids = np.array([[0, 12, 28], [43, 44, 46]])
    assert (atlas30.mask_ids(_palette_mask(ids, palette)) == ids).all()


def test_a_colour_mask_of_background_alone_is_refused(atlas30, atlas47):
    # An excluded frame could hide there; stored as ids the same frame is read.
    black = Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8))
    for table in (atlas30, atlas47):
        with pytest.raises(ValueError, match="background alone"):
            table.mask_ids(black)
    assert (atlas30.mask_ids(Image.fromarray(np.zeros((2, 3), dtype=np.uint8))) == 0).all()


def test_an_unknown_id_or_colour_in_a_mask_raises(atlas30):
    with pytest.raises(KeyError, match="mask id 47"):
        atlas30.mask_ids(Image.fromarray(np.array([[0, 47]], dtype=np.uint8)))
    rgb = np.zeros((1, 2, 3), dtype=np.uint8)
    rgb[0, 1] = (252, 186, 3)  # Pancreas in some embedded palettes, in no colour mask
    with pytest.raises(KeyError, match=r"\(252, 186, 3\)"):
        atlas30.mask_ids(Image.fromarray(rgb))


# Faults planted in a copy of the real table, which the loader has to refuse.

def _plant(tmp_path: Path, edit) -> Path:
    raw = json.loads(Path(load_table("atlas120k").path).read_text(encoding="utf-8"))
    edit(raw)
    out = tmp_path / "atlas120k.json"
    out.write_text(json.dumps(raw), encoding="utf-8")
    return out


def _original(raw, oid):
    return next(o for o in raw["original_ids"] if o["id"] == oid)


@pytest.mark.parametrize("fault, message", [
    (lambda r: _original(r, 32).pop("type"), "maps to background and needs a type"),
    (lambda r: _original(r, 12).update(type="tissue"), "has a type of its own"),
    (lambda r: _original(r, 12).update({"class": 30}), "maps to class 30, which is not"),
    (lambda r: _original(r, 12).pop("class"), "fields"),
    (lambda r: _original(r, 12).update(note="x"), "fields"),
    (lambda r: _original(r, 12).update(id=13), "original id 13 is given to two"),
    (lambda r: _original(r, 14).update(name="liver"), "original name 'liver' is given to two"),
    (lambda r: _original(r, 32).update(type="tisue"), "type 'tisue'"),
    (lambda r: r.pop("original_ids"), "top-level keys"),
    (lambda r: r["classes"][12].update(colour=[1, 2, 3]), "fields"),
    (lambda r: _original(r, 12).update({"class": True}), "maps to class True"),
    (lambda r: _original(r, 12).update({"class": 10.0}), "maps to class 10.0"),
    (lambda r: _original(r, 12).update({"class": [10]}), "maps to class \\[10\\]"),
    (lambda r: _original(r, 12).pop("colour"), "fields"),
    (lambda r: _original(r, 12).update(colour=None), "only an excluded marker"),
    (lambda r: _original(r, 12).update(colour=[0, 0, 255]), "colour .* is given to two"),
    (lambda r: _original(r, 42).update(colour=[0, 0, 0]), "colour .* is given to two"),
    (lambda r: _original(r, 33).update({"class": 28}), "no original id maps to class \\[26\\]"),
    (lambda r: r.update(original_ids=[]), "'original_ids' is not a non-empty list"),
    (lambda r: r.update(original_ids={"a": 1}), "'original_ids' is not a non-empty list"),
])
def test_a_planted_fault_is_refused(tmp_path, fault, message):
    path = _plant(tmp_path, fault)
    with pytest.raises(ValueError, match=message):
        load_table("atlas120k", path=path)


def test_the_planted_copy_loads_when_nothing_is_planted(tmp_path):
    path = _plant(tmp_path, lambda r: None)
    assert load_table("atlas120k", "atlas47", path=path).entries.keys() == set(range(47))
