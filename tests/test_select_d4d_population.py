"""Test the selection of D4D's sides on a census made up for the test, and the files in `d4d_meta/`.

The made-up census holds a side for each reason of the rule, on clips of two specimens and three
sessions. Each refusal has a test that plants its fault:

- a census cut short, and a census that lacks one clip;
- a clip the list does not hold, and a clip held twice;
- a side that carries an `error`, and a side with a point cloud and no `active`.

The files in `d4d_meta/` are checked against one another, against the rule's thresholds, and for
machine paths, email addresses and untranslated text.
"""

import json
import re
from pathlib import Path

import pytest

from pipeline.select_d4d_population import (
    MAX_GT_FRAME_GAP_S, MIN_GT_BLOCK_FRAC, check_census, classify, kept_clips, main, read_clips, select)

META = Path(__file__).resolve().parent.parent / "d4d_meta"
PRIVATE = re.compile(r"/(home|var/autofs|mnt|Users)/|[぀-ヿ一-鿿！-｠]|[\w.+-]+@[\w-]+\.[\w.]+")


def side(**kw) -> dict:
    """Return a census side that the rule keeps, with the fields in `kw` replaced."""
    return {"present": True, "gt_blk_frac": 0.5, "frame_minus_gt_s": 4.4, "active": False} | kw


def clip(key: str, start: dict, end: dict, moved: bool = False) -> dict:
    return {"key": key, "specimen": key.split("/")[0], "moved_camera": moved, "start": start, "end": end}


def census() -> list[dict]:
    """Return four clips: every reason of the rule, and kept sides of both cameras, two specimens and three sessions."""
    return [
        clip("specimen_1/s_a/Clip_1", side(), side(frame_minus_gt_s=-4.3)),
        clip("specimen_1/s_a/Clip_2", side(active=True), {"present": False, "active": False}, moved=True),
        clip("specimen_1/s_b/Clip_1", side(gt_blk_frac=0.1), side(frame_minus_gt_s=12.0), moved=True),
        clip("specimen_2/s_c/Clip_1", side(frame_minus_gt_s=-10.5), side(), moved=True),
    ]


def keys(c: list[dict]) -> list[str]:
    return [r["key"] for r in c]


def write_inputs(tmp_path, c: list[dict], clips: list[str]) -> list[str]:
    """Write a census and a list of clips; return the arguments that name them and an output directory."""
    (tmp_path / "census.json").write_text(json.dumps(c))
    (tmp_path / "clips.txt").write_text("\n".join(clips) + "\n")
    return ["--census", str(tmp_path / "census.json"), "--clips", str(tmp_path / "clips.txt"),
            "--out", str(tmp_path / "out")]


@pytest.mark.parametrize("fields, reason", [
    ({"present": False, "gt_blk_frac": 0.0, "active": True}, "no_gt"),
    ({"gt_blk_frac": 0.1, "active": True, "frame_minus_gt_s": 30.0}, "gt_not_visible"),
    ({"active": True, "frame_minus_gt_s": 30.0}, "tissue_moving"),
    ({"frame_minus_gt_s": -10.5}, "gt_stale"),
    ({"gt_blk_frac": MIN_GT_BLOCK_FRAC, "frame_minus_gt_s": -MAX_GT_FRAME_GAP_S}, "keep"),
])
def test_a_side_is_left_out_for_the_first_reason_of_the_rule_that_applies(fields, reason):
    assert classify(side(**fields)) == reason


def test_the_population_lists_the_kept_sides_and_counts_each_reason():
    population = select(census())
    # The keys keep the order the workbench wrote them in, and the reasons the order they first occur in.
    assert list(population) == ["n_clips_total", "n_sides_total", "exclusions", "n_sides_kept", "n_clips_kept",
                                "n_clips_moved", "n_clips_static", "n_sessions_kept", "n_specimens_kept",
                                "thresholds", "sides"]
    assert list(population["exclusions"].items()) == [("keep", 3), ("tissue_moving", 1), ("no_gt", 1),
                                                       ("gt_not_visible", 1), ("gt_stale", 2)]
    assert population == {
        "n_clips_total": 4, "n_sides_total": 8,
        "exclusions": population["exclusions"],
        "n_sides_kept": 3, "n_clips_kept": 2, "n_clips_moved": 1, "n_clips_static": 1,
        "n_sessions_kept": 2, "n_specimens_kept": 2,
        "thresholds": {"min_gt_block_frac": 0.2, "exclude_active": True, "max_gt_frame_gap_s": 10.0},
        "sides": [
            {"key": "specimen_1/s_a/Clip_1", "side": "start", "moved_camera": False, "specimen": "specimen_1"},
            {"key": "specimen_1/s_a/Clip_1", "side": "end", "moved_camera": False, "specimen": "specimen_1"},
            {"key": "specimen_2/s_c/Clip_1", "side": "end", "moved_camera": True, "specimen": "specimen_2"},
        ],
    }


def test_main_writes_the_population_and_the_clips_with_a_kept_side(tmp_path):
    c = census()
    main(write_inputs(tmp_path, c, keys(c)))
    out = tmp_path / "out"
    assert (out / "population.json").read_text() == json.dumps(select(c), indent=1)
    assert (out / "clips.txt").read_text() == "specimen_1/s_a/Clip_1\nspecimen_2/s_c/Clip_1\n"


def test_a_census_cut_short_is_refused(tmp_path):
    c = census()
    with pytest.raises(ValueError, match="lacks 2 of the 4 clips"):
        main(write_inputs(tmp_path, c[:2], keys(c)))
    assert not (tmp_path / "out").exists()


def test_a_census_that_lacks_one_clip_is_refused():
    c = census()
    with pytest.raises(ValueError, match="lacks 1 of the 4 clips it must hold, the first being specimen_1/s_b/Clip_1"):
        check_census(c[:2] + c[3:], keys(c))


def test_a_clip_the_list_does_not_hold_is_refused():
    c = census()
    with pytest.raises(ValueError, match=r"clips the list does not: \['specimen_2/s_c/Clip_1'\]"):
        check_census(c, keys(c)[:3])


def test_a_clip_held_twice_is_refused():
    c = census()
    with pytest.raises(ValueError, match=r"holds a clip twice: \['specimen_1/s_a/Clip_2'\]"):
        check_census(c + [c[1]], keys(c))


def test_a_side_that_carries_an_error_is_refused():
    # The census recorded a side it failed to measure as one without a point cloud, with the error beside it.
    c = census()
    c[0]["end"] = {"present": False, "error": "ValueError: empty point cloud"}
    with pytest.raises(ValueError, match="specimen_1/s_a/Clip_1 end: the census failed to measure the side"):
        check_census(c, keys(c))


def test_a_side_with_a_point_cloud_and_no_active_is_refused():
    # The census left `active` out when it found no record of the tissue's motion, and the rule kept such a side.
    c = census()
    del c[3]["end"]["active"]
    assert classify(side()) == "keep"
    with pytest.raises(ValueError, match="specimen_2/s_c/Clip_1 end: the side has a point cloud and no `active`"):
        check_census(c, keys(c))


def test_the_list_of_clips_holds_271_clips_once_in_order():
    clips = read_clips(str(META / "census_clips.txt"))
    assert len(clips) == len(set(clips)) == 271
    assert clips == sorted(clips)
    assert all(re.fullmatch(r"specimen_\d/\d{4}_\d\d_\d\d-\d\d_\d\d_\d\d/Clip_\d+", k) for k in clips)


def test_the_committed_population_was_selected_by_this_rule_from_every_listed_clip():
    population = json.loads((META / "population.json").read_text(encoding="utf-8"))
    clips = read_clips(str(META / "census_clips.txt"))
    assert population["thresholds"] == select([])["thresholds"]
    assert population["n_clips_total"] == len(clips)
    assert sum(population["exclusions"].values()) == population["n_sides_total"] == 2 * len(clips)
    assert population["exclusions"]["keep"] == population["n_sides_kept"] == len(population["sides"])
    kept = kept_clips(population)
    assert set(kept) <= set(clips)
    assert population["n_clips_kept"] == len(kept) == population["n_clips_moved"] + population["n_clips_static"]


def test_clips_txt_lists_the_clips_of_the_committed_population():
    population = json.loads((META / "population.json").read_text(encoding="utf-8"))
    assert (META / "clips.txt").read_text(encoding="utf-8") == "\n".join(kept_clips(population)) + "\n"


def test_no_file_carries_a_path_an_email_address_or_untranslated_text():
    for p in sorted(META.iterdir()):
        hits = PRIVATE.findall(p.read_text(encoding="utf-8"))
        assert not hits, (p.name, hits)


@pytest.mark.parametrize("text", ["/home/someone/datasets", "/var/autofs/data", "someone@example.com",
                                  "採点に使う"])
def test_the_private_pattern_finds_what_it_is_for(text):
    assert PRIVATE.search(text), text
