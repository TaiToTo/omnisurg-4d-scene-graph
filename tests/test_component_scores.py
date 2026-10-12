"""Component objects on frames one pixel high, each value worked out by hand, and the tool's refusals.

A frame is drawn as two strings of ten pixels: its GT, one letter per ATLAS-120k class (`L` Liver, `G`
Gallbladder, `T` Tools/camera, `X` the marker of an excluded frame), and its regions, one digit per region id,
`.` for no region. Every pixel has depth. The tool is called with the minimum sizes 0 and 4, which a frame of ten
pixels can tell apart.
"""
import json

import numpy as np
import pytest

import clip_dirs as C
from evalkit.classes import load_table
from evalkit.frame import score_view as evaluator_score_view
from evalkit.inputs import DEPTH_FILE
from evalkit.scored import scored_pixels
from evalkit.tools import component_scores
from evalkit.tools.component_scores import (
    component_objects, main, measure_condition, region_objects, score_view, tool_sha,
)
from evalkit.tools.condition_inventory import read_evals
from evalkit.tools.scores import ruler

TABLE = load_table("atlas120k")
LIVER, GALLBLADDER, TOOL, EXCLUDED = 12, 14, 1, 42
CODES = {"L": LIVER, "G": GALLBLADDER, "T": TOOL, "X": EXCLUDED}
SIZES = (0, 4)
ALL_VALID = np.ones((1, 10), dtype=bool)


def gt_map(s: str) -> np.ndarray:
    return np.array([[CODES[ch] for ch in s]], dtype=np.int32)


def region_map(s: str) -> np.ndarray:
    return np.array([[-1 if ch == "." else int(ch) for ch in s]], dtype=np.int32)


def scored(gt: str, view: str):
    return scored_pixels(gt_map(gt), TABLE, view, ALL_VALID)


# The liver cut in two by an instrument: pieces of 3 and 4 pixels. Regions 0 and 2 cover the two pieces, region 1
# the instrument.
CUT_LIVER = ("LLLTTTLLLL", "0001112222")
# Region 0 covers the liver and two gallbladder pixels (IoU 5/7), region 1 the other three (IoU 3/5).
SPILL = ("LLLLLGGGGG", "0000000111")


def test_a_class_cut_in_two_is_two_objects_of_one_class():
    s = scored(CUT_LIVER[0], "geometric")
    gt = component_objects(s.classes, s.mask, 0)
    assert gt.ids == (LIVER, LIVER) and gt.areas == (3, 4)
    every = scored(CUT_LIVER[0], "all")
    assert component_objects(every.classes, every.mask, 0).ids == (TOOL, LIVER, LIVER)


def test_pixels_touching_at_a_corner_are_one_piece():
    # The liver on the diagonal of a 2 x 2 frame: one piece of 2 pixels under 8-connectivity.
    classes = np.array([[LIVER, GALLBLADDER], [GALLBLADDER, LIVER]], dtype=np.int32)
    gt = component_objects(classes, np.ones((2, 2), dtype=bool), 0)
    assert gt.ids == (LIVER, GALLBLADDER) and gt.areas == (2, 2)


def test_the_minimum_size_drops_small_pieces_and_small_regions_alike():
    s = scored(CUT_LIVER[0], "geometric")
    assert component_objects(s.classes, s.mask, 4).areas == (4,)
    # Region 1 lies on the instrument, which the geometric view does not score, so it is no object at any size.
    assert region_objects(region_map(CUT_LIVER[1]), s.mask, 0).ids == (0, 2)
    assert region_objects(region_map(CUT_LIVER[1]), s.mask, 4).ids == (2,)


def test_a_region_on_each_piece_finds_both_where_the_evaluator_finds_one():
    gt, regions = gt_map(CUT_LIVER[0]), region_map(CUT_LIVER[1])
    ours = score_view(gt, regions, ALL_VALID, TABLE, "geometric", SIZES)
    assert (ours[0]["F1_50"], ours[0]["SQ"], ours[0]["hits"]) == (1.0, 1.0, 2)
    # At 4 pixels the 3-pixel piece and region 0 both go: one object each side, one hit.
    assert (ours[4]["gt"], ours[4]["predicted"], ours[4]["F1_50"]) == (1, 1, 1.0)
    # The evaluator's whole liver of 7 pixels: region 2 hits it at 4/7, region 0 is a false positive.
    theirs = evaluator_score_view(gt, regions, ALL_VALID, TABLE, "geometric")
    assert theirs.f1_50 == pytest.approx(2 / 3) and theirs.sq == pytest.approx(4 / 7)


def test_the_minimum_size_can_take_a_hit_away():
    ours = score_view(gt_map(SPILL[0]), region_map(SPILL[1]), ALL_VALID, TABLE, "geometric", SIZES)
    assert ours[0]["F1_50"] == 1.0 and ours[0]["SQ"] == pytest.approx((5 / 7 + 3 / 5) / 2)
    # Region 1 has 3 pixels, so at 4 the gallbladder is left without a region.
    assert (ours[4]["hits"], ours[4]["F1_50"], ours[4]["SQ"]) == (1, pytest.approx(2 / 3), pytest.approx(5 / 7))


def test_a_view_without_a_class_leaves_the_keys_undefined():
    ours = score_view(gt_map("TTTTTTTTTT"), region_map("0000000000"), ALL_VALID, TABLE, "geometric", SIZES)
    assert (ours[0]["F1_50"], ours[0]["SQ"], ours[0]["inst_BF"], ours[0]["predicted"]) == (None, None, None, 0)


def write_condition(tmp_path, frames, gt_frames=None):
    C.write_clip(tmp_path / "data", "VID01_a", [gt_map(g) for g, _ in frames], gt_frames=gt_frames,
                 dataset="atlas120k")
    C.write_labels(tmp_path / "tracks", "VID01_a", "c", {k: region_map(r) for k, (_, r) in enumerate(frames)})


def measure(tmp_path):
    return measure_condition("atlas120k", None, ["VID01_a"], tmp_path / "data", tmp_path / "tracks", "c",
                             "per_frame", SIZES)


def test_a_clip_on_disk_averages_each_size_over_its_scored_frames(tmp_path):
    write_condition(tmp_path, [CUT_LIVER, ("XLLLLLLLLL", "0000000000"), SPILL, ("TTTTTTTTTT", "0000000000")])
    summary = measure(tmp_path)
    assert summary["min_px"] == [0, 4] and summary["propagation"] == "per_frame"
    row = summary["per_clip"][0]
    assert (row["n_scored_frames"], row["n_excluded_frames"]) == (3, 1)
    at0, at4 = row["by_min_px"]["0"], row["by_min_px"]["4"]
    assert at0["F1_50/geometric"] == 1.0 and at0["SQ/geometric"] == pytest.approx((1 + (5 / 7 + 3 / 5) / 2) / 2)
    assert at4["F1_50/geometric"] == pytest.approx((1 + 2 / 3) / 2) and at4["SQ/geometric"] == pytest.approx((1 + 5 / 7) / 2)
    # The frame of instrument alone has no class in the geometric view, so the geometric means cover two frames.
    assert at0["n_frames"]["F1_50/geometric"] == 2 and at0["n_frames"]["F1_50/all"] == 3
    # In the all view every frame's instrument is an object, and every region hits.
    assert at0["F1_50/all"] == 1.0
    assert {k: at0["objects"]["all"][k] for k in ("gt", "predicted", "hits")} == {"gt": 6, "predicted": 6, "hits": 6}


def test_the_json_records_the_hash_of_the_tool(tmp_path):
    write_condition(tmp_path, [SPILL])
    assert measure(tmp_path)["tool_sha"] == tool_sha(component_scores.__file__)


def test_a_clip_whose_every_gt_frame_is_excluded_is_refused(tmp_path):
    write_condition(tmp_path, [("XLLLLLLLLL", "0000000000")] * 2)
    with pytest.raises(ValueError, match="every GT frame is excluded"):
        measure(tmp_path)


def test_an_excluded_frame_with_faulty_depth_is_refused(tmp_path):
    write_condition(tmp_path, [SPILL, ("XLLLLLLLLL", "0000000000")])
    path = tmp_path / "data" / "VID01_a" / DEPTH_FILE
    depth = np.load(path)["depth"]
    depth[1, 0, 5] = np.nan
    np.savez(path, depth=depth)
    with pytest.raises(ValueError, match="no valid depth"):
        measure(tmp_path)


def test_labels_without_a_rule_are_refused(tmp_path):
    write_condition(tmp_path, [SPILL])
    with pytest.raises(ValueError, match="state the rule"):
        measure_condition("atlas120k", None, ["VID01_a"], tmp_path / "data", tmp_path / "tracks", "c", None, SIZES)


def test_the_score_tools_refuse_its_json_as_a_score(tmp_path):
    write_condition(tmp_path, [SPILL])
    summary = measure(tmp_path)
    with pytest.raises(ValueError, match="so this one is neither"):
        ruler(summary)
    (tmp_path / "scores").mkdir()
    (tmp_path / "scores" / "c.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="so this one is neither"):
        read_evals(str(tmp_path / "scores"))


def test_the_command_line_writes_the_json_at_every_fixed_size(tmp_path, capsys):
    write_condition(tmp_path, [SPILL])
    (tmp_path / "clips.txt").write_text("VID01_a\n")
    out = tmp_path / "c.components.json"
    main(["--dataset", "atlas120k", "--clips", str(tmp_path / "clips.txt"), "--data-root", str(tmp_path / "data"),
          "--tracks-root", str(tmp_path / "tracks"), "--tag", "c", "--out", str(out), "--propagation", "per_frame"])
    written = json.loads(out.read_text())
    assert written["min_px"] == [0, 100, 300, 400] and set(written["per_clip"][0]["by_min_px"]) == {"0", "100", "300", "400"}
    # At 100 pixels and more nothing of a ten-pixel frame is left, so no key is defined.
    assert written["per_clip"][0]["by_min_px"]["0"]["F1_50/geometric"] == 1.0
    assert written["per_clip"][0]["by_min_px"]["100"]["F1_50/geometric"] is None
    assert "1 clips scored" in capsys.readouterr().out
