"""Curves along regions per frame, on JSONs written by hand, each value worked out by hand, and their refusals.

Two inputs on two clips. `rgb` has 9, 16 and 24 regions per frame at the grids 4, 8 and 24, with F1 0.45, 0.40
and 0.35; `normal` has 6 and 10 at 4 and 8, with 0.55 and 0.50. Both curves reach 9 and 10 regions.
"""
import json

import pytest

from evalkit.tools.region_curves import curves_of, main, matched

CLIPS = ["VID01_a", "VID02_b"]
KEY = "F1_50/geometric"
# Each condition: regions per frame, and its two clips' F1, whose mean is the point's value.
POINTS = {
    "rgb_pps4_sam_centre": (9.0, (0.40, 0.50)),
    "rgb_pps8_sam_centre": (16.0, (0.35, 0.45)),
    "rgb_pps24_sam_centre": (24.0, (0.30, 0.40)),
    "normal_da3_pps4_sam_centre": (6.0, (0.50, 0.60)),
    "normal_da3_pps8_sam_centre": (10.0, (0.45, 0.55)),
}


def score(name, values, clips=CLIPS, pilot=False):
    return {"eval_code_sha": "e", "dataset": "atlas120k", "class_set": "original", "views": ["all"], "pilot": pilot,
            "input_shas": {}, "versions": {}, "propagation": "both_ways_from_centre", "clips": clips,
            "track_dir_name": f"track_rgb_{name}", "per_clip": [{"clip": c, KEY: v} for c, v in zip(clips, values)]}


def write(scores_dir, name, regions, values, clips=CLIPS, **score_changes):
    (scores_dir / f"{name}.regions.json").write_text(json.dumps(
        {"track_dir_name": f"track_rgb_{name}", "clips": clips, "regions_per_frame": regions}))
    (scores_dir / f"{name}.json").write_text(json.dumps({**score(name, values, clips), **score_changes}))
    (scores_dir / f"{name}.components.json").write_text(json.dumps(
        {"track_dir_name": f"track_rgb_{name}", "min_px": [0, 300],
         "per_clip": [{"clip": c, "by_min_px": {"0": {KEY: v and v / 4}, "300": {KEY: v and v / 2}}}
                      for c, v in zip(clips, values)]}))


@pytest.fixture
def scores(tmp_path):
    for name, (regions, values) in POINTS.items():
        write(tmp_path, name, regions, values)
    return tmp_path


def test_each_input_is_a_curve_of_its_grids_ordered_by_regions(scores):
    result = curves_of(scores, KEY, None)
    assert list(result["curves"]) == ["normal", "rgb"]
    # The grid orders the points, not the file names, where 24 comes before 4.
    assert [(p["points_per_side"], p["regions_per_frame"]) for p in result["curves"]["rgb"]] == [
        (4, 9.0), (8, 16.0), (24, 24.0)]
    assert [p["value"] for p in result["curves"]["normal"]] == pytest.approx([0.55, 0.50])


def test_the_curves_are_read_at_every_count_both_reach(scores):
    rows = curves_of(scores, KEY, None)["matched"]
    assert [r["regions_per_frame"] for r in rows] == [9, 10]
    # rgb from 0.45 at 9 to 0.40 at 16; normal from 0.55 at 6 to 0.50 at 10.
    assert (rows[0]["rgb"], rows[0]["normal"]) == pytest.approx((0.45, 0.5125))
    assert (rows[1]["rgb"], rows[1]["normal"]) == pytest.approx((0.45 - 0.05 / 7, 0.50))


def test_component_objects_are_read_at_the_size_asked(scores):
    assert curves_of(scores, KEY, 300)["curves"]["rgb"][0]["value"] == pytest.approx(0.225)
    assert curves_of(scores, KEY, 0)["curves"]["rgb"][0]["value"] == pytest.approx(0.1125)
    with pytest.raises(ValueError, match="not 400"):
        curves_of(scores, KEY, 400)


def test_a_condition_with_another_rule_or_setting_is_left_out(scores):
    write(scores, "rgb_pps24_sam_forward", 20.0, (0.1, 0.1))
    write(scores, "normal-edge_da3_pps4_sam_centre_keepring", 5.0, (0.9, 0.9))
    assert set(curves_of(scores, KEY, None)["curves"]) == {"normal", "rgb"}


def test_an_input_at_one_grid_only_is_refused(scores):
    write(scores, "depth_da3_pps4_sam_centre", 5.0, (0.3, 0.3))
    with pytest.raises(ValueError, match=r"\['depth'\] have one grid only"):
        curves_of(scores, KEY, None)


def test_a_file_whose_name_breaks_the_rule_of_names_is_refused(scores):
    write(scores, "rgb_center18", 20.0, (0.1, 0.1))
    with pytest.raises(ValueError, match="rgb_center18"):
        curves_of(scores, KEY, None)


def test_conditions_on_other_clips_are_refused(scores):
    write(scores, "rgb_pps8_sam_centre", 16.0, (0.35, 0.45), clips=["VID01_a", "VID03_c"])
    with pytest.raises(ValueError, match="different sets of clips"):
        curves_of(scores, KEY, None)


def test_a_score_of_other_labels_than_the_count_is_refused(scores):
    write(scores, "rgb_pps8_sam_centre", 16.0, (0.35, 0.45), track_dir_name="track_rgb_rgb_center18")
    with pytest.raises(ValueError, match="the score read 'track_rgb_rgb_center18'"):
        curves_of(scores, KEY, None)


def test_a_score_and_a_count_on_other_clips_are_refused(scores):
    (scores / "rgb_pps8_sam_centre.regions.json").write_text(json.dumps(
        {"track_dir_name": "track_rgb_rgb_pps8_sam_centre", "clips": ["VID01_a"], "regions_per_frame": 16.0}))
    with pytest.raises(ValueError, match="other clips than the region count"):
        curves_of(scores, KEY, None)


def test_a_clip_without_the_key_is_left_out_of_the_mean(scores):
    write(scores, "rgb_pps4_sam_centre", 9.0, (0.40, None))
    p = curves_of(scores, KEY, None)["curves"]["rgb"][0]
    assert (p["value"], p["n_clips"]) == (0.40, 1)


def test_a_pilot_score_is_refused(scores):
    write(scores, "rgb_pps8_sam_centre", 16.0, (0.35, 0.45), pilot=True)
    with pytest.raises(ValueError, match="pilot mode"):
        curves_of(scores, KEY, None)


def test_a_curve_that_is_no_function_of_the_count_or_has_a_hole_is_refused():
    with pytest.raises(ValueError, match="two grids give one region count"):
        matched({"rgb": [{"regions_per_frame": 9.0, "value": 0.4}, {"regions_per_frame": 9.0, "value": 0.5}]})
    with pytest.raises(ValueError, match="hole"):
        matched({"rgb": [{"regions_per_frame": 9.0, "value": 0.4}, {"regions_per_frame": 10.0, "value": None}]})


def test_the_readings_start_at_the_first_whole_count_every_curve_reaches():
    rows = matched({"a": [{"regions_per_frame": 2.5, "value": 0.1}, {"regions_per_frame": 5.5, "value": 0.4}],
                    "b": [{"regions_per_frame": 3.2, "value": 0.2}, {"regions_per_frame": 7.0, "value": 0.6}]})
    assert [r["regions_per_frame"] for r in rows] == [4, 5]


def test_curves_that_reach_no_common_count_give_no_reading():
    assert matched({"rgb": [{"regions_per_frame": 9.0, "value": 0.4}, {"regions_per_frame": 16.0, "value": 0.5}],
                    "depth": [{"regions_per_frame": 3.0, "value": 0.4}, {"regions_per_frame": 6.0, "value": 0.5}]}) == []


def test_the_command_line_prints_the_curves_and_writes_them(scores, capsys):
    out = scores / "curves.json"
    main(["--scores-dir", str(scores), "--key", KEY, "--out", str(out)])
    printed = capsys.readouterr().out
    assert "pps4: 9.00 regions 0.4500" in printed and "  9  normal 0.5125  rgb 0.4500" in printed
    assert json.loads(out.read_text())["matched"][1]["regions_per_frame"] == 10
