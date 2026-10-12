"""The experiment plan: the committed plan is controlled, a plan that is not is refused, and names, options and
records agree with the stages that make the conditions."""
import copy
import json

import pytest

# The plan builds on `pipeline.condition_population`, whose stages draw the inputs with matplotlib: the `render` extra.
pytest.importorskip("matplotlib")

from evalkit.tools.condition_names import read_table  # noqa: E402
from pipeline.condition_population import build_parser, expected_record  # noqa: E402
from pipeline.experiment_plan import (  # noqa: E402
    PLAN, check_plan, command, conditions, label_dir, main, name_of_record, stage_options, status,
)

COMMITTED = json.loads(PLAN.read_text(encoding="utf-8"))
TABLE = read_table()


def plan_with(**changes) -> dict:
    plan = copy.deepcopy(COMMITTED)
    plan.update(changes)
    return plan


def comparison(pairs, varies, datasets=("atlas120k",)) -> dict:
    return {"what": "a planted comparison", "varies": list(varies), "datasets": list(datasets),
            "pairs": [list(p) for p in pairs]}


def test_the_committed_plan_is_controlled():
    check_plan(COMMITTED, TABLE)


def test_a_pair_that_varies_another_part_is_refused():
    # The per-frame side has a grid of 24, the tracked side of 8: the grid varies with the rule.
    plan = plan_with(compare=[comparison([("rgb_pps24_perframe", "rgb_pps8_sam_centre")], ["seed", "rule"])])
    with pytest.raises(ValueError, match=r"vary \['points_per_side', 'rule', 'seed'\]"):
        check_plan(plan, TABLE)


def test_a_pair_that_leaves_a_named_part_alone_is_refused():
    plan = plan_with(compare=[comparison([("rgb_pps24_perframe", "rgb_pps24_sam_centre")], ["seed", "rule", "depth"])])
    with pytest.raises(ValueError, match="varies \\['depth', 'rule', 'seed'\\]"):
        check_plan(plan, TABLE)


def test_a_compared_condition_the_plan_does_not_make_and_the_table_does_not_name_is_refused():
    # The plan makes no grid of 6, and CholecSeg8k's workbench conditions hold no Pi3X arm.
    plan = plan_with(compare=[comparison([("normal_da3_pps6_perframe", "normal_pi3x_pps6_perframe")], ["depth"],
                                         datasets=["cholecseg8k"])])
    with pytest.raises(ValueError, match="cholecseg8k: normal_pi3x_pps6_perframe is neither made"):
        check_plan(plan, TABLE)


def test_an_unknown_dataset_or_part_is_refused():
    plan = plan_with(compare=[comparison([("rgb_pps24_perframe", "rgb_pps24_sam_centre")], ["seed", "propagation"],
                                         datasets=["lapex"])])
    with pytest.raises(ValueError, match="'lapex' is no dataset") as e:
        check_plan(plan, TABLE)
    assert "['propagation'] are no parts of a name" in str(e.value)


def test_a_group_names_each_input_at_each_grid_and_each_depth_model_only_where_depth_is_read():
    # Per grid and rule: rgb once, the four inputs that read depth on DA3 and on Pi3X.
    names = conditions(COMMITTED, "atlas120k")
    assert len(names) == 9 * (6 + 1 + 1) and names == conditions(COMMITTED, "cholecseg8k")
    assert {"rgb_pps2_sam_forward", "depth_pi3x_pps48_sam_forward", "rgb-edge_da3_pps24_perframe",
            "normal_pi3x_pps24_sam_centre"} <= set(names)
    assert not any(n.startswith("rgb_da3") or n.startswith("rgb_pi3x") for n in names)


def test_a_stage_names_only_its_own_groups():
    by_stage = {s: conditions(COMMITTED, "cholecseg8k", s) for s in (1, 2, 3)}
    assert [len(by_stage[s]) for s in (1, 2, 3)] == [9 * 4, 9 * 3, 9]
    assert sorted(sum(by_stage.values(), [])) == sorted(conditions(COMMITTED, "cholecseg8k"))
    assert all("_pps2_" in n for n in by_stage[3])


@pytest.mark.parametrize("stage", [0, "1", True, None])
def test_a_group_without_a_whole_stage_is_refused(stage):
    group = {**copy.deepcopy(COMMITTED["make"][0]), "stage": stage}
    with pytest.raises(ValueError, match="is not a whole number from 1"):
        conditions(plan_with(make=[group]), "atlas120k")


def test_a_condition_made_twice_is_refused():
    twice = copy.deepcopy(COMMITTED["make"][0])
    with pytest.raises(ValueError, match="more than once"):
        conditions(plan_with(make=[twice, twice]), "atlas120k")


def test_a_name_gives_the_stage_and_every_option_the_name_says():
    stage, options = stage_options("normal-edge_pi3x_pps8_sam_forward_keepring")
    assert stage == "track"
    assert options == ["--sam-input", "normal_edge", "--depth-source", "pi3", "--points-per-side", "8",
                       "--seed-min-area", "400", "--keep-edge-ring", "--rule", "forward_from_first",
                       "--track-base", "rgb", "--seed-edge-gain", "1.0", "--seed-no-smooth"]
    assert stage_options("rgb_pps24_perframe") == (
        "per_frame", ["--sam-input", "rgb", "--depth-source", "da3", "--points-per-side", "24", "--seed-min-area",
                      "400", "--edge-gain", "1.0"])
    assert label_dir("rgb_pps24_perframe") == "track_rgb_rgb_pps24_perframe"


@pytest.mark.parametrize("name, why", [
    ("rgb_pps24_kmerge10_centre", "made outside the stages"),
    ("rgb_pps24_unmerged_centre", "made outside the stages"),
    ("normal-edge_da3_pps24_sam_centre_keepring_instrument-text", "not ported"),
])
def test_a_condition_the_stages_cannot_make_from_its_name_is_refused(name, why):
    with pytest.raises(ValueError, match=why):
        stage_options(name)


def record_of(name: str, **seed_input_changes) -> dict:
    """The record the stage writes for one clip of the named condition, from the options the name gives."""
    stage, options = stage_options(name)
    s = build_parser().parse_args([stage, "--input-dir", "d", "--clips", "c", "--tracks-root", "t", "--tag", name,
                                   "--gpus", "0", *options, "--sam-ckpt", "sam.pth"])
    record = json.loads(json.dumps(expected_record(stage, s)))
    record["seed_input"].update(seed_input_changes)
    if stage == "per_frame":
        return {**record, "clip": "VID01_a"}
    seed = 2 if record["bidir"] else 0
    return {**record, "clip": "VID01_a", "seed_frame": seed, "n_seed_regions": 7, "frames": [0, 1, 2, 3, 4]}


def test_the_record_of_every_planned_condition_shows_its_name():
    for dataset in ("atlas120k", "cholecseg8k"):
        for name in conditions(COMMITTED, dataset):
            assert name_of_record(record_of(name)) == name


def test_a_record_shows_a_kept_ring_and_the_pi3x_depth():
    for name in ("normal-edge_da3_pps8_sam_centre_keepring", "normal_pi3x_pps8_perframe"):
        assert name_of_record(record_of(name)) == name


@pytest.mark.parametrize("change, why", [
    ({"seed_edge_gain": 0.85}, "edges at gain 0.85"),
    ({"seed_smooth": True}, "smoothed normals"),
])
def test_a_record_with_the_tracking_stages_own_edge_defaults_shows_no_name(change, why):
    # Run without the name's options, the tracking stage draws the edges at 0.85 on smoothed normals.
    with pytest.raises(ValueError, match=why):
        name_of_record(record_of("normal-edge_da3_pps8_sam_centre", **change))


def test_the_edge_defaults_are_no_setting_in_an_input_without_edges():
    assert name_of_record(record_of("normal_da3_pps8_sam_centre", seed_edge_gain=0.85, seed_smooth=True)) == \
        "normal_da3_pps8_sam_centre"


def test_a_record_off_the_other_defaults_or_seeded_from_gt_shows_no_name():
    with pytest.raises(ValueError, match="seed regions of 300 pixels"):
        name_of_record({**record_of("rgb_pps24_sam_centre"), "seed_min_area": 300})
    with pytest.raises(ValueError, match="the tracker's input 'normal'"):
        name_of_record({**record_of("rgb_pps24_sam_centre"), "track_base": "normal"})
    with pytest.raises(ValueError, match="not from the tracker's own masks"):
        name_of_record({**record_of("rgb_pps24_sam_centre"), "seed_source": "gt"})


def write_record(tracks_root, clip, name, record):
    d = tracks_root / clip / label_dir(name)
    d.mkdir(parents=True)
    (d / "seed_info.json").write_text(json.dumps(record))


def test_status_counts_made_and_missing_clips_and_names_a_clip_made_as_another_condition(tmp_path):
    good, drifted = "rgb_pps4_sam_forward", "normal-edge_da3_pps4_sam_forward"
    write_record(tmp_path, "VID01_a", good, record_of(good))
    write_record(tmp_path, "VID01_a", drifted, record_of(drifted, seed_edge_gain=0.85))
    report = status(COMMITTED, "atlas120k", ["VID01_a", "VID02_b"], tmp_path)
    assert (report[good]["made"], report[good]["missing"], report[good]["other"]) == (["VID01_a"], ["VID02_b"], {})
    assert report[drifted]["other"]["VID01_a"].startswith("no name: the record holds settings no name says")


def run(tmp_path, *argv):
    (tmp_path / "clips.txt").write_text("VID01_a\n")
    main([*argv, "--dataset", "atlas120k", "--clips", str(tmp_path / "clips.txt"), "--tracks-root", str(tmp_path)])


def test_commands_print_only_the_conditions_still_to_make(tmp_path, capsys):
    made = "rgb_pps4_sam_forward"
    write_record(tmp_path, "VID01_a", made, record_of(made))
    run(tmp_path, "commands", "--input-dir", "/clips", "--sam-ckpt", "sam.pth", "--gpus", "0", "1")
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 72 - 1 and not any(f"--tag {made} " in line for line in lines)
    assert lines[0] == command("rgb_pps24_sam_forward", "/clips", str(tmp_path / "clips.txt"), str(tmp_path),
                               "sam.pth", [0, 1])


def test_commands_print_a_stage_alone(tmp_path, capsys):
    run(tmp_path, "commands", "--input-dir", "/clips", "--sam-ckpt", "sam.pth", "--gpus", "0", "--stage", "3")
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 9 and all("--points-per-side 2 " in line and "forward_from_first" in line for line in lines)


def test_commands_refuse_labels_made_as_another_condition(tmp_path):
    drifted = "normal-edge_da3_pps4_sam_forward"
    write_record(tmp_path, "VID01_a", drifted, record_of(drifted, seed_edge_gain=0.85))
    with pytest.raises(SystemExit, match=f"{drifted}: VID01_a: no name"):
        run(tmp_path, "commands", "--input-dir", "/clips", "--sam-ckpt", "sam.pth", "--gpus", "0")


def test_commands_print_every_condition_with_all(tmp_path, capsys):
    made = "rgb_pps4_sam_forward"
    write_record(tmp_path, "VID01_a", made, record_of(made))
    run(tmp_path, "commands", "--input-dir", "/clips", "--sam-ckpt", "sam.pth", "--gpus", "0", "--all")
    assert len(capsys.readouterr().out.splitlines()) == 72


def test_status_prints_each_conditions_counts(tmp_path, capsys):
    made = "rgb_pps4_sam_forward"
    write_record(tmp_path, "VID01_a", made, record_of(made))
    run(tmp_path, "status")
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 72 and lines[0].split() == [made, "1", "made,", "0", "missing"]
    assert lines[1].split() == ["rgb_pps24_sam_forward", "0", "made,", "1", "missing"]
