"""The experiment plan: the committed plan is controlled, a plan that is not is refused, and names, settings and
records agree with the stages that make the conditions."""
import copy
import json

import pytest

# The plan builds on `pipeline.condition_population`, whose stages draw the inputs with matplotlib: the `render` extra.
pytest.importorskip("matplotlib")

from pipeline.condition_population import build_parser, expected_record  # noqa: E402
from pipeline.experiment_plan import (  # noqa: E402
    PLAN, check_plan, command, conditions, label_dir, main, settings_of, status,
)

COMMITTED = json.loads(PLAN.read_text(encoding="utf-8"))


def plan_with(**changes) -> dict:
    plan = copy.deepcopy(COMMITTED)
    plan.update(changes)
    return plan


def comparison(pairs, varies, datasets=("atlas120k",)) -> dict:
    return {"what": "a planted comparison", "varies": list(varies), "datasets": list(datasets),
            "pairs": [list(p) for p in pairs]}


def test_the_committed_plan_is_controlled():
    check_plan(COMMITTED)


def test_a_pair_that_varies_another_part_is_refused():
    # The per-frame side has a grid of 24, the tracked side of 8: the grid varies with the rule.
    plan = plan_with(compare=[comparison([("rgb_pps24_perframe", "rgb_pps8_sam_centre")], ["seed", "rule"])])
    with pytest.raises(ValueError, match=r"vary \['points_per_side', 'rule', 'seed'\]"):
        check_plan(plan)


def test_a_pair_that_leaves_a_named_part_alone_is_refused():
    plan = plan_with(compare=[comparison([("rgb_pps24_perframe", "rgb_pps24_sam_centre")], ["seed", "rule", "depth"])])
    with pytest.raises(ValueError, match="varies \\['depth', 'rule', 'seed'\\]"):
        check_plan(plan)


def test_a_pair_under_two_rules_that_varies_more_than_the_rule_is_refused():
    # The pair varies exactly the parts it names, and is still refused: two rules are compared on the rule alone.
    pair = ("rgb_pps24_sam_forward", "rgb_pps4_sam_centre")
    plan = plan_with(compare=[comparison([pair], ["points_per_side", "rule"])])
    with pytest.raises(ValueError, match=r"differ in \['points_per_side'\] too") as e:
        check_plan(plan)
    assert "vary" not in str(e.value)


def test_a_compared_condition_the_plan_does_not_make_is_refused():
    # The plan makes no grid of 6.
    plan = plan_with(compare=[comparison([("normal_da3_pps6_perframe", "normal_pi3x_pps6_perframe")], ["depth"],
                                         datasets=["cholecseg8k"])])
    with pytest.raises(ValueError, match="cholecseg8k: the plan does not make normal_da3_pps6_perframe") as e:
        check_plan(plan)
    assert "cholecseg8k: the plan does not make normal_pi3x_pps6_perframe" in str(e.value)


def test_an_unknown_dataset_or_part_is_refused():
    plan = plan_with(compare=[comparison([("rgb_pps24_perframe", "rgb_pps24_sam_centre")], ["seed", "propagation"],
                                         datasets=["lapex"])])
    with pytest.raises(ValueError, match="'lapex' is no dataset") as e:
        check_plan(plan)
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


def test_a_stage_of_no_group_is_refused():
    with pytest.raises(ValueError, match=r"no group is of stage 4; the stages are \[1, 2, 3\]"):
        conditions(COMMITTED, "cholecseg8k", 4)


@pytest.mark.parametrize("stage", [0, "1", True, None])
def test_a_group_without_a_whole_stage_is_refused(stage):
    group = {**copy.deepcopy(COMMITTED["make"][0]), "stage": stage}
    with pytest.raises(ValueError, match="is not a whole number from 1"):
        conditions(plan_with(make=[group]), "atlas120k")


@pytest.mark.parametrize("empty", ["inputs", "depths", "points_per_side"])
def test_a_group_whose_list_names_nothing_is_refused(empty):
    group = {**copy.deepcopy(COMMITTED["make"][0]), empty: []}
    with pytest.raises(ValueError, match=rf"\['{empty}'\] name nothing"):
        conditions(plan_with(make=[group]), "atlas120k")


def test_a_group_of_inputs_that_read_no_depth_needs_no_depth_model():
    group = {**copy.deepcopy(COMMITTED["make"][0]), "inputs": ["rgb"], "depths": []}
    assert conditions(plan_with(make=[group]), "atlas120k") == ["rgb_pps4_sam_forward", "rgb_pps24_sam_forward"]


def test_a_condition_made_twice_is_refused():
    twice = copy.deepcopy(COMMITTED["make"][0])
    with pytest.raises(ValueError, match="more than once"):
        conditions(plan_with(make=[twice, twice]), "atlas120k")


def parsed(name: str, sam_ckpt: str = "sam.pth"):
    """The settings `condition_population`'s own parser reads from the command of the named condition."""
    argv = command(name, "/clips", "clips.txt", "/tracks", sam_ckpt, [0, 1]).split()
    return argv[3], build_parser().parse_args(argv[3:])


def test_a_command_gives_every_setting_the_name_says_and_the_stage_reads_them_back():
    name = "normal-edge_pi3x_pps8_sam_forward_keepring"
    assert command(name, "/clips", "clips.txt", "/tracks", "sam.pth", [0, 1]) == (
        "python -m pipeline.condition_population track --input-dir /clips --clips clips.txt --tracks-root /tracks "
        "--tag normal-edge_pi3x_pps8_sam_forward_keepring --gpus 0 1 --sam-input normal_edge --depth-source pi3 "
        "--points-per-side 8 --seed-min-area 400 --rule forward_from_first --track-base rgb --seed-edge-gain 1.0 "
        "--seed-no-smooth --keep-edge-ring --sam-ckpt sam.pth")
    assert command("rgb_pps24_perframe", "/clips", "clips.txt", "/tracks", "sam.pth", [2]) == (
        "python -m pipeline.condition_population per_frame --input-dir /clips --clips clips.txt --tracks-root /tracks "
        "--tag rgb_pps24_perframe --gpus 2 --sam-input rgb --depth-source da3 --points-per-side 24 --seed-min-area 400 "
        "--edge-gain 1.0 --sam-ckpt sam.pth")
    for dataset in ("atlas120k", "cholecseg8k"):
        for name in conditions(COMMITTED, dataset):
            stage, s = settings_of(name)
            read_stage, read = parsed(name)
            assert read_stage == stage and expected_record(stage, read) == expected_record(stage, s)
    assert label_dir("rgb_pps24_perframe") == "track_rgb_rgb_pps24_perframe"


@pytest.mark.parametrize("name, why", [
    ("rgb_pps24_kmerge10_centre", "made outside the stages"),
    ("rgb_pps24_unmerged_centre", "made outside the stages"),
    ("normal-edge_da3_pps24_sam_centre_keepring_instrument-text", "not ported"),
])
def test_a_condition_the_stages_cannot_make_from_its_name_is_refused(name, why):
    with pytest.raises(ValueError, match=why):
        settings_of(name)


def record_of(name: str, **changes) -> dict:
    """The record the stage writes for one clip of the named condition, from the options the command gives."""
    stage, s = parsed(name)
    record = json.loads(json.dumps(expected_record(stage, s)))
    record["seed_input"].update(changes.pop("seed_input", {}))
    record.update(changes)
    if stage == "per_frame":
        return {**record, "clip": "VID01_a"}
    seed = 2 if record["bidir"] else 0
    return {**record, "clip": "VID01_a", "seed_frame": seed, "n_seed_regions": 7, "frames": [0, 1, 2, 3, 4]}


def write_record(tracks_root, clip, name, record):
    d = tracks_root / clip / label_dir(name)
    d.mkdir(parents=True)
    (d / "seed_info.json").write_text(json.dumps(record) if isinstance(record, dict) else record)


def test_status_counts_made_and_missing_clips_and_names_a_clip_made_otherwise(tmp_path):
    good, drifted = "rgb_pps4_sam_forward", "normal-edge_da3_pps4_sam_forward"
    write_record(tmp_path, "VID01_a", good, record_of(good))
    # Run without the name's options, the tracking stage draws the edges at 0.85 on smoothed normals.
    write_record(tmp_path, "VID01_a", drifted,
                 record_of(drifted, seed_input={"seed_edge_gain": 0.85, "seed_smooth": True}))
    report = status(COMMITTED, "atlas120k", ["VID01_a", "VID02_b"], tmp_path)
    assert (report[good]["made"], report[good]["missing"], report[good]["other"]) == (["VID01_a"], ["VID02_b"], {})
    said = report[drifted]["other"]["VID01_a"]
    assert said.startswith("VID01_a's seed_info.json records other settings") and "seed_edge_gain" in said


@pytest.mark.parametrize("name, changes, said", [
    ("rgb_pps4_sam_forward", {"seed_input": {"seed_sam_kwargs": {"pred_iou_thresh": 0.5}}}, "seed_sam_kwargs"),
    ("rgb_pps4_sam_forward", {"stride": 2}, "stride 2, not 1"),
    ("rgb_pps4_sam_forward", {"seed_min_area": 300}, "seed_min_area 300, not 400"),
    ("rgb_pps4_sam_forward", {"track_base": "normal"}, "track_base 'normal', not 'rgb'"),
    ("rgb_pps4_sam_forward", {"seed_source": "gt"}, "seed_source 'gt', not 'sam'"),
    ("rgb_pps4_sam_forward", {"bidir": True}, "bidir True, not False"),
    ("rgb_pps24_perframe", {"seed_topk": 3}, "seed_topk 3, not 0"),
    ("rgb_pps24_perframe", {"tag": "rgb_pps24_perframe_old"}, "tag 'rgb_pps24_perframe_old'"),
    ("normal_pi3x_pps24_perframe", {"depth_source": "da3"}, "depth_source 'da3', not 'pi3'"),
])
def test_status_names_a_record_whose_settings_are_not_the_names(tmp_path, name, changes, said):
    write_record(tmp_path, "VID01_a", name, record_of(name, **changes))
    assert said in status(COMMITTED, "atlas120k", ["VID01_a"], tmp_path)[name]["other"]["VID01_a"]


def test_status_names_a_record_that_is_not_json(tmp_path):
    write_record(tmp_path, "VID01_a", "rgb_pps4_sam_forward", "{")
    report = status(COMMITTED, "atlas120k", ["VID01_a"], tmp_path)
    assert "cannot be read" in report["rgb_pps4_sam_forward"]["other"]["VID01_a"]


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


def test_commands_print_a_stage_alone_and_refuse_a_stage_of_no_group(tmp_path, capsys):
    run(tmp_path, "commands", "--input-dir", "/clips", "--sam-ckpt", "sam.pth", "--gpus", "0", "--stage", "3")
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 9 and all("--points-per-side 2 " in line and "forward_from_first" in line for line in lines)
    with pytest.raises(SystemExit, match="--stage: no group is of stage 4"):
        run(tmp_path, "status", "--stage", "4")


def test_commands_refuse_labels_made_otherwise(tmp_path):
    drifted = "normal-edge_da3_pps4_sam_forward"
    write_record(tmp_path, "VID01_a", drifted, record_of(drifted, seed_input={"seed_edge_gain": 0.85}))
    with pytest.raises(SystemExit, match=f"other settings:\n  {drifted}: VID01_a's seed_info.json records other"):
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


def test_check_prints_that_the_plan_is_controlled(capsys):
    main(["check"])
    assert capsys.readouterr().out.startswith("experiment_plan.json: every comparison varies only the parts it names")
