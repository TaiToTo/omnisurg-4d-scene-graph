"""The entry point end to end: a condition scored on clips written to disk, read back by the tools.

A prediction equal to the GT must score one, or zero for the variation of
information, on every key in every view. The JSON must pass the tools' own
readers, two conditions on the same inputs must be comparable, and one on
another GT must not. A run that fails writes nothing. The JSON records the
condition's propagation rule as every clip's `seed_info.json` gives it. A
seed that holds no rule, a seed from the GT among them, and a record on some
clips but not on the others are refused, in pilot mode too. Seeds made
outside the tracking stage are scored, recorded by their files' hash and by
whether the GT informed them, and compared only with seeds made alike.
"""
import json
import shutil

import numpy as np
import pytest

import clip_dirs as C
from evalkit.classes import VIEWS, load_table
from evalkit.code_sha import eval_code_sha
from evalkit.evaluate import check_table, main, read_population, score_condition, seed_file, write_scores
from evalkit.inputs import sha_of_files
from evalkit.keys import CLIP_METRICS, FRAME_METRICS, SIGNS, metric_key
from evalkit.time_iou import time_iou
from evalkit.tools.condition_inventory import read_evals
from evalkit.tools.scores import check_comparable, check_rows, load_scores, metric_keys, ruler

CLIPS = ["VID01_a", "VID02_b"]

# What the tracker records for a seed from its own masks on the centre of three frames, carried both
# ways, and on the first frame, carried forward.
BOTH_WAYS = {"seed_source": "sam", "seed_frame": 1, "bidir": True, "frames": [0, 1, 2]}
FORWARD = {"seed_source": "sam", "seed_frame": 0, "bidir": False, "frames": [0, 1, 2]}


def write_condition(tmp_path, tag, regions_of=lambda gt: gt.copy(), clips=CLIPS, data="data", seed_info=BOTH_WAYS):
    for k, clip in enumerate(clips):
        gt = [C.two_organs(cut=8 + 2 * i + k) for i in range(3)]
        if not (tmp_path / data / clip).exists():
            C.write_clip(tmp_path / data, clip, gt)
        C.write_labels(tmp_path / "tracks", clip, tag, {i: regions_of(g) for i, g in enumerate(gt)},
                       seed_info=seed_info)


# What the tracker records for regions it read from seed files, on the centre of three frames: the two
# regions of `write_seeds`, each kept at one pixel or more.
EXTERNAL = {**BOTH_WAYS, "seed_source": "external", "seed_min_area": 1, "n_seed_regions": 2}


def write_seeds(root, clips=CLIPS, frame=1):
    for k, clip in enumerate(clips):
        (root / clip).mkdir(parents=True)
        np.save(root / clip / f"label_{frame:04d}.npy", C.two_organs(cut=8 + k).astype(np.int16))


def score(tmp_path, tag, data="data", clips=CLIPS, propagation=None, pilot=False, seeds=None, gt_informed=None):
    return score_condition("cholecseg8k", None, clips, tmp_path / data, tmp_path / "tracks", tag, propagation,
                           seeds=seeds, seeds_gt_informed=gt_informed, pilot=pilot)


def test_a_perfect_prediction_scores_one_on_every_key_and_the_tools_read_the_json(tmp_path):
    write_condition(tmp_path, "perfect")
    out = tmp_path / "scores" / "perfect.json"
    out.parent.mkdir()
    write_scores(score(tmp_path, "perfect"), out)
    summary = load_scores(out)
    check_rows(summary, "perfect")
    assert ruler(summary).views == tuple(VIEWS) and not ruler(summary).pilot
    assert summary["clips"] == CLIPS and summary["track_dir_name"] == "perfect"
    for row in summary["per_clip"]:
        for key in metric_keys(summary)[:-len(CLIP_METRICS)]:
            best = 0.0 if SIGNS[key.split("/")[0]] <= 0 else 1.0
            assert row[key] == pytest.approx(best), key
        assert row["n_scored_frames"] == 3 and row["n_excluded_frames"] == 0
        assert set(row["n_frames"]) == {metric_key(m, v) for v in VIEWS for m in FRAME_METRICS}
        assert [f["frame"] for f in row["frames"]] == [100, 115, 130]
        assert sum(row["pixels"]["all"].values()) == 3 * 20 * 30
    assert summary["eval_code_sha"] == eval_code_sha()
    assert set(summary["input_shas"]) == set(CLIPS)
    assert set(summary["versions"]) == {"python", "numpy", "opencv", "pillow"}
    assert summary["seed"] == {"source": "sam", "gt_informed": False}
    assert read_evals(str(out.parent))["perfect"]["dir"] == "perfect"


def test_time_iou_pools_every_frame_with_a_prediction_in_time_order(tmp_path):
    gt = [C.two_organs(cut=c) for c in (8, 12, 10)]
    C.write_clip(tmp_path / "data", "VID01_a", gt, gt_frames=[0, 2], times=[0.0, 1.2, 0.6])
    C.write_labels(tmp_path / "tracks", "VID01_a", "t", {i: g.copy() for i, g in enumerate(gt)}, seed_info=BOTH_WAYS)
    row = score(tmp_path, "t", clips=["VID01_a"])["per_clip"][0]
    ones = [np.ones((20, 30), dtype=bool)] * 3
    assert row["time_IoU"] == pytest.approx(time_iou([gt[0], gt[2], gt[1]], ones))
    assert [f["frame"] for f in row["frames"]] == [100, 130]


def test_two_conditions_on_one_gt_compare_and_a_condition_on_another_gt_does_not(tmp_path):
    write_condition(tmp_path, "perfect")
    write_condition(tmp_path, "shifted", regions_of=lambda gt: np.roll(gt, 2, axis=1))
    a, b = score(tmp_path, "perfect"), score(tmp_path, "shifted")
    assert check_comparable(a, b)["clips"] == CLIPS
    shutil.copytree(tmp_path / "data", tmp_path / "other")
    mask = next((tmp_path / "other" / CLIPS[0] / "seg_masks").iterdir())
    mask.write_bytes(mask.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="read a different gt_masks"):
        check_comparable(a, score(tmp_path, "shifted", data="other"))


def test_a_class_table_outside_the_package_is_refused(tmp_path):
    check_table(load_table("cholecseg8k"))
    copy = tmp_path / "cholecseg8k.json"
    shutil.copy(load_table("cholecseg8k").path, copy)
    with pytest.raises(ValueError, match="eval_code_sha does not cover"):
        check_table(load_table("cholecseg8k", path=copy))


def test_a_run_that_fails_on_one_clip_writes_nothing(tmp_path):
    write_condition(tmp_path, "perfect")
    (tmp_path / "tracks" / CLIPS[1] / "perfect" / "label_0001.npy").unlink()
    (tmp_path / "clips.txt").write_text("\n".join(CLIPS) + "\n")
    out = tmp_path / "perfect.json"
    with pytest.raises(ValueError, match="no prediction"):
        main(["--dataset", "cholecseg8k", "--clips", str(tmp_path / "clips.txt"),
              "--data-root", str(tmp_path / "data"), "--tracks-root", str(tmp_path / "tracks"),
              "--tag", "perfect", "--out", str(out)])
    assert not out.exists() and not list(tmp_path.glob("*.partial"))


def test_the_command_line_writes_the_json(tmp_path, capsys):
    write_condition(tmp_path, "perfect")
    (tmp_path / "clips.txt").write_text("\n".join(CLIPS) + "\n")
    out = tmp_path / "perfect.json"
    main(["--dataset", "cholecseg8k", "--clips", str(tmp_path / "clips.txt"),
          "--data-root", str(tmp_path / "data"), "--tracks-root", str(tmp_path / "tracks"),
          "--tag", "perfect", "--out", str(out)])
    assert json.loads(out.read_text())["clips"] == CLIPS
    assert "2 clips scored" in capsys.readouterr().out


def test_a_population_with_no_clip_or_one_clip_twice_is_refused(tmp_path):
    p = tmp_path / "clips.txt"
    p.write_text("a\n\nb\n")
    assert read_population(p) == ["a", "b"]
    p.write_text("\n")
    with pytest.raises(ValueError, match="lists no clip"):
        read_population(p)
    p.write_text("a\nb\na\n")
    with pytest.raises(ValueError, match=r"\['a'\] more than once"):
        read_population(p)


def test_a_value_that_is_not_a_json_number_is_never_written(tmp_path):
    with pytest.raises(ValueError):
        write_scores({"per_clip": [{"clip": "a", "SQ/all": float("nan")}]}, tmp_path / "x.json")
    assert not (tmp_path / "x.json").exists()


@pytest.mark.parametrize("info, rule", [
    (BOTH_WAYS, "both_ways_from_centre"),
    (FORWARD, "forward_from_first"),
    ({"seed_source": "per_frame", "frames": "all"}, "per_frame"),
    ({"seed_source": "sam", "seed_frame": 1, "bidir": True, "frames": [2, 0, 1]}, "both_ways_from_centre"),
    ({"seed_source": "sam", "seed_frame": 0, "bidir": False, "frames": [1, 0, 2]}, "forward_from_first"),
])
def test_the_json_records_the_rule_each_clip_s_seed_info_gives(tmp_path, info, rule):
    write_condition(tmp_path, "c", seed_info=info)
    summary = score(tmp_path, "c")
    assert summary["propagation"] == rule


@pytest.mark.parametrize("info, why", [
    ({"seed_source": "gt", "seed_frame": 1, "bidir": True, "frames": [0, 1, 2]}, "the seed came from 'gt'"),
    ({"seed_source": "external", "seed_frame": 0, "bidir": True, "frames": [0, 1, 2]}, "the centre of the labelled frames is 1"),
    ({"seed_frame": 1, "bidir": True, "frames": [0, 1, 2]}, "names no seed_source"),
    ({"seed_source": "sam", "seed_frame": 2, "bidir": True, "frames": [0, 1, 2]}, "the centre of the labelled frames is 1"),
    ({"seed_source": "sam", "seed_frame": 0, "bidir": True, "frames": [0, 1, 2]}, "the centre of the labelled frames is 1"),
    ({"seed_source": "sam", "seed_frame": 1, "bidir": False, "frames": [1, 2]}, "the rule starts on frame 0"),
    ({"seed_source": "sam", "seed_frame": 1, "frames": [0, 1, 2]}, "lacks the seed frame, bidir"),
    ({"seed_source": "sam", "seed_frame": 1, "bidir": True, "frames": "all"}, "lacks the seed frame, bidir"),
    ({"seed_source": "sam", "seed_frame": 0, "bidir": True, "frames": [2, 0, 1]}, "the centre of the labelled frames is 1"),
    ({"seed_source": "sam", "seed_frame": 0, "bidir": "no", "frames": [0, 1, 2]}, "bidir is 'no', not true or false"),
    ({"seed_source": "sam", "seed_frame": 1, "bidir": 1, "frames": [0, 1, 2]}, "bidir is 1, not true or false"),
    ({"seed_source": "sam", "seed_frame": 1, "bidir": True, "frames": [0, 0, 1, 2, 3]}, "repeat a frame"),
    ({"seed_source": "sam", "seed_frame": True, "bidir": True, "frames": [0, 1, 2]}, "not all whole numbers"),
    ({"seed_source": "sam", "seed_frame": 1.0, "bidir": True, "frames": [0, 1, 2]}, "not all whole numbers"),
    ({"seed_source": "sam", "seed_frame": 0, "bidir": False, "frames": [0, 1.5, 2]}, "not all whole numbers"),
    ({"seed_source": "sam", "seed_frame": 0, "bidir": False, "frames": [0, "1", 2]}, "not all whole numbers"),
])
@pytest.mark.parametrize("pilot", [False, True])
def test_a_seed_that_holds_no_rule_is_refused(tmp_path, info, why, pilot):
    write_condition(tmp_path, "c", seed_info=info)
    with pytest.raises(ValueError, match=f"holds no propagation rule: .*{why}"):
        score(tmp_path, "c", pilot=pilot)


def test_a_record_on_some_clips_and_none_on_the_others_is_refused(tmp_path):
    write_condition(tmp_path, "c", clips=CLIPS[:1])
    write_condition(tmp_path, "c", clips=CLIPS[1:], seed_info=None)
    with pytest.raises(ValueError, match="on every clip or on none"):
        score(tmp_path, "c")
    with pytest.raises(ValueError, match="on every clip or on none"):
        score(tmp_path, "c", propagation="both_ways_from_centre")
    with pytest.raises(ValueError, match="on every clip or on none"):
        score(tmp_path, "c", pilot=True)


def test_two_rules_in_one_condition_are_refused(tmp_path):
    write_condition(tmp_path, "c", clips=CLIPS[:1])
    write_condition(tmp_path, "c", clips=CLIPS[1:], seed_info=FORWARD)
    with pytest.raises(ValueError, match="a condition has one rule"):
        score(tmp_path, "c")


def test_a_statement_that_contradicts_the_records_is_refused(tmp_path):
    write_condition(tmp_path, "c")
    with pytest.raises(ValueError, match="a condition has one rule"):
        score(tmp_path, "c", propagation="forward_from_first")


@pytest.mark.parametrize("pilot", [False, True])
def test_labels_without_a_record_need_the_rule_stated(tmp_path, pilot):
    write_condition(tmp_path, "c", seed_info=None)
    with pytest.raises(ValueError, match="state the rule with --propagation"):
        score(tmp_path, "c", pilot=pilot)
    assert score(tmp_path, "c", propagation="per_frame", pilot=pilot)["propagation"] == "per_frame"


@pytest.mark.parametrize("order", [CLIPS, CLIPS[::-1]])
def test_pilot_mode_refuses_a_gt_seed_beside_a_clip_without_a_record_in_either_order(tmp_path, order):
    # Every clip is read before the condition's rule is settled, so the population's order cannot hide a refusal.
    write_condition(tmp_path, "c", clips=CLIPS[:1], seed_info={"seed_source": "gt", "seed_frame": 1,
                                                               "bidir": True, "frames": [0, 1, 2]})
    write_condition(tmp_path, "c", clips=CLIPS[1:], seed_info=None)
    with pytest.raises(ValueError, match="the seed came from 'gt'"):
        score(tmp_path, "c", clips=list(order), pilot=True)


def test_pilot_mode_refuses_a_record_it_cannot_parse(tmp_path):
    write_condition(tmp_path, "c", seed_info=None)
    for clip in CLIPS:
        (tmp_path / "tracks" / clip / "c" / "seed_info.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="holds no propagation rule: Expecting"):
        score(tmp_path, "c", pilot=True)
    for clip in CLIPS:
        (tmp_path / "tracks" / clip / "c" / "seed_info.json").write_bytes(b"\xff\xfe{")
    with pytest.raises(ValueError, match="holds no propagation rule: 'utf-8' codec"):
        score(tmp_path, "c", pilot=True)


@pytest.mark.parametrize("pilot", [False, True])
def test_seeds_made_outside_the_stage_are_scored_and_recorded_by_their_files_hash(tmp_path, pilot):
    write_condition(tmp_path, "c", seed_info=EXTERNAL)
    write_seeds(tmp_path / "seeds")
    summary = score(tmp_path, "c", pilot=pilot, seeds=str(tmp_path / "seeds"), gt_informed=False)
    assert summary["propagation"] == "both_ways_from_centre"
    assert summary["seed"] == {"source": "external", "gt_informed": False}
    for clip in CLIPS:
        assert summary["input_shas"][clip]["seeds"] == sha_of_files([tmp_path / "seeds" / clip / "label_0001.npy"])
    # The hash names the files, not the path they were given by: a `{clip}` pattern or a link to the
    # same directory gives the same record.
    (tmp_path / "link").symlink_to(tmp_path / "seeds")
    for given in (str(tmp_path / "seeds" / "{clip}"), str(tmp_path / "link")):
        again = score(tmp_path, "c", pilot=pilot, seeds=given, gt_informed=False)
        assert again["input_shas"] == summary["input_shas"] and again["seed"] == summary["seed"]


def test_the_seed_file_is_where_the_tracking_stage_reads_it(tmp_path):
    track = pytest.importorskip("pipeline.track")
    for seeds in (str(tmp_path), str(tmp_path / "{clip}" / "track_rgb_k10")):
        assert seed_file(seeds, "VID01_a", 7) == track.seed_labels_dir(seeds, "VID01_a") / "label_0007.npy"


@pytest.mark.parametrize("seeds, gt_informed, why", [
    (None, False, "give their directory with --seeds"),
    ("seeds", None, "state it with --seeds-gt-informed"),
])
def test_external_seeds_need_their_directory_and_a_statement_on_the_gt(tmp_path, seeds, gt_informed, why):
    write_condition(tmp_path, "c", seed_info=EXTERNAL)
    write_seeds(tmp_path / "seeds")
    with pytest.raises(ValueError, match=why):
        score(tmp_path, "c", seeds=seeds and str(tmp_path / seeds), gt_informed=gt_informed)


@pytest.mark.parametrize("info, error, why", [
    ({**EXTERNAL, "n_seed_regions": 3}, ValueError, "holds 2 regions of 1 pixels or more, and the tracker's record counts 3"),
    ({**EXTERNAL, "seed_min_area": 500}, ValueError, "holds 0 regions of 500 pixels or more"),
    ({k: v for k, v in EXTERNAL.items() if k != "n_seed_regions"}, ValueError, "cannot be checked"),
    ({**EXTERNAL, "n_seed_regions": True}, ValueError, "cannot be checked"),
])
def test_a_seed_file_that_is_not_the_one_the_tracker_read_is_refused(tmp_path, info, error, why):
    write_condition(tmp_path, "c", seed_info=info)
    write_seeds(tmp_path / "seeds")
    with pytest.raises(error, match=why):
        score(tmp_path, "c", seeds=str(tmp_path / "seeds"), gt_informed=False)


def test_a_missing_seed_file_is_refused(tmp_path):
    write_condition(tmp_path, "c", seed_info=EXTERNAL)
    write_seeds(tmp_path / "seeds", frame=0)
    with pytest.raises(FileNotFoundError, match="no seed file .*label_0001.npy"):
        score(tmp_path, "c", seeds=str(tmp_path / "seeds"), gt_informed=False)


@pytest.mark.parametrize("info", [BOTH_WAYS, {"seed_source": "per_frame", "frames": "all"}])
def test_a_statement_the_record_already_decides_is_refused(tmp_path, info):
    write_condition(tmp_path, "c", seed_info=info)
    write_seeds(tmp_path / "seeds")
    with pytest.raises(ValueError, match="already says that no GT informed the seeds"):
        score(tmp_path, "c", gt_informed=False)
    with pytest.raises(ValueError, match="--seeds names seed files"):
        score(tmp_path, "c", seeds=str(tmp_path / "seeds"))


def test_labels_without_a_record_state_the_gt_unless_they_are_per_frame(tmp_path):
    write_condition(tmp_path, "c", seed_info=None)
    with pytest.raises(ValueError, match="no recorded source, so whether the GT made or informed them"):
        score(tmp_path, "c", propagation="both_ways_from_centre")
    summary = score(tmp_path, "c", propagation="both_ways_from_centre", gt_informed=True)
    assert summary["seed"] == {"source": None, "gt_informed": True}
    assert score(tmp_path, "c", propagation="per_frame")["seed"] == {"source": "per_frame", "gt_informed": False}


def test_seeds_made_two_ways_in_one_condition_are_refused(tmp_path):
    write_condition(tmp_path, "c", clips=CLIPS[:1])
    write_condition(tmp_path, "c", clips=CLIPS[1:], seed_info=EXTERNAL)
    with pytest.raises(ValueError, match=r"seed sources \['external', 'sam'\]; a condition's seeds are made one way"):
        score(tmp_path, "c", gt_informed=False)


def test_a_condition_the_gt_informed_is_compared_only_with_another(tmp_path):
    write_condition(tmp_path, "sam")
    write_condition(tmp_path, "kgt", seed_info=EXTERNAL)
    write_condition(tmp_path, "gt_seeded", seed_info=None)
    write_seeds(tmp_path / "seeds")
    sam = score(tmp_path, "sam")
    kgt = score(tmp_path, "kgt", seeds=str(tmp_path / "seeds"), gt_informed=True)
    gt_seeded = score(tmp_path, "gt_seeded", propagation="both_ways_from_centre", gt_informed=True)
    with pytest.raises(ValueError, match="whose seeds the GT made or informed cannot be compared"):
        check_comparable(sam, kgt)
    with pytest.raises(ValueError, match="whose seeds the GT made or informed cannot be compared"):
        check_comparable(kgt, score(tmp_path, "kgt", seeds=str(tmp_path / "seeds"), gt_informed=False))
    assert check_comparable(kgt, gt_seeded)["clips"] == CLIPS


def test_the_command_line_records_the_seeds_and_the_statement(tmp_path):
    write_condition(tmp_path, "c", seed_info=EXTERNAL)
    write_seeds(tmp_path / "seeds")
    (tmp_path / "clips.txt").write_text("\n".join(CLIPS) + "\n")
    out = tmp_path / "c.json"
    main(["--dataset", "cholecseg8k", "--clips", str(tmp_path / "clips.txt"),
          "--data-root", str(tmp_path / "data"), "--tracks-root", str(tmp_path / "tracks"),
          "--tag", "c", "--out", str(out), "--seeds", str(tmp_path / "seeds"), "--seeds-gt-informed", "yes"])
    assert json.loads(out.read_text())["seed"] == {"source": "external", "gt_informed": True}
