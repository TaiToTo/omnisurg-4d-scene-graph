"""The entry point end to end: a condition scored on clips written to disk, read back by the tools.

A prediction equal to the GT must score one, or zero for the variation of
information, on every key in every view. The JSON must pass the tools' own
readers, two conditions on the same inputs must be comparable, and one on
another GT must not. A run that fails writes nothing. The JSON records the
condition's propagation rule as each clip's `seed_info.json` gives it, and a
condition whose seed holds no rule is refused.
"""
import json
import shutil

import numpy as np
import pytest

import clip_dirs as C
from evalkit.classes import VIEWS, load_table
from evalkit.evaluate import check_table, main, read_population, score_condition, write_scores
from evalkit.keys import CLIP_METRICS, FRAME_METRICS, SIGNS, metric_key
from evalkit.time_iou import time_iou
from evalkit.tools.condition_inventory import read_evals
from evalkit.tools.scores import check_comparable, check_rows, load_scores, metric_keys, ruler

CLIPS = ["VID01_a", "VID02_b"]

# What the tracker records for a seed on the centre of three frames, carried both ways.
BOTH_WAYS = {"seed_source": "auto", "seed_frame": 1, "bidir": True, "frames": [0, 1, 2]}
FORWARD = {"seed_source": "auto", "seed_frame": 0, "bidir": False, "frames": [0, 1, 2]}


def write_condition(tmp_path, tag, regions_of=lambda gt: gt.copy(), clips=CLIPS, data="data", seed_info=BOTH_WAYS):
    for k, clip in enumerate(clips):
        gt = [C.two_organs(cut=8 + 2 * i + k) for i in range(3)]
        if not (tmp_path / data / clip).exists():
            C.write_clip(tmp_path / data, clip, gt)
        C.write_labels(tmp_path / "tracks", clip, tag, {i: regions_of(g) for i, g in enumerate(gt)},
                       seed_info=seed_info)


def score(tmp_path, tag, data="data", clips=CLIPS, propagation=None):
    return score_condition("cholecseg8k", None, clips, tmp_path / data, tmp_path / "tracks", tag, propagation)


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
    assert set(summary["input_shas"]) == set(CLIPS)
    assert set(summary["versions"]) == {"python", "numpy", "opencv", "pillow"}
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
])
def test_the_json_records_the_rule_each_clip_s_seed_info_gives(tmp_path, info, rule):
    write_condition(tmp_path, "c", seed_info=info)
    summary = score(tmp_path, "c")
    assert summary["propagation"] == rule


@pytest.mark.parametrize("info", [
    {"seed_source": "gt", "seed_frame": 2, "bidir": True, "frames": [0, 1, 2]},
    {"seed_source": "auto", "seed_frame": 1, "bidir": False, "frames": [1, 2]},
    {"seed_source": "auto", "seed_frame": 1, "frames": [0, 1, 2]},
])
def test_a_seed_that_holds_no_rule_is_refused(tmp_path, info):
    write_condition(tmp_path, "c", seed_info=info)
    with pytest.raises(ValueError, match="holds no propagation rule"):
        score(tmp_path, "c")


def test_two_rules_in_one_condition_are_refused(tmp_path):
    write_condition(tmp_path, "c", clips=CLIPS[:1])
    write_condition(tmp_path, "c", clips=CLIPS[1:], seed_info=FORWARD)
    with pytest.raises(ValueError, match="a condition has one rule"):
        score(tmp_path, "c")


def test_a_statement_that_contradicts_the_records_is_refused(tmp_path):
    write_condition(tmp_path, "c")
    with pytest.raises(ValueError, match="a condition has one rule"):
        score(tmp_path, "c", propagation="forward_from_first")


def test_labels_without_a_record_need_the_rule_stated(tmp_path):
    write_condition(tmp_path, "c", seed_info=None)
    with pytest.raises(ValueError, match="state the rule with --propagation"):
        score(tmp_path, "c")
    assert score(tmp_path, "c", propagation="per_frame")["propagation"] == "per_frame"
