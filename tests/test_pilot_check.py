"""The check against the pilot evaluator fires on every planted difference, and on nothing else."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from evalkit.tools import pilot_check as PC
from evalkit.tools.scores import PILOT_DOMAINS, PILOT_EVAL_CODE_SHA, metric_key

REPO = Path(__file__).resolve().parent.parent
CLIPS = ["VID01_s15_80_crop", "VID02_s15_80_crop"]


def pilot_json(clips=CLIPS, sha=PILOT_EVAL_CODE_SHA):
    rows = []
    for i, c in enumerate(clips):
        r = dict(clip=c, extra_ignore=[], GT_mIoU=0.5 + i / 10, boundary_F=0.4, boundary_R_raw=0.3,
                 VI_split=1.25, VI_merge=0.75, time_IoU=0.0, inst_F1_75=0.1, PQ=0.2)
        for d in PILOT_DOMAINS:
            s = "" if d == "full" else f"_{d}"
            r[f"inst_F1_50{s}"] = 0.6 if d == "full" else None
            r[f"SQ{s}"] = None if d != "full" else 0.7
            r[f"inst_BF{s}"] = 0.0 if d == "full" else None
            r[f"n_inst_frames{s}"] = 3 if d == "full" else 0
            r[f"n_SQ_frames{s}"] = r[f"n_BF_frames{s}"] = 2 if d == "full" else 0
        r.update(n_gt_frames=3, n_vi_frames=3)
        rows.append(r)
    return dict(eval_code_sha=sha, eval_code_tag="t", eval_version=2, dataset="cholec",
                tissue_ignore=[5], clips=list(clips), per_clip=rows)


def ours_json(clips=CLIPS, pilot=True, views=PILOT_DOMAINS):
    rows = []
    for i, c in enumerate(clips):
        r = dict(clip=c, time_IoU=0.0, **{metric_key("mIoU", "full"): 0.5 + i / 10,
                                          metric_key("boundary_F", "full"): 0.4,
                                          metric_key("boundary_R_raw", "full"): 0.3,
                                          metric_key("VI_split", "labeled"): 1.25,
                                          metric_key("VI_merge", "labeled"): 0.75})
        for d in views:
            r[metric_key("F1_50", d)] = 0.6 if d == "full" else None
            r[metric_key("SQ", d)] = None if d != "full" else 0.7
            r[metric_key("inst_BF", d)] = 0.0 if d == "full" else None
        r["n_frames"] = {ok: pilot_json()["per_clip"][0][pk] for pk, ok in PC.COUNTS}
        rows.append(r)
    return dict(eval_code_sha="a" * 64, dataset="cholecseg8k", pilot=pilot, class_set="original",
                views=list(views), clips=list(clips), input_shas={c: {} for c in clips}, versions={},
                propagation="both_ways_from_centre", per_clip=rows)


def test_the_shared_keys_are_the_table_s():
    assert ("inst_F1_50", "F1_50/full") in PC.SHARED
    assert ("inst_F1_50_labeled_tissue", "F1_50/labeled_tissue") in PC.SHARED
    assert ("GT_mIoU", "mIoU/full") in PC.SHARED and ("time_IoU", "time_IoU") in PC.SHARED
    # The pilot's VI is on the valid pixels whose GT is not background: the
    # `labeled` domain's mask, not `full`'s.
    assert ("VI_split", "VI_split/labeled") in PC.SHARED and ("VI_merge", "VI_merge/labeled") in PC.SHARED
    assert not any(ok == "VI_split/full" for _, ok in PC.SHARED)
    assert not any(pk in ("inst_F1_75", "PQ", "GT_mDice", "boundary_P_raw") for pk, _ in PC.SHARED)
    assert len(PC.SHARED) == 3 * len(PILOT_DOMAINS) + 5 + 1


def test_equal_scores_give_no_difference():
    assert PC.diff_shared(pilot_json(), ours_json()) == []


def test_a_value_that_differs_is_reported_with_both_values():
    ours = ours_json()
    ours["per_clip"][1][metric_key("SQ", "full")] = 0.7001
    d = PC.diff_shared(pilot_json(), ours)
    assert d == [f"{CLIPS[1]}.SQ/full: 0.7001 != pilot SQ=0.7"]


def test_a_none_where_the_pilot_wrote_zero_is_a_difference():
    ours = ours_json()
    ours["per_clip"][0][metric_key("inst_BF", "full")] = None
    assert any("inst_BF/full: None != pilot inst_BF=0.0" in x for x in PC.diff_shared(pilot_json(), ours))


def test_a_zero_where_the_pilot_wrote_none_is_a_difference():
    # The pilot writes None for `SQ` outside `full` on a clip with no GT
    # object there; a driver that wrote 0 in its place would be wrong.
    ours = ours_json()
    ours["per_clip"][0][metric_key("SQ", "labeled")] = 0.0
    assert any("SQ/labeled: 0.0 != pilot SQ_labeled=None" in x for x in PC.diff_shared(pilot_json(), ours))


def test_a_shared_key_the_evaluator_lacks_is_a_difference():
    ours = ours_json()
    del ours["per_clip"][0]["time_IoU"]
    assert any("time_IoU: missing" in x for x in PC.diff_shared(pilot_json(), ours))


def test_a_pilot_row_without_a_shared_key_is_refused_not_skipped():
    # Skipped, a row of `clip` alone would read as "diff 0".
    pilot = pilot_json()
    del pilot["per_clip"][1]["VI_split"]
    with pytest.raises(ValueError, match=f"{CLIPS[1]}: the pilot evaluator's row lacks VI_split"):
        PC.diff_shared(pilot, ours_json())
    pilot = pilot_json()
    pilot["per_clip"] = [dict(clip=c) for c in CLIPS]
    with pytest.raises(ValueError, match="row lacks"):
        PC.diff_shared(pilot, ours_json())


def test_a_pilot_row_that_removed_an_extra_ignore_is_refused():
    pilot = pilot_json()
    pilot["per_clip"][0]["extra_ignore"] = [7]
    with pytest.raises(ValueError, match=f"{CLIPS[0]}: the pilot evaluator's row removed extra_ignore"):
        PC.diff_shared(pilot, ours_json())


def test_a_clip_named_twice_or_a_population_that_disagrees_with_its_rows_is_refused():
    pilot = pilot_json()
    pilot["per_clip"].append(dict(pilot["per_clip"][0]))
    with pytest.raises(ValueError, match="the pilot evaluator's JSON: per_clip holds more than one row"):
        PC.diff_shared(pilot, ours_json())
    ours = ours_json()
    ours["clips"] = CLIPS[:1]
    with pytest.raises(ValueError, match="the pilot-mode JSON: the JSON's clips and its per_clip rows"):
        PC.diff_shared(pilot_json(), ours)


def test_two_jsons_without_a_clip_are_refused():
    with pytest.raises(ValueError, match="neither JSON holds a clip"):
        PC.diff_shared(pilot_json(clips=[]), ours_json(clips=[]))


def test_the_pilot_s_replaced_keys_are_not_compared():
    pilot = pilot_json()
    pilot["per_clip"][0]["PQ"] = 0.999
    assert PC.diff_shared(pilot, ours_json()) == []


def test_a_clip_only_one_side_scored_is_a_difference():
    d = PC.diff_shared(pilot_json(), ours_json(clips=CLIPS[:1]))
    assert d == [f"{CLIPS[1]}: scored by the pilot evaluator only"]


def test_a_json_that_is_not_the_pilot_s_is_refused():
    with pytest.raises(ValueError, match="given as the pilot evaluator's is not its"):
        PC.diff_shared(pilot_json(sha="b" * 64), ours_json())
    with pytest.raises(ValueError, match="given as the pilot evaluator's is not its"):
        PC.diff_shared(ours_json(), ours_json())


def test_a_normal_mode_json_is_refused():
    with pytest.raises(ValueError, match="not a pilot-mode score"):
        PC.diff_shared(pilot_json(), ours_json(pilot=False))
    with pytest.raises(ValueError, match="not a pilot-mode score"):
        PC.diff_shared(pilot_json(), pilot_json())


def test_a_pilot_mode_json_over_other_views_is_refused():
    with pytest.raises(ValueError, match="pilot's domains"):
        PC.diff_shared(pilot_json(), ours_json(views=("all", "tissue", "geometric")))


def test_the_command_diffs_every_condition_by_tag(tmp_path):
    pd, ed = tmp_path / "pilot", tmp_path / "ours"
    pd.mkdir(), ed.mkdir()
    for tag in ("a", "b"):
        (pd / f"{tag}.json").write_text(json.dumps(pilot_json()))
        (ed / f"{tag}.json").write_text(json.dumps(ours_json()))
    cmd = [sys.executable, "-m", "evalkit.tools.pilot_check", "--pilot-dir", str(pd), "--eval-dir", str(ed)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0 and "all 2 conditions" in r.stdout
    # The evaluator's sha is named with the result: it is the one the freeze records.
    assert f"evaluator {'a' * 16} reproduces the pilot evaluator" in r.stdout
    assert f"against the pilot evaluator {PILOT_EVAL_CODE_SHA[:16]}" in r.stdout
    ours = ours_json()
    ours["per_clip"][0][metric_key("F1_50", "full")] = 0.61
    (ed / "b.json").write_text(json.dumps(ours))
    (pd / "c.json").write_text(json.dumps(pilot_json()))
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 1 and "2 of 3 conditions differ" in r.stdout and "not scored in pilot mode" in r.stdout


def test_the_command_refuses_a_directory_scored_by_two_evaluators(tmp_path):
    pd, ed = tmp_path / "pilot", tmp_path / "ours"
    pd.mkdir(), ed.mkdir()
    for tag, sha in (("a", "a" * 64), ("b", "b" * 64)):
        (pd / f"{tag}.json").write_text(json.dumps(pilot_json()))
        ours = ours_json()
        ours["eval_code_sha"] = sha
        (ed / f"{tag}.json").write_text(json.dumps(ours))
    cmd = [sys.executable, "-m", "evalkit.tools.pilot_check", "--pilot-dir", str(pd), "--eval-dir", str(ed)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert r.returncode != 0 and "made by 2 evaluators" in r.stderr and "Traceback" not in r.stderr


def test_the_command_names_the_tag_it_could_not_check(tmp_path):
    pd, ed = tmp_path / "pilot", tmp_path / "ours"
    pd.mkdir(), ed.mkdir()
    (pd / "a.json").write_text(json.dumps(pilot_json()))
    (ed / "a.json").write_text(json.dumps(ours_json(pilot=False)))
    cmd = [sys.executable, "-m", "evalkit.tools.pilot_check", "--pilot-dir", str(pd), "--eval-dir", str(ed)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert r.returncode != 0 and r.stderr.startswith("a: the JSON given as the evaluator's is not a pilot-mode score")


def test_the_command_names_the_tag_and_the_path_of_a_json_it_cannot_read(tmp_path):
    pd, ed = tmp_path / "pilot", tmp_path / "ours"
    pd.mkdir(), ed.mkdir()
    (pd / "a.json").write_text(json.dumps(pilot_json()))
    (ed / "a.json").write_text("{not json")
    cmd = [sys.executable, "-m", "evalkit.tools.pilot_check", "--pilot-dir", str(pd), "--eval-dir", str(ed)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert r.returncode != 0 and "Traceback" not in r.stderr
    assert r.stderr.startswith(f"a: the pilot-mode JSON {ed / 'a.json'} cannot be read")
    # A pilot JSON that will not open, here a directory under its name, is refused the same way.
    (ed / "a.json").write_text(json.dumps(ours_json()))
    (pd / "a.json").unlink()
    (pd / "a.json").mkdir()
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert r.returncode != 0 and "Traceback" not in r.stderr
    assert r.stderr.startswith(f"a: the pilot evaluator's JSON {pd / 'a.json'} cannot be read")


def test_the_counts_pair_each_shared_key_with_the_frames_behind_it():
    assert ("n_inst_frames_tissue", "F1_50/tissue") in PC.COUNTS and ("n_gt_frames", "mIoU/full") in PC.COUNTS
    assert ("n_vi_frames", "VI_merge/labeled") in PC.COUNTS
    assert {ok for _, ok in PC.COUNTS} == {ok for _, ok in PC.SHARED} - {"time_IoU"}


def test_a_frame_left_out_is_a_difference_even_where_the_mean_hides_it():
    ours = ours_json()
    ours["per_clip"][0]["n_frames"]["mIoU/full"] = 2
    diffs = PC.diff_shared(pilot_json(), ours)
    assert diffs == [f"{CLIPS[0]}.n_frames[mIoU/full]: 2 != pilot n_gt_frames=3"]


def test_a_count_the_evaluator_lacks_is_a_difference_and_one_the_pilot_lacks_is_refused():
    ours = ours_json()
    del ours["per_clip"][1]["n_frames"]["SQ/labeled"]
    assert PC.diff_shared(pilot_json(), ours) == [f"{CLIPS[1]}.n_frames[SQ/labeled]: missing; the pilot wrote n_SQ_frames_labeled=0"]
    pilot = pilot_json()
    del pilot["per_clip"][0]["n_vi_frames"]
    with pytest.raises(ValueError, match="lacks the count n_vi_frames"):
        PC.diff_shared(pilot, ours_json())

