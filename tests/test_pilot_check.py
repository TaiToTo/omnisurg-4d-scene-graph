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
        rows.append(r)
    return dict(eval_code_sha=sha, eval_code_tag="t", eval_version=2, dataset="cholec",
                tissue_ignore=[5], clips=list(clips), per_clip=rows)


def ours_json(clips=CLIPS, pilot=True, views=PILOT_DOMAINS):
    rows = []
    for i, c in enumerate(clips):
        r = dict(clip=c, time_IoU=0.0, **{metric_key("mIoU", "full"): 0.5 + i / 10,
                                          metric_key("boundary_F", "full"): 0.4,
                                          metric_key("boundary_R_raw", "full"): 0.3,
                                          metric_key("VI_split", "full"): 1.25,
                                          metric_key("VI_merge", "full"): 0.75})
        for d in views:
            r[metric_key("F1_50", d)] = 0.6 if d == "full" else None
            r[metric_key("SQ", d)] = None if d != "full" else 0.7
            r[metric_key("inst_BF", d)] = 0.0 if d == "full" else None
        rows.append(r)
    return dict(eval_code_sha="a" * 64, dataset="cholecseg8k", pilot=pilot, class_set="original",
                views=list(views), clips=list(clips), input_shas={c: {} for c in clips}, versions={},
                per_clip=rows)


def test_the_shared_keys_are_the_table_s():
    assert ("inst_F1_50", "F1_50/full") in PC.SHARED
    assert ("inst_F1_50_labeled_tissue", "F1_50/labeled_tissue") in PC.SHARED
    assert ("GT_mIoU", "mIoU/full") in PC.SHARED and ("time_IoU", "time_IoU") in PC.SHARED
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


def test_a_shared_key_the_evaluator_lacks_is_a_difference():
    ours = ours_json()
    del ours["per_clip"][0]["time_IoU"]
    assert any("time_IoU: missing" in x for x in PC.diff_shared(pilot_json(), ours))


def test_a_key_the_pilot_did_not_write_is_not_compared():
    pilot = pilot_json()
    for r in pilot["per_clip"]:
        del r["VI_split"]
    assert PC.diff_shared(pilot, ours_json()) == []


def test_the_pilot_s_replaced_keys_are_not_compared():
    pilot = pilot_json()
    pilot["per_clip"][0]["PQ"] = 0.999
    assert PC.diff_shared(pilot, ours_json()) == []


def test_a_clip_only_one_side_scored_is_a_difference():
    d = PC.diff_shared(pilot_json(), ours_json(clips=CLIPS[:1]))
    assert d == [f"{CLIPS[1]}: scored by the pilot evaluator only"]


def test_a_json_that_is_not_the_pilot_s_is_refused():
    with pytest.raises(ValueError, match="not the pilot evaluator's"):
        PC.diff_shared(pilot_json(sha="b" * 64), ours_json())
    with pytest.raises(ValueError, match="not the pilot evaluator's"):
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
    ours = ours_json()
    ours["per_clip"][0][metric_key("F1_50", "full")] = 0.61
    (ed / "b.json").write_text(json.dumps(ours))
    (pd / "c.json").write_text(json.dumps(pilot_json()))
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 1 and "2 of 3 conditions differ" in r.stdout and "not scored in pilot mode" in r.stdout
