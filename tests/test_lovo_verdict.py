"""The videos a verdict rests on: each check of `lovo_verdict` is shown to refuse what it should, or to name the video."""
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from evalkit.tools import lovo_verdict as LV
from evalkit.tools import paired_stats as PS
from evalkit.tools.scores import CLIP_METRICS, FRAME_METRICS, PILOT_EVAL_CODE_SHA, metric_key

pytest.importorskip("scipy", reason="paired_stats needs the `tools` extra")

REPO = Path(__file__).resolve().parent.parent

VIDEOS = ("adrenalectomy__16GPCUPkXYQ", "appendectomy__41RKDh3INiU", "gastric_surgery__4FHGGFZsPzw",
          "lar___7H7G-4sevQ", "rectopexy__IJfgUvxoTkM", "hemicolectomy__5YDMlxTl0k8", "nephrectomy__abc123")
LOSER = VIDEOS[-1]


def pilot_pair() -> tuple[dict, dict]:
    """A base and a cond pilot JSON on 7 videos, 2 clips each.

    The cond gains a little on six videos and loses 0.4 on the last, on
    `inst_F1_50` (higher is better) and on `underseg_error` (lower is
    better) alike. `SQ` is None on the first clip.
    """
    rnd = random.Random(3)
    clips = [f"{v}__gt_{i:04d}" for v in VIDEOS for i in (1, 2)]
    rows = {"base": [], "cond": []}
    for n, c in enumerate(clips):
        base, cond = {"clip": c, "extra_ignore": []}, {"clip": c, "extra_ignore": []}
        for key, sign in (("inst_F1_50", +1), ("underseg_error", -1), ("SQ", +1), ("time_IoU", +1)):
            v = round(rnd.uniform(0.3, 0.6), 4)
            gain = -0.4 if c.startswith(LOSER) else round(rnd.uniform(0.01, 0.06), 4)
            base[key], cond[key] = v, round(v + sign * gain, 4)
            if key == "SQ" and n == 0:
                base[key] = None
        rows["base"].append(base)
        rows["cond"].append(cond)
    return tuple(dict(tag=tag, eval_code_sha=PILOT_EVAL_CODE_SHA, eval_code_tag="t", eval_version=2,
                      dataset="atlas", tissue_ignore=[1], clips=clips, per_clip=rows[tag])
                 for tag in ("base", "cond"))


def evaluator_scores(tag: str, shift: float, seed: int, rule: str = "both_ways_from_centre") -> dict:
    """An evaluator-layout score JSON on 4 videos, 2 clips each, every metric in the `all` view."""
    rnd = random.Random(seed)
    clips = [f"{v}__gt_{i:04d}" for v in VIDEOS[:4] for i in (1, 2)]
    keys = [metric_key(m, "all") for m in FRAME_METRICS] + list(CLIP_METRICS)
    rows = [{"clip": c, **{k: round(rnd.gauss(0.5 + shift, 0.05), 4) for k in keys}} for c in clips]
    return dict(tag=tag, eval_code_sha="a" * 64, dataset="atlas", pilot=False, class_set="original",
                views=["all"], clips=clips,
                input_shas={c: {"gt_masks": "g" * 64, "depth": "d" * 64, "predictions": f"p{c}"} for c in clips},
                versions={"python": "3.12.0"}, propagation=rule, per_clip=rows)


# What the workbench's `lovo_verdict.py` wrote on exactly `pilot_pair()`, so
# that the port is held to its numbers rather than only to its own tests.
WORKBENCH = {
    "inst_F1_50": {"n_clips": 14, "n_videos": 7, "sign": 1, "delta": -0.0236, "ci95_video": [-0.1506, 0.045],
                   "verdict": "", "n_flips": 1, "flips": [LOSER]},
    "underseg_error": {"n_clips": 14, "n_videos": 7, "sign": -1, "delta": 0.0228, "ci95_video": [-0.0422, 0.1492],
                       "verdict": "", "n_flips": 1, "flips": [LOSER]},
}
WORKBENCH_LOSER_ROW = {"video": LOSER, "n_clips": 2, "n_videos_wo": 6, "own_delta": -0.4, "delta_wo": 0.0392,
                       "ci95_wo": [0.0322, 0.0475], "ci_width_ratio": 0.0783, "verdict_wo": "★"}


def test_the_verdicts_are_the_workbench_s():
    base, cond = pilot_pair()
    res = LV.lovo_pair(base, cond, ["inst_F1_50", "underseg_error"])
    for key, want in WORKBENCH.items():
        assert {k: res[key][k] for k in want} == want, key
        assert [p["video"] for p in res[key]["per_video"]] == sorted(VIDEOS), key
    loser = next(p for p in res["inst_F1_50"]["per_video"] if p["video"] == LOSER)
    assert loser == WORKBENCH_LOSER_ROW


# ---------------------------------------------------------------- the video that holds a verdict


def test_the_video_that_makes_the_spread_is_named_and_the_interval_narrows():
    base, cond = pilot_pair()
    r = LV.lovo_pair(base, cond, ["inst_F1_50"])["inst_F1_50"]
    assert (r["verdict"], r["flips"]) == ("", [LOSER])
    loser = next(p for p in r["per_video"] if p["video"] == LOSER)
    assert loser["verdict_wo"] == "★" and loser["ci_width_ratio"] < 1
    # Every other video left out keeps the loser in, and with it no mark.
    assert all(p["verdict_wo"] == "" for p in r["per_video"] if p["video"] != LOSER)


def test_a_star_that_every_video_holds_names_no_video():
    # A flip is a change of mark, not a mark: every subset keeps the star here.
    d = np.array([0.05, 0.04, 0.06, 0.05, 0.03, 0.05, 0.04, 0.06])
    vids = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    r = LV.lovo(d, vids, +1)
    assert r["verdict"] == "★" and all(p["verdict_wo"] == "★" for p in r["per_video"])
    assert (r["n_flips"], r["flips"]) == (0, [])


def test_a_key_where_less_is_better_is_marked_in_its_direction():
    # The six videos where `underseg_error` fell are where the cond is better:
    # without the loser that is a star, where a verdict read without the
    # direction would give a cross.
    base, cond = pilot_pair()
    r = LV.lovo_pair(base, cond, ["underseg_error"])["underseg_error"]
    loser = next(p for p in r["per_video"] if p["video"] == LOSER)
    assert loser["delta_wo"] < 0 and loser["verdict_wo"] == "★"
    d = np.array([-0.05, -0.04, -0.06, -0.05, -0.03, -0.05])
    vids = np.array(["a", "a", "b", "b", "c", "c"])
    assert LV.lovo(d, vids, -1)["verdict"] == "★" and LV.lovo(d, vids, +1)["verdict"] == "✗"


def test_the_delta_and_interval_stay_as_cond_minus_base_whatever_the_direction():
    d = np.array([-0.05, -0.04, -0.06, -0.05, -0.03, -0.05])
    vids = np.array(["a", "a", "b", "b", "c", "c"])
    up, down = LV.lovo(d, vids, +1), LV.lovo(d, vids, -1)
    assert (up["delta"], up["ci95_video"]) == (down["delta"], down["ci95_video"])
    assert up["delta"] < 0 and (up["sign"], down["sign"]) == (1, -1)


def test_a_video_is_left_out_only_while_three_videos_remain():
    d = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
    three = np.array(["a", "a", "b", "b", "c", "c", "c", "c"])
    assert LV.lovo(d, three, +1)["per_video"] == []
    four = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    rows = LV.lovo(d, four, +1)["per_video"]
    assert [p["video"] for p in rows] == ["a", "b", "c", "d"]
    assert all(p["n_videos_wo"] == 3 and p["n_clips"] == 2 for p in rows)


def test_fewer_than_two_videos_is_refused():
    with pytest.raises(ValueError, match="1 video has no interval"):
        LV.lovo(np.array([0.1, 0.2]), np.array(["a", "a"]), +1)


def test_a_reference_value_is_refused():
    # No measure over time carries a star, so no video can hold its mark.
    base, cond = pilot_pair()
    with pytest.raises(ValueError, match="time_IoU is a reference value"):
        LV.lovo_pair(base, cond, ["time_IoU"])
    ev = evaluator_scores("base", 0.0, 1)
    for key in ("time_IoU", metric_key("unlabelled_share", "all")):
        with pytest.raises(ValueError, match="reference value"):
            LV.direction(ev, key)
    with pytest.raises(ValueError, match="sign is"):
        LV.lovo(np.array([0.1, 0.2, 0.3, 0.4]), np.array(["a", "a", "b", "b"]), 0)


def test_a_key_with_no_direction_is_refused():
    base, _ = pilot_pair()
    with pytest.raises(ValueError, match="no direction is known"):
        LV.direction(base, "no_such_key")
    with pytest.raises(ValueError, match="no direction is known"):
        LV.direction(evaluator_scores("base", 0.0, 1), "inst_F1_50")


def test_the_evaluator_s_keys_take_the_direction_of_the_metrics_table():
    ev = evaluator_scores("base", 0.0, 1)
    assert LV.direction(ev, metric_key("F1_50", "all")) == +1
    assert LV.direction(ev, metric_key("VI_split", "all")) == -1


def test_a_key_defined_on_too_few_clips_is_counted_on_the_clips_both_define():
    base, cond = pilot_pair()
    r = LV.lovo_pair(base, cond, ["SQ"])["SQ"]
    assert (r["n_clips"], r["n_videos"]) == (13, 7)


def test_a_key_defined_on_no_common_clip_is_left_out():
    base, cond = pilot_pair()
    for row in cond["per_clip"]:
        row["SQ"] = None
    assert LV.lovo_pair(base, cond, ["SQ", "inst_F1_50"]).keys() == {"inst_F1_50"}


def test_the_pair_refuses_two_evaluators():
    base, cond = pilot_pair()
    cond["eval_code_sha"] = "b" * 64
    with pytest.raises(ValueError, match="different evaluators"):
        LV.lovo_pair(base, cond, ["inst_F1_50"])


def test_the_verdicts_are_taken_on_the_clips_the_check_compared(monkeypatch):
    # One population: the one `check_comparable` returns, not one taken from the rows again.
    real = LV.check_comparable

    def four_videos(*args, **kwargs):
        chk = real(*args, **kwargs)
        return {**chk, "clips": chk["clips"][:8]}

    monkeypatch.setattr(LV, "check_comparable", four_videos)
    base, cond = pilot_pair()
    r = LV.lovo_pair(base, cond, ["inst_F1_50"])["inst_F1_50"]
    assert (r["n_clips"], r["n_videos"]) == (8, 4)


def test_a_value_that_rounds_to_zero_keeps_two_digits():
    # The mark follows the sign of the interval's end; -0.0 would hide it.
    assert LV._r(-2.4e-05) == -2.4e-05 and LV._r(0.0) == 0.0 and LV._r(0.123449) == 0.1234
    assert LV._f(-2.4e-05) == "-2.40e-05" and LV._f(0.0) == "+0.0000" and LV._f(0.05) == "+0.0500"


# ---------------------------------------------------------------- the command


def _write(tmp_path, **summaries):
    for tag, summary in summaries.items():
        (tmp_path / f"{tag}.json").write_text(json.dumps(summary), encoding="utf-8")


def _run(tmp_path, *args):
    return subprocess.run([sys.executable, "-m", "evalkit.tools.lovo_verdict", "--eval-dir", str(tmp_path), *args],
                          capture_output=True, text=True, cwd=REPO)


def test_the_command_writes_the_verdicts_and_names_the_video(tmp_path):
    base, cond = pilot_pair()
    _write(tmp_path, base=base, cond=cond)
    out = tmp_path / "lovo.json"
    run = _run(tmp_path, "--pairs", "base:cond", "--keys", "inst_F1_50,underseg_error", "--out", str(out))
    assert run.returncode == 0, run.stderr
    assert f"without {LOSER}" in run.stdout and "narrower, so this video made the spread" in run.stdout
    got = json.loads(out.read_text(encoding="utf-8"))
    assert list(got) == ["verdict_rule", "seed_scheme", "min_videos_after_drop", "pairs"]
    assert (got["verdict_rule"], got["seed_scheme"]) == (PS.VERDICT_RULE, PS.SEED_SCHEME)
    assert got["pairs"]["base:cond"] == LV.lovo_pair(base, cond, ["inst_F1_50", "underseg_error"])


def test_the_command_records_the_rule_of_evaluator_jsons(tmp_path):
    _write(tmp_path, base=evaluator_scores("base", 0.0, 1), cond=evaluator_scores("cond", 0.05, 2))
    out = tmp_path / "lovo.json"
    run = _run(tmp_path, "--pairs", "base:cond", "--keys", metric_key("F1_50", "all"), "--out", str(out))
    assert run.returncode == 0, run.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["propagation"] == "both_ways_from_centre"


def test_the_command_refuses_a_run_whose_pairs_hold_two_rules(tmp_path):
    _write(tmp_path, fwd=evaluator_scores("fwd", 0.0, 1, rule="forward_from_first"),
           ctr=evaluator_scores("ctr", 0.0, 2), pf=evaluator_scores("pf", 0.0, 3, rule="per_frame"))
    run = _run(tmp_path, "--pairs", "pf:fwd,pf:ctr", "--keys", metric_key("F1_50", "all"))
    assert run.returncode != 0 and "different rules" in run.stderr


def test_the_command_refuses_a_reference_value_before_any_bootstrap(tmp_path):
    base, cond = pilot_pair()
    _write(tmp_path, base=base, cond=cond)
    run = _run(tmp_path, "--pairs", "base:cond", "--keys", "inst_F1_50,time_IoU")
    assert run.returncode != 0 and "--keys: time_IoU is a reference value" in run.stderr
    assert "=====" not in run.stdout


@pytest.mark.parametrize("pairs, said", [("base", "<base>:<cond>"), ("base:cond:x", "<base>:<cond>"),
                                         ("base:nowhere", "nowhere.json")])
def test_the_command_refuses_a_bad_pair_with_a_message_not_a_traceback(tmp_path, pairs, said):
    base, cond = pilot_pair()
    _write(tmp_path, base=base, cond=cond)
    run = _run(tmp_path, "--pairs", pairs, "--keys", "inst_F1_50")
    assert run.returncode != 0 and said in run.stderr and "Traceback" not in run.stderr
