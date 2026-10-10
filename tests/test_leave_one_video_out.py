"""Each check of `leave_one_video_out` is shown to refuse what it should, or to name the video it should."""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from evalkit.tools import leave_one_video_out as LV
from evalkit.tools import paired_stats as PS
from evalkit.tools.scores import CLIP_METRICS, FRAME_METRICS, PILOT_EVAL_CODE_SHA, metric_key

pytest.importorskip("scipy", reason="paired_stats needs the `tools` extra")

REPO = Path(__file__).resolve().parent.parent

VIDEOS = ("adrenalectomy__16GPCUPkXYQ", "appendectomy__41RKDh3INiU", "gastric_surgery__4FHGGFZsPzw",
          "lar___7H7G-4sevQ", "rectopexy__IJfgUvxoTkM", "hemicolectomy__5YDMlxTl0k8", "nephrectomy__abc123")
LOSER = VIDEOS[-1]


def pilot_pair(videos: tuple[str, ...] = VIDEOS) -> tuple[dict, dict]:
    """A base and a cond pilot JSON, 2 clips per video.

    The cond gains a little on every video but the last, and loses 0.4 on
    the last, on `inst_F1_50` (higher is better) and on `underseg_error`
    (lower is better) alike. `SQ` is None on the first clip.
    """
    rnd = random.Random(3)
    loser = videos[-1]
    clips = [f"{v}__gt_{i:04d}" for v in videos for i in (1, 2)]
    rows = {"base": [], "cond": []}
    for n, c in enumerate(clips):
        base, cond = {"clip": c, "extra_ignore": []}, {"clip": c, "extra_ignore": []}
        for key, sign in (("inst_F1_50", +1), ("underseg_error", -1), ("SQ", +1), ("time_IoU", +1)):
            v = round(rnd.uniform(0.3, 0.6), 4)
            gain = -0.4 if c.startswith(loser) else round(rnd.uniform(0.01, 0.06), 4)
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


# What the workbench's `lovo_verdict.py` wrote on exactly `pilot_pair()`, with
# its `verdict` named `mark`, so that the port is held to its numbers rather
# than only to its own tests.
WORKBENCH = {
    "inst_F1_50": {"n_clips": 14, "n_videos": 7, "sign": 1, "delta": -0.0236, "ci95_video": [-0.1506, 0.045],
                   "mark": "", "n_flips": 1, "flips": [LOSER]},
    "underseg_error": {"n_clips": 14, "n_videos": 7, "sign": -1, "delta": 0.0228, "ci95_video": [-0.0422, 0.1492],
                       "mark": "", "n_flips": 1, "flips": [LOSER]},
}
WORKBENCH_LOSER_ROW = {"video": LOSER, "n_clips": 2, "n_videos_wo": 6, "own_delta": -0.4, "delta_wo": 0.0392,
                       "ci95_wo": [0.0322, 0.0475], "ci_width_ratio": 0.0783, "mark_wo": "★"}


def metrics_of(base: dict, cond: dict, keys: list[str]) -> dict:
    return LV.leave_each_out_of_pair(base, cond, keys)["metrics"]


def test_the_marks_are_the_workbench_s():
    base, cond = pilot_pair()
    res = metrics_of(base, cond, ["inst_F1_50", "underseg_error"])
    for key, want in WORKBENCH.items():
        assert {k: res[key][k] for k in want} == want, key
        assert [p["video"] for p in res[key]["per_video"]] == sorted(VIDEOS), key
    loser = next(p for p in res["inst_F1_50"]["per_video"] if p["video"] == LOSER)
    assert loser == WORKBENCH_LOSER_ROW


# ---------------------------------------------------------------- the video whose removal changes the mark


def test_the_video_that_hides_a_star_is_named_and_the_interval_narrows():
    base, cond = pilot_pair()
    r = metrics_of(base, cond, ["inst_F1_50"])["inst_F1_50"]
    assert (r["mark"], r["flips"]) == ("", [LOSER])
    loser = next(p for p in r["per_video"] if p["video"] == LOSER)
    assert loser["mark_wo"] == "★" and loser["ci_width_ratio"] < 1
    # Every other video left out keeps the loser in, and with it no mark.
    assert all(p["mark_wo"] == "" for p in r["per_video"] if p["video"] != LOSER)


def test_a_star_that_one_video_holds_is_lost_without_it_though_the_interval_narrows():
    # Video `a` gains 0.3 and the five others about zero: the star rests on `a` alone. Its interval
    # narrows without `a`, so a narrower interval does not say that the video only made the spread.
    d = np.array([0.3, 0.3, 0.028, 0.008, 0.018, 0.0, 0.021, 0.019, 0.004, -0.028, 0.024, 0.021])
    vids = np.array(list("aabbccddeeff"))
    r = LV.leave_each_out(d, vids, +1)
    assert (r["mark"], r["n_flips"], r["flips"]) == ("★", 1, ["a"])
    a = r["per_video"][0]
    assert a["video"] == "a" and a["mark_wo"] == "" and a["ci95_wo"][0] < 0 < a["ci95_wo"][1]
    assert a["ci_width_ratio"] < 1


def test_a_star_that_every_video_holds_names_no_video():
    # A flip is a change of mark, not a mark: every subset keeps the star here.
    d = np.array([0.05, 0.04, 0.06, 0.05, 0.03, 0.05, 0.04, 0.06])
    vids = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    r = LV.leave_each_out(d, vids, +1)
    assert r["mark"] == "★" and all(p["mark_wo"] == "★" for p in r["per_video"])
    assert (r["n_flips"], r["flips"]) == (0, [])


def test_a_key_where_less_is_better_is_marked_in_its_direction():
    # The six videos where `underseg_error` fell are where the cond is better:
    # without the loser that is a star, where a mark read without the
    # direction would be a cross.
    base, cond = pilot_pair()
    r = metrics_of(base, cond, ["underseg_error"])["underseg_error"]
    loser = next(p for p in r["per_video"] if p["video"] == LOSER)
    assert loser["delta_wo"] < 0 and loser["mark_wo"] == "★"
    d = np.array([-0.05, -0.04, -0.06, -0.05, -0.03, -0.05, -0.04, -0.05])
    vids = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    assert LV.leave_each_out(d, vids, -1)["mark"] == "★" and LV.leave_each_out(d, vids, +1)["mark"] == "✗"


def test_the_delta_and_interval_stay_as_cond_minus_base_whatever_the_direction():
    d = np.array([-0.05, -0.04, -0.06, -0.05, -0.03, -0.05, -0.04, -0.05])
    vids = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    up, down = LV.leave_each_out(d, vids, +1), LV.leave_each_out(d, vids, -1)
    assert (up["delta"], up["ci95_video"]) == (down["delta"], down["ci95_video"])
    assert up["delta"] < 0 and (up["sign"], down["sign"]) == (1, -1)


@pytest.mark.parametrize("vids", [list("aabbcccc"), list("aaaabbbb"), list("aaaaaaaa")])
def test_a_key_on_three_videos_or_fewer_is_refused_since_no_video_can_be_left_out(vids):
    # Leaving none out and printing that no video changes the mark would claim a check that never ran.
    d = np.array([0.05, 0.06, 0.04, 0.05, 0.03, 0.04, 0.05, 0.06])
    with pytest.raises(ValueError, match=f"from {len(set(vids))} video.s.: leaving one out would leave fewer than 3"):
        LV.leave_each_out(d, np.array(vids), +1)


def test_every_video_is_left_out_once():
    d = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
    four = np.array(["a", "a", "b", "b", "c", "c", "d", "d"])
    rows = LV.leave_each_out(d, four, +1)["per_video"]
    assert [p["video"] for p in rows] == ["a", "b", "c", "d"]
    assert all(p["n_videos_wo"] == 3 and p["n_clips"] == 2 for p in rows)


def test_a_reference_value_is_refused():
    # No measure over time carries a star, so leaving a video out cannot change its mark.
    base, cond = pilot_pair()
    with pytest.raises(ValueError, match="time_IoU has direction 0 and is never marked"):
        LV.leave_each_out_of_pair(base, cond, ["time_IoU"])
    ev = evaluator_scores("base", 0.0, 1)
    for key in ("time_IoU", metric_key("unlabelled_share", "all")):
        with pytest.raises(ValueError, match="has direction 0"):
            LV.direction(ev, key)
    with pytest.raises(ValueError, match="sign is"):
        LV.leave_each_out(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]), np.array(list("aabbccdd")), 0)


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


def test_a_key_undefined_on_a_clip_is_counted_on_the_clips_both_define():
    base, cond = pilot_pair()
    r = metrics_of(base, cond, ["SQ"])["SQ"]
    assert (r["n_clips"], r["n_videos"]) == (13, 7)


def test_a_key_in_no_row_is_refused():
    # A key the JSON does not hold is a wrong key, not a key undefined on every clip.
    base, cond = pilot_pair()
    for row in cond["per_clip"]:
        del row["SQ"]
    with pytest.raises(ValueError, match="SQ is in no row of cond"):
        LV.leave_each_out_of_pair(base, cond, ["SQ"])


def test_the_ratio_of_widths_is_none_when_the_whole_interval_has_no_width():
    r = LV.leave_each_out(np.full(8, 0.05), np.array(list("aabbccdd")), +1)
    assert r["ci95_video"] == [0.05, 0.05] and all(p["ci_width_ratio"] is None for p in r["per_video"])


def test_a_key_defined_on_no_common_clip_is_left_out():
    base, cond = pilot_pair()
    for row in cond["per_clip"]:
        row["SQ"] = None
    assert metrics_of(base, cond, ["SQ", "inst_F1_50"]).keys() == {"inst_F1_50"}


def test_the_pair_refuses_two_evaluators():
    base, cond = pilot_pair()
    cond["eval_code_sha"] = "b" * 64
    with pytest.raises(ValueError, match="different evaluators"):
        LV.leave_each_out_of_pair(base, cond, ["inst_F1_50"])


def test_the_marks_are_computed_on_the_clips_the_check_compared(monkeypatch):
    # One population: the one `check_comparable` returns, not one taken from the rows again.
    real = LV.check_comparable

    def four_videos(*args, **kwargs):
        chk = real(*args, **kwargs)
        return {**chk, "clips": chk["clips"][:8]}

    monkeypatch.setattr(LV, "check_comparable", four_videos)
    base, cond = pilot_pair()
    r = metrics_of(base, cond, ["inst_F1_50"])["inst_F1_50"]
    assert (r["n_clips"], r["n_videos"]) == (8, 4)


@pytest.mark.parametrize("n, said", [(3, "on 3 videos the interval is the range of the video means"),
                                     (4, "an interval on 4 videos is thin"), (5, None)])
def test_the_note_says_when_an_interval_is_the_range_of_the_means_or_thin(n, said):
    assert (said in LV._note(n)) if said else LV._note(n) == ""


def test_three_video_intervals_are_the_range_of_the_video_means():
    # What the note on three videos says, shown on the bootstrap itself.
    rng = np.random.default_rng(0)
    for _ in range(20):
        vids = np.array([v for v, n in zip("abc", rng.integers(1, 5, size=3)) for _ in range(n)])
        d = rng.normal(0.02, 0.05, size=len(vids))
        means = [d[vids == v].mean() for v in "abc"]
        assert np.allclose(PS.boot_ci(d, vids), (min(means), max(means)))


# ---------------------------------------------------------------- the command


def _write(tmp_path, **summaries):
    for tag, summary in summaries.items():
        (tmp_path / f"{tag}.json").write_text(json.dumps(summary), encoding="utf-8")


def _run(tmp_path, *args):
    return subprocess.run([sys.executable, "-m", "evalkit.tools.leave_one_video_out", "--eval-dir", str(tmp_path),
                           *args], capture_output=True, text=True, cwd=REPO)


def test_the_command_writes_the_marks_and_names_the_video(tmp_path):
    base, cond = pilot_pair()
    _write(tmp_path, base=base, cond=cond)
    out = tmp_path / "left_out.json"
    run = _run(tmp_path, "--pairs", "base:cond", "--keys", "inst_F1_50,underseg_error", "--out", str(out))
    assert run.returncode == 0, run.stderr
    assert f"without {LOSER}" in run.stdout
    # The row gives the mean's move and the width, and reads neither as the cause.
    assert "the mean moved +0.0628, the interval 0.08 times as wide)" in run.stdout
    assert "spread" not in run.stdout and "carried" not in run.stdout
    assert "<-" not in run.stdout
    got = json.loads(out.read_text(encoding="utf-8"))
    assert list(got) == ["mark_rule", "n_boot", "seed", "seed_scheme", "min_videos_after_drop", "pairs"]
    assert (got["mark_rule"], got["seed_scheme"]) == (PS.MARK_RULE, PS.SEED_SCHEME)
    assert (got["n_boot"], got["seed"]) == (PS.N_BOOT, PS.SEED)
    # A pilot JSON's pair records its evaluator as an evaluator JSON's does.
    assert got["pairs"]["base:cond"]["eval_code"] == PILOT_EVAL_CODE_SHA[:16]
    assert got["pairs"]["base:cond"] == LV.leave_each_out_of_pair(base, cond, ["inst_F1_50", "underseg_error"])


def test_the_command_records_the_rule_of_evaluator_jsons(tmp_path):
    _write(tmp_path, base=evaluator_scores("base", 0.0, 1), cond=evaluator_scores("cond", 0.05, 2))
    out = tmp_path / "left_out.json"
    run = _run(tmp_path, "--pairs", "base:cond", "--keys", metric_key("F1_50", "all"), "--out", str(out))
    assert run.returncode == 0, run.stderr
    got = json.loads(out.read_text(encoding="utf-8"))
    assert got["propagation"] == "both_ways_from_centre"
    assert got["pairs"]["base:cond"]["eval_code"] == "a" * 16


def test_the_printout_notes_a_thin_interval_and_a_range_of_three_means(tmp_path):
    # On four videos the whole interval is thin, and without the loser the three left give the range of their means.
    base, cond = pilot_pair(VIDEOS[:3] + (LOSER,))
    _write(tmp_path, base=base, cond=cond)
    run = _run(tmp_path, "--pairs", "base:cond", "--keys", "inst_F1_50")
    assert run.returncode == 0, run.stderr
    assert "(8 clips / 4 videos)  <- an interval on 4 videos is thin" in run.stdout
    row = next(line for line in run.stdout.splitlines() if f"without {LOSER}" in line)
    assert row.endswith("<- on 3 videos the interval is the range of the video means")


def test_out_takes_a_bare_file_name(tmp_path):
    base, cond = pilot_pair()
    _write(tmp_path, base=base, cond=cond)
    run = subprocess.run([sys.executable, "-m", "evalkit.tools.leave_one_video_out", "--eval-dir", ".",
                          "--pairs", "base:cond", "--keys", "inst_F1_50", "--out", "left_out.json"],
                         capture_output=True, text=True, cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(REPO)})
    assert run.returncode == 0, run.stderr
    assert (tmp_path / "left_out.json").exists()


def test_the_command_refuses_a_run_whose_pairs_hold_two_rules(tmp_path):
    _write(tmp_path, fwd=evaluator_scores("fwd", 0.0, 1, rule="forward_from_first"),
           ctr=evaluator_scores("ctr", 0.0, 2), pf=evaluator_scores("pf", 0.0, 3, rule="per_frame"))
    run = _run(tmp_path, "--pairs", "pf:fwd,pf:ctr", "--keys", metric_key("F1_50", "all"))
    assert run.returncode != 0 and "different rules" in run.stderr


def test_the_command_refuses_a_reference_value_before_any_bootstrap(tmp_path):
    base, cond = pilot_pair()
    _write(tmp_path, base=base, cond=cond)
    run = _run(tmp_path, "--pairs", "base:cond", "--keys", "inst_F1_50,time_IoU")
    assert run.returncode != 0 and "--keys: time_IoU has direction 0" in run.stderr
    assert "=====" not in run.stdout


def test_the_command_refuses_keys_that_name_no_key(tmp_path):
    base, cond = pilot_pair()
    _write(tmp_path, base=base, cond=cond)
    run = _run(tmp_path, "--pairs", "base:cond", "--keys", " , ")
    assert run.returncode != 0 and "--keys names no key" in run.stderr


@pytest.mark.parametrize("pairs, said", [("base", "<base>:<cond>"), ("base:cond:x", "<base>:<cond>"),
                                         ("base:nowhere", "nowhere.json")])
def test_the_command_refuses_a_bad_pair_with_a_message_not_a_traceback(tmp_path, pairs, said):
    base, cond = pilot_pair()
    _write(tmp_path, base=base, cond=cond)
    run = _run(tmp_path, "--pairs", pairs, "--keys", "inst_F1_50")
    assert run.returncode != 0 and said in run.stderr and "Traceback" not in run.stderr
