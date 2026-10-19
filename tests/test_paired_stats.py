"""The invariants of the statistics that give a claim a star, and the one definition of it.

The bootstrap tests are the workbench's: each one pins a property that
was once broken there. The scans that hold `AGENTS.md`'s mark rule over
every tracked Python file are in `tests/test_mark_rule.py`.
"""
import inspect
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from evalkit.tools import paired_stats as PS
from evalkit.tools.scores import CLIP_METRICS, FRAME_METRICS, PILOT_EVAL_CODE_SHA, PILOT_SIGNS, metric_key, sign_of_key

pytest.importorskip("scipy", reason="the Wilcoxon test needs the `tools` extra")

REPO = Path(__file__).resolve().parent.parent


def _data(n=27, n_vid=9, seed=7):
    rng = np.random.default_rng(seed)
    d = rng.normal(0.02, 0.05, n)
    vids = np.array([f"V{i % n_vid}" for i in range(n)])
    return d, vids


# ---------------------------------------------------------------- the bootstrap


def test_boot_ci_is_call_order_independent():
    d, vids = _data()
    first = PS.boot_ci(d, vids)
    other, ovids = _data(seed=99)
    for _ in range(5):
        PS.boot_ci(other, ovids)
        PS.boot_ci(d, None)
    assert PS.boot_ci(d, vids) == first


def test_boot_ci_differs_between_clip_and_video_units():
    d, vids = _data()
    assert PS.boot_ci(d, None) != PS.boot_ci(d, vids)


def test_boot_ci_video_unit_is_wider_when_clips_are_correlated():
    # The reason for resampling videos. On independent clips the video
    # interval is not wider, so the videos get an effect of their own.
    rng = np.random.default_rng(3)
    n_vid, per_vid = 9, 3
    vid_effect = rng.normal(0, 0.08, n_vid)
    d, vids = [], []
    for v in range(n_vid):
        for _ in range(per_vid):
            d.append(vid_effect[v] + rng.normal(0, 0.01))
            vids.append(f"V{v}")
    d, vids = np.array(d), np.array(vids)
    lo_c, hi_c = PS.boot_ci(d, None)
    lo_v, hi_v = PS.boot_ci(d, vids)
    assert (hi_v - lo_v) > (hi_c - lo_c)


def test_boot_ci_refuses_a_generator():
    d, vids = _data()
    with pytest.raises(TypeError, match="no longer takes a generator"):
        PS.boot_ci(d, vids, np.random.default_rng(0))


def test_boot_ci_seed_changes_the_draw():
    d, vids = _data()
    assert PS.boot_ci(d, vids, seed=1) != PS.boot_ci(d, vids, seed=2)


def test_boot_ci_takes_the_groups_as_a_list():
    # A list compared with one group was a single False, and the interval
    # came out (nan, nan): no mark, and no error.
    d, vids = _data()
    assert PS.boot_ci(d, list(vids)) == PS.boot_ci(d, vids)
    assert PS.boot_ci(list(d), None) == PS.boot_ci(d, None)


def test_boot_ci_refuses_groups_of_another_length():
    d, vids = _data()
    with pytest.raises(ValueError, match="units for"):
        PS.boot_ci(d, vids[:-1])


def test_boot_ci_refuses_fewer_than_two_units():
    # One video resampled is always that video, and the interval is a point:
    # a star or a cross unless the mean is exactly zero.
    with pytest.raises(ValueError, match="1 video has no interval"):
        PS.boot_ci(np.array([0.1, 0.09, 0.12]), np.array(["V1", "V1", "V1"]))
    with pytest.raises(ValueError, match="1 difference has no interval"):
        PS.boot_ci(np.array([0.1]), None)
    lo, hi = PS.boot_ci(np.array([0.1, 0.2]), np.array(["V1", "V2"]))
    assert lo < hi


def test_wilcoxon_per_video_is_undefined_below_six_videos():
    d, vids = _data(n=10, n_vid=5)
    assert PS.wilcoxon_video(d, vids) is None
    d, vids = _data(n=12, n_vid=6)
    assert PS.wilcoxon_video(d, vids) is not None


# ---------------------------------------------------------------- the mark


ENOUGH = PS.MIN_VIDEOS_FOR_MARK


def test_mark_of_reads_the_video_interval_alone():
    assert PS.mark_of([0.01, 0.05], n_videos=ENOUGH) == "★"
    assert PS.mark_of([-0.05, -0.01], n_videos=ENOUGH) == "✗"
    assert PS.mark_of([-0.01, 0.05], n_videos=ENOUGH) == ""
    assert PS.mark_of([0.0, 0.05], n_videos=ENOUGH) == "" and PS.mark_of([-0.05, 0.0], n_videos=ENOUGH) == ""
    assert PS.mark_of(None, n_videos=ENOUGH) == "" and PS.mark_of([None, 0.05], n_videos=ENOUGH) == ""


def test_mark_of_takes_the_interval_the_direction_and_the_videos_and_no_p_value():
    assert list(inspect.signature(PS.mark_of).parameters) == ["ci95_video", "sign", "n_videos"]


def test_the_fewest_videos_for_a_mark_is_six_and_the_rule_says_so():
    assert PS.MIN_VIDEOS_FOR_MARK == 6
    assert "on 6 videos or more" in PS.MARK_RULE


def test_three_video_intervals_are_the_range_of_the_video_means():
    # Why a mark needs more videos than an interval does, shown on the bootstrap itself.
    rng = np.random.default_rng(0)
    for _ in range(20):
        vids = np.array([v for v, n in zip("abc", rng.integers(1, 5, size=3)) for _ in range(n)])
        d = rng.normal(0.02, 0.05, size=len(vids))
        means = [d[vids == v].mean() for v in "abc"]
        assert np.allclose(PS.boot_ci(d, vids), (min(means), max(means)))


@pytest.mark.parametrize("sign, ci", [(+1, [0.01, 0.05]), (+1, [-0.05, -0.01]), (-1, [0.01, 0.05])])
def test_an_interval_on_five_videos_gives_no_mark_and_on_six_it_does(sign, ci):
    # Below the floor the interval is read for its sign only.
    assert PS.mark_of(ci, sign, n_videos=5) == ""
    assert PS.mark_of(ci, sign, n_videos=6) in ("★", "✗")
    assert PS.sign_only_note(5) == PS.sign_only_note(2) == "read for its sign only: a mark needs 6 videos"
    assert PS.sign_only_note(6) == ""
    # One video gives no interval, so there is nothing to read for its sign.
    assert PS.sign_only_note(1) == PS.sign_only_note(0) == ""


@pytest.mark.parametrize("n_videos", [-1, 6.0, True, np.int64(6), None])
def test_mark_of_refuses_a_count_of_videos_that_is_not_an_int_of_zero_or_more(n_videos):
    with pytest.raises(ValueError, match="n_videos is"):
        PS.mark_of([0.01, 0.05], n_videos=n_videos)


def test_mark_of_reads_the_interval_in_the_metric_s_direction():
    # `VI_split` falls when a method improves; `time_IoU` is a reference
    # value and is never marked, whichever way it moved.
    assert (PS.mark_of([0.01, 0.05], sign=-1, n_videos=ENOUGH) == "✗"
            and PS.mark_of([-0.05, -0.01], sign=-1, n_videos=ENOUGH) == "★")
    assert PS.mark_of([-0.01, 0.05], sign=-1, n_videos=ENOUGH) == ""
    assert (PS.mark_of([0.01, 0.05], sign=0, n_videos=ENOUGH) == ""
            and PS.mark_of([-0.05, -0.01], sign=0, n_videos=ENOUGH) == "")
    with pytest.raises(ValueError, match="sign is"):
        PS.mark_of([0.01, 0.05], sign=2, n_videos=ENOUGH)
    with pytest.raises(ValueError, match="sign is"):
        PS.mark_of([0.01, 0.05], sign=None, n_videos=ENOUGH)


@pytest.mark.parametrize("sign", [True, False, 1.0, -1.0, np.int64(1)])
def test_mark_of_refuses_a_sign_that_only_equals_an_int(sign):
    """`True == 1` and `1.0 == 1`, so a membership test alone lets a flag or a float through."""
    with pytest.raises(ValueError, match="sign is"):
        PS.mark_of([0.01, 0.05], sign=sign, n_videos=ENOUGH)


def test_is_star_matches_mark_of():
    assert (PS.is_star([0.01, 0.05], n_videos=ENOUGH) and not PS.is_star([-0.05, -0.01], n_videos=ENOUGH)
            and not PS.is_star([-0.01, 0.05], n_videos=ENOUGH))
    assert PS.is_star([-0.05, -0.01], sign=-1, n_videos=ENOUGH)
    assert not PS.is_star([0.01, 0.05], sign=-1, n_videos=ENOUGH)
    assert not PS.is_star([0.01, 0.05], n_videos=ENOUGH - 1)


def test_every_pilot_key_has_a_direction_from_the_one_table():
    assert set(PS.PILOT_KEYS) <= set(PILOT_SIGNS)
    assert PILOT_SIGNS["underseg_error"] == -1
    assert all(PILOT_SIGNS[k] == +1 for k in PS.PILOT_KEYS if k != "underseg_error")
    assert all(sign_of_key(pilot_scores("a", 0.0, 1), k) == PILOT_SIGNS[k] for k in PS.PILOT_KEYS)


def test_video_of():
    assert PS.video_of("gastric_surgery__4FHGGFZsPzw__gt_0004") == "gastric_surgery__4FHGGFZsPzw"
    assert PS.video_of("cholecystectomy___-aytJndMV4__gt_0002") == "cholecystectomy___-aytJndMV4"
    assert PS.video_of("VID26_s15_1855_crop") == "VID26"


# ---------------------------------------------------------------- the pair, against the workbench

VIDEOS = ("adrenalectomy__16GPCUPkXYQ", "appendectomy__41RKDh3INiU", "gastric_surgery__4FHGGFZsPzw",
          "lar___7H7G-4sevQ", "rectopexy__IJfgUvxoTkM", "hemicolectomy__5YDMlxTl0k8", "nephrectomy__abc123")


def pilot_scores(tag: str, shift: float, seed: int) -> dict:
    """A pilot-layout score JSON on 7 videos, 2 clips each, with `SQ` None on some clips; `seed` draws the values."""
    rnd = random.Random(seed)
    clips = [f"{v}__gt_{i:04d}" for v in VIDEOS for i in (1, 2)]
    rows = []
    for c in clips:
        r = {"clip": c, "extra_ignore": []}
        for k in PS.PILOT_KEYS:
            v = round(min(1.0, max(0.0, rnd.gauss(0.5 + shift, 0.15))), 4)
            r[k] = None if k == "SQ" and rnd.random() < 0.2 else v
        rows.append(r)
    return dict(tag=tag, eval_code_sha=PILOT_EVAL_CODE_SHA, eval_code_tag="t", eval_version=2,
                dataset="atlas", tissue_ignore=[1, 2], clips=clips, per_clip=rows)


def evaluator_scores(tag: str, shift: float, seed: int, views=("all",)) -> dict:
    """An evaluator-layout score JSON on 3 videos, 2 clips each, every metric in every view; `seed` draws the values."""
    rnd = random.Random(seed)
    clips = [f"{v}__gt_{i:04d}" for v in VIDEOS[:3] for i in (1, 2)]
    keys = [metric_key(m, v) for v in views for m in FRAME_METRICS] + list(CLIP_METRICS)
    rows = [{"clip": c, **{k: round(rnd.gauss(0.5 + shift, 0.15), 4) for k in keys}} for c in clips]
    return dict(tag=tag, eval_code_sha="a" * 64, dataset="atlas", pilot=False, class_set="original",
                views=list(views), clips=clips,
                input_shas={c: {"gt_masks": "g" * 64, "depth": "d" * 64, "crop": "c" * 64, "frames": "f" * 64, "predictions": f"p{c}"} for c in clips},
                versions={"python": "3.12.0", "numpy": "2.0.0"}, propagation="both_ways_from_centre",
                seed_source="sam", gt_informed=False, per_clip=rows)


# What the workbench's paired_stats wrote on exactly these two JSONs
# (`base` at shift 0 with seed 5, `cond` at shift 0.03 with seed 6), so that this port
# is held to its bytes rather than only to its own tests. The port adds
# `sign` to the row; everything the workbench wrote is checked unchanged.
WORKBENCH = {
    "inst_F1_50": {"n_clips": 14, "n_videos": 7, "base_mean": 0.5153, "base_sd": 0.1615, "cond_mean": 0.4957,
                   "cond_sd": 0.1531, "delta_mean": -0.0196, "delta_sd": 0.247, "delta_median": -0.0707,
                   "wins": 7, "losses": 7, "ties": 0, "wilcoxon_p": 0.9032, "wilcoxon_p_video": 0.6875,
                   "ci95_clip": [-0.1372, 0.1109], "ci95_video": [-0.1567, 0.1174]},
    # Four clips have no SQ on one side or the other, and with them one video.
    "SQ": {"n_clips": 10, "n_videos": 6, "base_mean": 0.5191, "base_sd": 0.135, "cond_mean": 0.5037,
           "cond_sd": 0.1298, "delta_mean": -0.0153, "delta_sd": 0.2067, "delta_median": 0.0814,
           "wins": 6, "losses": 4, "ties": 0, "wilcoxon_p": 0.625, "wilcoxon_p_video": 0.84375,
           "ci95_clip": [-0.1513, 0.0926], "ci95_video": [-0.1758, 0.0882]},
}


def test_wins_and_losses_are_counted_in_the_key_s_direction():
    """A key where smaller is better wins where it falls; a reference value counts no wins or losses."""
    base, cond = evaluator_scores("base", 0.0, 1), evaluator_scores("cond", 0.0, 1)
    vi = metric_key("VI_split", "all")
    for row in cond["per_clip"]:
        row[vi] = round(row[vi] - 0.1, 4)
        row["time_IoU"] = round(row["time_IoU"] + 0.1, 4)
    res = PS.compare_pair(base, cond, keys=[vi, "time_IoU"])
    got = res["metrics"][vi]
    assert (got["sign"], got["wins"], got["losses"], got["ties"]) == (-1, 6, 0, 0)
    got = res["metrics"]["time_IoU"]
    assert (got["sign"], got["wins"], got["losses"], got["ties"]) == (0, None, None, 0)


def test_the_pair_statistics_are_the_workbench_s():
    res = PS.compare_pair(pilot_scores("base", 0.0, 5), pilot_scores("cond", 0.03, 6))
    assert (res["n_clips"], res["n_videos"], res["population"]) == (14, 7, "identical")
    assert res["eval_code"] == PILOT_EVAL_CODE_SHA[:16]
    for key, want in WORKBENCH.items():
        got = res["metrics"][key]
        assert {k: got[k] for k in want} == want, key
        assert set(got) - set(want) == {"sign"}, key


def test_every_row_carries_the_direction_of_its_key():
    res = PS.compare_pair(pilot_scores("base", 0.0, 5), pilot_scores("cond", 0.03, 6))
    assert res["metrics"]["inst_F1_50"]["sign"] == +1 and res["metrics"]["underseg_error"]["sign"] == -1
    res = PS.compare_pair(evaluator_scores("base", 0.0, 5), evaluator_scores("cond", 0.03, 6))
    signs = {k: r["sign"] for k, r in res["metrics"].items()}
    assert signs[metric_key("F1_50", "all")] == +1
    assert signs[metric_key("VI_split", "all")] == -1 and signs[metric_key("VI_merge", "all")] == -1
    assert signs[metric_key("unlabelled_share", "all")] == 0 and signs["time_IoU"] == 0


def test_the_pair_refuses_two_evaluators():
    other = pilot_scores("cond", 0.03, 6)
    other["eval_code_sha"] = "b" * 64
    with pytest.raises(ValueError, match="different evaluators"):
        PS.compare_pair(pilot_scores("base", 0.0, 5), other)


def test_the_statistics_are_taken_on_the_clips_the_check_compared(monkeypatch):
    # One population: the one `check_comparable` returns. Taking it from the
    # rows again gave `compare_eval` and this tool two populations for one pair.
    real = PS.check_comparable

    def four_clips(*args, **kwargs):
        chk = real(*args, **kwargs)
        return {**chk, "clips": chk["clips"][:4]}

    monkeypatch.setattr(PS, "check_comparable", four_clips)
    res = PS.compare_pair(pilot_scores("base", 0.0, 5), pilot_scores("cond", 0.03, 6))
    assert (res["n_clips"], res["n_videos"]) == (4, 2)
    assert all(r["n_clips"] <= 4 for r in res["metrics"].values())


def test_dropping_a_video_shrinks_the_population_after_the_check():
    res = PS.compare_pair(pilot_scores("base", 0.0, 5), pilot_scores("cond", 0.03, 6), drop=[VIDEOS[0]])
    assert (res["n_clips"], res["n_videos"], res["dropped_videos"]) == (12, 6, [VIDEOS[0]])
    with pytest.raises(ValueError, match="left no clip"):
        PS.compare_pair(pilot_scores("base", 0.0, 5), pilot_scores("cond", 0.03, 6), drop=VIDEOS)


def test_dropping_a_video_the_scores_do_not_have_is_refused():
    # Otherwise the JSON records the video as dropped with nothing dropped,
    # and "the conclusion held without it" is read from an unchanged population.
    with pytest.raises(ValueError, match="names no video.*no_such_video"):
        PS.compare_pair(pilot_scores("base", 0.0, 5), pilot_scores("cond", 0.03, 6), drop=["no_such_video"])


def test_one_video_has_no_video_interval_and_so_no_mark():
    res = PS.compare_pair(pilot_scores("base", 0.0, 5), pilot_scores("cond", 0.03, 6), drop=VIDEOS[1:])
    assert (res["n_clips"], res["n_videos"]) == (2, 1)
    for key, r in res["metrics"].items():
        assert r["ci95_video"] is None and PS.mark_of(r["ci95_video"], r["sign"], n_videos=r["n_videos"]) == "", key
        # Two clips still give a clip interval and SDs; they are reference only.
        assert (r["ci95_clip"] is not None and r["delta_sd"] is not None) == (r["n_clips"] == 2), key


def test_one_clip_writes_none_and_not_nan():
    a, b = pilot_scores("base", 0.0, 5), pilot_scores("cond", 0.03, 6)
    a["per_clip"][0]["SQ"], b["per_clip"][0]["SQ"] = 0.5, 0.6
    for r in a["per_clip"][1:]:
        r["SQ"] = None
    res = PS.compare_pair(a, b)
    sq = res["metrics"]["SQ"]
    assert (sq["n_clips"], sq["n_videos"]) == (1, 1)
    assert sq["base_sd"] is None and sq["cond_sd"] is None and sq["delta_sd"] is None
    assert sq["ci95_clip"] is None and sq["ci95_video"] is None
    json.dumps(res, allow_nan=False)


def test_the_pair_computes_the_keys_it_is_named_and_only_those():
    a, b = pilot_scores("base", 0.0, 5), pilot_scores("cond", 0.03, 6)
    for x, y in zip(a["per_clip"], b["per_clip"]):
        x["inst_F1_50_labeled"], y["inst_F1_50_labeled"] = x["inst_F1_50"], y["inst_F1_50"] + 0.01
    assert "inst_F1_50_labeled" not in PS.compare_pair(a, b)["metrics"]
    res = PS.compare_pair(a, b, keys=["inst_F1_50_labeled"])
    assert list(res["metrics"]) == ["inst_F1_50_labeled"]
    got = res["metrics"]["inst_F1_50_labeled"]
    assert got["sign"] == PILOT_SIGNS["inst_F1_50_labeled"]
    assert got["delta_mean"] == round(WORKBENCH["inst_F1_50"]["delta_mean"] + 0.01, 4)
    # A key named by mistake is refused, even where no clip defines it.
    with pytest.raises(ValueError, match="not a key with a direction on this JSON"):
        PS.compare_pair(a, b, keys=["inst_F1_50_labelled"])
    ea, eb = evaluator_scores("base", 0.0, 5), evaluator_scores("cond", 0.03, 6)
    for spelt in ("F1_50/geometrc", "F1_50"):
        with pytest.raises(ValueError, match="not a key with a direction on this JSON"):
            PS.compare_pair(ea, eb, keys=[spelt])


def test_a_value_that_rounds_to_zero_keeps_two_digits():
    # An end written as -0.0 would lose its ✗.
    assert PS.round_keeping_sign(-2.4e-05) == -2.4e-05 and PS.round_keeping_sign(0.0) == 0.0
    assert PS.round_keeping_sign(0.123449) == 0.1234
    assert PS.format_signed(-2.4e-05) == "-2.40e-05" and PS.format_signed(0.0) == "+0.0000"
    assert PS.format_signed(0.05) == "+0.0500"


@pytest.mark.parametrize("x, written, printed", [(4.99e-05, 5e-05, "+5.00e-05"), (-4.99e-05, -5e-05, "-5.00e-05"),
                                                 (6e-05, 0.0001, "+0.0001")])
def test_a_value_is_printed_as_it_is_written_at_the_boundary_of_four_places(x, written, printed):
    # 5e-05 printed at four places would read 0.0001, twice the value written.
    assert PS.round_keeping_sign(x) == written and PS.format_signed(written) == printed


@pytest.mark.parametrize("ci, written, mark", [((2.03e-05, 0.01509), [2e-05, 0.0151], "★"),
                                               ((-0.01509, -2.03e-05), [-0.0151, -2e-05], "✗")])
def test_an_interval_end_near_zero_keeps_its_star_or_cross(monkeypatch, ci, written, mark):
    # Rounded to four places alone, either end would be written as 0.0, which gives neither ★ nor ✗.
    monkeypatch.setattr(PS, "boot_ci", lambda d, groups, **kw: ci)
    a, b = pilot_scores("base", 0.0, 5), pilot_scores("cond", 0.03, 6)
    r = PS.compare_pair(a, b, keys=["inst_F1_50"])["metrics"]["inst_F1_50"]
    assert r["ci95_video"] == written and PS.mark_of(r["ci95_video"], r["sign"], n_videos=r["n_videos"]) == mark


def test_a_key_defined_on_no_common_clip_is_left_out_not_zeroed():
    a, b = pilot_scores("base", 0.0, 5), pilot_scores("cond", 0.03, 6)
    for r in b["per_clip"]:
        r["SQ"] = None
    assert "SQ" not in PS.compare_pair(a, b)["metrics"]


def _run(tmp_path, *args):
    return subprocess.run([sys.executable, "-m", "evalkit.tools.paired_stats", "--eval-dir", str(tmp_path), *args],
                          capture_output=True, text=True, cwd=REPO)


def test_the_command_writes_the_pair_json(tmp_path):
    for tag, shift, seed in (("base", 0.0, 5), ("cond", 0.03, 6)):
        (tmp_path / f"{tag}.json").write_text(json.dumps(pilot_scores(tag, shift, seed)))
    out = tmp_path / "paired.json"
    run = _run(tmp_path, "--pairs", "base:cond", "--out", str(out))
    assert run.returncode == 0, run.stderr
    j = json.loads(out.read_text())
    assert j["seed_scheme"] == PS.SEED_SCHEME and j["n_boot"] == PS.N_BOOT
    assert j["pairs"]["base:cond"]["metrics"]["inst_F1_50"]["n_clips"] == 14


def test_the_command_prints_a_shrunken_population_with_its_videos(tmp_path):
    for tag, shift, seed in (("base", 0.0, 5), ("cond", 0.03, 6)):
        (tmp_path / f"{tag}.json").write_text(json.dumps(pilot_scores(tag, shift, seed)))
    run = _run(tmp_path, "--pairs", "base:cond", "--drop-video", ",".join(VIDEOS[1:]))
    assert run.returncode == 0, run.stderr
    assert "(2 clips / 1 videos)" in run.stdout and "videos resampled none" in run.stdout
    run = _run(tmp_path, "--pairs", "base:cond")
    assert "SQ   [10/14 clips, 6/7 videos]" in run.stdout


def test_the_command_says_when_an_interval_rests_on_too_few_videos_for_a_mark(tmp_path):
    for tag, shift, seed in (("base", 0.0, 5), ("cond", 0.03, 6)):
        (tmp_path / f"{tag}.json").write_text(json.dumps(pilot_scores(tag, shift, seed)))
    note = f"({PS.sign_only_note(5)})"
    run = _run(tmp_path, "--pairs", "base:cond", "--drop-video", ",".join(VIDEOS[:2]))
    assert run.returncode == 0, run.stderr
    assert "(10 clips / 5 videos)" in run.stdout and run.stdout.count(note) == len(PS.PILOT_KEYS)
    # Every key of the whole population rests on six videos or more, `SQ`'s on 6 of the 7.
    run = _run(tmp_path, "--pairs", "base:cond")
    assert "SQ   [10/14 clips, 6/7 videos]" in run.stdout and note not in run.stdout


def test_the_command_refuses_a_run_whose_pairs_hold_two_rules(tmp_path):
    # Both pairs share the per-frame condition and pass; the run's one JSON would hold two rules.
    for tag, rule in (("pf", "per_frame"), ("both", "both_ways_from_centre"), ("fwd", "forward_from_first")):
        j = evaluator_scores(tag, 0.0, 5)
        j["propagation"] = rule
        (tmp_path / f"{tag}.json").write_text(json.dumps(j))
    out = tmp_path / "paired.json"
    run = _run(tmp_path, "--pairs", "pf:both,pf:fwd", "--out", str(out))
    assert run.returncode != 0 and "one table per rule" in run.stderr and "Traceback" not in run.stderr
    assert not out.exists()
    run = _run(tmp_path, "--pairs", "pf:both", "--out", str(out))
    assert run.returncode == 0, run.stderr
    j = json.loads(out.read_text())
    assert j["propagation"] == "both_ways_from_centre"
    assert j["pairs"]["pf:both"]["propagation"] == "both_ways_from_centre"


def test_the_command_takes_a_run_whose_pairs_differ_in_the_rule_alone(tmp_path):
    # Each input's two rules make a pair; a third input under one rule alone would mix the table.
    for tag, name, rule in (("fwd", "rgb_pps24_sam_forward", "forward_from_first"),
                            ("ctr", "rgb_pps24_sam_centre", "both_ways_from_centre"),
                            ("n_fwd", "normal_da3_pps24_sam_forward", "forward_from_first"),
                            ("n_ctr", "normal_da3_pps24_sam_centre", "both_ways_from_centre"),
                            ("d_fwd", "depth_da3_pps24_sam_forward", "forward_from_first")):
        j = evaluator_scores(tag, 0.0, 5)
        j["propagation"], j["track_dir_name"] = rule, f"track_rgb_{name}"
        (tmp_path / f"{tag}.json").write_text(json.dumps(j))
    out = tmp_path / "paired.json"
    run = _run(tmp_path, "--pairs", "fwd:ctr,n_fwd:n_ctr", "--out", str(out))
    assert run.returncode == 0, run.stderr
    j = json.loads(out.read_text())
    assert j["propagation"] == "both_ways_from_centre and forward_from_first"
    assert j["pairs"]["fwd:ctr"]["propagation"] == "both_ways_from_centre and forward_from_first"
    run = _run(tmp_path, "--pairs", "fwd:ctr,n_fwd:d_fwd", "--out", str(out))
    assert run.returncode != 0 and "Traceback" not in run.stderr
    assert "['d_fwd', 'n_fwd'] lack the condition that differs from them in the rule alone" in run.stderr


def test_the_command_writes_no_rule_on_pilot_jsons(tmp_path):
    for tag, shift, seed in (("base", 0.0, 5), ("cond", 0.03, 6)):
        (tmp_path / f"{tag}.json").write_text(json.dumps(pilot_scores(tag, shift, seed)))
    out = tmp_path / "paired.json"
    assert _run(tmp_path, "--pairs", "base:cond", "--out", str(out)).returncode == 0
    j = json.loads(out.read_text())
    assert list(j) == ["n_boot", "seed", "seed_scheme", "pairs"] and "propagation" not in j["pairs"]["base:cond"]


def test_pairs_are_read_in_the_order_given():
    assert PS.parse_pairs(" b:a, c:d ,") == [("b", "a"), ("c", "d")]


@pytest.mark.parametrize("text", ["rgb", "rgb:", ":rgb", "a:b:c", "rgb:rgb", "", " , "])
def test_a_malformed_pair_is_refused(text):
    with pytest.raises(ValueError):
        PS.parse_pairs(text)


def test_shared_conditions_are_read_once(tmp_path, monkeypatch):
    summaries = {tag: {"tag": tag} for tag in ("base", "cond", "other")}
    for tag, summary in summaries.items():
        (tmp_path / f"{tag}.json").write_text(json.dumps(summary))
    reads = []
    real = PS.load_json

    def read_once(tag, eval_dir):
        assert tag not in reads, f"{tag} was read twice"
        reads.append(tag)
        return real(tag, eval_dir)

    monkeypatch.setattr(PS, "load_json", read_once)
    pairs, loaded = PS.load_pairs("base:cond,base:other,cond:other", str(tmp_path))
    assert pairs == [("base", "cond"), ("base", "other"), ("cond", "other")]
    assert loaded == summaries
    assert reads == list(summaries)


def test_a_later_bad_pair_is_refused_before_any_file_is_read(tmp_path):
    # No files exist, so reading the first pair before parsing the rest would raise OSError.
    with pytest.raises(ValueError, match="not one with itself"):
        PS.load_pairs("base:cond,other:other", str(tmp_path))


@pytest.mark.parametrize("pairs, said", [
    ("base", "a pair is <base>:<cond>"),
    ("base:cond:other", "a pair is <base>:<cond>"),
    ("base:base", "not one with itself"),
    (" , ", "no pair is given"),
    ("base:nothere", "nothere.json"),
])
def test_the_command_refuses_a_bad_pair_with_a_message_not_a_traceback(tmp_path, pairs, said):
    (tmp_path / "base.json").write_text(json.dumps(pilot_scores("base", 0.0, 5)))
    out = tmp_path / "paired.json"
    run = _run(tmp_path, "--pairs", pairs, "--out", str(out))
    assert run.returncode != 0 and said in run.stderr and "Traceback" not in run.stderr
    assert not out.exists() and not run.stdout

