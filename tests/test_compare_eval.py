"""compare_eval refuses the mixes it is there to refuse, and aligns each metric's population.

The aligned means are pinned to what the workbench's `compare_eval.py`
wrote on the same pair of pilot-layout JSONs, generated here and run
through the workbench version once.
"""
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from evalkit.tools import compare_eval as CE
from evalkit.tools.scores import PILOT_EVAL_CODE_SHA, PILOT_SIGNS, metric_key

REPO = Path(__file__).resolve().parent.parent
VIDEOS = ("adrenalectomy__16GPCUPkXYQ", "appendectomy__41RKDh3INiU", "gastric_surgery__4FHGGFZsPzw",
          "lar___7H7G-4sevQ", "rectopexy__IJfgUvxoTkM")
SHA = "a" * 64


def pilot_scores(tag: str, shift: float, seed: int) -> dict:
    """A pilot-layout score JSON on 5 videos, 2 clips each, `SQ` None on some clips, `time_IoU` absent."""
    rnd = random.Random(seed)
    clips = [f"{v}__gt_{i:04d}" for v in VIDEOS for i in (1, 2)]
    rows = []
    for c in clips:
        r = {"clip": c, "extra_ignore": []}
        for k in PILOT_SIGNS:
            if k == "time_IoU":
                continue
            v = round(min(1.0, max(0.0, rnd.gauss(0.5 + shift, 0.15))), 4)
            r[k] = None if k == "SQ" and rnd.random() < 0.3 else v
        rows.append(r)
    return dict(tag=tag, eval_code_sha=PILOT_EVAL_CODE_SHA, eval_code_tag="t", eval_version=2,
                dataset="atlas", tissue_ignore=[1, 2], clips=clips, per_clip=rows)


def evaluator_scores(pilot=False, class_set="original", views=("all", "tissue", "geometric"), shift=0.0, seed=1,
                     metrics=("F1_50", "SQ")):
    rnd = random.Random(seed)
    clips = [f"{v}__gt_0001" for v in VIDEOS]
    rows = [dict(clip=c, **{metric_key(m, v): round(rnd.random() + shift, 4) for v in views for m in metrics},
                 time_IoU=0.5) for c in clips]
    return dict(eval_code_sha=SHA, dataset="atlas120k", pilot=pilot, class_set=class_set, views=list(views),
                clips=clips, input_shas={c: {"gt_masks": "g", "depth": "d", "predictions": "p"} for c in clips},
                versions={"numpy": "2.0.0"}, per_clip=rows)


# What the workbench's compare_eval wrote on `base` (shift 0, seed 5) against
# `cond` (shift 0.03, seed 6).
WORKBENCH = {
    "n_clips": 10, "wins": 6,
    "inst_F1_50": {"base": 0.4352, "cond": 0.4794, "delta": 0.0442, "n_clips": 10},
    # Six clips have no SQ on one side or the other.
    "SQ": {"base": 0.6218, "cond": 0.6131, "delta": -0.0087, "n_clips": 4},
    "underseg_error": {"base": 0.5039, "cond": 0.572, "delta": 0.0681, "n_clips": 10},
}


def test_the_aligned_means_are_the_workbench_s():
    res = CE.compare(pilot_scores("base", 0.0, 5), pilot_scores("cond", 0.03, 6))
    assert (res["n_clips"], res["population"], res["eval_code"]) == (10, "identical", PILOT_EVAL_CODE_SHA[:16])
    assert res["wins"] == WORKBENCH["wins"]
    for k in ("inst_F1_50", "SQ", "underseg_error"):
        assert res["metrics"][k] == WORKBENCH[k], k
    assert "time_IoU" not in res["metrics"]


def test_sq_is_aligned_to_the_clips_both_define():
    a, b = pilot_scores("base", 0.0, 5), pilot_scores("cond", 0.03, 6)
    rows_a, rows_b = {r["clip"]: r for r in a["per_clip"]}, {r["clip"]: r for r in b["per_clip"]}
    both = [c for c in a["clips"] if rows_a[c]["SQ"] is not None and rows_b[c]["SQ"] is not None]
    res = CE.compare(a, b)["metrics"]["SQ"]
    assert res["n_clips"] == len(both) < 10
    assert res["base"] == round(sum(rows_a[c]["SQ"] for c in both) / len(both), 4)


# ---------------------------------------------------------------- what it refuses


def test_a_mix_of_shas_is_refused():
    b = pilot_scores("cond", 0.03, 6)
    b["eval_code_sha"] = "b" * 64
    with pytest.raises(ValueError, match="different evaluators"):
        CE.compare(pilot_scores("base", 0.0, 5), b)


def test_a_mix_of_modes_under_one_sha_is_refused():
    with pytest.raises(ValueError, match="pilot mode and the normal mode"):
        CE.compare(evaluator_scores(), evaluator_scores(pilot=True))


def test_a_mix_of_class_sets_under_one_sha_is_refused():
    with pytest.raises(ValueError, match="class sets"):
        CE.compare(evaluator_scores(), evaluator_scores(class_set="benchmark"))


def test_different_populations_are_refused_unless_a_subset_is_allowed():
    a, b = pilot_scores("base", 0.0, 5), pilot_scores("cond", 0.03, 6)
    b["clips"] = b["clips"][:6]
    b["per_clip"] = b["per_clip"][:6]
    with pytest.raises(ValueError, match="clip sets differ"):
        CE.compare(a, b)
    res = CE.compare(a, b, allow_subset=True)
    assert (res["population"], res["n_clips"], len(res["clips"])) == ("intersection", 6, 6)


# ---------------------------------------------------------------- the evaluator's JSONs


def test_the_evaluator_s_keys_and_primary_metric_come_from_the_json():
    a, b = evaluator_scores(seed=1), evaluator_scores(seed=1, shift=0.1)
    assert CE.primary_key(a) == ("F1_50/geometric", +1)
    res = CE.compare(a, b)
    assert set(res["metrics"]) == {"F1_50/all", "SQ/all", "F1_50/tissue", "SQ/tissue",
                                   "F1_50/geometric", "SQ/geometric", "time_IoU"}
    assert res["wins"] == 5
    assert dict(CE.metrics_of(a))["SQ/geometric"] == +1 and dict(CE.metrics_of(a))["time_IoU"] == 0


def test_a_clip_where_the_primary_metric_is_undefined_is_neither_a_win_nor_a_crash():
    # `F1_50` is None on a clip with no GT object in the geometric view. The
    # pilot evaluator wrote 0 there, so the workbench version never met a
    # None and subtracted the values directly.
    a, b = evaluator_scores(seed=1), evaluator_scores(seed=1, shift=0.1)
    b["per_clip"][0]["F1_50/geometric"] = None
    res = CE.compare(a, b)
    assert res["wins"] == 4 and res["n_clips"] == 5
    assert res["metrics"]["F1_50/geometric"]["n_clips"] == 4


def test_a_primary_metric_defined_on_no_common_clip_is_refused():
    a, b = evaluator_scores(seed=1), evaluator_scores(seed=1, shift=0.1)
    for r in b["per_clip"]:
        r["F1_50/geometric"] = None
    with pytest.raises(ValueError, match="defined on no common clip"):
        CE.compare(a, b)


def test_the_command_lists_only_the_clips_where_the_primary_metric_is_defined(tmp_path):
    a, b = evaluator_scores(seed=1), evaluator_scores(seed=1, shift=0.1)
    b["per_clip"][0]["F1_50/geometric"] = None
    (tmp_path / "base.json").write_text(json.dumps(a))
    (tmp_path / "cond.json").write_text(json.dumps(b))
    r = subprocess.run([sys.executable, "-m", "evalkit.tools.compare_eval", "--base", "base", "--cond", "cond",
                        "--dir", str(tmp_path)], capture_output=True, text=True, cwd=REPO, check=True)
    assert "rose: 4/4   [F1_50/geometric undefined on 1 clips]" in r.stdout
    # The clip's name appears nowhere but the per-clip list.
    assert a["per_clip"][0]["clip"] not in r.stdout


def test_an_evaluator_json_without_the_geometric_view_has_no_default_key():
    with pytest.raises(ValueError, match="no geometric view"):
        CE.primary_key(evaluator_scores(views=("all",)))
    assert CE.primary_key(evaluator_scores(views=("all",)), "SQ/all") == ("SQ/all", +1)


# ---------------------------------------------------------------- the key and its direction


def test_the_pilot_directions_are_the_specification_s():
    signs = dict(CE.metrics_of(pilot_scores("base", 0.0, 5)))
    # `time_IoU` is a reference value and `n_regions_mean` a count: neither has a better way.
    assert signs["time_IoU"] == 0 and signs["n_regions_mean"] == 0
    assert signs["underseg_error"] == -1 and signs["overseg_mean"] == -1
    assert all(s == +1 for k, s in signs.items() if k not in ("time_IoU", "n_regions_mean", "underseg_error", "overseg_mean"))


def test_the_summary_records_the_key_unless_it_is_the_workbench_s_on_a_pilot_json():
    a, b = pilot_scores("base", 0.0, 5), pilot_scores("cond", 0.03, 6)
    # The workbench's summary has no such field, and the pilot summary must stay its bytes.
    assert "key" not in CE.compare(a, b) and "sign" not in CE.compare(a, b)
    res = CE.compare(a, b, key="underseg_error")
    assert (res["key"], res["sign"]) == ("underseg_error", -1)
    res = CE.compare(evaluator_scores(seed=1), evaluator_scores(seed=1, shift=0.1))
    assert (res["key"], res["sign"]) == ("F1_50/geometric", +1)


def test_a_key_where_lower_is_better_counts_a_fall_as_a_win():
    a = evaluator_scores(seed=1, metrics=("F1_50", "VI_split"))
    b = evaluator_scores(seed=1, shift=0.1, metrics=("F1_50", "VI_split"))
    # `cond` is higher on every clip: every clip is a win on F1_50 and none on VI_split.
    assert CE.compare(a, b)["wins"] == 5
    assert CE.compare(a, b, key="VI_split/all")["wins"] == 0
    assert CE.compare(b, a, key="VI_split/all")["wins"] == 5


def test_a_key_in_no_row_of_a_pilot_json_is_refused_by_name():
    # The pilot's list names every key the pilot evaluator ever wrote;
    # `time_IoU` is in the list and in none of these rows, and the refusal
    # has to say so, not that it is defined on no common clip.
    a, b = pilot_scores("base", 0.0, 5), pilot_scores("cond", 0.03, 6)
    for r in a["per_clip"] + b["per_clip"]:
        r["time_IoU"] = 0.5
        del r["inst_BF"]
    with pytest.raises(ValueError, match="inst_BF is in no row"):
        CE.compare(a, b, key="inst_BF")


def test_a_reference_value_or_a_key_the_json_does_not_report_is_refused():
    a, b = evaluator_scores(seed=1), evaluator_scores(seed=1, shift=0.1)
    with pytest.raises(ValueError, match="reference value with no direction"):
        CE.compare(a, b, key="time_IoU")
    with pytest.raises(ValueError, match="not a key this JSON reports"):
        CE.compare(a, b, key="inst_F1_50")
    with pytest.raises(ValueError, match="not a key this JSON reports"):
        CE.compare(pilot_scores("base", 0.0, 5), pilot_scores("cond", 0.03, 6), key="F1_50/geometric")


def run(tmp_path, a, b, *more):
    (tmp_path / "base.json").write_text(json.dumps(a))
    (tmp_path / "cond.json").write_text(json.dumps(b))
    return subprocess.run([sys.executable, "-m", "evalkit.tools.compare_eval", "--base", "base", "--cond", "cond",
                           "--dir", str(tmp_path), *more], capture_output=True, text=True, cwd=REPO)


def directions(stdout: str) -> dict[str, str]:
    """Each metric line's direction phrase, by metric."""
    table = stdout.split("\n\nclip")[0].splitlines()[2:]
    return {line.split()[0]: next(d for d in CE.DIRECTION.values() if d in line) for line in table}


def test_the_table_prints_a_direction_and_no_mark_that_reads_as_a_verdict(tmp_path):
    # The verdict is `paired_stats.verdict` alone; a mark on a sign would be
    # a second one, read as "better" on a difference of 0.0004.
    a, b = pilot_scores("base", 0.0, 5), pilot_scores("cond", 0.03, 6)
    forward, backward = run(tmp_path, a, b), run(tmp_path, b, a)
    assert forward.returncode == 0 and backward.returncode == 0, forward.stderr + backward.stderr
    for r in (forward, backward):
        assert not any(m in r.stdout for m in ("○", "×", "★", "✗")), r.stdout
    # Swapping the conditions flips every delta and changes no direction.
    assert directions(forward.stdout) == directions(backward.stdout)
    d = directions(forward.stdout)
    assert d["inst_F1_50"] == "higher is better" and d["underseg_error"] == "lower is better"
    assert d["time_IoU"] == "reference, never marked"
    # Every pilot key has a line, whole, and the direction column is one
    # column: a name wider than the column would push its line's direction
    # to the right.
    table = forward.stdout.split("\n\nclip")[0].splitlines()[2:]
    assert {line.split()[0] for line in table} == set(PILOT_SIGNS)
    assert len({line.index(d[line.split()[0]]) for line in table}) == 1
    assert "unlabelled_share" not in forward.stdout


def test_the_command_takes_the_key_and_says_which_way_it_moved(tmp_path):
    a = evaluator_scores(seed=1, metrics=("F1_50", "VI_split"))
    b = evaluator_scores(seed=1, shift=0.1, metrics=("F1_50", "VI_split"))
    r = run(tmp_path, a, b, "--key", "VI_split/tissue", "--out_json", str(tmp_path / "cmp.json"))
    assert r.returncode == 0, r.stderr
    assert "clips on which VI_split/tissue fell: 0/5" in r.stdout
    j = json.loads((tmp_path / "cmp.json").read_text())
    assert (j["key"], j["sign"], j["wins"]) == ("VI_split/tissue", -1, 0)
    r = run(tmp_path, a, b, "--key", "time_IoU")
    assert r.returncode != 0 and "reference value" in r.stderr and "Traceback" not in r.stderr


def test_differing_versions_are_carried_into_the_summary():
    b = evaluator_scores()
    b["versions"]["numpy"] = "2.1.0"
    assert CE.compare(evaluator_scores(), b)["versions_differ"] == {"numpy": ("2.0.0", "2.1.0")}


# ---------------------------------------------------------------- the command


def test_the_command_writes_the_summary_and_the_chart(tmp_path):
    pytest.importorskip("matplotlib", reason="the chart needs the `tools` extra")
    d = tmp_path / "scores"
    d.mkdir()
    for tag, shift, seed in (("base", 0.0, 5), ("cond", 0.03, 6)):
        (d / f"{tag}.json").write_text(json.dumps(pilot_scores(tag, shift, seed)))
    out = tmp_path / "cmp.json"
    subprocess.run([sys.executable, "-m", "evalkit.tools.compare_eval", "--base", "base", "--cond", "cond",
                    "--dir", str(d), "--out_json", str(out), "--plot"], check=True, capture_output=True, cwd=REPO)
    j = json.loads(out.read_text())
    assert (j["base"], j["cond"], j["n_clips"], j["population"]) == ("base", "cond", 10, "identical")
    assert j["metrics"]["inst_F1_50"] == WORKBENCH["inst_F1_50"]
    assert (tmp_path / "figs" / "delta__base_vs_cond.png").stat().st_size > 0


def test_the_chart_is_drawn_when_the_key_is_undefined_on_a_clip(tmp_path):
    # Drawn over the clips both define, like the list; over all of them, the
    # None is subtracted and the command dies after printing the table.
    pytest.importorskip("matplotlib", reason="the chart needs the `tools` extra")
    d = tmp_path / "scores"
    d.mkdir()
    a, b = evaluator_scores(seed=1), evaluator_scores(seed=1, shift=0.1)
    b["per_clip"][0]["F1_50/geometric"] = None
    r = run(d, a, b, "--plot")
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "figs" / "delta__base_vs_cond.png").stat().st_size > 0


def test_the_summary_is_not_written_with_a_nan_in_it(tmp_path):
    a, b = pilot_scores("base", 0.0, 5), pilot_scores("cond", 0.03, 6)
    b["per_clip"][0]["inst_F1_50"] = float("nan")
    out = tmp_path / "cmp.json"
    r = run(tmp_path, a, b, "--out_json", str(out))
    assert r.returncode != 0 and "NaN" in r.stderr and "Traceback" not in r.stderr
    assert not out.exists()


def test_the_command_stops_on_a_mix_of_shas(tmp_path):
    a, b = pilot_scores("base", 0.0, 5), pilot_scores("cond", 0.03, 6)
    b["eval_code_sha"] = "b" * 64
    r = run(tmp_path, a, b)
    assert r.returncode != 0 and "different evaluators" in r.stderr


def test_the_command_stops_on_a_missing_or_broken_json(tmp_path):
    (tmp_path / "base.json").write_text(json.dumps(pilot_scores("base", 0.0, 5)))
    r = subprocess.run([sys.executable, "-m", "evalkit.tools.compare_eval", "--base", "base", "--cond", "cond",
                        "--dir", str(tmp_path)], capture_output=True, text=True, cwd=REPO)
    assert r.returncode != 0 and "cond.json" in r.stderr and "Traceback" not in r.stderr
    (tmp_path / "cond.json").write_text("{not json")
    r = subprocess.run([sys.executable, "-m", "evalkit.tools.compare_eval", "--base", "base", "--cond", "cond",
                        "--dir", str(tmp_path)], capture_output=True, text=True, cwd=REPO)
    assert r.returncode != 0 and "Expecting" in r.stderr and "Traceback" not in r.stderr
