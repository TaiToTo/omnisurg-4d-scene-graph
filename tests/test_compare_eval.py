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
from evalkit.tools.scores import PILOT_EVAL_CODE_SHA, metric_key

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
        for k, _ in CE.PILOT_METRICS:
            if k == "time_IoU":
                continue
            v = round(min(1.0, max(0.0, rnd.gauss(0.5 + shift, 0.15))), 4)
            r[k] = None if k == "SQ" and rnd.random() < 0.3 else v
        rows.append(r)
    return dict(tag=tag, eval_code_sha=PILOT_EVAL_CODE_SHA, eval_code_tag="t", eval_version=2,
                dataset="atlas", tissue_ignore=[1, 2], clips=clips, per_clip=rows)


def evaluator_scores(pilot=False, class_set="original", views=("all", "tissue", "geometric"), shift=0.0, seed=1):
    rnd = random.Random(seed)
    clips = [f"{v}__gt_0001" for v in VIDEOS]
    rows = [dict(clip=c, **{metric_key(m, v): round(rnd.random() + shift, 4) for v in views for m in ("F1_50", "SQ")},
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
    assert CE.primary_key(a) == "F1_50/geometric"
    res = CE.compare(a, b)
    assert set(res["metrics"]) == {"F1_50/all", "SQ/all", "F1_50/tissue", "SQ/tissue",
                                   "F1_50/geometric", "SQ/geometric", "time_IoU"}
    assert res["wins"] == 5
    assert dict(CE.metrics_of(a))["SQ/geometric"] == +1 and dict(CE.metrics_of(a))["time_IoU"] == 0


def test_an_evaluator_json_without_the_geometric_view_has_no_primary_metric():
    with pytest.raises(ValueError, match="no geometric view"):
        CE.primary_key(evaluator_scores(views=("all",)))


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


def test_the_command_stops_on_a_mix_of_shas(tmp_path):
    a, b = pilot_scores("base", 0.0, 5), pilot_scores("cond", 0.03, 6)
    b["eval_code_sha"] = "b" * 64
    (tmp_path / "base.json").write_text(json.dumps(a))
    (tmp_path / "cond.json").write_text(json.dumps(b))
    r = subprocess.run([sys.executable, "-m", "evalkit.tools.compare_eval", "--base", "base", "--cond", "cond",
                        "--dir", str(tmp_path)], capture_output=True, text=True, cwd=REPO)
    assert r.returncode != 0 and "different evaluators" in r.stderr
