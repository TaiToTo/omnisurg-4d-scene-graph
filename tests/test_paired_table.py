"""paired_table prints the workbench's row on the same scores, and refuses a table that mixes what must not mix.

The pinned rows are what the workbench's `summary97.py` printed on the
pilot-layout JSONs generated here, run through it once with the row
labelled `pair`.
"""
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from evalkit.tools import paired_table as PT
from evalkit.tools.scores import CLIP_METRICS, FRAME_METRICS, PILOT_EVAL_CODE_SHA, metric_key

pytest.importorskip("scipy", reason="paired_stats needs the `tools` extra")

REPO = Path(__file__).resolve().parent.parent
VIDEOS = ("adrenalectomy__16GPCUPkXYQ", "appendectomy__41RKDh3INiU", "gastric_surgery__4FHGGFZsPzw",
          "lar___7H7G-4sevQ", "rectopexy__IJfgUvxoTkM", "hemicolectomy__5YDMlxTl0k8", "nephrectomy__abc123")
PILOT_ROW_KEYS = PT.PILOT_COLUMNS + ("underseg_error", "SQ", "time_IoU")


def pilot_scores(tag: str, shift: dict | None = None, seed: int = 1, n_regions: float = 8.0,
                 videos=VIDEOS, per_video: int = 3) -> dict:
    """A pilot-layout score JSON: every key drawn by seed 0, then moved by `shift` with noise drawn by `seed`."""
    base, noise = random.Random(0), random.Random(seed)
    clips = [f"{v}__gt_{i:04d}" for v in videos for i in range(1, per_video + 1)]
    rows = []
    for c in clips:
        r = {"clip": c, "extra_ignore": []}
        for k in PILOT_ROW_KEYS:
            v = base.gauss(0.5, 0.1)
            if shift is not None:
                v += shift.get(k, 0.0) + noise.gauss(0.0, 0.03)
            r[k] = round(v, 4)
        rows.append(r)
    return dict(tag=tag, eval_code_sha=PILOT_EVAL_CODE_SHA, eval_code_tag="t", eval_version=2, dataset="atlas",
                tissue_ignore=[1], clips=clips, n_regions_mean=n_regions, per_clip=rows)


def evaluator_scores(tag: str, shift: float = 0.0, seed: int = 1, propagation: str = "both_ways_from_centre",
                     class_set: str = "original", sha: str = "a" * 64) -> dict:
    """An evaluator-layout score JSON on 7 videos, 3 clips each, every metric in two views."""
    rnd = random.Random(seed)
    views = ("all", "geometric")
    clips = [f"{v}__gt_{i:04d}" for v in VIDEOS for i in (1, 2, 3)]
    keys = [metric_key(m, v) for v in views for m in FRAME_METRICS] + list(CLIP_METRICS)
    rows = [{"clip": c, **{k: round(rnd.gauss(0.5 + shift, 0.05), 4) for k in keys}} for c in clips]
    return dict(tag=tag, eval_code_sha=sha, dataset="atlas120k", pilot=False, class_set=class_set, views=list(views),
                clips=clips, input_shas={c: {"gt_masks": "g", "depth": "d", "predictions": f"p{tag}{c}"} for c in clips},
                versions={"numpy": "2.0.0"}, propagation=propagation, per_clip=rows)


def one_row(cond: str = "cond", base: str = "base", label: str = "pair") -> tuple:
    return (("heading", ((label, cond, base),)),)


def row_lines(lines: list[str], label: str = "pair") -> list[str]:
    """The row labelled `label` and every line under it, up to the next row or block."""
    start = next(i for i, line in enumerate(lines) if line.startswith(f"  {label} "))
    end = start + 1
    while end < len(lines) and lines[end].startswith(" " * 30):
        end += 1
    return lines[start:end]


# The pair the workbench's table was run on: F1 rose, the boundary F fell,
# the boundary recall did not move, and the class map rose a little.
SHIFT = {"inst_F1_50": 0.03, "boundary_F": -0.03, "GT_mIoU": 0.005}


def workbench_pair() -> dict:
    return {"base": pilot_scores("base"), "cond": pilot_scores("cond", SHIFT, seed=2, n_regions=6.5)}


def against_the_clips_pair() -> dict:
    """A pair where F1 rose by 0.2 or more on one clip of each video and fell by 0.01 on the other two; nothing else moved."""
    base = pilot_scores("base")
    cond = {**json.loads(json.dumps(base)), "tag": "cond"}
    for i, r in enumerate(cond["per_clip"]):
        r["inst_F1_50"] = round(r["inst_F1_50"] + (0.2 + 0.02 * (i // 3) if i % 3 == 0 else -0.01), 4)
    return {"base": base, "cond": cond}


def one_video_pair() -> dict:
    """The workbench's pair, on six clips of one video."""
    return {"base": pilot_scores("base", videos=VIDEOS[:1], per_video=6),
            "cond": pilot_scores("cond", SHIFT, seed=2, n_regions=6.5, videos=VIDEOS[:1], per_video=6)}


# What the workbench's summary97 printed for each pair: the row, and the
# interval line's bracket, wins, losses and videos.
WORKBENCH = {
    "pair": ("  pair                          +0.0381       ★  -0.0404       ✗  -0.0043          +0.0009"
             "           6.50/8.00", "[+0.0292,+0.0482]", "19-2 / 7"),
    # The workbench wrote `★(符)` where the port writes `★(sign)`.
    "against_the_clips": ("  pair                          +0.0800 ★(sign)  +0.0000          +0.0000          +0.0000"
                          "           8.00/8.00", "[+0.0705,+0.0895]", "7-14 / 7"),
}


@pytest.mark.parametrize("name, pair", [("pair", workbench_pair), ("against_the_clips", against_the_clips_pair)])
def test_a_row_is_the_workbench_s(name, pair):
    row, ci = row_lines(PT.table_lines(one_row(), pair()))
    want_row, want_ci, want_counts = WORKBENCH[name]
    assert row == want_row
    assert ci.strip() == f"inst_F1_50 CI {want_ci} wins-losses {want_counts} videos"


def test_a_mark_the_clips_move_against_is_noted_both_ways():
    assert PT.mark_text({"mark": "★", "wins": 3, "losses": 3}) == "★(sign)"
    assert PT.mark_text({"mark": "★", "wins": 4, "losses": 3}) == "★"
    assert PT.mark_text({"mark": "✗", "wins": 3, "losses": 3}) == "✗(sign)"
    assert PT.mark_text({"mark": "✗", "wins": 2, "losses": 3}) == "✗"
    assert PT.mark_text({"mark": "", "wins": 0, "losses": 9}) == " "


def test_one_video_gives_no_value_where_the_workbench_marked_a_point():
    # The workbench printed `★ ✗ ✗(符) ✗` here, each from the interval
    # [m, m] that one video gives; the port prints no value and no mark.
    lines = PT.table_lines(one_row(), one_video_pair())
    (row,) = row_lines(lines)
    assert row.split()[1:] == [PT.NO_VALUE] * 4 + ["6.50/8.00"]
    assert lines[0].startswith("atlas: 6 clips / 1 videos")


def test_fewer_than_five_clips_give_no_value_and_a_shrunken_population_is_said():
    pair = workbench_pair()
    for r in pair["cond"]["per_clip"][4:]:
        r["SQ"] = None
    for r in pair["base"]["per_clip"][:2]:
        r["inst_F1_50"] = None
    lines = PT.table_lines(one_row(), pair, keys=["inst_F1_50", "SQ"])
    row, ci, *notes = row_lines(lines)
    assert row.split()[3] == PT.NO_VALUE and "CI [" in ci
    assert [n.strip() for n in notes] == ["[inst_F1_50: 19/21 clips, 7/7 videos]", "[SQ: 4/21 clips, 2/7 videos]"]


@pytest.mark.parametrize("shift, mark", [(-0.05, "★"), (+0.05, "✗")])
def test_the_mark_reads_the_interval_in_the_key_s_direction(shift, mark):
    # underseg_error counts error, so a fall is the better way.
    pair = {"base": pilot_scores("base"), "cond": pilot_scores("cond", {"underseg_error": shift}, seed=2)}
    (row, _) = row_lines(PT.table_lines(one_row(), pair, keys=["underseg_error"]))
    assert float(row.split()[1]) * shift > 0 and row.split()[2] == mark


@pytest.mark.parametrize("make, keys, said", [
    (lambda: pilot_scores("x"), ["time_IoU"], "time_IoU is a reference value"),
    (lambda: pilot_scores("x"), ["n_regions_mean"], "n_regions_mean is a reference value"),
    (lambda: evaluator_scores("x"), ["time_IoU"], "time_IoU is a reference value"),
    (lambda: evaluator_scores("x"), [metric_key("unlabelled_share", "all")], "is a reference value"),
    (lambda: evaluator_scores("x"), [metric_key("F1_50", "tissue")], "not a key this JSON reports"),
    (lambda: pilot_scores("x"), ["F1_50/all"], "not a key this JSON reports"),
    (lambda: evaluator_scores("x"), [], "not settled; name them with --keys"),
])
def test_a_key_the_table_cannot_report_is_refused(make, keys, said):
    with pytest.raises(ValueError, match=said):
        PT.columns_of(make(), keys)


def test_a_condition_without_a_score_json_is_refused():
    pair = workbench_pair()
    del pair["base"]
    with pytest.raises(ValueError, match=r"no score JSON for \['base'\]"):
        PT.table_lines(one_row(), pair)


def test_a_key_in_no_row_is_refused():
    pair = workbench_pair()
    for r in pair["base"]["per_clip"]:
        del r["GT_mIoU"]
    with pytest.raises(ValueError, match=r"base: \['GT_mIoU'\] in no row"):
        PT.table_lines(one_row(), pair)


def test_a_pilot_json_without_its_count_of_regions_is_refused():
    pair = workbench_pair()
    del pair["cond"]["n_regions_mean"]
    with pytest.raises(ValueError, match=r"\['cond'\]: a pilot JSON records n_regions_mean"):
        PT.table_lines(one_row(), pair)


def test_the_evaluator_s_table_names_its_rule_and_has_no_column_of_regions():
    pair = {"base": evaluator_scores("base", propagation="per_frame"), "cond": evaluator_scores("cond", 0.05, seed=2)}
    lines = PT.table_lines(one_row(), pair, keys=[metric_key("F1_50", "geometric"), metric_key("VI_split", "all")])
    assert lines[0] == "atlas120k: 21 clips / 7 videos / eval_code_sha aaaaaaaa / propagation both_ways_from_centre"
    assert any(line.startswith("regions: the evaluator records no count of regions") for line in lines)
    row, _ = row_lines(lines)
    # VI_split is better lower, so a rise is a cross.
    assert row.split()[2] == "★" and row.split()[4] == "✗" and len(row.split()) == 5


def two_rows(c1="c1", b1="b1", c2="c2", b2="b2") -> tuple:
    return (("heading", (("first", c1, b1), ("second", c2, b2))),)


@pytest.mark.parametrize("plant, said", [
    (lambda t: t["c2"].update(eval_code_sha="b" * 64), "different evaluators"),
    (lambda t: (t["c2"].update(class_set="benchmark"), t["b2"].update(class_set="benchmark")), "class_set"),
    (lambda t: (t["c2"].update(pilot=True), t["b2"].update(pilot=True)), "pilot=True"),
    (lambda t: (t["c2"].update(views=["all"]), t["b2"].update(views=["all"])), "views"),
    (lambda t: (t["c2"].update(dataset="cholecseg8k"), t["b2"].update(dataset="cholecseg8k")), "different datasets"),
    (lambda t: (t["c2"].update(propagation="forward_from_first"), t["b2"].update(propagation="forward_from_first")),
     "different rules"),
])
def test_a_table_whose_rows_differ_in_what_must_match_is_refused(plant, said):
    # Each row's two conditions match; the rows do not, so only a check
    # across the whole table refuses it.
    tables = {t: evaluator_scores(t, seed=i) for i, t in enumerate(("c1", "b1", "c2", "b2"))}
    plant(tables)
    with pytest.raises(ValueError, match=said):
        PT.table_lines(two_rows(), tables, keys=[metric_key("F1_50", "all")])


def test_two_rules_beside_one_per_frame_condition_are_refused():
    # Each row sets a tracked condition beside the per-frame one, which
    # either rule may be compared with; the table still holds two rules.
    tables = {"p": evaluator_scores("p", propagation="per_frame"), "a": evaluator_scores("a", seed=2),
              "f": evaluator_scores("f", seed=3, propagation="forward_from_first")}
    with pytest.raises(ValueError, match="different rules"):
        PT.table_lines(two_rows("p", "a", "p", "f"), tables, keys=[metric_key("F1_50", "all")])


def test_a_mix_the_first_condition_cannot_see_is_refused():
    # A pilot JSON with no `eval_version` records no domain, and passes
    # against any; the two after it remove different classes as `tissue`.
    tables = {"c1": pilot_scores("c1"), "b1": pilot_scores("b1", {}, seed=2),
              "c2": pilot_scores("c2", {}, seed=3), "b2": pilot_scores("b2", {}, seed=4)}
    del tables["c1"]["eval_version"]
    tables["b2"]["tissue_ignore"] = [1, 5]
    with pytest.raises(ValueError, match="different domains"):
        PT.table_lines(two_rows(), tables)


def test_different_populations_are_refused_where_the_workbench_compared_the_clips_both_held():
    pair = workbench_pair()
    pair["cond"]["per_clip"].pop()
    pair["cond"]["clips"].pop()
    with pytest.raises(ValueError, match="clip sets differ"):
        PT.table_lines(one_row(), pair)


def test_differing_library_versions_are_noted_not_refused():
    pair = {"base": evaluator_scores("base"), "cond": evaluator_scores("cond", seed=2)}
    pair["base"]["versions"] = {"numpy": "2.1.0"}
    lines = PT.table_lines(one_row(), pair, keys=[metric_key("F1_50", "all")])
    assert "note: library versions differ between cond and base: {'numpy': ('2.0.0', '2.1.0')}" in lines


def test_the_paper_s_table_names_each_row_once():
    rows = [(cond, base) for _, block in PT.BLOCKS for _, cond, base in block]
    labels = [label for _, block in PT.BLOCKS for label, _, _ in block]
    assert len(set(rows)) == len(rows) and len(set(labels)) == len(labels)


def _write(tmp_path, tags):
    for i, tag in enumerate(tags):
        (tmp_path / f"{tag}.json").write_text(json.dumps(pilot_scores(tag, {"inst_F1_50": 0.01 * i}, seed=i)))


def _run(tmp_path, *args):
    return subprocess.run([sys.executable, "-m", "evalkit.tools.paired_table", "--eval-dir", str(tmp_path), *args],
                          capture_output=True, text=True, cwd=REPO)


def test_the_command_prints_every_row_of_the_paper_s_table(tmp_path):
    _write(tmp_path, PT.tags_of(PT.BLOCKS))
    out = _run(tmp_path)
    assert out.returncode == 0, out.stderr
    for heading, rows in PT.BLOCKS:
        assert heading in out.stdout
        for label, _, _ in rows:
            assert f"\n  {label} " in out.stdout
    assert out.stdout.count(" CI [") == sum(len(rows) for _, rows in PT.BLOCKS)


def test_the_command_stops_on_a_missing_condition_with_a_message_not_a_traceback(tmp_path):
    tags = PT.tags_of(PT.BLOCKS)
    _write(tmp_path, tags[1:])
    out = _run(tmp_path)
    assert out.returncode != 0 and "Traceback" not in out.stderr
    assert f"{tags[0]}: no score JSON to read" in out.stderr


def test_the_command_stops_on_a_reference_value(tmp_path):
    _write(tmp_path, PT.tags_of(PT.BLOCKS))
    out = _run(tmp_path, "--keys", "inst_F1_50,time_IoU")
    assert out.returncode != 0 and "Traceback" not in out.stderr
    assert "time_IoU is a reference value" in out.stderr
