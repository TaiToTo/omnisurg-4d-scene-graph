"""arms_paired prints the workbench's rows, marks a line only in its key's direction, and refuses a mixed table.

`scores.check_comparable_table`, which refuses a mixed table, is tested
with `paired_table`, which uses it too; here the command shows it is used.

The rows are pinned to what the workbench's `arms_paired.py` printed on
the same per-clip values, generated here and run through the workbench
version once.
"""
import copy
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from evalkit.tools import arms_paired as AP
from evalkit.tools.paired_stats import MARK_RULE, boot_ci, video_of
from evalkit.tools.scores import PILOT_EVAL_CODE_SHA, metric_key

pytest.importorskip("scipy", reason="the Wilcoxon test needs the `tools` extra")

REPO = Path(__file__).resolve().parent.parent
VIDEOS = ("adrenalectomy__16GPCUPkXYQ", "appendectomy__41RKDh3INiU", "gastric_surgery__4FHGGFZsPzw",
          "lar___7H7G-4sevQ", "rectopexy__IJfgUvxoTkM", "cholecystectomy__1ud3syYKD3A")
CLIPS = [f"{v}__gt_{i:04d}" for v in VIDEOS for i in (1, 2, 3)]
SHA = "a" * 64

# Each condition's shift from the base and its seed. A video's clips share
# an effect, as clips of one video do.
CONDITIONS = {"rgb_pf": (0.0, 1), "rgb_edge_pf": (0.06, 2), "normal_pf": (-0.06, 3), "normal_edge_pf": (0.01, 4)}
PAIRS = [("rgb_pf", "rgb_edge_pf"), ("rgb_pf", "normal_pf"), ("normal_pf", "normal_edge_pf"), ("rgb_pf", "normal_edge_pf")]


def pilot_scores(shift: float, seed: int) -> dict:
    """A pilot-layout score JSON on six videos of three clips, with `SQ` None on some clips."""
    rnd = random.Random(seed)
    effect = {v: rnd.gauss(0, 0.03) for v in VIDEOS}
    rows = []
    for c in CLIPS:
        e = effect[video_of(c)]
        rows.append({
            "clip": c, "extra_ignore": [],
            "inst_F1_50": round(0.5 + shift + e + rnd.gauss(0, 0.04), 4),
            "underseg_error": round(0.4 - shift + e + rnd.gauss(0, 0.04), 4),
            "time_IoU": round(0.6 + shift + e + rnd.gauss(0, 0.04), 4),
            "SQ": None if rnd.random() < 0.25 else round(0.7 + shift + rnd.gauss(0, 0.04), 4),
        })
    return dict(tag="t", eval_code_sha=PILOT_EVAL_CODE_SHA, eval_code_tag="t", eval_version=2,
                dataset="atlas", tissue_ignore=[1, 2], clips=list(CLIPS), per_clip=rows)


def table_of_pilot_scores() -> dict:
    return {tag: pilot_scores(*CONDITIONS[tag]) for tag in CONDITIONS}


def evaluator_scores(propagation="both_ways_from_centre", class_set="original", views=("all", "tissue", "geometric"),
                     pilot=False, seed=1) -> dict:
    rnd = random.Random(seed)
    rows = [dict(clip=c, **{metric_key("F1_50", v): round(rnd.random(), 4) for v in views}) for c in CLIPS]
    return dict(eval_code_sha=SHA, dataset="atlas120k", pilot=pilot, class_set=class_set, views=list(views),
                clips=list(CLIPS), input_shas={c: {"gt_masks": "g", "depth": "d", "predictions": "p"} for c in CLIPS},
                versions={"numpy": "2.0.0"}, propagation=propagation, per_clip=rows)


def rows(key: str, summaries=None, pairs=PAIRS) -> list[str]:
    table = AP.compare_pairs(summaries or table_of_pilot_scores(), pairs, key)
    return [AP.format_row(b, c, r, len(table["clips"]), table["n_videos"]) for b, c, r in table["rows"]]


# What the workbench's arms_paired printed for `PAIRS` on three keys of
# `table_of_pilot_scores()`, each condition named by its tag, with the clips
# in sorted order, as `pps_f1.py` wrote them and `check_comparable` returns
# them. The bootstrap's seed is drawn from the data in its order, so the
# clips in the order of `CLIPS` give other intervals. It marked every key
# as if higher were better.
WORKBENCH_ROWS = {
    "inst_F1_50": [
        "rgb_edge_pf − rgb_pf               +0.0822   [+0.0519,+0.1117]      18-0-0   0.0000 ★",
        "normal_pf − rgb_pf                 -0.0663   [-0.0969,-0.0329]      4-0-14   0.0019 ✗",
        "normal_edge_pf − normal_pf         +0.0887   [+0.0640,+0.1213]      16-0-2   0.0001 ★",
        "normal_edge_pf − rgb_pf            +0.0224   [-0.0192,+0.0595]      13-0-5   0.1187",
    ],
    "time_IoU": [
        "rgb_edge_pf − rgb_pf               +0.0714   [+0.0327,+0.1056]      14-0-4   0.0004 ★",
        "normal_pf − rgb_pf                 -0.0658   [-0.1152,-0.0070]      4-0-14   0.0077 ✗",
        "normal_edge_pf − normal_pf         +0.0849   [+0.0400,+0.1158]      16-0-2   0.0002 ★",
        "normal_edge_pf − rgb_pf            +0.0191   [-0.0118,+0.0537]      12-0-6   0.1540",
    ],
    "underseg_error": [
        "rgb_edge_pf − rgb_pf               -0.0468   [-0.0861,-0.0039]      5-0-13   0.0210 ✗",
        "normal_pf − rgb_pf                 +0.0843   [+0.0413,+0.1221]      16-0-2   0.0007 ★",
        "normal_edge_pf − normal_pf         -0.0788   [-0.1076,-0.0523]      1-0-17   0.0000 ✗",
        "normal_edge_pf − rgb_pf            +0.0055   [-0.0481,+0.0533]      10-0-8   0.5798",
    ],
}


# ---------------------------------------------------------------- the rows


def test_each_row_is_the_workbench_s():
    assert rows("inst_F1_50") == WORKBENCH_ROWS["inst_F1_50"]


def test_a_reference_value_is_never_marked():
    # The workbench marked `time_IoU` as it marked F1. The numbers stay; the marks go.
    unmarked = [r.removesuffix(" ★").removesuffix(" ✗") for r in WORKBENCH_ROWS["time_IoU"]]
    assert unmarked != WORKBENCH_ROWS["time_IoU"]
    assert rows("time_IoU") == unmarked


def test_a_lower_is_better_key_is_marked_in_its_direction():
    # `underseg_error` falling is a gain: the workbench drew it as a loss.
    swap = {"★": "✗", "✗": "★"}
    flipped = [r[:-1] + swap[r[-1]] if r[-1] in swap else r for r in WORKBENCH_ROWS["underseg_error"]]
    assert rows("underseg_error") == flipped


def test_a_key_left_undefined_on_some_clips_is_taken_on_the_clips_both_define():
    summaries = table_of_pilot_scores()
    table = AP.compare_pairs(summaries, PAIRS[:1], "SQ")
    base, cond, row = table["rows"][0]
    a = {r["clip"]: r["SQ"] for r in summaries[base]["per_clip"]}
    b = {r["clip"]: r["SQ"] for r in summaries[cond]["per_clip"]}
    both = [c for c in sorted(CLIPS) if a[c] is not None and b[c] is not None]
    d = np.array([b[c] - a[c] for c in both])
    assert row["n_clips"] == len(both) < len(CLIPS)
    assert row["delta"] == d.mean()
    assert row["ci95_video"] == boot_ci(d, np.array([video_of(c) for c in both]))
    note = f"   [{len(both)}/{len(CLIPS)} clips, {row['n_videos']}/{len(VIDEOS)} videos]"
    assert AP.format_row(base, cond, row, len(CLIPS), len(VIDEOS)).endswith(note)


def test_a_key_defined_on_one_video_has_no_interval_and_no_mark():
    summaries = table_of_pilot_scores()
    for tag, s in summaries.items():
        for r in s["per_clip"]:
            r["SQ"] = 0.5 + CONDITIONS[tag][0] if video_of(r["clip"]) == VIDEOS[0] else None
    base, cond, row = AP.compare_pairs(summaries, PAIRS[:1], "SQ")["rows"][0]
    assert (row["n_videos"], row["ci95_video"], row["mark"]) == (1, None, "")
    assert "none" in AP.format_row(base, cond, row, len(CLIPS), len(VIDEOS))


def test_a_pair_that_never_moves_has_no_p():
    summaries = table_of_pilot_scores()
    summaries["rgb_pf_again"] = copy.deepcopy(summaries["rgb_pf"])
    base, cond, row = AP.compare_pairs(summaries, [("rgb_pf", "rgb_pf_again")], "inst_F1_50")["rows"][0]
    assert (row["up"], row["same"], row["down"], row["p_clip"], row["mark"]) == (0, len(CLIPS), 0, None, "")
    assert AP.format_row(base, cond, row, len(CLIPS), len(VIDEOS)).endswith(f"0-{len(CLIPS)}-0     none")


def test_only_a_difference_of_exactly_zero_stays():
    # 0.1 + 0.2 − 0.3 is 5.6e-17, not zero: the workbench counted it as staying, below 1e-9.
    summaries = table_of_pilot_scores()
    summaries["rgb_pf_again"] = copy.deepcopy(summaries["rgb_pf"])
    summaries["rgb_pf"]["per_clip"][0]["inst_F1_50"] = 0.3
    summaries["rgb_pf_again"]["per_clip"][0]["inst_F1_50"] = 0.1 + 0.2
    row = AP.compare_pairs(summaries, [("rgb_pf", "rgb_pf_again")], "inst_F1_50")["rows"][0][2]
    assert (row["up"], row["same"], row["down"]) == (1, len(CLIPS) - 1, 0)


# ---------------------------------------------------------------- what a table refuses


def test_the_header_names_the_table_s_rule(capsys):
    table = AP.compare_pairs({"pf": evaluator_scores(propagation="per_frame", seed=1), "both": evaluator_scores(seed=2)},
                             [("pf", "both")], "F1_50/geometric")
    AP.print_table(table)
    assert capsys.readouterr().out.startswith(f"atlas120k: {len(CLIPS)} clips / {len(VIDEOS)} videos, "
                                              "eval_code=aaaaaaaaaaaaaaaa, propagation=both_ways_from_centre\n")


def test_different_library_versions_are_reported_not_refused(capsys):
    summaries = {"a": evaluator_scores(seed=1), "b": evaluator_scores(seed=2), "c": evaluator_scores(seed=3)}
    summaries["c"]["versions"] = {"numpy": "2.1.0"}
    table = AP.compare_pairs(summaries, [("a", "b"), ("a", "c")], "F1_50/geometric")
    assert table["versions_differ"] == {("a", "c"): {"numpy": ("2.0.0", "2.1.0")}}
    AP.print_table(table)
    assert "note: library versions differ between a and c: {'numpy': ('2.0.0', '2.1.0')}" in capsys.readouterr().out


def test_a_key_the_jsons_do_not_report_is_refused():
    with pytest.raises(ValueError, match="not a key these JSONs report"):
        AP.compare_pairs({"a": evaluator_scores(seed=1), "b": evaluator_scores(seed=2)}, [("a", "b")], "F1_50/nowhere")


def test_a_pilot_key_with_no_direction_is_refused_for_that():
    # The pilot's JSONs hold VI_split, but no direction for it.
    with pytest.raises(ValueError, match="VI_split has no direction in scores.PILOT_SIGNS"):
        AP.compare_pairs(table_of_pilot_scores(), PAIRS, "VI_split")


def test_a_key_in_no_row_is_refused():
    summaries = table_of_pilot_scores()
    for s in summaries.values():
        for r in s["per_clip"]:
            del r["time_IoU"]
    with pytest.raises(ValueError, match="rgb_pf:rgb_edge_pf: time_IoU is defined on no clip"):
        AP.compare_pairs(summaries, PAIRS, "time_IoU")


def test_pairs_are_read_in_the_order_given():
    assert AP.parse_pairs(" b:a, c:d ,") == [("b", "a"), ("c", "d")]


@pytest.mark.parametrize("text", ["rgb", "rgb:", ":rgb", "a:b:c", "rgb:rgb", "", " , "])
def test_a_malformed_pair_is_refused(text):
    with pytest.raises(ValueError):
        AP.parse_pairs(text)


# ---------------------------------------------------------------- the command


def _write(tmp_path: Path, summaries: dict) -> None:
    for tag, s in summaries.items():
        (tmp_path / f"{tag}.json").write_text(json.dumps(s), encoding="utf-8")


def _run(tmp_path: Path, key: str = "inst_F1_50", pairs=PAIRS) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "evalkit.tools.arms_paired", "--eval-dir", str(tmp_path), "--key", key,
           "--pairs", ",".join(f"{b}:{c}" for b, c in pairs)]
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)


def test_the_command_prints_the_workbench_s_rows(tmp_path):
    _write(tmp_path, table_of_pilot_scores())
    res = _run(tmp_path)
    assert res.returncode == 0, res.stderr
    lines = res.stdout.splitlines()
    assert lines[0] == f"atlas: {len(CLIPS)} clips / {len(VIDEOS)} videos, eval_code={PILOT_EVAL_CODE_SHA[:16]}"
    head = next(i for i, line in enumerate(lines) if line.startswith("pair"))
    assert lines[head + 1:head + 1 + len(PAIRS)] == WORKBENCH_ROWS["inst_F1_50"]
    assert lines[-1] == f"★ / ✗: {MARK_RULE}."


def test_the_command_refuses_a_mixed_table_and_names_the_two_conditions(tmp_path):
    # Each pair passes alone: `normal_pf` and `normal_edge_pf` share the other evaluator.
    summaries = table_of_pilot_scores()
    for tag in ("normal_pf", "normal_edge_pf"):
        summaries[tag]["eval_code_sha"] = "b" * 64
    _write(tmp_path, summaries)
    res = _run(tmp_path, pairs=[("rgb_pf", "rgb_edge_pf"), ("normal_pf", "normal_edge_pf")])
    assert res.returncode != 0 and res.stdout == ""
    assert "rgb_pf and normal_pf cannot share a table" in res.stderr and "different evaluators" in res.stderr


def test_the_command_refuses_a_condition_with_no_score_json(tmp_path):
    # The workbench skipped such a pair, and the table lost a line with nothing said.
    summaries = table_of_pilot_scores()
    del summaries["normal_edge_pf"]
    _write(tmp_path, summaries)
    res = _run(tmp_path)
    assert res.returncode != 0 and res.stdout == ""
    assert "normal_edge_pf.json" in res.stderr
