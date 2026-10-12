"""The StereoMIS table on rows whose differences are known.

Six sequences, the fewest a mark needs, hold two clips each, one `moving`
and one `slow`. `good` scores 0.2 below the base on every clip, `bad` 0.1
above it, and `mixed` 0.1 below on two sequences' clips and 0.1 above on
the others'.
"""

import json
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

import trajectory_eval.tools.stereomis_table as T  # noqa: E402
from evalkit.tools.paired_stats import MARK_RULE, MIN_VIDEOS_FOR_MARK, boot_ci  # noqa: E402

SHA = "c" * 64
SEQS = ("P1", "P2_0", "P2_1", "P2_2", "P2_3", "P2_4")
OFFSET = {"floor_static": lambda s: 0.0, "good": lambda s: -0.2, "bad": lambda s: 0.1,
          "mixed": lambda s: -0.1 if s in ("P1", "P2_0") else 0.1}


def rows_of(conds=tuple(OFFSET)) -> list[dict]:
    rows = []
    for i, s in enumerate(SEQS):
        for k, stratum in enumerate(("moving", "slow")):
            base = 0.9 - 0.1 * i - 0.05 * k
            for c in conds:
                rows.append(dict(clip=f"{s}__clip_{k:04d}", seq=s, stratum=stratum, span_mm=20.0 - 15 * k, cond=c,
                                 ate_rel=round(base + OFFSET[c](s), 4), rpe_trans_rel=0.5, rpe_rot_deg=1.0,
                                 scale_ratio=None if c == "floor_static" else 2.0, trajectory_code_sha=SHA))
    return rows


def test_a_condition_below_the_base_is_marked_better_and_one_above_worse():
    rows = rows_of()
    good, bad = T.compare(rows, "floor_static", "good"), T.compare(rows, "floor_static", "bad")
    assert (good["mark"], good["delta"], good["better"], good["worse"]) == ("★", -0.2, 12, 0)
    assert (bad["mark"], bad["delta"], bad["better"], bad["worse"]) == ("✗", 0.1, 0, 12)
    assert (good["n"], good["n_seq"], good["key"], good["stratum"]) == (12, 6, "ate_rel", "")


def test_a_pair_on_five_sequences_gets_no_mark_and_says_so_and_one_on_six_does(capsys):
    assert len(SEQS) == MIN_VIDEOS_FOR_MARK
    five = [r for r in rows_of() if r["seq"] != SEQS[-1]]
    assert T.compare(five, "floor_static", "good")["mark"] == ""
    assert "[read for its sign only: a mark needs 6 videos]" in capsys.readouterr().out
    assert T.compare(rows_of(), "floor_static", "good")["mark"] == "★"
    assert "read for its sign only" not in capsys.readouterr().out


def test_the_interval_is_the_paired_stats_bootstrap_over_sequences():
    rows = rows_of()
    pair = T.compare(rows, "floor_static", "mixed")
    by = {(r["clip"], r["cond"]): r["ate_rel"] for r in rows}
    clips = sorted({r["clip"] for r in rows})
    d = np.array([by[c, "mixed"] - by[c, "floor_static"] for c in clips])
    np.testing.assert_allclose(d, [-0.1] * 4 + [0.1] * 8, atol=1e-9)
    lo, hi = boot_ci(d, np.repeat(SEQS, 2))
    assert pair["ci95_seq"] == [round(lo, 4), round(hi, 4)] and pair["delta"] == round(d.mean(), 4)
    assert pair["mark"] == "" and pair["better"] == 4 and pair["worse"] == 8


def test_an_interval_end_near_zero_keeps_its_sign_its_mark_and_its_printed_digits(capsys):
    rows = rows_of(("floor_static",))
    rows += [dict(r, cond="tiny", ate_rel=r["ate_rel"] - 2e-5) for r in rows]
    pair = T.compare(rows, "floor_static", "tiny")
    assert pair["mark"] == "★" and pair["ci95_seq"] == [-2e-5, -2e-5]
    assert "95%CI(seq) [-2.00e-05, -2.00e-05]" in capsys.readouterr().out


def test_a_stratum_pairs_its_own_clips():
    pair = T.compare(rows_of(), "floor_static", "good", stratum="slow")
    assert (pair["n"], pair["n_seq"], pair["stratum"]) == (6, 6, "slow")


def test_every_condition_is_paired_with_the_base_overall_and_in_each_stratum(capsys):
    out = T.summarize(rows_of(), "floor_static")
    assert [(p["cond"], p["stratum"]) for p in out["pairs"]] == [
        (c, st) for st in ("", "moving", "slow") for c in ("bad", "good", "mixed")]
    printed = capsys.readouterr().out
    assert MARK_RULE in printed and "★ good - floor_static: delta -0.2000" in printed
    assert "P1       2          +0.1000         -0.2000         -0.1000" in printed


def test_a_condition_scored_on_other_clips_than_the_base_is_refused():
    rows = [r for r in rows_of() if not (r["cond"] == "good" and r["clip"] == "P2_1__clip_0001")]
    with pytest.raises(ValueError, match=r"other clips than the base 'floor_static': "
                                         r"\[\"good \(lacks 1 of the base's clips, adds 0\)\"\]"):
        T.summarize(rows, "floor_static")


@pytest.mark.parametrize("field, value", [("seq", "P2_0"), ("stratum", "moving"), ("span_mm", 5.001)])
def test_a_clip_cut_in_two_ways_is_refused(field, value):
    rows = [dict(r, **{field: value}) if r["cond"] == "good" and r["clip"] == "P1__clip_0001" else r
            for r in rows_of()]
    with pytest.raises(ValueError, match=r"differs between conditions: \['P1__clip_0001'\]"):
        T.summarize(rows, "floor_static")


def test_a_pair_of_one_sequence_is_refused_with_its_name():
    rows = [r for r in rows_of() if r["seq"] == "P1"]
    with pytest.raises(ValueError, match="bad - floor_static: a bootstrap over 1 "):
        T.summarize(rows, "floor_static")


def test_a_base_with_no_row_is_refused():
    with pytest.raises(ValueError, match=r"no row of the base 'floor_line'"):
        T.summarize(rows_of(), "floor_line")


def test_the_command_reads_the_score_files_and_refuses_two_versions_of_the_scorer(tmp_path, monkeypatch):
    rows = rows_of()
    a, b = tmp_path / "controls.json", tmp_path / "methods.json"
    a.write_text(json.dumps([r for r in rows if r["cond"] == "floor_static"]))
    b.write_text(json.dumps([r for r in rows if r["cond"] != "floor_static"]))
    out = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", ["stereomis_table", str(a), str(b), "--out", str(out)])
    T.main()
    written = json.loads(out.read_text())
    assert len(written["pairs"]) == 9 and written["trajectory_code_sha"] == SHA
    b.write_text(json.dumps([dict(r, trajectory_code_sha="d" * 64) for r in rows if r["cond"] != "floor_static"]))
    with pytest.raises(SystemExit, match="2 versions of the scorer"):
        T.main()
