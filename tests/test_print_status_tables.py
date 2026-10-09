"""The four tables of `print_status_tables`: each mark is the verdict's, and each refusal is shown on a planted fault."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from evalkit.tools import print_status_tables as PST

REPO = Path(__file__).resolve().parent.parent

UP, DOWN, ACROSS = [0.01, 0.02], [-0.02, -0.01], [-0.01, 0.02]


def cell(delta: float, ci: list[float], **more) -> dict:
    return {"delta": delta, "ci95_video": list(ci), **more}


def input_track() -> dict:
    """An input-track JSON: every arm the table prints, `full` up, `labeled` down, the rest across zero."""
    keys = [k for k, _ in PST.INPUT_TRACK_KEYS]
    cis = {"full": UP, "labeled": DOWN}
    arms = {f"in_{i}_k{k}": {"n": 6.0 + k / 10} for i in (*PST.INPUTS, "rgb") for k in PST.KS}
    pairs = {f"in_{i}_k{k}": {key: cell(0.01, cis.get(key, ACROSS)) for key in keys}
             for i in PST.INPUTS for k in PST.KS}
    # A procedure with no difference is not counted as positive.
    procedures = {"p1": {"delta": {"full": 0.1, "boundary_F": -0.1}}, "p2": {"delta": {"full": 0.2, "boundary_F": 0.1}},
                  "p3": {"delta": {"full": 0.0, "boundary_F": 0.0}}}
    matched = {f"in_{i}_k0": {**{key: cell(-0.02, cis.get(key, ACROSS)) for key in keys}, "n_reached": 8.0,
                              "n_clamped": 3, "k_target": 0, "broken": False, "by_procedure": procedures}
               for i in PST.INPUTS}
    return dict(dataset="atlas97", stride=1, n_clips=4, n_videos=3, n_frames=40, eval_code_sha="1f8a813a" + "0" * 56,
                base_name="rgb", arms=arms, pairs=pairs, matched=matched)


def identity() -> dict:
    """An identity JSON whose intervals lie wholly on one side of zero, so a mark could be read on each."""
    metrics = {k: {"delta_mean": 0.2, "ci95_video": UP if n % 2 else DOWN, "wins": 3, "losses": 1}
               for n, (k, _) in enumerate(PST.IDENTITY_KEYS)}
    cond = dict(hold_mean=0.8, idf1=0.6, idsw_per_track=0.02, frag_per_track=0.07, re_recovery=0.97, re_n_gaps=12,
                n_gt_tracks=100, gt_inst_per_frame=6.9)
    return dict(dataset="atlas97", n_boot=10000, params={"min_area": 400},
                conditions={"rgb/a": dict(cond), "rgb/b": dict(cond)},
                pairs={"rgb/a:rgb/b": {"n_clips": 4, "n_videos": 3, "metrics": metrics}})


def kmerge_pairs() -> dict:
    pair = dict(a="op_x", b="op_y", at="matched", delta=0.017, ci95_video=UP, verdict="★", n_clips=4, n_videos=3,
                win=3, lose=1)
    return dict(dataset="atlas97", stride=5, k_list=[2, 8], pairs=[pair, {**pair, "at": "k8", "ci95_video": ACROSS,
                                                                           "verdict": ""}])


def classic(degenerate: bool = False, **setting) -> dict:
    """A classic-segmentation JSON of one row, every key's interval below zero, its stored marks the verdict's."""
    stored = "" if degenerate else "✗"
    pair = {k: cell(-0.1, DOWN, verdict=stored) for k, _ in PST.CLASSIC_KEYS}
    row = dict(n_clips=27, n_videos=17, degenerate=degenerate, arms={"normal": {"n": 2.7}, "rgb": {"n": 3.6}}, pair=pair)
    return dict(dataset="cholec", k=8, markers="peak", **{"stride": 5, "drop_instruments": True, **setting},
                by_sigma={"33.0": row})


# ---------------------------------------------------------------- marks


def test_a_mark_is_the_verdict_on_the_stored_interval():
    lines = PST.input_track_lines(input_track())
    row = next(line for line in lines if line.startswith("in_edge_k0 "))
    assert row == ("in_edge_k0         6.0/ 6.0 +0.0100 ★ [+0.010,+0.020] +0.0100 ✗ [-0.020,-0.010]"
                   + " +0.0100    [-0.010,+0.020]" * 3)


def test_a_mark_reads_the_interval_in_the_key_s_direction(monkeypatch):
    # Every key the four tables print is one where higher is better, so the
    # direction is planted: an interval below zero is a star where lower is better.
    monkeypatch.setitem(PST.PILOT_KEY_OF, "planted", "underseg_error")
    assert PST.mark(DOWN, "planted") == "★" and PST.mark(UP, "planted") == "✗"


def test_the_identity_table_carries_no_mark():
    # No measure over time carries a star, though every interval here lies on one side of zero.
    lines = PST.identity_lines(identity())
    assert not any("★" in line or "✗" in line for line in lines)
    assert lines[4].startswith("rgb/a:rgb/b") and "+0.2000 [+0.010,+0.020]   3-  1" in lines[4]


def test_a_stored_mark_that_is_not_the_verdict_is_refused():
    d = kmerge_pairs()
    d["pairs"][1]["verdict"] = "★"
    with pytest.raises(ValueError, match="stores the mark '★'"):
        PST.kmerge_pairs_lines(d)
    d = classic()
    d["by_sigma"]["33.0"]["pair"]["boundary_F"]["verdict"] = ""
    with pytest.raises(ValueError, match="boundary_F .* stores the mark ''"):
        PST.classic_lines([d])


def test_a_degenerate_row_has_no_mark_where_its_interval_excludes_zero():
    lines = PST.classic_lines([classic(degenerate=True), classic()])
    assert lines[2].endswith("  degenerate") and "✗" not in lines[2]
    assert lines[3].count("✗") == len(PST.CLASSIC_KEYS)
    # A degenerate row whose JSON stores a mark was decided by another rule.
    d = classic(degenerate=True)
    d["by_sigma"]["33.0"]["pair"]["labeled"]["verdict"] = "✗"
    with pytest.raises(ValueError, match="stores the mark '✗'"):
        PST.classic_lines([d])


# ---------------------------------------------------------------- refusals


def test_the_input_track_table_refuses_pairs_that_are_not_its_rows():
    d = input_track()
    d["pairs"]["in_edge_k20"] = d["pairs"]["in_edge_k0"]
    with pytest.raises(ValueError, match=r"not printed \['in_edge_k20'\]"):
        PST.input_track_lines(d)
    d = input_track()
    del d["pairs"]["in_depth_k12"]
    with pytest.raises(ValueError, match=r"missing \['in_depth_k12'\]"):
        PST.input_track_lines(d)


def test_the_input_track_table_refuses_a_broken_matched_row():
    d = input_track()
    d["matched"]["in_normal_k0"]["broken"] = True
    with pytest.raises(ValueError, match="in_normal_k0: the analysis found the matched row broken"):
        PST.input_track_lines(d)


def test_the_matched_rows_count_the_procedures_with_a_positive_difference():
    row = next(line for line in PST.input_track_lines(input_track()) if line.startswith("in_depth_k0    reached"))
    assert row.endswith("| procedures with a positive Δ, of 3: F1 2 / boundary F 1")
    assert "reached 8.00 / clamped   3 clips" in row


def test_the_identity_table_refuses_conditions_measured_against_different_gt_tracks():
    d = identity()
    d["conditions"]["rgb/b"]["n_gt_tracks"] = 99
    with pytest.raises(ValueError, match="different GT tracks"):
        PST.identity_lines(d)


@pytest.mark.parametrize("setting", [{"stride": 1}, {"drop_instruments": False}])
def test_the_classic_table_refuses_files_with_another_setting(setting):
    with pytest.raises(ValueError, match="differ in stride or in dropping instruments"):
        PST.classic_lines([classic(), classic(**setting)])


# ---------------------------------------------------------------- the command


def _write(tmp_path, name, d):
    path = tmp_path / name
    path.write_text(json.dumps(d), encoding="utf-8")
    return str(path)


def _run(*args):
    return subprocess.run([sys.executable, "-m", "evalkit.tools.print_status_tables", *args],
                          capture_output=True, text=True, cwd=REPO)


def test_the_command_prints_each_table_given_between_separators(tmp_path):
    run = _run("--input-track", _write(tmp_path, "e1.json", input_track()),
               "--identity", _write(tmp_path, "e2.json", identity()),
               "--kmerge-pairs", _write(tmp_path, "h2.json", kmerge_pairs()),
               "--classic", _write(tmp_path, "c2.json", classic()), _write(tmp_path, "c1.json", classic(True)))
    assert run.returncode == 0, run.stderr
    sections = run.stdout.split("\n\n" + PST.SEPARATOR + "\n")
    assert [s.split(" ", 1)[0] for s in sections] == ["Inputs", "Identity", "Merged", "Classic"]
    # The classic JSONs are printed in sorted order, whatever order they are named in.
    assert sections[3].splitlines()[2].endswith("degenerate")


def test_the_command_prints_one_table_without_a_separator(tmp_path):
    run = _run("--kmerge-pairs", _write(tmp_path, "h2.json", kmerge_pairs()))
    assert run.returncode == 0, run.stderr
    assert PST.SEPARATOR not in run.stdout and run.stdout.startswith("Merged pairs")


def test_the_command_refuses_no_table_and_prints_nothing_of_a_run_it_refuses(tmp_path):
    run = _run()
    assert run.returncode != 0 and "at least one analysis JSON" in run.stderr
    bad = copy.deepcopy(kmerge_pairs())
    bad["pairs"][0]["verdict"] = ""
    run = _run("--input-track", _write(tmp_path, "e1.json", input_track()),
               "--kmerge-pairs", _write(tmp_path, "h2.json", bad))
    assert run.returncode != 0 and "h2.json" in run.stderr and run.stdout == ""


def test_the_command_refuses_a_value_that_is_not_a_number(tmp_path):
    path = tmp_path / "h2.json"
    path.write_text(json.dumps(kmerge_pairs()).replace("0.017", "NaN"), encoding="utf-8")
    run = _run("--kmerge-pairs", str(path))
    assert run.returncode != 0 and "NaN" in run.stderr
