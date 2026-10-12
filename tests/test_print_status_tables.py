"""The two tables of `print_status_tables`: each mark is `mark_of`'s, and each refusal is shown on a planted fault."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from evalkit.tools import print_status_tables as PST
from evalkit.tools.paired_stats import sign_only_note

REPO = Path(__file__).resolve().parent.parent

UP, DOWN, ACROSS = [0.01, 0.02], [-0.02, -0.01], [-0.01, 0.02]


def cell(delta: float, ci: list[float]) -> dict:
    return {"delta": delta, "ci95_video": list(ci)}


def input_track(n_videos: int = 6) -> dict:
    """An input-track JSON: every arm the table prints, `full` up, `labeled` down, the rest across zero."""
    keys = [k for k, _ in PST.INPUT_TRACK_KEYS]
    cis = {"full": UP, "labeled": DOWN}
    arms = {f"in_{i}_k{k}": {"n": 6.0 + k / 10} for i in (*PST.INPUTS, "rgb") for k in PST.KS}
    pairs = {f"in_{i}_k{k}": {key: cell(0.01, cis.get(key, ACROSS)) for key in keys}
             for i in PST.INPUTS for k in PST.KS}
    # A procedure with no difference is not counted as positive.
    procedures = {"p1": {"delta": {"full": 0.1, "boundary_F": -0.1}}, "p2": {"delta": {"full": 0.2, "boundary_F": 0.1}},
                  "p3": {"delta": {"full": 0.0, "boundary_F": 0.0}}}
    matched = {f"in_{i}_k{k}": {**{key: cell(-0.02, cis.get(key, ACROSS)) for key in keys}, "n_reached": 8.0,
                                "n_clamped": 3, "k_target": k, "broken": False, "by_procedure": procedures}
               for i in PST.INPUTS for k in PST.KS}
    return dict(dataset="atlas97", mode="input_track", stride=1, n_clips=12, n_videos=n_videos, n_frames=40,
                eval_code_sha="1f8a813a" + "0" * 56, base_name="rgb", arms=arms, pairs=pairs, matched=matched)


def identity() -> dict:
    """An identity JSON whose intervals lie wholly on one side of zero, so a mark could be read on each."""
    metrics = {k: {"delta_mean": 0.2, "ci95_video": UP if n % 2 else DOWN, "wins": 3, "losses": 1}
               for n, (k, _, _) in enumerate(PST.IDENTITY_KEYS)}
    cond = dict(hold_mean=0.8, idf1=0.6, idsw_per_track=0.02, frag_per_track=0.07, re_recovery=0.97, re_n_gaps=12,
                n_gt_tracks=100, gt_inst_per_frame=6.9)
    return dict(dataset="atlas97", n_boot=10000, params={"min_area": 400}, primary="hold_mean",
                conditions={"rgb/a": dict(cond), "rgb/b": dict(cond)},
                pairs={"rgb/a:rgb/b": {"n_clips": 4, "n_videos": 3, "metrics": metrics}})


# ---------------------------------------------------------------- marks


def test_a_mark_is_mark_of_on_the_stored_interval():
    lines = PST.input_track_lines(input_track())
    row = next(line for line in lines if line.startswith("in_edge_k0 "))
    assert row == ("in_edge_k0         6.0/ 6.0 +0.0100 ★ [+0.0100,+0.0200] +0.0100 ✗ [-0.0200,-0.0100]"
                   + " +0.0100   [-0.0100,+0.0200]" * 3)
    assert not any("sign only" in line for line in lines)


def test_an_analysis_on_five_videos_marks_nothing_and_says_so():
    lines = PST.input_track_lines(input_track(n_videos=5))
    assert not any("★" in line or "✗" in line for line in lines[2:])
    assert lines[2] == f"every interval is on 5 videos, {sign_only_note(5)}"


def test_a_mark_reads_the_interval_in_the_key_s_direction(monkeypatch):
    # Every key the input-track table prints is one where higher is better, so the
    # direction is planted: an interval below zero is a star where lower is better.
    monkeypatch.setitem(PST.PILOT_KEY_OF, "planted", "underseg_error")
    assert PST.mark_text(DOWN, "planted", 6) == "★" and PST.mark_text(UP, "planted", 6) == "✗"


def test_the_identity_table_carries_no_mark():
    # No measure over time carries a star, though every interval here lies on one side of zero.
    lines = PST.identity_lines(identity())
    assert not any("★" in line or "✗" in line for line in lines)
    assert lines[4].startswith("rgb/a:rgb/b") and "+0.2000 [+0.0100,+0.0200]   3-  1" in lines[4]


def test_the_identity_table_counts_wins_and_losses_in_the_key_s_direction():
    # The analysis stores 3 clips with a positive and 1 with a negative difference on every key.
    # Where lower is better (IDsw/track, Frag/track), those 3 are losses.
    row = PST.identity_lines(identity())[4]
    assert re.findall(r"\] +(\d+- +\d+)", row) == ["3-  1", "3-  1", "1-  3", "1-  3"]


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


def test_the_input_track_table_refuses_matched_rows_that_are_not_the_pairs():
    d = input_track()
    del d["matched"]["in_normal_k0"]
    with pytest.raises(ValueError, match=r"the matched rows are not the pairs: .* missing \['in_normal_k0'\]"):
        PST.input_track_lines(d)


def test_the_input_track_table_refuses_another_mode_of_the_analysis():
    # The per-frame mode writes the same arm names from other labels, so only `mode` tells the JSONs apart.
    d = input_track()
    d["mode"] = "input"
    with pytest.raises(ValueError, match="ran with --mode input, not input_track"):
        PST.input_track_lines(d)


def test_a_table_refuses_a_row_whose_interval_is_null():
    d = input_track()
    d["pairs"]["in_depth_k8"]["labeled"]["ci95_video"] = None
    with pytest.raises(ValueError, match="in_depth_k8 labeled: the video-level interval is null"):
        PST.input_track_lines(d)
    d = identity()
    d["pairs"]["rgb/a:rgb/b"]["metrics"]["idf1"]["ci95_video"] = None
    with pytest.raises(ValueError, match="rgb/a:rgb/b idf1: the video-level interval is null"):
        PST.identity_lines(d)


def test_the_matched_rows_count_the_procedures_with_a_positive_difference():
    row = next(line for line in PST.input_track_lines(input_track()) if line.startswith("in_depth_k0    reached"))
    assert row.endswith("| procedures with a positive Δ, of 3: F1 2 / boundary F 1")
    assert "reached 8.00 / clamped   3 clips" in row


def test_the_identity_table_refuses_conditions_measured_against_different_gt_tracks():
    d = identity()
    d["conditions"]["rgb/b"]["n_gt_tracks"] = 99
    with pytest.raises(ValueError, match="different GT tracks"):
        PST.identity_lines(d)
    d = identity()
    d["conditions"]["rgb/b"]["gt_inst_per_frame"] = 7.1
    with pytest.raises(ValueError, match="different GT tracks"):
        PST.identity_lines(d)


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
               "--identity", _write(tmp_path, "e2.json", identity()))
    assert run.returncode == 0, run.stderr
    sections = run.stdout.split("\n\n" + PST.SEPARATOR + "\n")
    assert [s.split(" ", 1)[0] for s in sections] == ["Inputs", "Identity"]


def test_the_command_prints_one_table_without_a_separator(tmp_path):
    run = _run("--identity", _write(tmp_path, "e2.json", identity()))
    assert run.returncode == 0, run.stderr
    assert PST.SEPARATOR not in run.stdout and run.stdout.startswith("Identity over time")


def test_the_command_refuses_no_table_and_prints_nothing_of_a_run_it_refuses(tmp_path):
    run = _run()
    assert run.returncode != 0 and "at least one analysis JSON" in run.stderr
    bad = identity()
    bad["conditions"]["rgb/b"]["n_gt_tracks"] = 99
    run = _run("--input-track", _write(tmp_path, "e1.json", input_track()),
               "--identity", _write(tmp_path, "e2.json", bad))
    assert run.returncode != 0 and "e2.json" in run.stderr and run.stdout == ""


def test_the_command_refuses_a_value_that_is_not_a_number(tmp_path):
    path = tmp_path / "e2.json"
    path.write_text(json.dumps(identity()).replace("0.97", "NaN"), encoding="utf-8")
    run = _run("--identity", str(path))
    assert run.returncode != 0 and "NaN" in run.stderr
