"""Test `evalkit.tools.d4d_front_behind_verdict` on made-up outputs of `d4d_front_behind`.

Each refusal has a test that plants its fault: outputs scored on other regions, over other answer pixels or by
other code, and outputs whose sides hold other pairs.
"""

import json

import pytest

from evalkit.tools import d4d_front_behind_verdict as v

# Six clips in three sessions of two specimens; the first clip's camera moves.
KEYS = ["s1/a/c1", "s1/a/c2", "s1/b/c1", "s1/b/c2", "s2/c/c1", "s2/c/c2"]


def rec(key, ok, side="end"):
    """Return a side's record whose pairs are answered `ok`, with gaps of 1, 3, 6, ... mm."""
    n = len(ok)
    return dict(key=key, side=side, n_pairs=n, n_correct=sum(ok), pairs=[[0, i + 1] for i in range(n)],
                gap_mm=[1.0 + 3 * i for i in range(n)], ok=list(ok))


def output(answer, oks, regions="ours", pixels="all", sha="abc"):
    return dict(answer=answer, regions=regions, answer_pixels=pixels, front_behind_code_sha=sha,
                per_frame=[rec(k, ok) for k, ok in zip(KEYS, oks)])


AHEAD = [[1, 1, 1, 0]] * 6
FLOOR = [[1, 0, 0, 1]] * 6


def test_an_answer_ahead_of_the_floor_in_every_unit_is_a():
    out = v.compare(output("mono", AHEAD), output("area", FLOOR), {"s1/a/c1"})
    assert out["verdict"] == "A" and not out["flip_floor"]
    assert {u: (d["n_units"], d["delta"], d["wins"], d["losses"]) for u, d in out["main"].items()} == {
        "session": (3, 0.25, 3, 0), "clip": (6, 0.25, 6, 0), "specimen": (2, 0.25, 2, 0)}
    assert (out["strata"]["moved"]["n_units"], out["strata"]["static"]["n_units"]) == (1, 5)
    assert [(r["n_pairs"], r["acc"]) for r in out["gap_curve_answer"]] == [
        (24, 0.75), (18, 0.6667), (12, 0.5), (6, 0.0), (0, None)]


def test_a_floor_below_one_half_is_read_flipped():
    worse = [[0, 0, 0, 1]] * 6
    out = v.compare(output("mono", AHEAD), output("area", worse), {"s1/a/c1"})
    assert out["flip_floor"] and out["main"]["clip"]["acc_floor"] == 0.75 and out["verdict"] == "B"


def test_a_stratum_with_no_clip_is_left_empty():
    assert v.compare(output("mono", AHEAD), output("area", FLOOR), set())["strata"]["moved"] is None


def test_the_verdict_reads_the_signs_and_the_intervals():
    unit = lambda delta, lo, hi: dict(delta=delta, lo=lo, hi=hi)
    assert v.verdict({"session": unit(0.1, 0.05, 0.2), "clip": unit(0.1, 0.02, 0.2)}) == "A"
    assert v.verdict({"session": unit(-0.1, -0.2, -0.05), "clip": unit(-0.1, -0.2, -0.01)}) == "C"
    assert v.verdict({"session": unit(0.1, 0.05, 0.2), "clip": unit(0.1, -0.01, 0.2)}) == "B"
    assert v.verdict({"session": unit(0.1, 0.05, 0.2), "clip": unit(-0.1, -0.2, -0.01)}) == "undecided"


@pytest.mark.parametrize("floor, message", [
    (output("area", FLOOR, regions="kmeans"), "other regions"),
    (output("area", FLOOR, sha="def"), "other code"),
    (output("screen", FLOOR, pixels="gt"), "other pixels"),
])
def test_outputs_not_scored_alike_are_refused(floor, message):
    with pytest.raises(ValueError, match=message):
        v.compare(output("mono", AHEAD), floor, set())


def test_answer_pixels_are_not_compared_against_the_area_floor():
    assert v.compare(output("mono", AHEAD, pixels="gt"), output("area", FLOOR), {"s1/a/c1"})["verdict"] == "A"


def test_a_side_with_pairs_in_one_output_only_is_refused():
    floor = output("area", FLOOR)
    floor["per_frame"][0].update(n_pairs=0, n_correct=0, pairs=[], gap_mm=[], ok=[])
    with pytest.raises(ValueError, match=r"in the answer only: \[\('s1/a/c1', 'end'\)\]"):
        v.compare(output("mono", AHEAD), floor, set())


def test_a_side_with_other_pairs_is_refused():
    floor = output("area", FLOOR)
    floor["per_frame"][2]["pairs"][0] = [1, 2]
    with pytest.raises(ValueError, match=r"1 side\(s\) hold other pairs.*s1/b/c1"):
        v.compare(output("mono", AHEAD), floor, set())


def test_main_writes_the_verdict_and_refuses_outputs_not_scored_alike(tmp_path):
    for name, out in (("mono", output("mono", AHEAD)), ("area", output("area", FLOOR)),
                      ("kmeans", output("area", FLOOR, regions="kmeans"))):
        (tmp_path / f"{name}.json").write_text(json.dumps(out))
    population = tmp_path / "population.json"
    population.write_text(json.dumps({"sides": [{"key": k, "side": "end", "moved_camera": k == "s1/a/c1"}
                                                for k in KEYS]}))
    v.main(["--answer", str(tmp_path / "mono.json"), "--floor", str(tmp_path / "area.json"),
            "--population", str(population), "--out", str(tmp_path / "verdict.json")])
    written = json.loads((tmp_path / "verdict.json").read_text())
    assert written["verdict"] == "A" and written["strata"]["moved"]["n_units"] == 1
    with pytest.raises(SystemExit, match="refused: the outputs were scored on other regions"):
        v.main(["--answer", str(tmp_path / "mono.json"), "--floor", str(tmp_path / "kmeans.json"),
                "--population", str(population)])
