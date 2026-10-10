"""The invariants of the statistics that give a claim a star, and the repository's one definition of it.

The bootstrap tests are the workbench's: each one pins a property that
was once broken there. The three scans at the end hold `AGENTS.md`'s rule
over every tracked Python file: a star is decided by `paired_stats.verdict`
and nothing else, never by a p-value, and never by a truth value that has
dropped the direction. Each scan is shown to catch a planted fault, and to
let the allowed forms through.
"""
import ast
import inspect
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from evalkit.tools import paired_stats as PS
from evalkit.tools.scores import CLIP_METRICS, FRAME_METRICS, PILOT_EVAL_CODE_SHA, PILOT_SIGNS, metric_key

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


# ---------------------------------------------------------------- the verdict


def test_verdict_is_the_video_interval_alone():
    assert PS.verdict([0.01, 0.05]) == "★"
    assert PS.verdict([-0.05, -0.01]) == "✗"
    assert PS.verdict([-0.01, 0.05]) == ""
    assert PS.verdict([0.0, 0.05]) == "" and PS.verdict([-0.05, 0.0]) == ""
    assert PS.verdict(None) == "" and PS.verdict([None, 0.05]) == ""


def test_verdict_takes_the_interval_and_the_direction_and_no_p_value():
    assert list(inspect.signature(PS.verdict).parameters) == ["ci95_video", "sign"]


def test_verdict_reads_the_interval_in_the_metric_s_direction():
    # `VI_split` falls when a method improves; `time_IoU` is a reference
    # value and is never marked, whichever way it moved.
    assert PS.verdict([0.01, 0.05], sign=-1) == "✗" and PS.verdict([-0.05, -0.01], sign=-1) == "★"
    assert PS.verdict([-0.01, 0.05], sign=-1) == ""
    assert PS.verdict([0.01, 0.05], sign=0) == "" and PS.verdict([-0.05, -0.01], sign=0) == ""
    with pytest.raises(ValueError, match="sign is"):
        PS.verdict([0.01, 0.05], sign=2)
    with pytest.raises(ValueError, match="sign is"):
        PS.verdict([0.01, 0.05], sign=None)


def test_is_star_matches_verdict():
    assert PS.is_star([0.01, 0.05]) and not PS.is_star([-0.05, -0.01]) and not PS.is_star([-0.01, 0.05])
    assert PS.is_star([-0.05, -0.01], sign=-1) and not PS.is_star([0.01, 0.05], sign=-1)


def test_every_pilot_key_has_a_direction_from_the_one_table():
    assert set(PS.PILOT_KEYS) <= set(PILOT_SIGNS)
    assert PILOT_SIGNS["underseg_error"] == -1
    assert all(PILOT_SIGNS[k] == +1 for k in PS.PILOT_KEYS if k != "underseg_error")
    assert all(PS.sign_of_key(pilot_scores("a", 0.0, 1), k) == PILOT_SIGNS[k] for k in PS.PILOT_KEYS)


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
                input_shas={c: {"gt_masks": "g" * 64, "depth": "d" * 64, "predictions": f"p{c}"} for c in clips},
                versions={"python": "3.12.0", "numpy": "2.0.0"}, propagation="both_ways_from_centre", per_clip=rows)


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
        assert r["ci95_video"] is None and PS.verdict(r["ci95_video"], r["sign"]) == "", key
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
    with pytest.raises(KeyError):
        PS.compare_pair(a, b, keys=["inst_F1_50_labelled"])
    ea, eb = evaluator_scores("base", 0.0, 5), evaluator_scores("cond", 0.03, 6)
    for spelt in ("F1_50/geometrc", "F1_50"):
        with pytest.raises(KeyError, match="not keys the evaluator writes"):
            PS.compare_pair(ea, eb, keys=[spelt])


def test_a_value_that_rounds_to_zero_keeps_two_digits():
    # An end written as -0.0 would lose its ✗.
    assert PS.round_keeping_sign(-2.4e-05) == -2.4e-05 and PS.round_keeping_sign(0.0) == 0.0
    assert PS.round_keeping_sign(0.123449) == 0.1234
    assert PS.format_signed(-2.4e-05) == "-2.40e-05" and PS.format_signed(0.0) == "+0.0000"
    assert PS.format_signed(0.05) == "+0.0500"


@pytest.mark.parametrize("ci, written, mark", [((2.03e-05, 0.01509), [2e-05, 0.0151], "★"),
                                               ((-0.01509, -2.03e-05), [-0.0151, -2e-05], "✗")])
def test_an_interval_end_near_zero_keeps_its_star_or_cross(monkeypatch, ci, written, mark):
    # Rounded to four places alone, either end would be written as 0.0, which gives neither ★ nor ✗.
    monkeypatch.setattr(PS, "boot_ci", lambda d, groups, **kw: ci)
    a, b = pilot_scores("base", 0.0, 5), pilot_scores("cond", 0.03, 6)
    r = PS.compare_pair(a, b, keys=["inst_F1_50"])["metrics"]["inst_F1_50"]
    assert r["ci95_video"] == written and PS.verdict(r["ci95_video"], r["sign"]) == mark


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


def test_the_command_writes_no_rule_on_pilot_jsons(tmp_path):
    for tag, shift, seed in (("base", 0.0, 5), ("cond", 0.03, 6)):
        (tmp_path / f"{tag}.json").write_text(json.dumps(pilot_scores(tag, shift, seed)))
    out = tmp_path / "paired.json"
    assert _run(tmp_path, "--pairs", "base:cond", "--out", str(out)).returncode == 0
    j = json.loads(out.read_text())
    assert list(j) == ["n_boot", "seed", "seed_scheme", "pairs"] and "propagation" not in j["pairs"]["base:cond"]


@pytest.mark.parametrize("pairs, said", [
    ("base", "--pairs takes <base>:<cond>"),
    ("base:cond:other", "--pairs takes <base>:<cond>"),
    ("base:nothere", "nothere.json"),
])
def test_the_command_refuses_a_bad_pair_with_a_message_not_a_traceback(tmp_path, pairs, said):
    (tmp_path / "base.json").write_text(json.dumps(pilot_scores("base", 0.0, 5)))
    run = _run(tmp_path, "--pairs", pairs)
    assert run.returncode != 0 and said in run.stderr and "Traceback" not in run.stderr


# ---------------------------------------------------------------- the rule, over the repository
#
# The scans read syntax trees, not text, so a statement split over lines,
# an `if`, an f-string or a name standing in for 0.05 are one statement
# each. What they do not see is a decision spread over two statements
# (`sig = p < 0.05`, then `mark = "★" if sig else ""`); they are a
# tripwire, not a proof.


def _tracked_python_files():
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z", "*.py"], check=True, capture_output=True).stdout
    for rel in out.decode().split("\0"):
        if rel and rel != "evalkit/tools/paired_stats.py":
            yield rel, (REPO / rel).read_text(encoding="utf-8")


def _package_of(rel: str) -> str:
    """The package a tracked file belongs to, for its relative imports."""
    return ".".join(Path(rel).parts[:-1])


def _alpha_names(tree) -> set[str]:
    """Names bound to 0.05 anywhere in the module, in a tuple assignment too, so that `ALPHA = 0.05` hides nothing."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            pairs = []
            for t in node.targets:
                if isinstance(t, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    pairs += zip(t.elts, node.value.elts)
                else:
                    pairs.append((t, node.value))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            pairs = [(node.target, node.value)]
        else:
            continue
        names |= {t.id for t, v in pairs
                  if isinstance(t, ast.Name) and isinstance(v, ast.Constant) and v.value == 0.05}
    return names


def _compares_with_alpha(node, alpha: set[str]) -> bool:
    """Whether a comparison in `node` has 0.05, or a name for it, on either side, whichever way it reads."""
    def is_alpha(x):
        return (isinstance(x, ast.Constant) and x.value == 0.05) or (isinstance(x, ast.Name) and x.id in alpha)

    return any(any(is_alpha(part) for part in (cmp_.left, *cmp_.comparators))
               for cmp_ in ast.walk(node) if isinstance(cmp_, ast.Compare))


MARK_NAMES = {"star", "mark", "sig", "beats", "verdict", "sep", "clear", "p_clear"}


def _names_in(node) -> set[str]:
    """The names a statement touches: its variables, and the string keys it subscripts (`row["star"]`)."""
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    names |= {n.slice.value for n in ast.walk(node)
              if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str)}
    return names


def marks_decided_by_p(src: str) -> list[str]:
    """`<line>  <statement>` for every statement that decides a mark from a p-value.

    Printing p beside a mark is allowed; comparing p with 0.05 in a
    statement that also touches a mark, by name or by the character, is
    not, whichever way the comparison reads.
    """
    tree = ast.parse(src)
    alpha = _alpha_names(tree)
    bad = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return, ast.Expr)):
            test = node
        elif isinstance(node, (ast.If, ast.IfExp, ast.While)):
            test = node.test
        else:
            continue
        if not _compares_with_alpha(test, alpha):
            continue
        seg = ast.get_source_segment(src, node) or ""
        if "★" in seg or "✗" in seg or (_names_in(node) & MARK_NAMES):
            bad.setdefault(node.lineno, f"{node.lineno}  {seg.splitlines()[0].strip()}")
    return [bad[k] for k in sorted(bad)]


def marks_bound_to_a_truth_value(src: str) -> list[str]:
    """`<line>  <statement>` for every assignment that binds `bool(verdict(...))` to a mark's name.

    `star = bool(verdict(ci))` is True for a cross too, and four readers
    once drew a loss as a star through it. Counting with bool() is fine;
    binding it to a mark's name is not.
    """
    bad = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Assign):
            continue
        targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
        for t in node.targets:
            if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str):
                targets.add(t.slice.value)
        if not (targets & MARK_NAMES):
            continue
        for call in (n for n in ast.walk(node.value) if isinstance(n, ast.Call)):
            if isinstance(call.func, ast.Name) and call.func.id == "bool" and "verdict" in ast.dump(call).lower():
                seg = ast.get_source_segment(src, node) or ""
                bad.append(f"{node.lineno}  {seg.splitlines()[0].strip()}")
    return bad


# Every glyph that reads as "better" or "worse" in a table: the verdict's own
# two, and the circle and cross that read as a verdict on a sign alone.
MARKS = ("★", "✗", "○", "×")


def _emits_a_mark(tree) -> bool:
    docstrings = set()
    for n in ast.walk(tree):
        body = getattr(n, "body", None)
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and body:
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docstrings.add(id(first.value))
    return any(isinstance(n, ast.Constant) and isinstance(n.value, str) and any(m in n.value for m in MARKS)
               and id(n) not in docstrings for n in ast.walk(tree))


def _dotted(node) -> str | None:
    """`a.b.c` for an attribute chain that starts at a name; None for any other expression."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _module_of(node: ast.ImportFrom, package: str) -> str:
    """The module a `from` import names, a relative one resolved against `package`."""
    if not node.level:
        return node.module
    base = package.split(".")
    base = base[: len(base) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


VERDICTS = {"evalkit.tools.paired_stats.verdict", "evalkit.tools.paired_stats.is_star"}


def _borrows_verdict(tree, package: str) -> bool:
    """Whether the module calls `paired_stats.verdict` or `is_star`, through whatever import form binds it.

    Every name an import binds is resolved to the dotted thing it stands
    for, and every name or attribute chain the module uses is read through
    that table; so `from evalkit.tools import paired_stats as PS` and
    `PS.verdict(...)` count, as `import evalkit.tools.paired_stats` and the
    full chain do.
    """
    bound = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                bound[a.asname] = a.name
                if a.asname is None:
                    bound[a.name.split(".")[0]] = a.name.split(".")[0]
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                bound[a.asname or a.name] = f"{_module_of(n, package)}.{a.name}"
    bound.pop(None, None)
    for n in ast.walk(tree):
        dotted = _dotted(n) if isinstance(n, (ast.Attribute, ast.Name)) else None
        if dotted is None:
            continue
        head, _, rest = dotted.partition(".")
        if bound.get(head, head) + (f".{rest}" if rest else "") in VERDICTS:
            return True
    return False


def prints_a_mark_without_the_verdict(src: str, package: str = "") -> bool:
    """Whether a module writes a star or a cross outside a docstring without borrowing `verdict`."""
    tree = ast.parse(src)
    return _emits_a_mark(tree) and not _borrows_verdict(tree, package)


def test_no_statement_decides_a_mark_from_a_p_value():
    bad = [f"{rel}:{hit}" for rel, src in _tracked_python_files() for hit in marks_decided_by_p(src)]
    assert not bad, "a mark is decided by p; the rule is paired_stats.VERDICT_RULE:\n  " + "\n  ".join(bad)


def test_no_mark_is_bound_to_a_truth_value_that_dropped_the_direction():
    bad = [f"{rel}:{hit}" for rel, src in _tracked_python_files() for hit in marks_bound_to_a_truth_value(src)]
    assert not bad, "bool(verdict(...)) bound to a mark's name drops the direction:\n  " + "\n  ".join(bad)


def test_every_file_that_prints_a_mark_borrows_the_verdict():
    # The tests are left out: they spell the marks to check them.
    bad = [rel for rel, src in _tracked_python_files()
           if not rel.startswith("tests/") and prints_a_mark_without_the_verdict(src, _package_of(rel))]
    assert not bad, "a mark is printed without borrowing paired_stats.verdict:\n  " + "\n  ".join(bad)


# The planted faults. Each scan is shown to catch the forms it is for, the
# negated and the indirect ones included, and to let the allowed ones pass.

DECIDED_BY_P = [
    'mark = "★" if p < 0.05 else ""',
    'mark = "" if p > 0.05 else "★"',
    'mark = "★" if 0.05 > p else ""',
    'mark = "" if 0.05 < p else "★"',
    'if p <= 0.05:\n    mark = "★"',
    'while p < 0.05:\n    star = True',
    'row["star"] = p < 0.05',
    'print(f"{k} {\'★\' if p < 0.05 else \'\'}")',
    'ALPHA, BETA = 0.05, 0.1\nmark = "★" if p < ALPHA else ""',
    'ALPHA: float = 0.05\nmark = "★" if p < ALPHA else ""',
    'def f(p):\n    alpha = 0.05\n    return "★" if p < alpha else ""',
    'mark = ("★"\n        if p\n        < 0.05 else "")',
]

P_BESIDE_A_MARK = [
    'print(f"{mark} p={p:.4f}")',
    'mark = verdict(ci)',
    'if p < 0.05:\n    print("a small p, printed, not decisive")',
    'alpha = 0.05\nprint(f"alpha={alpha}")',
    'sig = wilcoxon_p_video',
    'row["wilcoxon_p"] = round(p, 5)',
]


@pytest.mark.parametrize("src", DECIDED_BY_P)
def test_the_p_scan_sees_a_planted_decision(src):
    assert marks_decided_by_p(src), src


@pytest.mark.parametrize("src", P_BESIDE_A_MARK)
def test_the_p_scan_lets_p_be_printed_beside_a_mark(src):
    assert not marks_decided_by_p(src), src


BOUND_TO_A_TRUTH_VALUE = [
    'star = bool(verdict(ci))',
    'row["star"] = bool(PS.verdict(ci))',
    'sig = bool(paired_stats.verdict(r["ci95_video"], r["sign"]))',
]

KEEPS_THE_DIRECTION = [
    'n_marked = sum(bool(verdict(c)) for c in cis)',
    'star = verdict(ci) == "★"',
    'star = is_star(ci)',
    'mark = verdict(ci)',
]


@pytest.mark.parametrize("src", BOUND_TO_A_TRUTH_VALUE)
def test_the_bool_scan_sees_a_planted_binding(src):
    assert marks_bound_to_a_truth_value(src), src


@pytest.mark.parametrize("src", KEEPS_THE_DIRECTION)
def test_the_bool_scan_lets_counting_and_comparing_pass(src):
    assert not marks_bound_to_a_truth_value(src), src


PRINTS_WITHOUT_BORROWING = [
    'print("★" if lo > 0 else "")',
    'mark = "○" if d * sign > 0 else "×"',
    'from evalkit.tools.paired_stats import boot_ci\nmark = "★" if boot_ci(d)[0] > 0 else ""',
    'import evalkit.tools.paired_stats\nmark = "★"',
    'from evalkit.tools import paired_stats as PS\nmark = "★"\nvideo = PS.video_of(c)',
]

LEGEND = 'print("  ★ better   ✗ worse")\n'

BORROWS = [
    ('from evalkit.tools.paired_stats import verdict\n' + LEGEND + 'print(verdict(ci, sign))', ""),
    ('from evalkit.tools.paired_stats import is_star\n' + LEGEND + 'print(is_star(ci))', ""),
    ('from evalkit.tools import paired_stats as PS\n' + LEGEND + 'print(PS.verdict(ci, sign))', ""),
    ('from evalkit.tools import paired_stats\n' + LEGEND + 'print(paired_stats.verdict(ci))', ""),
    ('import evalkit.tools.paired_stats as P\n' + LEGEND + 'print(P.is_star(ci))', ""),
    ('import evalkit.tools.paired_stats\n' + LEGEND + 'print(evalkit.tools.paired_stats.verdict(ci))', ""),
    ('from evalkit import tools\n' + LEGEND + 'print(tools.paired_stats.verdict(ci))', ""),
    ('from .paired_stats import verdict\n' + LEGEND + 'print(verdict(ci))', "evalkit.tools"),
    ('from . import paired_stats\n' + LEGEND + 'print(paired_stats.verdict(ci))', "evalkit.tools"),
    ('from ..tools import paired_stats\n' + LEGEND + 'print(paired_stats.verdict(ci))', "evalkit.other"),
]


@pytest.mark.parametrize("src", PRINTS_WITHOUT_BORROWING)
def test_the_mark_scan_sees_a_planted_mark_without_the_verdict(src):
    assert prints_a_mark_without_the_verdict(src), src


@pytest.mark.parametrize("src, package", BORROWS)
def test_the_mark_scan_knows_every_import_form(src, package):
    assert not prints_a_mark_without_the_verdict(src, package), src


def test_the_mark_scan_ignores_a_mark_in_a_docstring():
    assert not prints_a_mark_without_the_verdict('"""A ★ is a star and a ✗ a cross."""\nx = 1')
    assert not prints_a_mark_without_the_verdict('def f():\n    """★"""\n    return 1')
