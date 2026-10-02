"""The invariants of the statistics that give a claim a star, and the repository's one definition of it.

The bootstrap tests are the workbench's: each one pins a property that
was once broken there. The two scans at the end hold `AGENTS.md`'s rule
over every tracked Python file: a star is decided by `paired_stats.verdict`
and nothing else, never by a p-value, and never by a truth value that has
dropped the direction.
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
from evalkit.tools.scores import PILOT_EVAL_CODE_SHA

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


def test_verdict_takes_no_p_value():
    assert list(inspect.signature(PS.verdict).parameters) == ["ci95_video"]


def test_is_star_matches_verdict():
    assert PS.is_star([0.01, 0.05]) and not PS.is_star([-0.05, -0.01]) and not PS.is_star([-0.01, 0.05])


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


# What the workbench's paired_stats wrote on exactly these two JSONs
# (`base` at shift 0 with seed 5, `cond` at shift 0.03 with seed 6), so that this port
# is held to its bytes rather than only to its own tests.
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
        assert res["metrics"][key] == want, key


def test_the_pair_refuses_two_evaluators():
    other = pilot_scores("cond", 0.03, 6)
    other["eval_code_sha"] = "b" * 64
    with pytest.raises(ValueError, match="different evaluators"):
        PS.compare_pair(pilot_scores("base", 0.0, 5), other)


def test_dropping_a_video_shrinks_the_population_after_the_check():
    res = PS.compare_pair(pilot_scores("base", 0.0, 5), pilot_scores("cond", 0.03, 6), drop=[VIDEOS[0]])
    assert (res["n_clips"], res["n_videos"], res["dropped_videos"]) == (12, 6, [VIDEOS[0]])
    with pytest.raises(ValueError, match="left no clip"):
        PS.compare_pair(pilot_scores("base", 0.0, 5), pilot_scores("cond", 0.03, 6), drop=VIDEOS)


def test_a_key_defined_on_no_common_clip_is_left_out_not_zeroed():
    a, b = pilot_scores("base", 0.0, 5), pilot_scores("cond", 0.03, 6)
    for r in b["per_clip"]:
        r["SQ"] = None
    assert "SQ" not in PS.compare_pair(a, b)["metrics"]


def test_the_command_writes_the_pair_json(tmp_path):
    for tag, shift, seed in (("base", 0.0, 5), ("cond", 0.03, 6)):
        (tmp_path / f"{tag}.json").write_text(json.dumps(pilot_scores(tag, shift, seed)))
    out = tmp_path / "paired.json"
    subprocess.run([sys.executable, "-m", "evalkit.tools.paired_stats", "--eval-dir", str(tmp_path),
                    "--pairs", "base:cond", "--out", str(out)], check=True, capture_output=True, cwd=REPO)
    j = json.loads(out.read_text())
    assert j["seed_scheme"] == PS.SEED_SCHEME and j["n_boot"] == PS.N_BOOT
    assert j["pairs"]["base:cond"]["metrics"]["inst_F1_50"]["n_clips"] == 14


# ---------------------------------------------------------------- the rule, over the repository


def _tracked_python_files():
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z", "*.py"], check=True, capture_output=True).stdout
    for rel in out.decode().split("\0"):
        if rel and rel != "evalkit/tools/paired_stats.py":
            yield rel, (REPO / rel).read_text(encoding="utf-8")


def _alpha_aliases(tree) -> set[str]:
    """Names bound to 0.05 at module level, so that `ALPHA = 0.05` hides nothing."""
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and node.value.value == 0.05:
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return names


def _tests_significance(node, alpha: set[str]) -> bool:
    def is_alpha(x):
        return (isinstance(x, ast.Constant) and x.value == 0.05) or (isinstance(x, ast.Name) and x.id in alpha)

    for cmp_ in (n for n in ast.walk(node) if isinstance(n, ast.Compare)):
        parts = [cmp_.left, *cmp_.comparators]
        for i, op in enumerate(cmp_.ops):
            if is_alpha(parts[i + 1]) and isinstance(op, (ast.Lt, ast.LtE)):
                return True
            if is_alpha(parts[i]) and isinstance(op, (ast.Gt, ast.GtE)):
                return True
    return False


MARK_NAMES = {"star", "mark", "sig", "beats", "verdict", "sep", "clear", "p_clear"}


def test_no_statement_decides_a_mark_from_a_p_value():
    # Printing p beside a mark is allowed; deciding the mark by it is not.
    # Statements are read as syntax trees, so an assignment split over two
    # lines, an `if`, an f-string or a constant standing in for 0.05 are
    # all seen.
    bad = []
    for rel, src in _tracked_python_files():
        tree = ast.parse(src)
        alpha = _alpha_aliases(tree)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return, ast.Expr)):
                test = node
            elif isinstance(node, (ast.If, ast.IfExp, ast.While)):
                test = node.test
            else:
                continue
            if not _tests_significance(test, alpha):
                continue
            seg = ast.get_source_segment(src, node) or ""
            used = {t.id for t in ast.walk(node) if isinstance(t, ast.Name)}
            if "★" in seg or "✗" in seg or (used & MARK_NAMES):
                bad.append(f"{rel}:{node.lineno}  {seg.splitlines()[0].strip()}")
    assert not bad, "a mark is decided by p; the rule is paired_stats.VERDICT_RULE:\n  " + "\n  ".join(bad)


def test_no_mark_is_bound_to_a_truth_value_that_dropped_the_direction():
    # `star = bool(verdict(ci))` is True for a cross too, and four readers
    # once drew a loss as a star through it. Counting with bool() is fine;
    # binding it to a mark's name is not.
    bad = []
    for rel, src in _tracked_python_files():
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
                    bad.append(f"{rel}:{node.lineno}  {seg.splitlines()[0].strip()}")
    assert not bad, "bool(verdict(...)) bound to a mark's name drops the direction:\n  " + "\n  ".join(bad)


def _emits_a_mark(tree) -> bool:
    docstrings = set()
    for n in ast.walk(tree):
        body = getattr(n, "body", None)
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and body:
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docstrings.add(id(first.value))
    return any(isinstance(n, ast.Constant) and isinstance(n.value, str) and ("★" in n.value or "✗" in n.value)
               and id(n) not in docstrings for n in ast.walk(tree))


def _borrows_verdict(tree) -> bool:
    aliases = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module == "evalkit.tools.paired_stats":
            if any(a.name in ("verdict", "is_star") for a in n.names):
                return True
            aliases |= {a.asname or a.name for a in n.names if a.name == "paired_stats"}
        if isinstance(n, ast.Import):
            aliases |= {a.asname or a.name for a in n.names if a.name == "evalkit.tools.paired_stats"}
    return any(isinstance(n, ast.Attribute) and n.attr in ("verdict", "is_star")
               and isinstance(n.value, ast.Name) and n.value.id in aliases for n in ast.walk(tree))


def test_every_file_that_prints_a_mark_borrows_the_verdict():
    bad = [rel for rel, src in _tracked_python_files()
           if _emits_a_mark(tree := ast.parse(src)) and not _borrows_verdict(tree) and not rel.startswith("tests/")]
    assert not bad, "a mark is printed without borrowing paired_stats.verdict:\n  " + "\n  ".join(bad)
