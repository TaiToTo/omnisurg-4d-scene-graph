"""Test that the claims table's cells are `paired_stats`' and that the table refuses what would mislead.

Each check is shown refusing a planted fault:

- a score JSON without the key, or with a key that has no direction;
- a stage the propagation rules contradict, and two rules in one table;
- a column of two datasets, two columns of one, and a JSON with no dataset;
- a pair that is not comparable;
- a directory that is missing or holds none of the table's conditions.
"""
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from evalkit.tools import claims_table as CT
from evalkit.tools import paired_stats as PS
from evalkit.tools.scores import PILOT_EVAL_CODE_SHA, metric_key

pytest.importorskip("scipy", reason="the Wilcoxon test needs the `tools` extra")

REPO = Path(__file__).resolve().parent.parent
PILOT_KEY = "inst_F1_50_labeled"
KEY = metric_key("F1_50", "all")
VIDEOS = {"atlas120k": ("adrenalectomy__16GPCUPkXYQ", "appendectomy__41RKDh3INiU", "lar___7H7G-4sevQ",
                        "rectopexy__IJfgUvxoTkM"),
          "cholecseg8k": ("VID01", "VID12", "VID25", "VID26")}
PILOT_DATASET = {"atlas120k": "atlas", "cholecseg8k": "cholec"}


def clips_of(column: str) -> list[str]:
    if column == "atlas120k":
        return [f"{v}__gt_{i:04d}" for v in VIDEOS[column] for i in (1, 2)]
    return [f"{v}_s15_{i}_crop" for v in VIDEOS[column] for i in (100, 200)]


def values(column: str, seed: int, shift: float) -> dict[str, float]:
    """A value per clip; the same `seed` gives the same values, so `shift` alone separates two conditions."""
    rnd = random.Random(seed)
    return {c: round(rnd.uniform(0.3, 0.6) + shift, 4) for c in clips_of(column)}


def pilot_json(column: str, tag: str, vals: dict[str, float]) -> dict:
    rows = [{"clip": c, "extra_ignore": [], PILOT_KEY: v, "inst_F1_50": v} for c, v in vals.items()]
    return dict(tag=tag, eval_code_sha=PILOT_EVAL_CODE_SHA, eval_code_tag="t", eval_version=2,
                dataset=PILOT_DATASET[column], tissue_ignore=[1], clips=list(vals), per_clip=rows)


def evaluator_json(column: str, tag: str, vals: dict[str, float], rule: str) -> dict:
    rows = [{"clip": c, KEY: v, metric_key("VI_split", "all"): v} for c, v in vals.items()]
    return dict(tag=tag, eval_code_sha="a" * 64, dataset=column, pilot=False, class_set="original",
                views=["all"], clips=list(vals),
                input_shas={c: {"gt_masks": "g" * 64, "depth": "d" * 64, "predictions": f"p{tag}{c}"} for c in vals},
                versions={"python": "3.12.0"}, propagation=rule, per_clip=rows)


def rule_of(tag: str, tracked: str = "both_ways_from_centre") -> str:
    """The rule each claim's stage asks of a tag: per frame for the per-frame conditions, else `tracked`."""
    return "per_frame" if "perframe" in tag else tracked


def write_table(tmp_path: Path, shifts: dict | None = None, layout: str = "pilot", skip=(),
                tracked: dict | None = None) -> dict[str, str]:
    """Write every condition of every column; a condition's values are its base's plus its `shifts` entry.

    Args:
        shifts: `(column, tag)` to the amount added; 0 for any other.
        layout: `pilot` or `evaluator`.
        skip: `(column, tag)` pairs to leave without a score JSON.
        tracked: Each column's rule for its propagated conditions.
    """
    shifts = shifts or {}
    dirs = {}
    for column in CT.DATASETS:
        d = tmp_path / column
        d.mkdir()
        tags = sorted({t for c in CT.CLAIMS for t in c.pairs[column]})
        for tag in tags:
            if (column, tag) in skip:
                continue
            vals = values(column, 1, shifts.get((column, tag), 0.0))
            if layout == "pilot":
                doc = pilot_json(column, tag, vals)
            else:
                rule = rule_of(tag, (tracked or {}).get(column, "both_ways_from_centre"))
                doc = evaluator_json(column, tag, vals, rule)
            (d / f"{tag}.json").write_text(json.dumps(doc), encoding="utf-8")
        dirs[column] = str(d)
    return dirs


def rewrite(dirs: dict, column: str, tag: str, change) -> None:
    path = Path(dirs[column]) / f"{tag}.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    change(doc)
    path.write_text(json.dumps(doc), encoding="utf-8")


MERGE = CT.CLAIMS[0]


# ---------------------------------------------------------------- the cells


def test_a_cell_is_the_pair_statistics_of_paired_stats_on_the_key(tmp_path):
    dirs = write_table(tmp_path, {("atlas120k", "t5_k10"): 0.05})
    rows, rule = CT.build(dirs, PILOT_KEY)
    assert rule is None
    claim, cells = rows[0]
    assert claim is MERGE
    base, cond = (PS.load_json(t, dirs["atlas120k"]) for t in MERGE.pairs["atlas120k"])
    assert cells[0] == PS.compare_pair(base, cond, keys=[PILOT_KEY])
    r = cells[0]["metrics"][PILOT_KEY]
    assert CT.format_cell(cells[0], PILOT_KEY) == (
        f"{r['delta_mean']:+.4f} [{r['ci95_video'][0]:+.3f},{r['ci95_video'][1]:+.3f}] "
        f"p={r['wilcoxon_p']}★ (8 clips, 4 videos)")


def test_a_row_is_starred_in_both_only_when_both_datasets_give_a_star(tmp_path):
    dirs = write_table(tmp_path, {("atlas120k", "t5_k10"): 0.05, ("cholecseg8k", "t5_k10"): 0.05,
                                  ("atlas120k", "rgb_center18"): 0.05, ("cholecseg8k", "ch_rgb_center"): -0.05})
    rows, _ = CT.build(dirs, PILOT_KEY)
    both = {claim.name: all(CT.star_in(p, PILOT_KEY) for p in cells) for claim, cells in rows}
    # Every claim whose compared condition is `t5_k10` gets a star in both;
    # the rgb tracking claim gets a star and a cross, and the edge one nothing.
    assert [both[c.name] for c in CT.CLAIMS] == [True, False, False, True, True]
    assert "→ 3 / 5 rows get a star in both datasets" in CT.render(rows, None, PILOT_KEY, markdown=False)


def test_a_key_where_less_is_better_is_read_in_its_direction(tmp_path):
    dirs = write_table(tmp_path, {("atlas120k", "t5_k10"): -0.05, ("cholecseg8k", "t5_k10"): -0.05},
                       layout="evaluator")
    rows, _ = CT.build(dirs, metric_key("VI_split", "all"))
    assert all(CT.star_in(p, metric_key("VI_split", "all")) for p in rows[0][1])
    rows, _ = CT.build(dirs, KEY)
    assert "✗" in CT.format_cell(rows[0][1][0], KEY)


def test_a_condition_without_a_score_json_is_not_measured_and_its_row_gets_no_star(tmp_path):
    dirs = write_table(tmp_path, {("atlas120k", "t5_k10"): 0.05, ("cholecseg8k", "t5_k10"): 0.05},
                       skip={("cholecseg8k", "t5_floor")})
    rows, _ = CT.build(dirs, PILOT_KEY)
    assert rows[0][1][1] is None and CT.format_cell(None, PILOT_KEY) == "not measured"
    assert not all(CT.star_in(p, PILOT_KEY) for p in rows[0][1])


def test_a_score_json_without_the_key_is_refused_not_shown_as_not_measured(tmp_path):
    dirs = write_table(tmp_path)
    rewrite(dirs, "cholecseg8k", "t5_k10", lambda d: [r.pop(PILOT_KEY) for r in d["per_clip"]])
    with pytest.raises(ValueError, match=r"Merging fixes over-splitting.*\(cholecseg8k\).*t5_k10.*holds no"):
        CT.build(dirs, PILOT_KEY)


def test_a_key_defined_on_no_common_clip_is_said_so(tmp_path):
    dirs = write_table(tmp_path)
    rewrite(dirs, "atlas120k", "t5_k10", lambda d: [r.update({PILOT_KEY: None}) for r in d["per_clip"]])
    rows, _ = CT.build(dirs, PILOT_KEY)
    assert CT.format_cell(rows[0][1][0], PILOT_KEY) == "not defined on any common clip"


def test_a_shrunken_population_is_shown_against_the_pair_s(tmp_path):
    dirs = write_table(tmp_path)
    rewrite(dirs, "atlas120k", "t5_k10", lambda d: d["per_clip"][0].update({PILOT_KEY: None}))
    rows, _ = CT.build(dirs, PILOT_KEY)
    assert CT.format_cell(rows[0][1][0], PILOT_KEY).endswith("(7/8 clips, 4/4 videos)")


def test_a_key_with_no_direction_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not a key with a direction"):
        CT.build(write_table(tmp_path), "no_such_key")


# ---------------------------------------------------------------- the stage and the rule


def test_the_stages_are_checked_against_the_rules_the_evaluator_records(tmp_path):
    rows, rule = CT.build(write_table(tmp_path, layout="evaluator"), KEY)
    assert rule == "both_ways_from_centre"
    assert "each stage is checked" in CT.render(rows, rule, KEY, markdown=False)


@pytest.mark.parametrize("column, tag, said", [
    ("atlas120k", "t5_floor", r"Merging fixes.*\(atlas120k\).*measured 'propagated'.*t5_floor.*'per_frame'"),
    ("cholecseg8k", "ch_rgb_center", r"\(rgb, over-split\) \(cholecseg8k\).*ch_rgb_center the rule 'per_frame'"),
])
def test_a_claim_on_propagated_regions_whose_condition_is_per_frame_is_refused(tmp_path, column, tag, said):
    dirs = write_table(tmp_path, layout="evaluator")
    rewrite(dirs, column, tag, lambda d: d.update(propagation="per_frame"))
    with pytest.raises(ValueError, match=said):
        CT.build(dirs, KEY)


def test_a_per_frame_base_that_was_propagated_is_refused(tmp_path):
    dirs = write_table(tmp_path, layout="evaluator")
    rewrite(dirs, "cholecseg8k", "ch_edge_perframe", lambda d: d.update(propagation="both_ways_from_centre"))
    with pytest.raises(ValueError, match=r"normal_edge.*\(cholecseg8k\).*per frame → propagated"):
        CT.build(dirs, KEY)


def test_a_table_of_two_propagation_rules_is_refused(tmp_path):
    dirs = write_table(tmp_path, layout="evaluator", tracked={"cholecseg8k": "forward_from_first"})
    with pytest.raises(ValueError, match="different rules"):
        CT.build(dirs, KEY)


def test_pilot_jsons_leave_the_stages_declared_and_the_table_says_so(tmp_path):
    rows, rule = CT.build(write_table(tmp_path), PILOT_KEY)
    assert rule is None
    assert "each stage is as declared" in CT.render(rows, rule, PILOT_KEY, markdown=True)


# ---------------------------------------------------------------- the columns


def test_a_column_that_holds_two_datasets_is_refused(tmp_path):
    dirs = write_table(tmp_path)
    rewrite(dirs, "atlas120k", "rgb_center18", lambda d: d.update(dataset="cholec"))
    with pytest.raises(ValueError, match="atlas120k: the score JSONs record more than one dataset"):
        CT.build(dirs, PILOT_KEY)


def test_two_columns_of_one_dataset_are_refused(tmp_path):
    dirs = write_table(tmp_path)
    for tag in {t for c in CT.CLAIMS for t in c.pairs["cholecseg8k"]}:
        rewrite(dirs, "cholecseg8k", tag, lambda d: d.update(dataset="atlas"))
    with pytest.raises(ValueError, match="both hold scores of 'atlas'"):
        CT.build(dirs, PILOT_KEY)


def test_a_json_that_records_no_dataset_is_refused(tmp_path):
    dirs = write_table(tmp_path)
    rewrite(dirs, "atlas120k", "t5_floor", lambda d: d.pop("dataset"))
    with pytest.raises(ValueError, match="records no dataset"):
        CT.build(dirs, PILOT_KEY)


def test_a_pair_that_is_not_comparable_is_refused_with_its_claim_named(tmp_path):
    dirs = write_table(tmp_path)
    rewrite(dirs, "cholecseg8k", "t5_k10", lambda d: d.update(eval_code_sha="b" * 64))
    with pytest.raises(ValueError, match=r"Merging fixes over-splitting.*\(cholecseg8k\).*different evaluators"):
        CT.build(dirs, PILOT_KEY)


def test_a_directory_that_holds_none_of_the_conditions_is_refused(tmp_path):
    dirs = write_table(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="none of the table's conditions"):
        CT.build({**dirs, "cholecseg8k": str(empty)}, PILOT_KEY)
    with pytest.raises(ValueError, match="no such directory"):
        CT.build({**dirs, "cholecseg8k": str(tmp_path / "missing")}, PILOT_KEY)


@pytest.mark.parametrize("values, said", [
    (["atlas120k=a"], "no directory for \\['cholecseg8k'\\]"),
    (["atlas120k=a", "cholecseg8k=b", "atlas120k=c"], "twice"),
    (["atlas120k=a", "lapex=b"], "'lapex'"),
    (["atlas120k"], "DATASET=DIR"),
])
def test_scores_name_each_dataset_once(values, said):
    with pytest.raises(ValueError, match=said):
        CT.parse_scores(values)


# ---------------------------------------------------------------- the claims and the command


def test_every_claim_names_a_pair_in_every_dataset_and_a_known_stage():
    for claim in CT.CLAIMS:
        assert set(claim.pairs) == set(CT.DATASETS), claim.name
        assert claim.stage in CT.STAGES, claim.name
        assert all(base != cond for base, cond in claim.pairs.values()), claim.name


def test_the_command_prints_the_markdown_table(tmp_path):
    dirs = write_table(tmp_path, {("atlas120k", "t5_k10"): 0.05, ("cholecseg8k", "t5_k10"): 0.05})
    out = subprocess.run([sys.executable, "-m", "evalkit.tools.claims_table", "--key", PILOT_KEY, "--markdown",
                          *[f"--scores={ds}={d}" for ds, d in dirs.items()]],
                         capture_output=True, text=True, cwd=REPO)
    assert out.returncode == 0, out.stderr
    assert "| claim | stage | atlas120k | cholecseg8k | both |" in out.stdout
    assert f"| {MERGE.name} | propagated | " in out.stdout and out.stdout.count("| ★★ |") == 3


def test_the_command_refuses_with_a_message_not_a_traceback(tmp_path):
    out = subprocess.run([sys.executable, "-m", "evalkit.tools.claims_table", "--key", PILOT_KEY,
                          "--scores", f"atlas120k={tmp_path}"], capture_output=True, text=True, cwd=REPO)
    assert out.returncode != 0 and "Traceback" not in out.stderr
    assert "no directory for ['cholecseg8k']" in out.stderr
