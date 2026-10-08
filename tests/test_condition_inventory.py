"""The inventory fires on every planted fault and stays quiet on a sound tree.

A sound tree of two conditions on three clips, with clip lengths that
differ (5, 9, 13 labels: a clip's length is its own), is built in a
temporary directory, and one fault at a time is planted in it: a mixed
condition, a short one, provenance without labels, labels without
provenance, scores that cannot be compared, a missing root, and the
mixes the evaluator's JSONs make possible.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from evalkit.tools import condition_inventory as CI
from evalkit.tools.scores import PILOT_EVAL_CODE_SHA

REPO = Path(__file__).resolve().parent.parent
CLIPS = ["c1", "c2", "c3"]
LABELS = {"c1": 5, "c2": 9, "c3": 13}


def provenance(sam_input: str, pps: int) -> dict:
    return dict(sam_input=sam_input, depth_source="da3", frames="all", seed_source="per_frame",
                track_base="rgb", seed_min_area=400, seed_topk=0, point_grids=None,
                seed_input=dict(points_per_side=pps, produced_by="sam", seed_edge_gain=1.0,
                                seed_smooth=False, edge_ring_masked=None, seed_sam_kwargs={}))


def pilot_score(track_dir: str, sha: str = PILOT_EVAL_CODE_SHA) -> dict:
    return dict(track_dir_name=track_dir, eval_code_sha=sha, eval_code_tag="T", eval_version=2,
                dataset="atlas", n_clips=3, n_missing=0, n_failed=0, tissue_ignore=[1],
                clips=CLIPS, per_clip=[dict(clip=c, extra_ignore=[]) for c in CLIPS])


def evaluator_score(track_dir: str, pilot: bool = False, class_set: str = "original", clips=CLIPS,
                    depth: str = "d") -> dict:
    return dict(track_dir_name=track_dir, eval_code_sha="a" * 64, dataset="atlas120k", pilot=pilot,
                class_set=class_set, views=["all", "tissue", "geometric"], n_clips=len(clips), n_missing=0,
                n_failed=0, clips=list(clips), versions={}, propagation="both_ways_from_centre",
                input_shas={c: {"gt_masks": "g", "depth": depth, "predictions": track_dir} for c in clips},
                per_clip=[dict(clip=c) for c in clips])


def plant(base: Path, spec: dict) -> None:
    """`{tag: [(clip, provenance, label count), ...]}` into `base/<clip>/<tag>/`."""
    for tag, items in spec.items():
        for clip, prov, n in items:
            d = base / clip / tag
            d.mkdir(parents=True, exist_ok=True)
            (d / CI.PROV_NAME).write_text(json.dumps(prov))
            for i in range(n):
                (d / f"label_{i:04d}.npy").write_bytes(b"")


def write_score(ev: Path, tag: str, score: dict) -> None:
    (ev / f"{tag}.json").write_text(json.dumps(score))


@pytest.fixture
def tree(tmp_path):
    """A sound tree: two conditions on three clips, whose two scores are comparable."""
    tr, ev = tmp_path / "tracks", tmp_path / "scores"
    ev.mkdir()
    plant(tr, {f"track_rgb_{t}": [(c, provenance(i, 8), LABELS[c]) for c in CLIPS]
               for t, i in (("a_rgb", "rgb"), ("a_normal", "normal"))})
    for t in ("a_rgb", "a_normal"):
        write_score(ev, t, pilot_score(f"track_rgb_{t}"))
    return tr, ev


def problems(tr, ev, **kw):
    return CI.inventory([(str(tr), str(ev), "T")], **kw)


# ---------------------------------------------------------------- the sound tree


def test_a_sound_tree_raises_no_problem(tree, capsys):
    assert problems(*tree) == []
    assert "comparable groups: 1" in capsys.readouterr().out


def test_a_sound_tree_scored_by_the_evaluator_raises_no_problem(tree, capsys):
    tr, ev = tree
    for t in ("a_rgb", "a_normal"):
        write_score(ev, t, evaluator_score(f"track_rgb_{t}"))
    assert problems(tr, ev) == []
    assert "comparable groups: 1" in capsys.readouterr().out


def test_a_group_is_what_the_comparison_tool_calls_comparable(tree, capsys):
    # The order of `tissue_ignore` and the evaluator's tag do not stop the
    # comparison tool from comparing these two, so the inventory must not
    # split them.
    tr, ev = tree
    s = pilot_score("track_rgb_a_normal")
    s["tissue_ignore"], s["eval_code_tag"] = [1], "another tag"
    s2 = pilot_score("track_rgb_a_rgb")
    s2["tissue_ignore"] = [1]
    write_score(ev, "a_normal", s)
    write_score(ev, "a_rgb", s2)
    assert problems(tr, ev) == []
    assert "comparable groups: 1" in capsys.readouterr().out


# ---------------------------------------------------------------- the planted faults


def test_a_condition_mixed_across_clips_is_reported(tree):
    tr, ev = tree
    plant(tr, {"track_rgb_a_rgb": [("c3", provenance("rgb", 24), 13)]})
    assert any("conditions are mixed" in p for p in problems(tr, ev))


def test_a_condition_on_fewer_clips_is_reported(tree):
    tr, ev = tree
    plant(tr, {"track_rgb_a_short": [("c1", provenance("depth", 8), 5)]})
    assert any("1 clips where the others have 3 (missing ['c2', 'c3'])" in p for p in problems(tr, ev))


def test_a_condition_on_other_clips_of_the_same_number_is_reported(tree):
    # Three clips, as the others have, but not the same three: a count
    # alone passed this.
    tr, ev = tree
    plant(tr, {"track_rgb_a_other": [(c, provenance("depth", 8), 5) for c in ("c1", "c2", "c4")]})
    assert any("a_other: 3 clips where the others have 3 (missing ['c3']) (extra ['c4'])" in p
               for p in problems(tr, ev))


def test_provenance_without_labels_is_reported(tree):
    tr, ev = tree
    plant(tr, {"track_rgb_a_stub": [(c, provenance("depth", 4), 0) for c in CLIPS]})
    assert any("no label" in p for p in problems(tr, ev))


def test_labels_without_provenance_are_reported_per_clip(tree):
    tr, ev = tree
    d = tr / "c2" / "track_rgb_a_rgb"
    (d / CI.PROV_NAME).unlink()
    assert any("no provenance" in p for p in problems(tr, ev))


def test_a_clip_covered_with_different_label_counts_is_reported(tree):
    tr, ev = tree
    (tr / "c2" / "track_rgb_a_rgb" / "label_0009.npy").write_bytes(b"")
    assert any("covered differently" in p for p in problems(tr, ev))


def test_a_clip_with_provenance_and_no_label_under_one_condition_is_reported(tree):
    # A run that died on one clip: the provenance was written, no label
    # followed, and the other condition covers the clip in full. The tag as
    # a whole has labels, so the "provenance but no label" check is silent.
    tr, ev = tree
    for p in (tr / "c2" / "track_rgb_a_rgb").glob("label_*.npy"):
        p.unlink()
    assert any("covered differently" in p and "c2: 0..9 labels" in p for p in problems(tr, ev))


def test_a_broken_provenance_is_counted_once_and_blamed_on_its_own_tag(tree):
    # Counted under "unreadable" and again under "no provenance", the
    # broken tag covered four clips of three, and the sound tag was the
    # one reported as having died half-way.
    tr, ev = tree
    (tr / "c2" / "track_rgb_a_rgb" / CI.PROV_NAME).write_text("{not json")
    probs = problems(tr, ev)
    assert any("a_rgb: 1 clips have no provenance" in p or "a_rgb" in p and "unreadable" in p for p in probs)
    assert not any("a_normal" in p for p in probs)
    assert not any("where the others have" in p for p in probs)


def test_a_condition_with_no_label_at_all_is_reported_once_not_per_clip(tree):
    tr, ev = tree
    plant(tr, {"track_rgb_a_stub": [(c, provenance("depth", 4), 0) for c in CLIPS]})
    probs = problems(tr, ev)
    assert any("no label" in p for p in probs)
    assert not any("covered differently" in p for p in probs)


def test_the_population_is_counted_from_the_clips_not_read_from_a_summary_field(tree):
    # The evaluator's JSONs are not required to record `n_clips`; two
    # absent counts compared equal, and two populations passed as comparable.
    tr, ev = tree
    for t, clips in (("a_rgb", CLIPS), ("a_normal", CLIPS[:2])):
        s = evaluator_score(f"track_rgb_{t}")
        del s["n_clips"]
        s["clips"], s["per_clip"], s["input_shas"] = clips, [dict(clip=c) for c in clips], {c: {} for c in clips}
        write_score(ev, t, s)
    assert any("2 groups" in p for p in problems(tr, ev))


def test_two_populations_of_the_same_size_are_two_groups(tree):
    tr, ev = tree
    write_score(ev, "a_normal", evaluator_score("track_rgb_a_normal", clips=["c1", "c2", "c4"]))
    write_score(ev, "a_rgb", evaluator_score("track_rgb_a_rgb"))
    assert any("2 groups" in p for p in problems(tr, ev))


def test_two_scores_that_read_different_depth_maps_are_two_groups(tree, capsys):
    tr, ev = tree
    write_score(ev, "a_normal", evaluator_score("track_rgb_a_normal", depth="other"))
    write_score(ev, "a_rgb", evaluator_score("track_rgb_a_rgb"))
    assert any("2 groups" in p for p in problems(tr, ev))
    assert "read a different depth" in capsys.readouterr().out


def test_two_shas_in_one_score_directory_are_two_groups_and_the_whole_reason_is_printed(tree, capsys):
    # The check's message runs over several lines; the first alone ended at
    # "cannot be compared: the difference" and said nothing of what differed.
    tr, ev = tree
    write_score(ev, "a_normal", pilot_score("track_rgb_a_normal", sha="b" * 64))
    assert any("2 groups" in p for p in problems(tr, ev))
    out = capsys.readouterr().out
    assert "!! a_rgb vs a_normal: two conditions scored by different evaluators" in out
    assert "between the methods would carry the difference between the evaluators" in out


def test_two_modes_under_one_sha_are_two_groups(tree):
    tr, ev = tree
    write_score(ev, "a_rgb", evaluator_score("track_rgb_a_rgb"))
    write_score(ev, "a_normal", evaluator_score("track_rgb_a_normal", pilot=True))
    assert any("2 groups" in p for p in problems(tr, ev))


def test_two_class_sets_under_one_sha_are_two_groups(tree):
    tr, ev = tree
    write_score(ev, "a_rgb", evaluator_score("track_rgb_a_rgb"))
    write_score(ev, "a_normal", evaluator_score("track_rgb_a_normal", class_set="benchmark"))
    assert any("2 groups" in p for p in problems(tr, ev))


def test_two_propagation_rules_in_one_score_directory_are_two_groups(tree, capsys):
    tr, ev = tree
    forward = evaluator_score("track_rgb_a_normal")
    forward["propagation"] = "forward_from_first"
    write_score(ev, "a_normal", forward)
    write_score(ev, "a_rgb", evaluator_score("track_rgb_a_rgb"))
    assert any("2 groups" in p for p in problems(tr, ev))
    assert "propagated under different rules" in capsys.readouterr().out


def test_a_per_frame_score_joins_a_group_of_either_rule_and_the_group_names_the_rule(tree, capsys):
    tr, ev = tree
    per_frame = evaluator_score("track_rgb_a_normal")
    per_frame["propagation"] = "per_frame"
    write_score(ev, "a_normal", per_frame)
    write_score(ev, "a_rgb", evaluator_score("track_rgb_a_rgb"))
    assert problems(tr, ev) == []
    assert "propagation=both_ways_from_centre → 2 conditions" in capsys.readouterr().out


def test_a_tag_tracked_both_ways_on_some_clips_and_forward_on_others_is_reported(tree):
    tr, ev = tree
    tracked = {**provenance("rgb", 8), "seed_source": "sam"}
    plant(tr, {"track_rgb_a_rgb": [("c3", {**tracked, "bidir": False}, 13)]})
    plant(tr, {"track_rgb_a_rgb": [(c, {**tracked, "bidir": True}, LABELS[c]) for c in ("c1", "c2")]})
    assert any("conditions are mixed" in p for p in problems(tr, ev))


def test_a_pilot_json_and_an_evaluator_json_are_two_groups(tree):
    tr, ev = tree
    write_score(ev, "a_rgb", evaluator_score("track_rgb_a_rgb"))
    assert any("2 groups" in p for p in problems(tr, ev))


def test_every_score_in_a_group_is_comparable_with_every_other_not_with_the_first_alone(tree):
    # A pilot JSON that records no domain is comparable with one of either
    # domain, and two of different domains are not: checked against the
    # first member alone, the three passed as one group.
    tr, ev = tree
    plant(tr, {"track_rgb_a_legacy": [(c, provenance("depth", 8), LABELS[c]) for c in CLIPS]})
    legacy = pilot_score("track_rgb_a_legacy")
    del legacy["eval_version"]
    write_score(ev, "a_legacy", legacy)
    one, two = pilot_score("track_rgb_a_normal"), pilot_score("track_rgb_a_rgb")
    one["tissue_ignore"], two["tissue_ignore"] = [1], [2]
    write_score(ev, "a_normal", one)
    write_score(ev, "a_rgb", two)
    assert any("2 groups" in p for p in problems(tr, ev))


def test_a_score_whose_labels_are_gone_is_reported(tree):
    tr, ev = tree
    write_score(ev, "ghost", pilot_score("track_rgb_a_ghost"))
    assert any("have no labels" in p and "ghost" in p for p in problems(tr, ev))


def test_labels_nobody_scored_are_a_problem_only_when_required(tree, capsys):
    tr, ev = tree
    plant(tr, {"track_rgb_a_unscored": [(c, provenance("depth", 8), LABELS[c]) for c in CLIPS]})
    assert problems(tr, ev) == []
    assert "nobody scored" in capsys.readouterr().out
    assert any("not scored" in p for p in problems(tr, ev, require_scored=True))


def test_a_score_is_matched_to_its_labels_by_track_dir_name_across_roots(tmp_path):
    # The propagation stage's labels are not scored in a directory of their
    # own; their score sits in the seed stage's directory and names them.
    seed, prop, ev = tmp_path / "seed", tmp_path / "prop", tmp_path / "scores"
    ev.mkdir()
    plant(seed, {"track_a": [(c, provenance("rgb", 8), LABELS[c]) for c in CLIPS]})
    plant(prop, {"track_a_t12": [(c, provenance("rgb", 8), LABELS[c]) for c in CLIPS]})
    write_score(ev, "a", pilot_score("track_a"))
    write_score(ev, "t12_a", pilot_score("track_a_t12"))
    assert CI.inventory([(str(seed), str(ev), "seed"), (str(prop), "", "propagated")], require_scored=True) == []


def test_the_table_says_scored_of_labels_scored_from_another_root(tmp_path, capsys):
    # The propagation root's table said "no" of labels the seed root's
    # directory scores, while the match across roots counted them as scored.
    seed, prop, ev = tmp_path / "seed", tmp_path / "prop", tmp_path / "scores"
    ev.mkdir()
    plant(seed, {"track_a": [(c, provenance("rgb", 8), LABELS[c]) for c in CLIPS]})
    plant(prop, {"track_a_t12": [(c, provenance("rgb", 8), LABELS[c]) for c in CLIPS]})
    write_score(ev, "a", pilot_score("track_a"))
    write_score(ev, "t12_a", pilot_score("track_a_t12"))
    CI.inventory([(str(seed), str(ev), "seed"), (str(prop), "", "propagated")])
    line = next(l for l in capsys.readouterr().out.splitlines() if l.startswith("track_a_t12"))
    assert "scored" in line.split() and "no" not in line.split()


# ---------------------------------------------------------------- what it refuses before counting


def test_a_missing_root_is_refused_before_anything_is_counted(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        CI.inventory([(str(tmp_path / "no_such_root"), "", "T")])


def test_a_file_is_not_a_root(tmp_path):
    (tmp_path / "tracks.json").write_text("{}")
    with pytest.raises(ValueError, match="does not exist"):
        CI.check_root_exists(str(tmp_path / "tracks.json"))


def test_a_score_root_that_does_not_exist_is_refused(tmp_path):
    (tmp_path / "tracks").mkdir()
    with pytest.raises(ValueError, match="score directory does not exist"):
        CI.inventory([(str(tmp_path / "tracks"), str(tmp_path / "scroes"), "T")])


def test_a_score_that_does_not_name_its_labels_is_refused(tree):
    tr, ev = tree
    s = evaluator_score("track_rgb_a_rgb")
    del s["track_dir_name"]
    write_score(ev, "a_rgb", s)
    with pytest.raises(ValueError, match="no track_dir_name"):
        problems(tr, ev)


def test_a_score_directory_with_something_that_is_not_a_score_is_refused(tree):
    tr, ev = tree
    (ev / "paired.json").write_text(json.dumps({"n_boot": 10000, "pairs": {}}))
    with pytest.raises(ValueError, match="not a score JSON"):
        problems(tr, ev)


def test_a_score_with_part_of_the_evaluator_s_record_is_refused(tree):
    tr, ev = tree
    s = evaluator_score("track_rgb_a_rgb")
    del s["versions"]
    write_score(ev, "a_rgb", s)
    with pytest.raises(ValueError, match="neither"):
        problems(tr, ev)


# ---------------------------------------------------------------- --root


def test_a_bare_root_uses_itself_as_the_heading():
    assert CI.parse_root("outputs/tracks/seed") == ("outputs/tracks/seed", "", "outputs/tracks/seed")


def test_the_score_root_is_the_second_field_and_the_heading_the_third():
    assert CI.parse_root("a:b") == ("a", "b", "a")
    assert CI.parse_root("a:b:seed stage") == ("a", "b", "seed stage")
    assert CI.parse_root("a::propagated")[1:] == ("", "propagated")


def test_surrounding_spaces_do_not_become_part_of_a_path():
    assert CI.parse_root(" a : b : c ") == ("a", "b", "c")


def test_an_empty_label_root_and_a_fourth_field_are_refused():
    with pytest.raises(ValueError, match="empty label root"):
        CI.parse_root(":outputs/eval")
    with pytest.raises(ValueError, match="at most"):
        CI.parse_root("a:b:c:d")


# ---------------------------------------------------------------- the matrix


def test_the_matrix_prints_letters_not_marks_and_takes_a_none(tree, capsys):
    tr, ev = tree
    none = provenance("rgb", 8)
    none["sam_input"] = None
    plant(tr, {"track_rgb_a_none": [(c, none, LABELS[c]) for c in CLIPS]})
    plant(tr, {"track_rgb_a_prov": [(c, provenance("depth", 16), 0) for c in CLIPS]})
    CI.matrix(str(tr), str(ev), "T")
    out = capsys.readouterr().out
    assert not any(m in out for m in ("○", "×", "△", "★", "✗"))
    assert f"{CI.SCORED} scored" in out and "None" in out
    # The rows: `rgb` scored at pps 8 and not run at 16, `depth` provenance
    # only at 16, `None` a row of its own and not a traceback.
    rows = {l.split()[0]: l.split()[1:] for l in out.splitlines() if l.strip().startswith(("rgb", "None", "depth "))}
    assert rows["rgb"] == [CI.SCORED, CI.NOT_RUN]
    assert rows["depth"] == [CI.NOT_RUN, CI.PROVENANCE_ONLY]
    assert rows["None"] == [CI.LABELS_ONLY, CI.NOT_RUN]


def test_the_matrix_keeps_a_condition_whose_points_per_side_is_none(tree, capsys):
    # A None points-per-side fell out of the columns, and the condition read as not run.
    tr, ev = tree
    plant(tr, {"track_rgb_a_nopps": [(c, provenance("depth", None), LABELS[c]) for c in CLIPS]})
    CI.matrix(str(tr), str(ev), "T")
    out = capsys.readouterr().out
    header = next(l for l in out.splitlines() if l.strip().startswith("input"))
    assert header.split()[1:] == ["pps8", "ppsNone"]
    rows = {l.split()[0]: l.split()[1:] for l in out.splitlines() if l.strip().startswith(("rgb", "depth "))}
    assert rows["rgb"] == [CI.SCORED, CI.NOT_RUN]
    assert rows["depth"] == [CI.NOT_RUN, CI.LABELS_ONLY]


def test_the_matrix_shows_every_condition_in_a_cell_not_the_one_found_first(tree, capsys):
    # Two tags with the same input, points-per-side and depth source, one
    # scored and one not: the cell showed whichever the glob found first.
    tr, ev = tree
    plant(tr, {"track_rgb_a_again": [(c, provenance("rgb", 8), LABELS[c]) for c in CLIPS]})
    CI.matrix(str(tr), str(ev), "T")
    rows = {l.split()[0]: l.split()[1:] for l in capsys.readouterr().out.splitlines() if l.strip().startswith("rgb")}
    assert rows["rgb"] == [f"{CI.SCORED}/{CI.LABELS_ONLY}"]


# ---------------------------------------------------------------- the command


def test_the_command_sets_the_exit_code_on_a_problem(tree):
    tr, ev = tree
    cmd = [sys.executable, "-m", "evalkit.tools.condition_inventory", "--root", f"{tr}:{ev}:T"]
    assert subprocess.run(cmd, capture_output=True, cwd=REPO).returncode == 0
    write_score(ev, "a_normal", pilot_score("track_rgb_a_normal", sha="b" * 64))
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 1 and "2 groups" in r.stdout
    r = subprocess.run(cmd[:-2] + ["--root", str(tr / "nowhere")], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 1 and "does not exist" in r.stderr
