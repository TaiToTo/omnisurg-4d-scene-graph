"""The inventory fires on every planted fault and stays quiet on a sound tree.

A sound tree of two conditions on three clips, with clip lengths that
differ (5, 9, 13 labels: a clip's length is its own), is built in a
temporary directory, and one fault at a time is planted in it: the
workbench's six, and the mixes the evaluator's JSONs make possible.
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


def evaluator_score(track_dir: str, pilot: bool = False, class_set: str = "original") -> dict:
    return dict(track_dir_name=track_dir, eval_code_sha="a" * 64, dataset="atlas120k", pilot=pilot,
                class_set=class_set, views=["all", "tissue", "geometric"], n_clips=3, n_missing=0,
                n_failed=0, clips=CLIPS, input_shas={c: {} for c in CLIPS}, versions={},
                per_clip=[dict(clip=c) for c in CLIPS])


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
    """A sound tree: two conditions on three clips, both scored by one ruler."""
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
    assert "rulers: 1" in capsys.readouterr().out


# ---------------------------------------------------------------- the planted faults


def test_a_condition_mixed_across_clips_is_reported(tree):
    tr, ev = tree
    plant(tr, {"track_rgb_a_rgb": [("c3", provenance("rgb", 24), 13)]})
    assert any("conditions are mixed" in p for p in problems(tr, ev))


def test_a_condition_on_fewer_clips_is_reported(tree):
    tr, ev = tree
    plant(tr, {"track_rgb_a_short": [("c1", provenance("depth", 8), 5)]})
    assert any("1 clips where the others have 3" in p for p in problems(tr, ev))


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


def test_two_shas_in_one_score_directory_are_a_split_ruler(tree):
    tr, ev = tree
    write_score(ev, "a_normal", pilot_score("track_rgb_a_normal", sha="b" * 64))
    assert any("2 rulers" in p for p in problems(tr, ev))


def test_two_modes_under_one_sha_are_a_split_ruler(tree):
    tr, ev = tree
    write_score(ev, "a_rgb", evaluator_score("track_rgb_a_rgb"))
    write_score(ev, "a_normal", evaluator_score("track_rgb_a_normal", pilot=True))
    assert any("2 rulers" in p for p in problems(tr, ev))


def test_two_class_sets_under_one_sha_are_a_split_ruler(tree):
    tr, ev = tree
    write_score(ev, "a_rgb", evaluator_score("track_rgb_a_rgb"))
    write_score(ev, "a_normal", evaluator_score("track_rgb_a_normal", class_set="benchmark"))
    assert any("2 rulers" in p for p in problems(tr, ev))


def test_a_pilot_json_and_an_evaluator_json_are_two_rulers(tree):
    tr, ev = tree
    write_score(ev, "a_rgb", evaluator_score("track_rgb_a_rgb"))
    assert any("2 rulers" in p for p in problems(tr, ev))


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


# ---------------------------------------------------------------- what it refuses before counting


def test_a_missing_root_is_refused_before_anything_is_counted(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        CI.inventory([(str(tmp_path / "no_such_root"), "", "T")])


def test_a_file_is_not_a_root(tmp_path):
    (tmp_path / "tracks.json").write_text("{}")
    with pytest.raises(ValueError, match="does not exist"):
        CI.check_root_exists(str(tmp_path / "tracks.json"))


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


# ---------------------------------------------------------------- the command


def test_the_command_sets_the_exit_code_on_a_problem(tree):
    tr, ev = tree
    cmd = [sys.executable, "-m", "evalkit.tools.condition_inventory", "--root", f"{tr}:{ev}:T"]
    assert subprocess.run(cmd, capture_output=True, cwd=REPO).returncode == 0
    write_score(ev, "a_normal", pilot_score("track_rgb_a_normal", sha="b" * 64))
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 1 and "2 rulers" in r.stdout
    r = subprocess.run(cmd[:-2] + ["--root", str(tr / "nowhere")], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 1 and "does not exist" in r.stderr
