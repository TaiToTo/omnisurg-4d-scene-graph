"""The comparability check refuses every mix it is there to refuse, on planted faults.

Two score JSONs are built for each case, one of the pilot evaluator's
layout and one of the evaluator's, and one field at a time is changed.
A check earns its place by failing when it should, so every rule has the
case that trips it, and one case checks that an untouched pair passes.
"""
import copy

import pytest

from evalkit.tools import scores
from evalkit.tools.scores import (
    PILOT_DOMAINS,
    PILOT_EVAL_CODE_SHA,
    check_comparable,
    clips_of,
    defined_clips,
    is_pilot_json,
    metric_key,
    metric_keys,
    ruler,
    sign_of,
    split_key,
)

CLIPS = ["VID01_s15_80_crop", "VID02_s15_80_crop"]
SHA = "a" * 64


def pilot_json(sha=PILOT_EVAL_CODE_SHA, clips=CLIPS, dataset="cholec", version=2):
    return dict(
        eval_code_sha=sha, eval_code_tag="t", eval_version=version, dataset=dataset,
        tissue_ignore=[5, 9], clips=list(clips),
        per_clip=[dict(clip=c, extra_ignore=[], inst_F1_50=0.5, SQ=None) for c in clips],
    )


def evaluator_json(sha=SHA, clips=CLIPS, dataset="cholecseg8k", pilot=False, class_set="original",
                   views=("all", "tissue", "geometric")):
    return dict(
        eval_code_sha=sha, dataset=dataset, pilot=pilot, class_set=class_set, views=list(views),
        clips=list(clips),
        input_shas={c: {"gt_masks": "g" * 64, "depth": "d" * 64, "predictions": f"p{c}"} for c in clips},
        versions={"python": "3.12.0", "numpy": "2.0.0"},
        per_clip=[dict(clip=c, **{metric_key("F1_50", v): 0.5 for v in views}) for c in clips],
    )


# ---------------------------------------------------------------- which JSON is which


def test_a_json_with_none_of_the_evaluator_fields_is_the_pilot_evaluator_s():
    assert is_pilot_json(pilot_json())
    r = ruler(pilot_json())
    assert (r.pilot, r.class_set, r.views) == (True, "original", PILOT_DOMAINS)
    assert r.domain == ("cholec", (5, 9), ((),))


def test_a_json_with_all_the_evaluator_fields_is_the_evaluator_s():
    assert not is_pilot_json(evaluator_json())
    r = ruler(evaluator_json(pilot=True, class_set="benchmark"))
    assert (r.pilot, r.class_set, r.views, r.domain) == (True, "benchmark", ("all", "tissue", "geometric"), None)


def test_a_json_with_some_of_the_evaluator_fields_is_refused():
    j = evaluator_json()
    del j["input_shas"]
    with pytest.raises(ValueError, match="neither"):
        is_pilot_json(j)
    with pytest.raises(ValueError, match="neither"):
        check_comparable(j, evaluator_json())


def test_the_pilot_s_first_jsons_record_no_domain():
    assert ruler(pilot_json(version=None)).domain is None


def test_clips_come_from_per_clip_when_the_json_predates_the_clips_field():
    j = pilot_json()
    del j["clips"]
    assert clips_of(j) == CLIPS


# ---------------------------------------------------------------- the pair that passes


def test_identical_rulers_and_populations_pass():
    chk = check_comparable(pilot_json(), pilot_json())
    assert chk == dict(clips=sorted(CLIPS), population="identical", eval_code=PILOT_EVAL_CODE_SHA[:16])
    chk = check_comparable(evaluator_json(), evaluator_json())
    assert chk == dict(clips=sorted(CLIPS), population="identical", eval_code=SHA[:16])


# ---------------------------------------------------------------- the sha


def test_different_shas_are_refused_even_on_the_same_population():
    with pytest.raises(ValueError, match="different evaluators"):
        check_comparable(pilot_json(sha="b" * 64), pilot_json())
    with pytest.raises(ValueError, match="different evaluators"):
        check_comparable(evaluator_json(sha="b" * 64), evaluator_json())


def test_a_missing_sha_is_refused_unless_said_so():
    a, b = pilot_json(sha=None), pilot_json(sha=None)
    with pytest.raises(ValueError, match="eval_code_sha"):
        check_comparable(a, b)
    assert check_comparable(a, b, allow_legacy_code=True)["eval_code"] == "legacy-unverified"


# ---------------------------------------------------------------- what the sha does not say


def test_a_pilot_json_and_an_evaluator_json_are_never_compared():
    # Even under one sha: the check against the pilot evaluator is a
    # verification of its own, not a comparison.
    with pytest.raises(ValueError, match="pilot evaluator's and the other"):
        check_comparable(pilot_json(sha=SHA), evaluator_json())


def test_pilot_mode_and_normal_mode_under_one_sha_are_refused():
    with pytest.raises(ValueError, match="pilot mode and the normal mode share it"):
        check_comparable(evaluator_json(pilot=True), evaluator_json())


def test_two_class_sets_under_one_sha_are_refused():
    with pytest.raises(ValueError, match="class sets"):
        check_comparable(evaluator_json(class_set="benchmark"), evaluator_json())


def test_two_sets_of_views_under_one_sha_are_refused():
    with pytest.raises(ValueError, match="views"):
        check_comparable(evaluator_json(views=("all",)), evaluator_json())


def test_two_datasets_under_one_sha_are_refused():
    with pytest.raises(ValueError, match="dataset"):
        check_comparable(evaluator_json(dataset="atlas120k"), evaluator_json())


def test_two_pilot_domains_under_one_sha_are_refused():
    b = pilot_json(dataset="atlas")
    b["tissue_ignore"] = [1]
    with pytest.raises(ValueError, match="different domains"):
        check_comparable(pilot_json(), b)


def test_a_pilot_json_without_a_domain_is_not_checked_for_one():
    # The pilot's own check skipped the domain when one side had none; its
    # sha check already refuses that pair unless legacy code is allowed.
    a, b = pilot_json(sha=None, version=None), pilot_json(sha=None, dataset="atlas")
    assert check_comparable(a, b, allow_legacy_code=True)["population"] == "identical"


# ---------------------------------------------------------------- the population


def test_different_clip_sets_are_refused_unless_a_subset_is_allowed():
    a, b = pilot_json(clips=CLIPS), pilot_json(clips=CLIPS[:1])
    with pytest.raises(ValueError, match="clip sets differ"):
        check_comparable(a, b)
    chk = check_comparable(a, b, allow_subset=True)
    assert (chk["clips"], chk["population"]) == (CLIPS[:1], "intersection")


def test_no_common_clip_is_refused_even_with_a_subset_allowed():
    with pytest.raises(ValueError, match="no clip is common"):
        check_comparable(pilot_json(clips=CLIPS[:1]), pilot_json(clips=CLIPS[1:]), allow_subset=True)


# ---------------------------------------------------------------- the inputs and the versions


def test_two_scores_that_read_a_different_gt_are_refused():
    b = evaluator_json()
    b["input_shas"][CLIPS[0]]["gt_masks"] = "x" * 64
    with pytest.raises(ValueError, match="different gt_masks"):
        check_comparable(evaluator_json(), b)


def test_two_scores_that_read_a_different_depth_map_are_refused():
    b = evaluator_json()
    b["input_shas"][CLIPS[1]]["depth"] = "x" * 64
    with pytest.raises(ValueError, match="different depth"):
        check_comparable(evaluator_json(), b)


def test_different_predictions_are_the_point_and_pass():
    b = evaluator_json()
    b["input_shas"][CLIPS[0]]["predictions"] = "other"
    assert check_comparable(evaluator_json(), b)["population"] == "identical"


def test_a_clip_without_input_shas_is_refused():
    b = evaluator_json()
    del b["input_shas"][CLIPS[0]]
    with pytest.raises(ValueError, match="no input shas"):
        check_comparable(evaluator_json(), b)


def test_inputs_are_checked_on_the_compared_clips_only():
    # The clip that is not compared may differ in anything.
    a, b = evaluator_json(), evaluator_json(clips=CLIPS[:1])
    a["input_shas"][CLIPS[1]]["depth"] = "x" * 64
    assert check_comparable(a, b, allow_subset=True)["clips"] == CLIPS[:1]


def test_different_versions_are_reported_not_refused():
    b = evaluator_json()
    b["versions"]["numpy"] = "2.1.0"
    chk = check_comparable(evaluator_json(), b)
    assert chk["versions_differ"] == {"numpy": ("2.0.0", "2.1.0")}
    assert "versions_differ" not in check_comparable(evaluator_json(), evaluator_json())


# ---------------------------------------------------------------- the keys


def test_the_evaluator_s_keys_are_one_per_metric_per_view_then_time_iou():
    keys = metric_keys(evaluator_json(views=("all", "geometric")))
    assert keys[0] == "F1_50/all"
    assert keys[len(scores.FRAME_METRICS)] == "F1_50/geometric"
    assert keys[-1] == "time_IoU"
    assert len(keys) == 2 * len(scores.FRAME_METRICS) + 1


def test_a_pilot_json_has_no_key_list_of_its_own():
    with pytest.raises(ValueError, match="pilot JSON"):
        metric_keys(pilot_json())


def test_keys_split_back_into_metric_and_view():
    assert split_key(metric_key("SQ", "geometric")) == ("SQ", "geometric")
    assert split_key("time_IoU") == ("time_IoU", None)
    assert split_key("inst_F1_50_tissue") == ("inst_F1_50_tissue", None)


def test_every_metric_has_a_direction_and_the_reference_values_have_none():
    assert {sign_of(metric_key(m, "all")) for m in scores.FRAME_METRICS} == {+1, -1, 0}
    assert sign_of("VI_split") == -1 and sign_of("time_IoU") == 0 and sign_of("unlabelled_share") == 0
    with pytest.raises(KeyError):
        sign_of("inst_F1_50")


def test_defined_clips_keeps_the_clips_where_both_sides_have_a_value():
    rows = {c: dict(SQ=0.5) for c in CLIPS}
    other = copy.deepcopy(rows)
    other[CLIPS[1]]["SQ"] = None
    assert defined_clips(rows, other, CLIPS, "SQ") == CLIPS[:1]
    assert defined_clips(rows, other, CLIPS, "absent") == []
