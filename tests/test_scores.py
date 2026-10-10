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
    sign_of_key,
    signs_of,
    split_key,
)

CLIPS = ["VID01_s15_80_crop", "VID02_s15_80_crop"]
SHA = "a" * 64
# Two propagation rules, by the names the paper's conditions use.
BOTH_WAYS, FORWARD = "both_ways_from_centre", "forward_from_first"


def pilot_json(sha=PILOT_EVAL_CODE_SHA, clips=CLIPS, dataset="cholec", version=2):
    return dict(
        eval_code_sha=sha, eval_code_tag="t", eval_version=version, dataset=dataset,
        tissue_ignore=[5, 9], clips=list(clips),
        per_clip=[dict(clip=c, extra_ignore=[], inst_F1_50=0.5, SQ=None) for c in clips],
    )


def evaluator_json(sha=SHA, clips=CLIPS, dataset="cholecseg8k", pilot=False, class_set="original",
                   views=("all", "tissue", "geometric"), propagation=BOTH_WAYS):
    return dict(
        eval_code_sha=sha, dataset=dataset, pilot=pilot, class_set=class_set, views=list(views),
        clips=list(clips), propagation=propagation,
        input_shas={c: {"gt_masks": "g" * 64, "depth": "d" * 64, "predictions": f"p{c}"} for c in clips},
        versions={"python": "3.12.0", "numpy": "2.0.0"},
        per_clip=[dict(clip=c, **{metric_key("F1_50", v): 0.5 for v in views}) for c in clips],
    )


# ---------------------------------------------------------------- which JSON is which


def test_a_json_with_none_of_the_evaluator_fields_is_the_pilot_evaluator_s():
    assert is_pilot_json(pilot_json())
    r = ruler(pilot_json())
    assert (r.pilot, r.class_set, r.views) == (True, "original", PILOT_DOMAINS)
    assert r.domain == ("cholec", (5, 9))


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
    assert chk == dict(clips=sorted(CLIPS), population="identical", eval_code=SHA[:16], propagation=BOTH_WAYS)


# ---------------------------------------------------------------- the propagation rule


def test_two_conditions_propagated_under_different_rules_are_refused():
    with pytest.raises(ValueError, match="different rules") as e:
        check_comparable(evaluator_json(), evaluator_json(propagation=FORWARD))
    assert BOTH_WAYS in str(e.value) and FORWARD in str(e.value)


def test_a_per_frame_condition_is_comparable_with_either_rule():
    for rule in (BOTH_WAYS, FORWARD):
        chk = check_comparable(evaluator_json(propagation=rule), evaluator_json(propagation=scores.PER_FRAME))
        assert chk["propagation"] == rule
        chk = check_comparable(evaluator_json(propagation=scores.PER_FRAME), evaluator_json(propagation=rule))
        assert chk["propagation"] == rule
    both = check_comparable(evaluator_json(propagation=scores.PER_FRAME), evaluator_json(propagation=scores.PER_FRAME))
    assert both["propagation"] == scores.PER_FRAME


def test_a_rule_that_is_not_a_name_is_refused():
    for rule in (None, "", 1):
        with pytest.raises(ValueError, match="names no rule"):
            check_comparable(evaluator_json(), evaluator_json(propagation=rule))


def test_an_evaluator_json_without_a_rule_is_neither_kind():
    j = evaluator_json()
    del j["propagation"]
    with pytest.raises(ValueError, match="neither"):
        check_comparable(evaluator_json(), j)


def test_a_pilot_json_records_no_rule_and_two_of_them_compare_as_before():
    assert scores.propagation_rule_of(pilot_json()) is None
    assert "propagation" not in check_comparable(pilot_json(), pilot_json())


def test_one_table_holds_one_rule_besides_per_frame():
    pf = evaluator_json(propagation=scores.PER_FRAME)
    assert scores.check_one_rule({"a": evaluator_json(), "b": pf, "c": evaluator_json()}) == BOTH_WAYS
    assert scores.check_one_rule({"b": pf}) == scores.PER_FRAME
    assert scores.check_one_rule({"p": pilot_json(), "q": pilot_json()}) is None


def test_a_table_whose_pairs_each_pass_but_that_holds_two_rules_is_refused():
    # Both pairs share the per-frame condition and pass; the table holds two rules and fails.
    tables = {"both": evaluator_json(), "pf": evaluator_json(propagation=scores.PER_FRAME),
              "fwd": evaluator_json(propagation=FORWARD)}
    check_comparable(tables["pf"], tables["both"])
    check_comparable(tables["pf"], tables["fwd"])
    with pytest.raises(ValueError, match="one table per rule") as e:
        scores.check_one_rule(tables)
    assert "['both']" in str(e.value) and "['fwd']" in str(e.value)


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


def test_swapped_extra_ignores_under_one_set_of_values_are_refused():
    # The same two values, each on the other clip: what `tissue` removed
    # from a clip differs even though the values agree as a set.
    a, b = pilot_json(), pilot_json()
    a["per_clip"][0]["extra_ignore"], a["per_clip"][1]["extra_ignore"] = [3], [7]
    b["per_clip"][0]["extra_ignore"], b["per_clip"][1]["extra_ignore"] = [7], [3]
    assert check_comparable(a, copy.deepcopy(a))["population"] == "identical"
    with pytest.raises(ValueError, match="extra_ignore"):
        check_comparable(a, b)


def test_extra_ignores_are_checked_on_the_compared_clips_only():
    # The clip that is not compared may differ in anything.
    a, b = pilot_json(), pilot_json(clips=CLIPS[:1])
    a["per_clip"][1]["extra_ignore"] = [7]
    assert check_comparable(a, b, allow_subset=True)["clips"] == CLIPS[:1]


def test_a_pilot_json_without_a_domain_is_not_checked_for_one():
    # The pilot's own check skipped the domain when one side had none; its
    # sha check already refuses that pair unless legacy code is allowed.
    a, b = pilot_json(sha=None, version=None), pilot_json(sha=None)
    assert check_comparable(a, b, allow_legacy_code=True)["population"] == "identical"


def test_two_datasets_are_refused_even_when_one_pilot_json_records_no_domain():
    # The pilot's own check let this pair through: without a domain on both
    # sides it compared nothing but the clips. The dataset is recorded on
    # both, and the specification says it must match.
    a, b = pilot_json(sha=None, version=None), pilot_json(sha=None, dataset="atlas")
    with pytest.raises(ValueError, match="different datasets"):
        check_comparable(a, b, allow_legacy_code=True)


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


def test_an_empty_clips_field_is_a_claim_and_not_a_missing_one():
    # `clips: []` says nothing was scored; falling back on a leftover
    # `per_clip` would compare clips the JSON never claims.
    j = pilot_json()
    j["clips"] = []
    assert clips_of(j) == []


def test_two_scores_of_no_clips_at_all_are_refused():
    with pytest.raises(ValueError, match="nothing to compare"):
        check_comparable(pilot_json(clips=[]), pilot_json(clips=[]))


@pytest.mark.parametrize("make", [pilot_json, evaluator_json])
def test_a_json_whose_clips_and_rows_disagree_is_refused(make):
    # The population is read from `clips` and the values from `per_clip`;
    # where they disagree, a pair is checked on one set of clips and
    # averaged over another, with nothing failing.
    fewer_claimed = make()
    fewer_claimed["clips"] = CLIPS[:1]
    with pytest.raises(ValueError, match=r"(?s)cond: .*name different clips.*per_clip only: \['VID02_s15_80_crop'\]"):
        check_comparable(make(), fewer_claimed)
    fewer_rows = make()
    fewer_rows["per_clip"] = fewer_rows["per_clip"][:1]
    with pytest.raises(ValueError, match=r"(?s)base: .*name different clips.*clips only: \['VID02_s15_80_crop'\]"):
        check_comparable(fewer_rows, make())
    twice = make()
    twice["per_clip"].append(copy.deepcopy(twice["per_clip"][0]))
    with pytest.raises(ValueError, match=r"more than one row for \['VID01_s15_80_crop'\]"):
        check_comparable(make(), twice)


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


def test_input_shas_that_name_no_input_beyond_the_predictions_are_refused():
    # `{}` on both sides slips past a name-by-name comparison: there is no
    # name to compare, and nothing says the two read the same GT.
    a, b = evaluator_json(), evaluator_json()
    a["input_shas"][CLIPS[0]], b["input_shas"][CLIPS[0]] = {}, {}
    with pytest.raises(ValueError, match="no input beyond the predictions"):
        check_comparable(a, b)
    a["input_shas"][CLIPS[0]] = {"predictions": "p"}
    b["input_shas"][CLIPS[0]] = {"predictions": "q"}
    with pytest.raises(ValueError, match="no input beyond the predictions"):
        check_comparable(a, b)


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


def test_a_key_takes_its_direction_from_the_pilot_s_table_or_from_its_metric():
    assert signs_of(pilot_json()) == dict(scores.PILOT_SIGNS)
    assert sign_of_key(pilot_json(), "underseg_error") == -1
    assert sign_of_key(evaluator_json(), metric_key("VI_split", "tissue")) == -1
    assert sign_of_key(evaluator_json(), "time_IoU") == 0


@pytest.mark.parametrize("make, key", [
    (pilot_json, "VI_split"),  # in the pilot's rows, with no direction in its table
    (pilot_json, metric_key("F1_50", "all")),
    (lambda: evaluator_json(views=("all",)), metric_key("F1_50", "tissue")),  # a view not scored
    (evaluator_json, "inst_F1_50"),
])
def test_a_key_with_no_direction_on_the_json_is_refused(make, key):
    with pytest.raises(ValueError, match="not a key with a direction on this JSON"):
        sign_of_key(make(), key)


def test_defined_clips_keeps_the_clips_where_both_sides_have_a_value():
    rows = {c: dict(SQ=0.5) for c in CLIPS}
    other = copy.deepcopy(rows)
    other[CLIPS[1]]["SQ"] = None
    assert defined_clips(rows, other, CLIPS, "SQ") == CLIPS[:1]
    assert defined_clips(rows, other, CLIPS, "absent") == []


def test_a_json_that_holds_a_nan_or_an_infinity_is_refused(tmp_path):
    # `json` reads a bare `NaN` as a float, and a mean over it is a NaN that
    # prints like a value and counts as a loss.
    p = tmp_path / "s.json"
    for token in ("NaN", "Infinity", "-Infinity"):
        p.write_text('{"per_clip": [{"clip": "c", "SQ": ' + token + '}]}')
        with pytest.raises(ValueError, match=f"holds {token}"):
            scores.load_scores(p)
    p.write_text('{"per_clip": [{"clip": "c", "SQ": 0.5}]}')
    assert scores.load_scores(p)["per_clip"][0]["SQ"] == 0.5
