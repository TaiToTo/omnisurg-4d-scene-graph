"""Check Holm's correction, the paired tests and the summary intervals."""
import numpy as np
import pandas as pd

from aecai_rev import stats


def test_holm_matches_the_step_down_rule():
    # m = 3: sorted p 0.01, 0.02, 0.04 -> 0.03, 0.04, 0.04 (monotone), in input order.
    assert stats.holm([0.04, 0.01, 0.02]) == [0.04, 0.03, 0.04]


def test_holm_keeps_a_missing_test_missing():
    out = stats.holm([0.01, None, 0.5])
    assert out[1] is None
    assert out[0] == 0.02 and out[2] == 0.5


def test_holm_caps_at_one():
    assert stats.holm([0.6, 0.7]) == [1.0, 1.0]


def _long(diff):
    rng = np.random.default_rng(0)
    base = rng.uniform(0.3, 0.7, size=20)
    clips = [f"VID{(i % 10) + 1:02d}_s15_{i}_crop" for i in range(20)]
    rows = []
    for c, b in zip(clips, base):
        rows.append(dict(clip=c, metric="f1", modality="rgb", value=b))
        rows.append(dict(clip=c, metric="f1", modality="normal", value=b + diff))
    return pd.DataFrame(rows)


def test_a_constant_shift_gives_an_interval_on_it():
    t = stats.paired_tests(_long(0.05), ["metric"], [("normal", "rgb")])
    row = t.iloc[0]
    assert np.isclose(row["mean_diff"], 0.05)
    assert np.isclose(row["ci_low"], 0.05) and np.isclose(row["ci_high"], 0.05)
    assert row["p_raw"] < 0.001


def test_no_difference_gives_no_test():
    t = stats.paired_tests(_long(0.0), ["metric"], [("normal", "rgb")])
    assert t.iloc[0]["p_raw"] is None or pd.isna(t.iloc[0]["p_raw"])


def test_summary_interval_contains_the_mean_and_has_a_video_interval():
    s = stats.summarize(_long(0.1), ["modality", "metric"])
    for _, r in s.iterrows():
        assert r["ci_low"] <= r["mean"] <= r["ci_high"]
        assert r["ci_low_video"] <= r["mean"] <= r["ci_high_video"]
