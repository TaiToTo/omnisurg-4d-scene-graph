"""The clip list in `stereomis_meta/` agrees with the rule that made it and with its README.

The list was made from the dataset, which no test reads, so these tests pin
what the rule gives without it: the grid, the offsets, the counts the README
gives, and the masks the README says every frame of P2 has.
"""

from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("scipy")

import surgical_core.stereomis as S  # noqa: E402

LIST = Path(__file__).resolve().parent.parent / "stereomis_meta" / "clips.json"


@pytest.fixture(scope="module")
def clip_list():
    return S.read_clip_list(LIST)


def test_the_list_holds_86_clips_of_which_43_are_usable(clip_list):
    assert len(clip_list) == 86
    assert Counter(c["drop"] for c in clip_list) == {"": 43, "gt_static": 26, "gt_jump": 17}
    assert Counter(c["seq"] for c in clip_list if c["usable"]) == {
        "P1": 6, "P2_0": 6, "P2_1": 4, "P2_2": 4, "P2_4": 5, "P2_5": 5, "P2_6": 3, "P2_7": 6, "P2_8": 4}


def test_each_clip_lies_on_the_fixed_grid_with_the_measured_row_offset(clip_list):
    for c in clip_list:
        k = int(c["name"][-4:])
        span = S.CLIP_FRAMES * c["stride"]
        assert c["stride"] == 12, c["name"]
        assert c["frames"][0] == S.CLIP_GRID_START + k * span, c["name"]
        assert c["gt_row_offset"] == S.GT_ROW_OFFSET[c["seq"]], c["name"]
        assert c["stratum"] == S.stratum_of(c["span_mm"]), c["name"]


def test_every_frame_of_a_usable_p2_clip_has_the_mask_of_its_own_number(clip_list):
    """P1's masks cover part of the record only: 504 of the 672 frames of its usable clips have one."""
    own = {s: [m == f + 1 for c in clip_list if c["usable"] and c["seq"] == s for f, m in zip(c["frames"], c["masks"])]
           for s in S.SEQUENCES}
    assert all(all(v) for s, v in own.items() if s != "P1")
    assert (sum(own["P1"]), len(own["P1"])) == (504, 672)
    assert all(m is None for c in clip_list if c["usable"] and c["seq"] == "P1"
               for f, m in zip(c["frames"], c["masks"]) if m != f + 1)
