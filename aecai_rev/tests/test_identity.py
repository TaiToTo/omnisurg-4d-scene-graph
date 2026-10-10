"""Check GT linking, survival and reappearance on hand-made sequences."""
import numpy as np

from aecai_rev import identity


def _blob(y, x, h=20, w=20, size=8):
    m = np.zeros((h, w), bool)
    m[y:y + size, x:x + size] = True
    return m


def test_link_follows_overlap_and_breaks_at_a_gap():
    insts = {0: [(_blob(0, 0), 2)], 1: [(_blob(1, 1), 2)], 5: [(_blob(1, 1), 2)]}
    any_gap = identity.link_gt(insts, [0, 1, 5], 0.0, None)
    adjacent = identity.link_gt(insts, [0, 1, 5], 0.0, 1)
    assert any_gap[0] == any_gap[1] == any_gap[5]
    assert adjacent[0] == adjacent[1] != adjacent[5]


def test_link_keeps_classes_apart():
    insts = {0: [(_blob(0, 0), 2)], 1: [(_blob(0, 0), 5)]}
    t = identity.link_gt(insts, [0, 1], 0.0, None)
    assert t[0] != t[1]


def test_survival_and_one_reappearance():
    lab = [np.full((4, 4), -1) for _ in range(7)]
    for f in (0, 1, 3, 4, 5, 6):          # region 3 vanishes on frame 2 only
        lab[f][0, 0] = 3
    out = identity.survival_and_reappearance(lab, 3, [3])
    assert out[0]["fwd"] == 1.0
    assert np.isclose(out[0]["bwd"], 2 / 3)
    assert out[0]["events"] == [(2, 1, "backward")]
