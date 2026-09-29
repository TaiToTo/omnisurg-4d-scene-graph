"""v2's metrics on the scenes the metric guide draws, against values worked out by hand.

`docs/metrics/index.html` explains each metric by drawing a scene from
`scenes.py` and printing what v2 computes on it. This file says what those
numbers must be, derived from the scene's areas rather than by calling v2
twice. When the guide and the code disagree, a test here fails.

Several tests pin a behaviour of v2 that `docs/eval_v3.md` changes. They
describe v2 as it is, which is what v3 is checked against, not as it should be.
"""
import itertools
import math

import cv2
import numpy as np
import pytest

import scenes as S
import v2_import

E = v2_import.load()
AREA = S.H * S.HALF            # one class, 3,000 px


def _frame(scene):
    return E.eval_frame(scene.lab, scene.gt, scene.valid)


def _h(p: float) -> float:
    """Binary entropy in bits."""
    return 0.0 if p in (0.0, 1.0) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


# ── Objects found and how well they fit: F1@t, SQ, PQ, F1_avg ───────────────

def test_exact():
    fr = _frame(S.exact())
    i = fr["inst"]
    assert (i["f1@0.5"], i["f1@0.75"], i["pq"], i["sq"], i["f1_avg"]) == (1.0, 1.0, 1.0, 1.0, 1.0)
    assert fr["ious"] == {1: 1.0, 2: 1.0}
    assert fr["bf"][2] == pytest.approx(1.0)
    assert fr["underseg"] == 0.0
    assert fr["vi_split"] == pytest.approx(0.0) and fr["vi_merge"] == pytest.approx(0.0)


@pytest.mark.parametrize("k", [1, 3, 10])
def test_shifted_boundary(k):
    """F1@0.5 ignores the shift; the IoU-based metrics and UE see it."""
    i = _frame(S.shifted(k))["inst"]
    iou1, iou2 = S.HALF / (S.HALF + k), (S.HALF - k) / S.HALF
    assert i["f1@0.5"] == 1.0
    assert i["sq"] == pytest.approx((iou1 + iou2) / 2)
    assert i["pq"] == pytest.approx((iou1 + iou2) / 2)
    fr = _frame(S.shifted(k))
    assert fr["ious"] == {1: pytest.approx(iou1), 2: pytest.approx(iou2)}
    # The spilled band, S.H × k px, is counted once on each side of the boundary.
    assert fr["underseg"] == pytest.approx(2 * S.H * k / (2 * AREA))


def test_split_class():
    """Over-segmentation: one GT object, two regions, each with IoU exactly 0.5."""
    fr = _frame(S.split())
    i = fr["inst"]
    tp, fp, fn = 2, 1, 0                   # class 2 (IoU 1) and one half of class 1 (IoU 0.5)
    assert i["f1@0.5"] == pytest.approx(2 * tp / (2 * tp + fp + fn))
    assert i["sq"] == pytest.approx((1.0 + 0.5) / 2)
    assert i["pq"] == pytest.approx((1.0 + 0.5) / (tp + fp / 2 + fn / 2))
    assert i["f1@0.75"] == pytest.approx(2 * 1 / (2 * 1 + 2 + 1))    # only class 2 is above 0.75
    assert i["f1_avg"] == pytest.approx((0.8 + 9 * 0.4) / 10)         # 0.50 at TP=2, the rest at TP=1
    assert fr["ious"] == {1: 1.0, 2: 1.0}  # both halves vote for class 1: the class map is exact
    assert fr["overseg"] == {1: 2, 2: 1}
    assert fr["underseg"] == 0.0
    assert fr["vi_split"] == pytest.approx(0.5 * 1.0)   # half the pixels, split 50/50: 1 bit
    assert fr["vi_merge"] == pytest.approx(0.0)
    assert fr["br_raw"] == (pytest.approx(0.5), pytest.approx(1.0))  # extra boundary costs precision only


def test_merged_classes():
    """Under-segmentation: one region over both classes."""
    fr = _frame(S.merged())
    i = fr["inst"]
    assert i["f1@0.5"] == pytest.approx(2 * 1 / (2 * 1 + 0 + 1))     # matches one class at IoU 0.5
    assert i["sq"] == pytest.approx(0.5)
    assert i["pq"] == pytest.approx(0.5 / (1 + 0 + 0.5))
    assert i["f1@0.75"] == 0.0
    # The vote is a tie; argmax takes the smaller id, class 1.
    assert fr["ious"] == {1: pytest.approx(0.5), 2: 0.0}
    assert fr["dices"] == {1: pytest.approx(2 * AREA / (2 * AREA + AREA)), 2: 0.0}
    assert fr["underseg"] == pytest.approx(1.0)        # the maximum
    assert fr["vi_merge"] == pytest.approx(1.0) and fr["vi_split"] == pytest.approx(0.0)


def test_gap_without_region():
    """Unassigned pixels are no false positive, but they wreck the boundary and count in VI."""
    fr = _frame(S.gap(10))
    i = fr["inst"]
    assert i["f1@0.5"] == 1.0
    assert i["pq"] == pytest.approx(0.9)               # 45 of 50 columns
    # The class map is background in the gap, so its boundaries sit 5 px from
    # the true one: nothing within 2 px, all within 5 px.
    assert fr["bf_tol"][2][2] == 0.0 and fr["bf_tol"][5][2] == pytest.approx(1.0)
    # VI treats -1 as one more label: a tenth of each class is "region -1".
    assert fr["vi_split"] == pytest.approx(_h(0.1))
    assert fr["vi_merge"] == pytest.approx(0.1 * 1.0)


def test_gap_painted_as_region():
    fr = _frame(S.gap(10, painted=True))
    i = fr["inst"]
    assert i["f1@0.5"] == pytest.approx(2 * 2 / (2 * 2 + 1 + 0))     # the band is a false positive
    assert i["pq"] == pytest.approx((0.9 + 0.9) / (2 + 0.5))


def test_region_on_background():
    sc = S.on_background()
    i = _frame(sc)["inst"]
    assert i["f1@0.5"] == pytest.approx(2 * 2 / (2 * 2 + 1 + 0))
    assert i["sq"] == 1.0
    labeled = sc.valid & (sc.gt != E.BACKGROUND)
    assert E.instance_metrics(sc.lab, sc.gt, labeled)["f1@0.5"] == 1.0


@pytest.mark.parametrize("px, counted", [(E.INSTANCE_MIN_CC - 1, False), (E.INSTANCE_MIN_CC, True)])
def test_sliver_threshold(px, counted):
    i = _frame(S.sliver(px))["inst"]
    assert i["n_pred"] == (3 if counted else 2)
    assert i["f1@0.5"] == pytest.approx(0.8 if counted else 1.0)


# ── The class map: mIoU and boundary ────────────────────────────────────────

@pytest.mark.parametrize("k", [1, 2, 3, 4, 10])
@pytest.mark.parametrize("tol", [1, 2, 3, 5])
def test_boundary_tolerance(k, tol):
    """Both boundaries are drawn 2 px wide. Each of the two predicted boundary
    columns counts if it lies within `tol` of the nearer true one, and vice versa."""
    near = ((k - 1 <= tol) + (k <= tol)) / 2
    p, r, f = _frame(S.shifted(k))["bf_tol"][tol]
    assert (p, r) == (pytest.approx(near), pytest.approx(near))
    assert f == pytest.approx(near, abs=1e-6)


def test_tolerance_never_lowers_precision_or_recall():
    rng = np.random.default_rng(0)
    for _ in range(40):
        gt = np.repeat(rng.integers(0, 4, (6, 10)), 10, axis=0).repeat(10, axis=1)
        lab = np.repeat(rng.integers(-1, 6, (6, 10)), 10, axis=0).repeat(10, axis=1)
        fr = E.eval_frame(lab.astype(np.int32), gt.astype(np.int32), np.ones(gt.shape, bool))
        for which in ("bf_tol", "br_raw_tol"):
            for pos in (0, 1):
                seq = [fr[which][t][pos] for t in E.BOUNDARY_TOLS]
                assert seq == sorted(seq), (which, pos, seq)


# ── Splitting and merging counts ─────────────────────────────────────────────

@pytest.mark.parametrize("width, counted", [(3, False), (4, False), (5, True)])
def test_overseg_needs_200_px_and_5_percent(width, counted):
    """A piece counts towards `overseg` only if it is ≥ 200 px *and* ≥ 5 % of the class.

    v2's comment says "or". Class 2 here is 5,040 px (5 % = 252). A 4-column
    spill is 240 px: enough under "or", not under the code's "and".
    """
    fr = _frame(S.spill(width))
    assert fr["overseg"][2] == (2 if counted else 1)


# ── Time ─────────────────────────────────────────────────────────────────────

def test_time_iou():
    sc = S.moved(5)
    assert E.time_iou([sc.lab, sc.next_lab], [sc.valid] * 2) == pytest.approx(
        (S.HALF / (S.HALF + 5) + (S.HALF - 5) / S.HALF) / 2)
    sc = S.swapped()
    assert E.time_iou([sc.lab, sc.next_lab], [sc.valid] * 2) == 0.0


# ── Exact relations between metrics ─────────────────────────────────────────

ALL = [S.exact(), S.shifted(1), S.shifted(3), S.shifted(10), S.split(), S.merged(),
       S.gap(10), S.gap(10, painted=True), S.on_background(), S.sliver(300)]


@pytest.mark.parametrize("scene", ALL, ids=lambda s: s.key)
def test_relations(scene):
    fr = _frame(scene)
    i = fr["inst"]
    if i["sq"] is not None:
        assert i["pq"] == pytest.approx(i["sq"] * i["f1@0.5"])
    assert 2 * i["pq"] - i["f1@0.5"] - 1e-12 <= i["f1_avg"] <= 2 * i["pq"] - 0.9 * i["f1@0.5"] + 1e-12
    for c, iou in fr["ious"].items():
        assert fr["dices"][c] == pytest.approx(2 * iou / (1 + iou))


def test_matching_at_exactly_one_half_is_optimal():
    """F1@0.5 counts pairs with IoU ≥ 0.5, ties at exactly 0.5 included, found greedily.

    The earlier property test covers IoU > 0.5 only, where every object has at
    most one partner. This one draws partitions on a coarse grid, so exact
    halves are common, and compares the greedy count with a maximum matching
    found by brute force.
    """
    rng = np.random.default_rng(1)
    ties = caught = 0
    for _ in range(300):
        gt = np.repeat(rng.integers(1, 4, 5), 20)[None, :].repeat(S.H, 0).astype(np.int32)
        lab = np.repeat(rng.integers(0, 4, 10), 10)[None, :].repeat(S.H, 0).astype(np.int32)
        valid = np.ones(gt.shape, bool)
        gid, ga = E._gt_instance_map(gt, valid)
        pid, pa, _ = E._pred_region_map(lab, valid)
        greedy = sum(1 for iou, _, _ in E._match_pairs(gid, ga, pid, pa) if iou >= 0.5)
        ok, pairs = set(), []
        for g in range(len(ga)):
            for p in range(len(pa)):
                inter = int(((gid == g + 1) & (pid == p + 1)).sum())
                iou = inter / (ga[g] + pa[p] - inter)
                ties += iou == 0.5
                pairs.append((iou, g, p))
                if iou >= 0.5:
                    ok.add((g, p))
        best = max((len(m) for r in range(len(ok) + 1) for m in itertools.combinations(ok, r)
                    if len({g for g, _ in m}) == r == len({p for _, p in m})), default=0)
        assert greedy == best
        # The planted fault: greedy in the wrong order, weakest overlap first.
        caught += _greedy_count(sorted(p for p in pairs if p[0] > 0)) < best
    assert ties > 20                       # exact halves were actually exercised
    assert caught > 0                      # and the comparison does catch a worse matcher


def _greedy_count(pairs) -> int:
    """Matches with IoU ≥ 0.5 when `pairs` are taken greedily in the order given."""
    gused, pused, n = set(), set(), 0
    for iou, g, p in pairs:
        if g in gused or p in pused:
            continue
        gused.add(g); pused.add(p)
        n += iou >= 0.5
    return n


# ── The CholecSeg8k colour table ─────────────────────────────────────────────

def test_v2_reads_the_real_hepatic_vein_colour_as_background(tmp_path):
    """v2 maps Hepatic Vein to (0, 255, 0); the masks draw it as (0, 50, 128).

    Checked against the dataset's watershed masks (see docs/eval_v3.md): no
    mask contains (0, 255, 0), and (0, 50, 128) is the only colour carrying
    Hepatic Vein's code. v2 therefore never scores the class. v3 fixes the
    table; this pins what v2 does, since v3 is compared against it.
    """
    from surgical_core.cholec import mask_to_id_map
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[:2] = (0, 50, 128)
    rgb[2:] = (0, 255, 0)
    path = str(tmp_path / "mask.png")
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    idm = mask_to_id_map(path, 4, 4)
    assert (idm[:2] == 0).all()            # the real colour: background, silently
    assert (idm[2:] == 11).all()           # the table's colour, which no mask uses
