"""Check the parameterised area cut and matching against the tagged evaluator."""
import numpy as np

from aecai_rev import score


def _scene(rng, h=96, w=128, n_cls=6, n_reg=25):
    """Return a random class map, region map and valid mask with blobs."""
    gt = np.zeros((h, w), np.int64)
    for _ in range(30):
        c = int(rng.integers(0, n_cls))
        y, x = rng.integers(0, h), rng.integers(0, w)
        gt[max(0, y - 15):y + 15, max(0, x - 20):x + 20] = c
    lab = np.full((h, w), -1, np.int32)
    for r in range(n_reg):
        y, x = rng.integers(0, h), rng.integers(0, w)
        lab[max(0, y - 12):y + 12, max(0, x - 12):x + 12] = r
    valid = np.ones((h, w), bool)
    valid[:, :5] = False
    return gt, lab, valid


def test_cut_equals_tagged_at_its_threshold(et):
    rng = np.random.default_rng(0)
    for _ in range(20):
        gt, lab, valid = _scene(rng)
        cut = et.INSTANCE_MIN_CC
        mine = [m for m, _ in score.gt_instances(gt, valid, cut, {et.BACKGROUND})]
        theirs = et._gt_instances(gt, valid)
        assert len(mine) == len(theirs)
        assert all((a == b).all() for a, b in zip(mine, theirs))
        mine_p = [m for m, _ in score.pred_regions(lab, valid, cut)]
        theirs_p = et._pred_regions(lab, valid)
        assert len(mine_p) == len(theirs_p)
        assert all((a == b).all() for a, b in zip(mine_p, theirs_p))


def test_a_larger_cut_drops_small_components(et):
    gt = np.zeros((50, 50), np.int64)
    gt[0:10, 0:10] = 2          # 100 px
    gt[20:50, 20:50] = 3        # 900 px
    valid = np.ones_like(gt, bool)
    assert len(score.gt_instances(gt, valid, 0, {0})) == 2
    assert len(score.gt_instances(gt, valid, 101, {0})) == 1


def test_match_pairs_gives_the_tagged_ious(et):
    rng = np.random.default_rng(1)
    for _ in range(20):
        gt, lab, valid = _scene(rng)
        gts = et._gt_instances(gt, valid)
        preds = et._pred_regions(lab, valid)
        ious, _, _ = et._match_ious(gts, preds)
        assert [p[0] for p in score.match_pairs(gts, preds)] == ious


def test_frame_scores_equal_tagged_eval_frame(et):
    rng = np.random.default_rng(2)
    for _ in range(10):
        gt, lab, valid = _scene(rng)
        ref = et.eval_frame(lab, gt, valid)
        mine = score.frame_scores(et, lab, gt, valid, et.INSTANCE_MIN_CC, (0.5, 0.75),
                                  {et.BACKGROUND})
        assert mine["n_reg"] == ref["n_regions"]
        assert np.isclose(mine["miou"], np.mean(list(ref["ious"].values())))
        if ref["inst"] is None:
            assert mine["f1@0.5"] is None
        else:
            assert np.isclose(mine["f1@0.5"], ref["inst"]["f1@0.5"])
            assert np.isclose(mine["f1@0.75"], ref["inst"]["f1@0.75"])


def test_merge_joins_the_fragments_of_one_instance():
    lab = np.full((10, 10), -1, np.int32)
    lab[:, 0:3] = 4
    lab[:, 3:5] = 7
    lab[:, 5:10] = 9
    g = np.zeros((10, 10), bool)
    g[:, 0:5] = True            # regions 4 and 7 lie on one instance
    h = np.zeros((10, 10), bool)
    h[:, 6:10] = True
    out = score.merge_by_instance(lab, np.ones_like(g), [g, h])
    assert set(np.unique(out[:, 0:5])) == {4}
    assert set(np.unique(out[:, 5:10])) == {9}


def test_merge_leaves_a_region_without_instance():
    lab = np.zeros((4, 4), np.int32)
    lab[:, 2:] = 1
    g = np.zeros((4, 4), bool)
    g[:, :2] = True
    out = score.merge_by_instance(lab, np.ones_like(g), [g])
    assert (out == lab).all()


def test_region_to_class_equals_tagged(et):
    rng = np.random.default_rng(3)
    for _ in range(20):
        gt, lab, valid = _scene(rng)
        assert score.region_to_class(lab, gt, valid) == et._region_to_class(lab, gt, valid)


def test_iou_pairs_equal_mask_ious(et):
    rng = np.random.default_rng(4)
    for _ in range(20):
        gt, lab, valid = _scene(rng)
        gts = et._gt_instances(gt, valid)
        preds = et._pred_regions(lab, valid)
        ref = []
        for gi, g in enumerate(gts):
            for pj, p in enumerate(preds):
                u = int((g | p).sum())
                inter = int((g & p).sum())
                if u and inter:
                    ref.append((inter / u, gi, pj))
        assert sorted(score.iou_pairs(gts, preds)) == sorted(ref)
