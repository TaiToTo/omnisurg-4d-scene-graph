"""The per-frame driver, on the 60 x 100 scenes of `tests/scenes.py` read through the real tables.

The scenes' class 1 and 2 become ATLAS-120k's Liver (12) and Gallbladder
(14), both `tissue`, so that every view scores them; 0 stays Background.
The tests pin a few keys by hand and check the rest against the modules
that define them, called on the same pixels, so that the driver cannot hand
a module other arrays than the specification says.
"""
import numpy as np
import pytest

import scenes as S
from evalkit.boundary import boundary_pixels, boundary_score
from evalkit.classes import VIEWS, load_table
from evalkit.classmap import class_map, class_scores
from evalkit.frame import FrameScores, ViewScores, score_frame, score_view
from evalkit.objects import gt_objects, instance_scores, predicted_objects
from evalkit.scored import PixelCounts
from evalkit.vi import variation_of_information

ALL = np.ones((S.H, S.W), dtype=bool)
DEPTH = np.ones((S.H, S.W), dtype=np.float32)
LIVER, GALLBLADDER, TOOL, VENA_CAVA, BACKGROUND, EXCLUDED = 12, 14, 1, 11, 0, 42
OMENTUM, MESENTERIUM = 9, 20     # one benchmark class, two original ids


@pytest.fixture(scope="module")
def atlas():
    return load_table("atlas120k")


@pytest.fixture(scope="module")
def benchmark():
    return load_table("atlas120k", "benchmark")


def atlas_ids(gt: np.ndarray) -> np.ndarray:
    """The scene's ids 0, 1, 2 as ATLAS-120k's Background, Liver, Gallbladder."""
    return np.array([BACKGROUND, LIVER, GALLBLADDER], dtype=np.int32)[gt]


def test_an_exact_prediction_scores_one_or_zero_on_every_key_in_every_view(atlas):
    r = score_frame(atlas_ids(S.exact().gt), S.exact().lab, DEPTH, atlas)
    assert not r.excluded and set(r.views) == set(VIEWS)
    for view, v in r.views.items():
        assert v.view == view
        assert v.metrics() == {
            "F1_50": 1.0, "SQ": 1.0, "inst_BF": 1.0, "mIoU": 1.0, "boundary_F": 1.0,
            "boundary_R_raw": 1.0, "VI_split": pytest.approx(0.0), "VI_merge": pytest.approx(0.0),
            "unlabelled_share": 0.0,
        }
        assert (v.n_gt_objects, v.n_pred_objects, v.n_hits, v.n_inst_bf_hits) == (2, 2, 2, 2)
        assert v.ious == {LIVER: 1.0, GALLBLADDER: 1.0}
        assert v.counts == PixelCounts(0, 0, 0, 0, S.H * S.W)


def test_each_key_reads_its_own_field():
    # Every field a different value, so that two keys swapped (F1_50 and
    # SQ, boundary_F and boundary_R_raw, VI_split and VI_merge) show in the
    # JSON the clip driver writes through `metrics()`.
    v = ViewScores(
        view="all", counts=PixelCounts(0, 0, 0, 0, 1), f1_50=0.1, sq=0.2, inst_bf=0.3, miou=0.4,
        boundary_f=0.5, boundary_r_raw=0.6, vi_split=0.7, vi_merge=0.8, unlabelled_share=0.9,
        n_gt_objects=1, n_pred_objects=2, n_hits=3, n_inst_bf_hits=4, ious={},
    )
    assert v.metrics() == {
        "F1_50": 0.1, "SQ": 0.2, "inst_BF": 0.3, "mIoU": 0.4, "boundary_F": 0.5,
        "boundary_R_raw": 0.6, "VI_split": 0.7, "VI_merge": 0.8, "unlabelled_share": 0.9,
    }


def test_a_spilled_region_is_scored_as_each_module_defines_it(atlas):
    # Region 0 covers the 16 columns of class 1 and spills 10 into class 2.
    # Both objects are found (IoU 960/1560 and 4440/5040); the class map
    # names region 0 after class 1, so its edge, like the regions' own, is
    # 10 px off the true one and the boundary keys are 0.
    scene = S.spill(10)
    gt, lab = atlas_ids(scene.gt), scene.lab
    v = score_frame(gt, lab, DEPTH, atlas).views["geometric"]
    assert v.f1_50 == 1.0
    assert v.sq == pytest.approx((960 / 1560 + 4440 / 5040) / 2)
    assert v.inst_bf == 0.0 and v.boundary_f == 0.0 and v.boundary_r_raw == 0.0
    assert v.ious == {LIVER: pytest.approx(16 / 26), GALLBLADDER: pytest.approx(74 / 84)}
    assert v.miou == pytest.approx((16 / 26 + 74 / 84) / 2)
    # The rest against the modules on the same pixels.
    vi = variation_of_information(gt, lab, ALL)
    assert (v.vi_split, v.vi_merge) == (vi.split, vi.merge)
    cmap = class_map(gt, lab, ALL)
    assert v.miou == class_scores(gt, cmap, ALL).miou
    gb = boundary_pixels(gt, ALL)
    assert v.boundary_f == boundary_score(boundary_pixels(cmap.classes, ALL), gb).f
    assert v.boundary_r_raw == boundary_score(boundary_pixels(lab, ALL), gb).recall


def test_boundary_f_reads_the_class_map_and_boundary_r_raw_the_regions(atlas):
    # Class 1 split in two regions: the regions' boundary has an extra cut,
    # which `boundary_R_raw` forgives (recall 1) while the class map, where
    # both pieces are named class 1, has no cut at all (F 1). A driver that
    # fed the regions to `boundary_F` would see the precision drop.
    scene = S.split()
    v = score_frame(atlas_ids(scene.gt), scene.lab, DEPTH, atlas).views["all"]
    assert v.boundary_f == 1.0 and v.boundary_r_raw == 1.0
    assert v.miou == 1.0
    assert v.f1_50 == pytest.approx(2 * 2 / (2 + 3))
    assert (v.n_gt_objects, v.n_pred_objects, v.n_hits) == (2, 3, 2)
    assert boundary_score(boundary_pixels(scene.lab, ALL), boundary_pixels(scene.gt, ALL)).precision == 0.5


def test_boundary_r_raw_reads_the_regions_and_not_the_class_map(atlas):
    # Region 1 is liver columns 0-14 and gallbladder columns 50-59, named
    # liver by majority; region 0 is the liver between. The regions keep
    # the true edge at column 50, so `boundary_R_raw` is 1, while the class
    # map, liver on both sides of it, loses it: a driver that fed the class
    # map to `boundary_R_raw` would see 0, the value `boundary_F` has.
    gt = atlas_ids(S.exact().gt)
    lab = np.full((S.H, S.W), 2, dtype=np.int32)
    lab[:, 15:50] = 0
    lab[:, :15] = lab[:, 50:60] = 1
    v = score_frame(gt, lab, DEPTH, atlas).views["all"]
    assert v.boundary_r_raw == 1.0 and v.boundary_f == 0.0


def test_boundary_f_is_the_f_score_and_not_the_precision(atlas):
    # Region 1 is liver columns 0-9 and gallbladder columns 50-64, named
    # gallbladder by majority, so the class map has an extra edge at column
    # 10 beside the true one: precision 1/2, recall 1, F 2/3.
    gt = atlas_ids(S.exact().gt)
    lab = S.exact().lab.copy()
    lab[:, 50:] = 2
    lab[:, :10] = lab[:, 50:65] = 1
    v = score_frame(gt, lab, DEPTH, atlas).views["all"]
    cmap = class_map(gt, lab, ALL)
    assert boundary_score(boundary_pixels(cmap.classes, ALL), boundary_pixels(gt, ALL)).precision == 0.5
    assert v.boundary_f == pytest.approx(2 / 3) and v.boundary_r_raw == 1.0


def test_the_views_differ_only_in_the_pixels_they_score(atlas):
    # Columns 0-39 liver, 40-49 a tool, 50-89 gallbladder, 90-99 the vena
    # cava (`appearance`). The prediction is exact. The `all` view scores
    # four objects, `tissue` three, `geometric` two; each view's counts say
    # what it removed, and every key stays 1 or 0 because nothing is wrong.
    # In the geometric view the tool band separates the two tissues, so no
    # two scored neighbours differ: that view has no GT boundary, and its
    # boundary keys are undefined rather than 1.
    gt = np.empty((S.H, S.W), dtype=np.int32)
    gt[:, :40], gt[:, 40:50], gt[:, 50:90], gt[:, 90:] = LIVER, TOOL, GALLBLADDER, VENA_CAVA
    lab = np.empty((S.H, S.W), dtype=np.int32)
    lab[:, :40], lab[:, 40:50], lab[:, 50:90], lab[:, 90:] = 0, 1, 2, 3
    r = score_frame(gt, lab, DEPTH, atlas)
    assert {view: v.n_gt_objects for view, v in r.views.items()} == {"all": 4, "tissue": 3, "geometric": 2}
    assert r.views["all"].counts == PixelCounts(0, 0, 0, 0, 6000)
    assert r.views["tissue"].counts == PixelCounts(0, 0, 0, 600, 5400)
    assert r.views["geometric"].counts == PixelCounts(0, 0, 0, 1200, 4800)
    assert set(r.views["geometric"].ious) == {LIVER, GALLBLADDER}
    for v in r.views.values():
        assert v.f1_50 == 1.0 and v.sq == 1.0 and v.miou == 1.0
    assert r.views["all"].boundary_f == 1.0 and r.views["tissue"].boundary_f == 1.0
    g = r.views["geometric"]
    assert g.boundary_f is None and g.boundary_r_raw is None and g.inst_bf is None
    assert (g.n_hits, g.n_inst_bf_hits) == (2, 0)


def test_a_region_on_unlabelled_tissue_is_no_object_and_a_spill_onto_it_is_counted(atlas):
    # The top ten rows are unlabelled; a region there alone is no object and
    # costs nothing. A region that runs from the liver over those rows is
    # scored as if it stopped, and the spill shows in `unlabelled_share`.
    scene = S.on_background(10)
    v = score_frame(atlas_ids(scene.gt), scene.lab, DEPTH, atlas).views["geometric"]
    assert v.counts.background == 1000 and v.n_pred_objects == 2 and v.f1_50 == 1.0
    assert v.unlabelled_share == 0.0
    lab = scene.lab.copy()
    lab[:10, :S.HALF] = 0              # region 0 grows over the unlabelled rows on its side
    v = score_frame(atlas_ids(scene.gt), lab, DEPTH, atlas).views["geometric"]
    assert v.f1_50 == 1.0 and v.sq == 1.0
    assert v.unlabelled_share == pytest.approx(500 / (2500 + 500 + 2500))


def test_a_region_is_named_by_its_scored_pixels_and_not_by_the_background_under_it(atlas):
    # Region 2 covers the 15 top-left rows: 500 px of unlabelled background
    # over 250 px of liver. Named by its scored pixels it is liver and
    # mIoU is 1; a vote over every valid pixel, which is the pilot
    # evaluator's rule, would name it background.
    scene = S.on_background(10)
    lab = scene.lab.copy()
    lab[:15, :S.HALF] = 2
    v = score_frame(atlas_ids(scene.gt), lab, DEPTH, atlas).views["geometric"]
    assert v.miou == 1.0 and v.ious == {LIVER: 1.0, GALLBLADDER: 1.0}
    assert v.counts.background == 1000


def test_unlabelled_share_counts_background_and_not_the_pixels_a_view_removed(atlas):
    # Columns 0-39 liver, 40-49 a tool, 50-99 gallbladder; region 0 runs
    # over the tool band. The geometric view removes the band for being a
    # tool, not for being unlabelled, so the spill is in neither count and
    # the share is 0; counted as background it would be 600 / 6000.
    gt = np.empty((S.H, S.W), dtype=np.int32)
    gt[:, :40], gt[:, 40:50], gt[:, 50:] = LIVER, TOOL, GALLBLADDER
    v = score_frame(gt, S.exact().lab, DEPTH, atlas).views["geometric"]
    assert v.counts == PixelCounts(0, 0, 0, 600, 5400)
    assert v.unlabelled_share == 0.0 and v.f1_50 == 1.0


def test_the_gt_is_the_class_sets_class_and_not_the_mask_id(benchmark):
    # Omentum and Mesenterium are two original ids and one benchmark class.
    # Under one region they are one GT object, found; read as mask ids they
    # would be two, and F1_50 2 / 3.
    gt = np.empty((S.H, S.W), dtype=np.int32)
    gt[:, :S.HALF], gt[:, S.HALF:] = OMENTUM, MESENTERIUM
    lab = np.zeros((S.H, S.W), dtype=np.int32)
    v = score_frame(gt, lab, DEPTH, benchmark).views["all"]
    assert (v.n_gt_objects, v.n_pred_objects, v.n_hits, v.f1_50) == (1, 1, 1, 1.0)
    assert v.miou == 1.0 and set(v.ious) == {benchmark.class_of(OMENTUM)} == {benchmark.class_of(MESENTERIUM)}
    assert instance_scores(gt_objects(gt, ALL), predicted_objects(lab, ALL)).n_gt == 2


def test_a_view_that_removes_every_pixel_defines_no_key(atlas):
    # The vena cava alone: the geometric view scores nothing, and every key is None
    # rather than 0, with the counts saying why.
    gt = np.full((S.H, S.W), VENA_CAVA, dtype=np.int32)
    r = score_frame(gt, S.exact().lab, DEPTH, atlas)
    v = r.views["geometric"]
    assert v.counts == PixelCounts(0, 0, 0, 6000, 0)
    assert all(value is None for value in v.metrics().values())
    assert (v.n_gt_objects, v.n_pred_objects, v.n_hits, v.n_inst_bf_hits) == (0, 0, 0, 0)
    assert v.ious == {}
    # The `all` view scores it: one object, two regions over it.
    assert r.views["all"].f1_50 == pytest.approx(2 * 1 / (1 + 2))


def test_a_frame_with_no_gt_boundary_leaves_the_boundary_keys_undefined(atlas):
    # Liver alone, predicted as two regions: there is no GT boundary to
    # recover, so the boundary keys and inst_BF are None while F1_50 and
    # VI_split are not.
    gt = np.full((S.H, S.W), LIVER, dtype=np.int32)
    v = score_frame(gt, S.exact().lab, DEPTH, atlas).views["geometric"]
    assert v.boundary_f is None and v.boundary_r_raw is None and v.inst_bf is None
    assert v.f1_50 == pytest.approx(2 / 3) and v.vi_split == pytest.approx(1.0)
    assert v.n_hits == 1 and v.n_inst_bf_hits == 0


def test_an_excluded_frame_is_returned_skipped_and_scored_in_no_view(atlas):
    gt = atlas_ids(S.exact().gt)
    gt[:10, :10] = EXCLUDED
    r = score_frame(gt, S.exact().lab, DEPTH, atlas)
    assert r == FrameScores(excluded=True, views={})
    # In the benchmark set too, where the marker maps to background.
    assert score_frame(gt, S.exact().lab, DEPTH, load_table("atlas120k", "benchmark")).excluded


def test_a_pixel_without_valid_depth_refuses_the_frame(atlas):
    depth = DEPTH.copy()
    depth[0, 0] = np.nan
    with pytest.raises(ValueError, match="no valid depth"):
        score_frame(atlas_ids(S.exact().gt), S.exact().lab, depth, atlas)
    # An excluded frame too: the depth map is checked before the marker, so
    # a faulty depth map is found whether or not the frame is scored.
    gt = atlas_ids(S.exact().gt)
    gt[:10, :10] = EXCLUDED
    with pytest.raises(ValueError, match="no valid depth"):
        score_frame(gt, S.exact().lab, depth, atlas)
    with pytest.raises(ValueError, match="only pilot mode"):
        score_view(atlas_ids(S.exact().gt), S.exact().lab, np.isfinite(depth), atlas, "all")


def test_the_three_maps_must_share_one_shape(atlas):
    gt = atlas_ids(S.exact().gt)
    with pytest.raises(ValueError, match="`regions` has shape"):
        score_frame(gt, S.exact().lab[:, :-1], DEPTH, atlas)
    with pytest.raises(ValueError, match="`depth` has shape"):
        score_frame(gt, S.exact().lab, DEPTH[:-1], atlas)
    with pytest.raises(ValueError, match="integer"):
        score_frame(gt.astype(np.float32), S.exact().lab, DEPTH, atlas)


def test_a_mask_id_the_table_does_not_have_raises(atlas):
    gt = atlas_ids(S.exact().gt)
    gt[0, 0] = 47
    with pytest.raises(KeyError, match="47"):
        score_frame(gt, S.exact().lab, DEPTH, atlas)
