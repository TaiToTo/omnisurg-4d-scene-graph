"""Pilot mode against the pilot evaluator's values, worked out by hand on the drawn scenes.

The values are the pilot evaluator's, derived from each scene's areas for the
metric guide; pilot mode must give them, so they are kept rather than derived
again under the evaluator's rules. Then the clip: how the pilot averaged,
what it wrote where nothing was defined, the order it read the frames in, and
the frames it scored.
"""
import json
import math

import cv2
import numpy as np
import pytest

import clip_dirs as C
import scenes as S
from evalkit.evaluate import score_condition
from evalkit.inputs import MANIFEST
from evalkit.keys import CLIP_METRICS, FRAME_METRICS, metric_key
from evalkit.pilot import PILOT_DOMAINS, PILOT_MIN_CC_PX
from evalkit.pilot_clip import _pilot_gt, pilot_frame, pilot_row, pilot_time_iou, score_pilot_clip
from evalkit.tools.pilot_check import SHARED

TOOLS = frozenset({5, 9})


def frame(scene):
    return pilot_frame(scene.gt.astype(np.int32), scene.lab.astype(np.int32), scene.valid, TOOLS)


def binary_entropy(p: float) -> float:
    return 0.0 if p in (0.0, 1.0) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def test_exact():
    f = frame(S.exact())
    assert (f["inst"]["full"]["f1_50"], f["inst"]["full"]["sq"]) == (1.0, 1.0)
    assert f["miou"] == 1.0 and f["boundary_f"] == pytest.approx(1.0)
    assert f["vi_split"] == pytest.approx(0.0) and f["vi_merge"] == pytest.approx(0.0)


@pytest.mark.parametrize("k", [1, 2, 3, 4, 10])
def test_shifted_boundary(k):
    """F1_50 ignores the shift; SQ and mIoU see it; the boundary counts the columns within 2 px."""
    f = frame(S.shifted(k))
    iou1, iou2 = S.HALF / (S.HALF + k), (S.HALF - k) / S.HALF
    assert f["inst"]["full"]["f1_50"] == 1.0
    assert f["inst"]["full"]["sq"] == pytest.approx((iou1 + iou2) / 2)
    assert f["miou"] == pytest.approx((iou1 + iou2) / 2)
    near = ((k - 1 <= 2) + (k <= 2)) / 2
    assert f["boundary_f"] == pytest.approx(near, abs=1e-6)


def test_split_class():
    """One GT object, two regions, each with IoU exactly 0.5."""
    f = frame(S.split())
    assert f["inst"]["full"]["f1_50"] == pytest.approx(2 * 2 / (2 * 2 + 1))
    assert f["inst"]["full"]["sq"] == pytest.approx(0.75)
    assert f["miou"] == 1.0                     # both halves vote for class 1
    assert f["vi_split"] == pytest.approx(0.5) and f["vi_merge"] == pytest.approx(0.0)
    assert f["boundary_r_raw"] == pytest.approx(1.0)


def test_merged_classes():
    f = frame(S.merged())
    assert f["inst"]["full"]["f1_50"] == pytest.approx(2 / 3)
    assert f["inst"]["full"]["sq"] == pytest.approx(0.5)
    assert f["miou"] == pytest.approx((0.5 + 0.0) / 2)   # the tie goes to class 1
    assert f["vi_merge"] == pytest.approx(1.0) and f["vi_split"] == pytest.approx(0.0)


def test_gap_without_region():
    """The class map is background in the gap, so its boundary sits 5 px from the true one."""
    f = frame(S.gap(10))
    assert f["inst"]["full"]["f1_50"] == 1.0
    assert f["boundary_f"] == 0.0
    assert f["vi_split"] == pytest.approx(binary_entropy(0.1))
    assert f["vi_merge"] == pytest.approx(0.1)


def test_gap_painted_as_region():
    assert frame(S.gap(10, painted=True))["inst"]["full"]["f1_50"] == pytest.approx(4 / 5)


def test_region_on_background_is_an_object_in_full_and_none_in_labeled():
    f = frame(S.on_background())
    assert f["inst"]["full"]["f1_50"] == pytest.approx(4 / 5) and f["inst"]["full"]["sq"] == 1.0
    assert f["inst"]["labeled"]["f1_50"] == 1.0


@pytest.mark.parametrize("px, f1", [(PILOT_MIN_CC_PX - 1, 1.0), (PILOT_MIN_CC_PX, 0.8)])
def test_a_sliver_is_an_object_from_the_cut_up(px, f1):
    assert frame(S.sliver(px))["inst"]["full"]["f1_50"] == pytest.approx(f1)


def test_time_iou():
    sc = S.moved(5)
    assert pilot_time_iou([sc.lab, sc.next_lab], [sc.valid] * 2) == pytest.approx(
        (S.HALF / (S.HALF + 5) + (S.HALF - 5) / S.HALF) / 2)
    sc = S.swapped()
    assert pilot_time_iou([sc.lab, sc.next_lab], [sc.valid] * 2) == 0.0
    assert pilot_time_iou([sc.lab], [sc.valid]) == 0.0      # nothing pooled: the pilot wrote 0


def test_the_pilot_s_colours_read_hepatic_vein_s_real_colour_as_background(tmp_path):
    """The pilot drew Hepatic Vein as (0, 255, 0), which no mask holds, and read the
    mask's (0, 50, 128) and the region line's white as background."""
    rgb = np.zeros((6, 4, 3), np.uint8)
    rgb[:2], rgb[2:4], rgb[4:] = (0, 50, 128), (0, 255, 0), (255, 255, 255)
    cv2.imwrite(str(tmp_path / "000000_color_mask.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    ids, _ = _pilot_gt(tmp_path, 0, (6, 4))
    assert (ids[:2] == 0).all() and (ids[2:4] == 11).all() and (ids[4:] == 0).all()
    assert _pilot_gt(tmp_path, 1, (6, 4)) is None


def test_where_nothing_is_defined_the_row_holds_what_the_pilot_wrote():
    """No GT object anywhere: F1_50 is 0 in `full` and None in the other domains."""
    background = np.zeros((S.H, S.W), np.int32)
    f = pilot_frame(background, S.exact().lab.astype(np.int32), np.ones((S.H, S.W), bool), TOOLS)
    row = pilot_row("c", [(0, f)], 0.0)
    assert row["F1_50/full"] == 0.0 and row["SQ/full"] is None and row["inst_BF/full"] is None
    for d in PILOT_DOMAINS[1:]:
        assert row[f"F1_50/{d}"] is None and row["n_frames"][f"F1_50/{d}"] == 0
    assert row["VI_split/labeled"] is None and row["n_frames"]["VI_split/labeled"] == 0


def test_the_clip_means_are_rounded_to_four_decimals_and_counted():
    frames = [(0, frame(S.shifted(3))), (1, frame(S.exact())), (2, frame(S.merged()))]
    row = pilot_row("c", frames, 1 / 3)
    expected = round(float(np.mean([v["miou"] for _, v in frames])), 4)
    assert row["mIoU/full"] == expected and row["time_IoU"] == 0.3333
    assert row["n_frames"]["mIoU/full"] == 3 and row["n_frames"]["SQ/full"] == 3
    assert [f["frame"] for f in row["frames"]] == [0, 1, 2]


def test_every_key_written_is_the_evaluator_s_and_holds_what_pilot_check_compares():
    row = pilot_row("c", [(0, frame(S.split()))], 0.5)
    keys = {metric_key(m, d) for m in FRAME_METRICS for d in PILOT_DOMAINS} | set(CLIP_METRICS)
    written = {k for k in row if k not in ("clip", "n_frames", "frames")}
    assert written <= keys
    assert {ours for _, ours in SHARED} == written


def pilot_clip_dir(tmp_path, names):
    gt = [C.two_organs(cut=8 + 4 * i) for i in range(len(names))]
    C.write_clip(tmp_path / "data", "c", gt)
    out = tmp_path / "tracks" / "c" / "t"
    out.mkdir(parents=True)
    for i, name in enumerate(names):
        np.save(out / name, gt[i].astype(np.int16))
    return gt


def test_the_frames_come_in_file_name_order_as_the_pilot_read_them(tmp_path):
    names = [f"label_{i}.npy" for i in range(11)]
    gt = pilot_clip_dir(tmp_path, names)
    row, shas = score_pilot_clip(tmp_path / "data", tmp_path / "tracks", "t", "c", "cholecseg8k")
    order = sorted(range(11), key=lambda i: f"label_{i}.npy")      # 0, 1, 10, 2, ...
    assert [f["frame"] for f in row["frames"]] == order
    ones = [np.ones((20, 30), bool)] * 11
    assert row["time_IoU"] == round(pilot_time_iou([gt[i] for i in order], ones), 4)
    assert set(shas) == {"gt_masks", "depth", "predictions"}


def test_a_mask_the_viewer_wrote_is_not_scored_as_gt(tmp_path):
    # The pilot's data root held the annotated masks only; the viewer's masks, marked by seg_provenance, are
    # not among the frames it scored.
    pilot_clip_dir(tmp_path, [f"label_{i:04d}.npy" for i in range(3)])
    path = tmp_path / "data" / "c" / MANIFEST
    m = json.loads(path.read_text())
    m["frames"][1].update(is_anchor=False, seg_provenance="sam3_gt_propagated")
    path.write_text(json.dumps(m))
    row, _ = score_pilot_clip(tmp_path / "data", tmp_path / "tracks", "t", "c", "cholecseg8k")
    assert [f["frame"] for f in row["frames"]] == [0, 2]


def test_a_mask_the_manifest_does_not_flag_is_refused(tmp_path):
    pilot_clip_dir(tmp_path, [f"label_{i:04d}.npy" for i in range(3)])
    path = tmp_path / "data" / "c" / MANIFEST
    m = json.loads(path.read_text())
    m["frames"][1]["has_seg_mask"] = False
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="frame 1 is flagged has_seg_mask=False but its mask is there"):
        score_pilot_clip(tmp_path / "data", tmp_path / "tracks", "t", "c", "cholecseg8k")


def test_a_clip_whose_predicted_frames_are_not_gt_frames_is_refused(tmp_path):
    pilot_clip_dir(tmp_path, [f"label_{i:04d}.npy" for i in range(2)])
    path = tmp_path / "data" / "c" / MANIFEST
    m = json.loads(path.read_text())
    for f in m["frames"]:
        f.update(is_anchor=False, seg_provenance="sam3_gt_propagated")
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="no predicted frame is a GT frame"):
        score_pilot_clip(tmp_path / "data", tmp_path / "tracks", "t", "c", "cholecseg8k")


def test_a_label_outside_the_depth_is_refused(tmp_path):
    pilot_clip_dir(tmp_path, ["label_0000.npy", "label_0009.npy"])
    with pytest.raises(ValueError, match="names frame 9"):
        score_pilot_clip(tmp_path / "data", tmp_path / "tracks", "t", "c", "cholecseg8k")


def test_a_pilot_mode_json_says_so_and_scores_the_pilot_s_domains(tmp_path):
    pilot_clip_dir(tmp_path, [f"label_{i:04d}.npy" for i in range(3)])
    summary = score_condition("cholecseg8k", None, ["c"], tmp_path / "data", tmp_path / "tracks", "t", pilot=True)
    assert summary["pilot"] is True and summary["views"] == list(PILOT_DOMAINS)
    assert summary["per_clip"][0]["F1_50/labeled_tissue"] == 1.0
