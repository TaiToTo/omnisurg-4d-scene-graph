"""One clip's inputs read from disk: time order, shapes, fingerprints, and what is refused.

The clips are written by `tests/clip_dirs.py` as the pipeline lays them out.
The refusals include the two faults the real CholecSeg8k clips showed: an
image that appears twice, at one timestamp, and a stale mask left on a frame
the manifest says has none. A mask the viewer's step wrote under the GT's
name, marked by `seg_provenance`, is not GT and is not scored. CholecSeg8k's
region line is filled and counted.
"""
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

import clip_dirs as C
from evalkit.inputs import DEPTH_FILE, MANIFEST, MASK_DIR, depth_sha, read_clip

TAG = "cond"


def clip_with_labels(tmp_path, n=4, **kw):
    gt = [C.two_organs(cut=10 + i) for i in range(n)]
    C.write_clip(tmp_path / "data", "c", gt, **kw)
    C.write_labels(tmp_path / "tracks", "c", TAG, {i: g.copy() for i, g in enumerate(gt)}, scale=2)
    return gt


def read(tmp_path):
    return read_clip(tmp_path / "data", tmp_path / "tracks", TAG, "c", C.TABLE)


def test_frames_come_in_time_order_and_every_input_at_the_depth_s_shape(tmp_path):
    gt = clip_with_labels(tmp_path, times=[0.0, 1.2, 0.6, 1.8])
    clip = read(tmp_path)
    assert clip.order == (0, 2, 1, 3)
    assert clip.numbers == {0: 100, 1: 115, 2: 130, 3: 145}
    assert clip.depth.shape == (4, 20, 30)
    for i in range(4):
        assert np.array_equal(clip.gt[i], gt[i]) and np.array_equal(clip.regions[i], gt[i])
        assert clip.gt[i].dtype == np.int32 and clip.regions[i].dtype == np.int32


def test_without_timestamps_the_frames_are_ordered_by_their_number(tmp_path):
    clip_with_labels(tmp_path)
    path = tmp_path / "data" / "c" / MANIFEST
    m = json.loads(path.read_text())
    for f, number in zip(m["frames"], [300, 100, 400, 200]):
        del f["timestamp_sec"]
        f["native_frame"] = number
    path.write_text(json.dumps(m))
    assert read(tmp_path).order == (1, 3, 0, 2)


def another_gt_mask(tmp_path):
    big = np.kron(C.two_organs(cut=3), np.ones((2, 2), dtype=np.int32))
    Image.fromarray(C.colour_mask(big)).save(tmp_path / "data" / "c" / MASK_DIR / f"000002{C.MASK_SUFFIX['cholecseg8k']}")


def another_depth(tmp_path):
    np.savez(tmp_path / "data" / "c" / DEPTH_FILE, depth=np.full((4, 20, 30), 2.0, dtype=np.float32))


def another_crop(tmp_path):
    path = tmp_path / "data" / "c" / MANIFEST
    m = json.loads(path.read_text())
    m["crop_info"]["x0"] += 1
    path.write_text(json.dumps(m))


def another_time_order(tmp_path):
    path = tmp_path / "data" / "c" / MANIFEST
    m = json.loads(path.read_text())
    m["frames"][1]["timestamp_sec"], m["frames"][2]["timestamp_sec"] = (
        m["frames"][2]["timestamp_sec"], m["frames"][1]["timestamp_sec"])
    path.write_text(json.dumps(m))


def another_prediction_on_a_frame_without_gt(tmp_path):
    np.save(tmp_path / "tracks" / "c" / TAG / "label_0001.npy", np.zeros((40, 60), dtype=np.int16))


@pytest.mark.parametrize("name, plant", [
    ("gt_masks", another_gt_mask), ("depth", another_depth), ("crop", another_crop),
    ("frames", another_time_order), ("predictions", another_prediction_on_a_frame_without_gt),
])
def test_each_fingerprint_moves_with_its_own_input_only(tmp_path, name, plant):
    clip_with_labels(tmp_path, gt_frames=[0, 2, 3])
    before = dict(read(tmp_path).shas)
    assert set(before) == {"gt_masks", "depth", "crop", "frames", "predictions"}
    with np.load(tmp_path / "data" / "c" / DEPTH_FILE) as z:
        assert before["depth"] == hashlib.sha256(z["depth"].astype(np.float32).tobytes()).hexdigest()
    plant(tmp_path)
    after = dict(read(tmp_path).shas)
    assert {k for k in before if before[k] != after[k]} == {name}


def test_the_depth_fingerprint_is_the_depth_manifest_s():
    d = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    assert depth_sha(d) == hashlib.sha256(d.astype(np.float32).tobytes()).hexdigest()


def test_a_frame_without_gt_still_brings_its_prediction(tmp_path):
    clip_with_labels(tmp_path, gt_frames=[0, 2, 3])
    clip = read(tmp_path)
    assert set(clip.gt) == {0, 2, 3} and set(clip.regions) == {0, 1, 2, 3}


def edit_manifest(tmp_path, change):
    path = tmp_path / "data" / "c" / MANIFEST
    m = json.loads(path.read_text())
    change(m)
    path.write_text(json.dumps(m))


def test_a_gt_frame_without_a_prediction_is_refused(tmp_path):
    clip_with_labels(tmp_path)
    (tmp_path / "tracks" / "c" / TAG / "label_0002.npy").unlink()
    with pytest.raises(ValueError, match=r"GT frames \[2\] have no prediction"):
        read(tmp_path)


def test_one_image_twice_at_one_timestamp_is_refused(tmp_path):
    clip_with_labels(tmp_path, times=[0.0, 0.6, 0.6, 1.2])
    with pytest.raises(ValueError, match="share one timestamp_sec"):
        read(tmp_path)


def test_a_mask_on_a_frame_the_manifest_says_has_none_is_refused(tmp_path):
    clip_with_labels(tmp_path)
    edit_manifest(tmp_path, lambda m: m["frames"][3].update(has_seg_mask=False))
    with pytest.raises(ValueError, match="frame 3 is flagged has_seg_mask=False but its mask is there"):
        read(tmp_path)


@pytest.mark.parametrize("change, match", [
    (lambda m: m["frames"][1].pop("timestamp_sec"), "1 of 4|3 of 4"),
    (lambda m: m["frames"][2].update(seq_idx=5), "says seq_idx 5"),
    (lambda m: m["frames"].pop(), "the manifest lists 3 frames"),
    (lambda m: m.pop("crop_info"), "crop rectangle"),
    (lambda m: m.update(crop={"x": 0}), "crop rectangle"),
])
def test_a_manifest_that_disagrees_with_itself_or_the_depth_is_refused(tmp_path, change, match):
    clip_with_labels(tmp_path)
    edit_manifest(tmp_path, change)
    with pytest.raises(ValueError, match=match):
        read(tmp_path)


@pytest.mark.parametrize("name, array, match", [
    ("label_7.npy", np.zeros((20, 30), np.int16), "names frame 7"),
    ("label_1.npy", np.zeros((20, 30), np.int16), "frame 1 has two predictions"),
    ("label_x.npy", np.zeros((20, 30), np.int16), "is not label_<index>.npy"),
])
def test_a_label_file_that_names_no_frame_or_one_twice_is_refused(tmp_path, name, array, match):
    clip_with_labels(tmp_path)
    np.save(tmp_path / "tracks" / "c" / TAG / name, array)
    with pytest.raises(ValueError, match=match):
        read(tmp_path)


def test_a_prediction_that_is_not_an_integer_map_is_refused(tmp_path):
    clip_with_labels(tmp_path)
    np.save(tmp_path / "tracks" / "c" / TAG / "label_0000.npy", np.zeros((20, 30), np.float32))
    with pytest.raises(ValueError, match="not an \\(H, W\\) integer map"):
        read(tmp_path)


@pytest.mark.parametrize("big", [2 ** 32 + 2, -(2 ** 31) - 1])
def test_a_region_id_outside_int32_is_refused(tmp_path, big):
    clip_with_labels(tmp_path)
    label = np.load(tmp_path / "tracks" / "c" / TAG / "label_0000.npy").astype(np.int64)
    label[label == label.max()] = big
    np.save(tmp_path / "tracks" / "c" / TAG / "label_0000.npy", label)
    with pytest.raises(ValueError, match="outside int32"):
        read(tmp_path)


def test_a_condition_without_a_directory_for_the_clip_is_refused(tmp_path):
    clip_with_labels(tmp_path)
    with pytest.raises(FileNotFoundError, match="no prediction directory"):
        read_clip(tmp_path / "data", tmp_path / "tracks", "other", "c", C.TABLE)


def test_a_clip_with_no_mask_is_refused(tmp_path):
    clip_with_labels(tmp_path, gt_frames=[])
    with pytest.raises(ValueError, match="no frame is a GT frame"):
        read(tmp_path)


def test_a_mask_colour_the_table_does_not_have_is_refused(tmp_path):
    clip_with_labels(tmp_path)
    path = next((tmp_path / "data" / "c" / MASK_DIR).iterdir())
    rgb = np.array(Image.open(path))
    rgb[0, 0] = (1, 2, 3)
    Image.fromarray(rgb).save(path)
    with pytest.raises(KeyError, match="not in the class table"):
        read(tmp_path)


def test_a_mask_marked_by_seg_provenance_is_not_gt_and_needs_no_prediction(tmp_path):
    gt = [C.two_organs(cut=10 + i) for i in range(4)]
    C.write_clip(tmp_path / "data", "c", gt, gt_frames=[0, 3], sam_frames=[1, 2])
    C.write_labels(tmp_path / "tracks", "c", TAG, {0: gt[0].copy(), 3: gt[3].copy()}, scale=2)
    clip = read(tmp_path)
    assert sorted(clip.gt) == [0, 3]


def test_the_gt_masks_fingerprint_covers_gt_frames_only(tmp_path):
    clip_with_labels(tmp_path, gt_frames=[0, 3])
    before = read(tmp_path).shas["gt_masks"]
    edit_manifest(tmp_path, lambda m: m["frames"][3].update(seg_provenance="sam3_gt_propagated"))
    assert read(tmp_path).shas["gt_masks"] != before


@pytest.mark.parametrize("change, match", [
    (lambda m: m["frames"][2].update(is_anchor=False), "frame 2 has a mask but is neither an anchor"),
    (lambda m: m["frames"][2].pop("is_anchor"), "frame 2 has a mask but is neither an anchor"),
    (lambda m: m["frames"][1].pop("has_seg_mask"), "frame 1 has no has_seg_mask flag"),
])
def test_a_frame_whose_gt_the_manifest_leaves_unknown_is_refused(tmp_path, change, match):
    clip_with_labels(tmp_path)
    edit_manifest(tmp_path, change)
    with pytest.raises(ValueError, match=match):
        read(tmp_path)


def test_a_gt_flag_without_a_mask_file_is_refused(tmp_path):
    clip_with_labels(tmp_path)
    (tmp_path / "data" / "c" / MASK_DIR / f"{2:06d}_color_mask.png").unlink()
    with pytest.raises(ValueError, match="frame 2 is flagged has_seg_mask=True but its mask is missing"):
        read(tmp_path)


def test_the_region_line_is_filled_on_the_mask_as_stored_and_counted_at_the_depth_s_shape(tmp_path):
    # The mask is twice the depth's size, and the line is drawn on it from an odd column, so that the
    # resize keeps the line's right half only. Filled on the mask, that half is gallbladder, from the
    # pixel beside it. Filled after the resize, the line would be 1 px wide, as near to both organs,
    # and take the liver's smaller id: the orders differ on column 13.
    gt = C.two_organs(cut=12)
    C.write_clip(tmp_path / "data", "c", [gt, C.two_organs()])
    big = np.kron(gt, np.ones((2, 2), dtype=np.int32))
    big[:, 24], big[:, 25:27] = C.LIVER, C.REGION_LINE
    big[10:12, 40:42] = C.REGION_LINE          # a dot inside the gallbladder, 1 px at the depth's shape
    Image.fromarray(C.colour_mask(big)).save(tmp_path / "data" / "c" / MASK_DIR / f"{0:06d}_color_mask.png")
    C.write_labels(tmp_path / "tracks", "c", TAG, {0: C.two_organs(), 1: C.two_organs()}, scale=2)
    clip = read(tmp_path)
    assert np.array_equal(clip.gt[0], C.two_organs(cut=13))
    assert dict(clip.filled) == {0: 21, 1: 0}


def test_a_mask_of_the_region_line_alone_is_refused(tmp_path):
    clip_with_labels(tmp_path)
    path = tmp_path / "data" / "c" / MASK_DIR / f"{2:06d}_color_mask.png"
    Image.fromarray(np.full((40, 60, 3), 255, dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match="c: frame 2: every pixel holds a filled id"):
        read(tmp_path)
