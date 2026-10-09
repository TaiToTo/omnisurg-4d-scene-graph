"""Run `pipeline.track` through a stand-in segmenter and tracker, and check what it writes and what it refuses.

The stand-ins answer without torch or weights: the segmenter cuts every
image into fixed regions, and the tracker returns its seed masks on every
frame it is asked for. `pipeline.byte_check` checks the real models against
the workbench's stage. The segmenter inputs need the `render` extra.
"""

import json
import sys

import cv2
import numpy as np
import pytest
from PIL import Image

pytest.importorskip("matplotlib")

import pipeline.track as track  # noqa: E402
from pipeline.track import collapse, paint_masks, read_clip, run_track, seed_masks, window  # noqa: E402
from sam3_wrapper import FrameResult  # noqa: E402
from surgical_core.geometry.render import sam_input_image  # noqa: E402

H, W, N = 12, 16, 5


class StandInSegmenter:
    """Cut every image into a left half (region 0), a right half (1) and a 2x2 corner (2)."""

    def __init__(self):
        self.images = []

    def label_map(self, image, points_per_side):
        self.images.append(image.copy())
        labels = np.zeros(image.shape[:2], int)
        labels[:, W // 2:] = 1
        labels[:2, :2] = 2
        return labels


class StandInTracker:
    """Return the seed masks on every frame from the start frame on, in the direction asked."""

    def __init__(self):
        self.calls = []

    def init_video(self, frames):
        self.frames = frames

    def add_masks(self, frame_idx, masks, obj_ids):
        self.masks, self.ids = masks, obj_ids

    def propagate(self, start_frame_idx, reverse=False):
        self.calls.append((start_frame_idx, reverse))
        order = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, len(self.frames))
        for f in order:
            yield FrameResult(f, list(self.ids), np.stack(self.masks), np.linspace(0.9, 0.5, len(self.ids)))


def _clip(tmp_path, hole=False, name="VID01_s15_80_crop", scale=1):
    """Write a clip of `N` frames with DA3 depth, a slope with a hole when asked, and return its directory.

    The images are `scale` times the depth's size, as a clip's images are larger than its depth.
    """
    d = tmp_path / "clips" / name
    (d / "input_images").mkdir(parents=True)
    (d / "exports" / "mini_npz").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(N):
        Image.fromarray(rng.integers(0, 256, (H * scale, W * scale, 3), dtype=np.uint8)).save(
            d / "input_images" / f"{i:06d}.png")
    yy, xx = np.mgrid[0:H, 0:W]
    depth = np.stack([0.3 + 0.01 * xx + 0.005 * yy + 0.001 * i for i in range(N)]).astype(np.float32)
    if hole:
        depth[:, 5:8, 6:10] = 0.0
    K = np.tile(np.array([[20.0, 0, W / 2], [0, 20.0, H / 2], [0, 0, 1]]), (N, 1, 1))
    np.savez(d / "exports" / "mini_npz" / "results.npz", depth=depth, intrinsics=K, extrinsics=np.zeros((N, 3, 4)))
    return d


def _run(tmp_path, clip, **kw):
    seg, trk = kw.pop("segmenter", StandInSegmenter()), kw.pop("tracker", StandInTracker())
    args = dict(tag="t", rule="both_ways_from_centre", sam_input="rgb", track_base="rgb", seed_min_area=1)
    args.update(kw)
    return run_track(clip, tmp_path / "tracks", segmenter=seg, tracker=trk, **args), seg, trk


def test_both_ways_seeds_the_middle_frame_and_carries_both_ways(tmp_path):
    lab_dir, _, trk = _run(tmp_path, _clip(tmp_path))
    assert lab_dir.name == "track_rgb_t"
    assert trk.calls == [(2, False), (2, True)]
    assert sorted(p.name for p in lab_dir.glob("label_*.npy")) == [f"label_{i:04d}.npy" for i in range(N)]
    info = json.loads((lab_dir / "seed_info.json").read_text())
    assert (info["seed_frame"], info["bidir"], info["frames"], info["n_seed_regions"]) == (2, True, list(range(N)), 3)
    assert info["seed_input"]["edge_ring_masked"] is None
    assert (tmp_path / "tracks" / "VID01_s15_80_crop" / "viz" / "montage_track_rgb_t.png").is_file()


def test_forward_from_first_seeds_frame_0_and_carries_forwards(tmp_path):
    lab_dir, _, trk = _run(tmp_path, _clip(tmp_path), rule="forward_from_first")
    assert trk.calls == [(0, False)]
    info = json.loads((lab_dir / "seed_info.json").read_text())
    assert (info["seed_frame"], info["bidir"]) == (0, False)


def test_the_two_rules_place_the_seed_as_the_evaluator_reads_them():
    assert window("both_ways_from_centre", 30) == (list(range(30)), 15, True)
    assert window("forward_from_first", 30) == (list(range(30)), 0, False)
    with pytest.raises(ValueError, match="unknown rule"):
        window("seed_from_gt", 30)


def test_a_seed_region_under_the_minimum_is_dropped(tmp_path):
    lab_dir, _, trk = _run(tmp_path, _clip(tmp_path), seed_min_area=5)
    assert trk.ids == [1, 2]
    with pytest.raises(ValueError, match="no seed region"):
        _run(tmp_path, _clip(tmp_path, name="VID02_s15_80_crop"), seed_min_area=10_000)


def test_the_edge_ring_setting_reaches_the_seed_and_the_tracker(tmp_path):
    clip = _clip(tmp_path, hole=True)
    lab_dir, seg, trk = _run(tmp_path, clip, sam_input="normal_edge", track_base="rgb_edge", mask_ring=False)
    depth, K = read_clip(clip, "da3")
    rgb = np.array(Image.open(clip / "input_images" / "000002.png").convert("RGB"))
    assert np.array_equal(seg.images[0], sam_input_image("normal_edge", depth[2], K[2], None, rgb, mask_ring=False))
    assert not np.array_equal(seg.images[0], sam_input_image("normal_edge", depth[2], K[2], None, rgb, mask_ring=True))
    assert np.array_equal(trk.frames[2], sam_input_image("rgb_edge", depth[2], K[2], None, rgb, mask_ring=False))
    assert json.loads((lab_dir / "seed_info.json").read_text())["seed_input"]["edge_ring_masked"] is False


def test_the_higher_score_is_painted_last_and_a_missing_score_goes_under():
    a = np.zeros((2, 2), bool)
    a[0] = True
    full = np.ones((2, 2), bool)
    r = FrameResult(0, [7, 8, 9], np.stack([full, a, full]), np.array([0.2, 0.9, np.nan], np.float32))
    assert collapse(r, (2, 2)).tolist() == [[8, 8], [7, 7]]


def test_the_first_mask_wins_and_regions_are_numbered_from_0():
    top = np.zeros((2, 3), bool)
    top[:, 0] = True
    under = np.ones((2, 3), bool)
    under[:, 2] = False
    assert paint_masks([top, under], (2, 3)).tolist() == [[1, 0, -1], [1, 0, -1]]
    assert paint_masks([], (2, 2)).tolist() == [[-1, -1], [-1, -1]]


def test_seed_masks_name_each_region_by_its_id_plus_one():
    labels = np.array([[0, 0, 1], [-1, 2, 2]])
    masks, ids = seed_masks(labels, 2)
    assert ids == [1, 3] and [int(m.sum()) for m in masks] == [2, 2]


def test_labels_left_by_an_earlier_run_are_refused_unless_replaced(tmp_path):
    clip = _clip(tmp_path)
    lab_dir, _, _ = _run(tmp_path, clip)
    stale = lab_dir / "label_0099.npy"
    np.save(stale, np.zeros((H, W), np.int16))
    with pytest.raises(ValueError, match="pass --overwrite"):
        _run(tmp_path, clip)
    _run(tmp_path, clip, overwrite=True)
    assert not stale.exists()


def test_a_missing_image_is_refused(tmp_path):
    clip = _clip(tmp_path)
    (clip / "input_images" / "000003.png").unlink()
    with pytest.raises(FileNotFoundError, match="no image"):
        _run(tmp_path, clip)


def test_pi3x_depth_is_put_on_the_da3_grid_and_scale(tmp_path):
    clip = _clip(tmp_path)
    d = np.full((N, H // 2, W // 2), 4.0, np.float32)
    K = np.tile(np.array([[10.0, 0, 4.0], [0, 10.0, 3.0], [0, 0, 1]]), (N, 1, 1))
    np.savez(clip / "exports" / "mini_npz" / "results__pi3x.npz", depth=d, intrinsics=K)
    depth, K2 = read_clip(clip, "pi3")
    da3, _ = read_clip(clip, "da3")
    assert depth.shape == (N, H, W)
    assert np.allclose(np.median(depth), np.median(da3))
    assert np.allclose(K2[0], [[20.0, 0, 8.0], [0, 20.0, 6.0], [0, 0, 1]])


def test_pi3x_depth_of_another_length_or_missing_is_refused(tmp_path):
    clip = _clip(tmp_path)
    with pytest.raises(FileNotFoundError, match="results__pi3x"):
        read_clip(clip, "pi3")
    np.savez(clip / "exports" / "mini_npz" / "results__pi3x.npz", depth=np.ones((N - 1, H, W), np.float32),
             intrinsics=np.zeros((N - 1, 3, 3)))
    with pytest.raises(ValueError, match="Pi3X has 4 frames"):
        read_clip(clip, "pi3")


@pytest.mark.parametrize("kw, match", [
    (dict(sam_input="rgb_shade"), "unknown input mode"),
    (dict(track_base="normals"), "unknown input mode"),
    (dict(depth_source="da2"), "unknown depth source"),
])
def test_an_unknown_setting_is_refused(tmp_path, kw, match):
    with pytest.raises(ValueError, match=match):
        _run(tmp_path, _clip(tmp_path), **kw)


def _seeds(tmp_path, clip, frame, labels):
    d = tmp_path / "seeds" / clip.name
    d.mkdir(parents=True)
    np.save(d / f"label_{frame:04d}.npy", labels)
    return str(tmp_path / "seeds")


def test_seed_regions_made_outside_are_read_in_place_of_the_segmenter(tmp_path):
    clip = _clip(tmp_path)
    labels = np.zeros((H, W), int)
    labels[H // 2:] = 4
    seeds = _seeds(tmp_path, clip, 2, labels)
    lab_dir, _, trk = _run(tmp_path, clip, segmenter=None, seed_labels=seeds)
    assert trk.ids == [1, 2]
    info = json.loads((lab_dir / "seed_info.json").read_text())
    assert (info["seed_source"], info["seed_labels"]) == ("external", seeds)
    assert info["seed_input"] == dict(produced_by="external", cache_path=seeds, points_per_side=None,
                                      seed_sam_kwargs=None, seed_edge_gain=None, seed_smooth=None,
                                      edge_ring_masked=None)


def test_seed_regions_are_read_from_a_condition_and_numbered_from_0(tmp_path):
    clip = _clip(tmp_path)
    labels = np.full((H, W), 7)
    labels[:, :3] = 3
    labels[0, 0] = -1
    d = tmp_path / "tracks" / clip.name / "track_rgb_pf_k10"
    d.mkdir(parents=True)
    np.save(d / "label_0002.npy", labels)
    template = str(tmp_path / "tracks" / "{clip}" / "track_rgb_pf_k10")
    lab_dir, _, trk = _run(tmp_path, clip, segmenter=None, seed_labels=template)
    assert trk.ids == [1, 2]
    assert np.array_equal(trk.masks[0], labels == 3)
    assert json.loads((lab_dir / "seed_info.json").read_text())["seed_labels"] == template


@pytest.mark.parametrize("frame, labels, err, match", [
    (0, np.zeros((H, W), int), FileNotFoundError, "made for seed frame 2"),
    (2, np.zeros((H, W + 1), int), ValueError, "not the depth's"),
    (2, np.full((H, W), -1), ValueError, "holds no region"),
])
def test_seed_regions_made_outside_that_do_not_fit_are_refused(tmp_path, frame, labels, err, match):
    clip = _clip(tmp_path)
    seeds = _seeds(tmp_path, clip, frame, labels)
    with pytest.raises(err, match=match):
        _run(tmp_path, clip, segmenter=None, seed_labels=seeds)


class ReverseDiffersTracker(StandInTracker):
    """Return the seed masks forwards, and backwards the masks shifted to other ids, skipping frame 0."""

    def propagate(self, start_frame_idx, reverse=False):
        if not reverse:
            yield from super().propagate(start_frame_idx)
            return
        self.calls.append((start_frame_idx, True))
        for f in range(start_frame_idx, 0, -1):
            yield FrameResult(f, [i + 10 for i in self.ids], np.stack(self.masks), np.ones(len(self.ids)))


def test_the_backward_pass_keeps_the_forward_seed_frame_and_the_record_lists_the_frames_labelled(tmp_path):
    lab_dir, _, trk = _run(tmp_path, _clip(tmp_path), tracker=ReverseDiffersTracker())
    assert set(np.unique(np.load(lab_dir / "label_0002.npy"))) == {1, 2, 3}
    assert set(np.unique(np.load(lab_dir / "label_0001.npy"))) == {11, 12, 13}
    assert json.loads((lab_dir / "seed_info.json").read_text())["frames"] == [1, 2, 3, 4]


class SilentTracker(StandInTracker):
    """Carry the seed to no frame."""

    def propagate(self, start_frame_idx, reverse=False):
        return iter(())


def test_a_tracker_that_carries_no_frame_is_refused(tmp_path):
    with pytest.raises(ValueError, match="carried no frame"):
        _run(tmp_path, _clip(tmp_path), tracker=SilentTracker())


def test_images_larger_than_the_depth_are_resized_to_it_with_area_averaging(tmp_path):
    clip = _clip(tmp_path, scale=2)
    _, seg, trk = _run(tmp_path, clip)
    rgb = np.array(Image.open(clip / "input_images" / "000002.png").convert("RGB"))
    small = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    assert np.array_equal(seg.images[0], small)
    assert trk.frames.shape == (N, H, W, 3) and np.array_equal(trk.frames[2], small)


def test_the_seed_settings_of_edges_and_smoothing_do_not_reach_the_tracker(tmp_path):
    clip = _clip(tmp_path, hole=True)
    _, seg, trk = _run(tmp_path, clip, sam_input="normal_edge", track_base="normal_edge", seed_edge_gain=1.0,
                       seed_smooth=False)
    depth, K = read_clip(clip, "da3")
    seed = sam_input_image("normal_edge", depth[2], K[2], None, None, edge_gain=1.0, smooth=False, mask_ring=True)
    tracked = sam_input_image("normal_edge", depth[2], K[2], None, None, edge_gain=0.85, smooth=True, mask_ring=True)
    assert not np.array_equal(seed, tracked)
    assert np.array_equal(seg.images[0], seed)
    assert np.array_equal(trk.frames[2], tracked)


def test_seed_smoothing_is_turned_off_on_its_own(tmp_path):
    clip = _clip(tmp_path, hole=True)
    _, seg, _ = _run(tmp_path, clip, sam_input="normal_edge", seed_smooth=False)
    depth, K = read_clip(clip, "da3")
    off = sam_input_image("normal_edge", depth[2], K[2], None, None, smooth=False, mask_ring=True)
    assert not np.array_equal(off, sam_input_image("normal_edge", depth[2], K[2], None, None, smooth=True, mask_ring=True))
    assert np.array_equal(seg.images[0], off)


def test_a_clip_without_depth_is_refused(tmp_path):
    clip = _clip(tmp_path)
    (clip / "exports" / "mini_npz" / "results.npz").unlink()
    with pytest.raises(FileNotFoundError, match="run the depth stage first"):
        read_clip(clip, "da3")


def test_pi3x_depth_without_a_value_is_filled_with_its_median(tmp_path):
    clip = _clip(tmp_path)
    d = np.full((N, H, W), 4.0, np.float32)
    d[:, 0, 0] = np.nan
    K = np.tile(np.array([[20.0, 0, 8.0], [0, 20.0, 6.0], [0, 0, 1]]), (N, 1, 1))
    np.savez(clip / "exports" / "mini_npz" / "results__pi3x.npz", depth=d, intrinsics=K)
    depth, _ = read_clip(clip, "pi3")
    assert np.isfinite(depth).all() and depth[0, 0, 0] == np.median(depth)


class FailingTracker(StandInTracker):
    def propagate(self, start_frame_idx, reverse=False):
        raise RuntimeError("out of memory")


def test_a_run_that_fails_keeps_the_labels_an_earlier_run_left(tmp_path):
    clip = _clip(tmp_path)
    lab_dir, _, _ = _run(tmp_path, clip)
    montage = tmp_path / "tracks" / clip.name / "viz" / "montage_track_rgb_t.png"
    before = sorted(p.name for p in lab_dir.iterdir())
    with pytest.raises(RuntimeError, match="out of memory"):
        _run(tmp_path, clip, tracker=FailingTracker(), overwrite=True)
    assert sorted(p.name for p in lab_dir.iterdir()) == before and montage.is_file()
    assert [p.name for p in (tmp_path / "tracks" / clip.name).iterdir() if "partial" in p.name] == []



def test_a_run_that_fails_while_writing_leaves_no_partial_labels(tmp_path, monkeypatch):
    clip = _clip(tmp_path)
    lab_dir, _, _ = _run(tmp_path, clip)
    before = sorted(p.name for p in lab_dir.iterdir())

    def full_disk(*a):
        raise OSError("no space left on device")

    monkeypatch.setattr(track, "write_montage", full_disk)
    with pytest.raises(OSError, match="no space"):
        _run(tmp_path, clip, overwrite=True)
    assert sorted(p.name for p in lab_dir.iterdir()) == before
    assert [p.name for p in (tmp_path / "tracks" / clip.name).iterdir() if "partial" in p.name] == []

class _StandInSession(StandInTracker):
    device = "cpu"

    def __init__(self, *a, **kw):
        super().__init__()


def _main(monkeypatch, *argv):
    monkeypatch.setattr(track, "Sam3VideoInstanceSession", _StandInSession)
    monkeypatch.setattr(track, "SeedSegmenter", lambda ckpt, device: StandInSegmenter())
    monkeypatch.setattr(sys, "argv", ["track", *argv])
    track.main()


def _argv(tmp_path, *extra):
    return ["--input-dir", str(tmp_path / "clips"), "--tracks-root", str(tmp_path / "tracks"), "--tag", "t",
            "--rule", "forward_from_first", "--sam-input", "rgb", "--track-base", "rgb", "--seed-min-area", "1",
            *extra]


def test_main_runs_every_clip_with_depth(tmp_path, monkeypatch):
    _clip(tmp_path)
    _clip(tmp_path, name="VID02_s15_80_crop")
    _main(monkeypatch, *_argv(tmp_path, "--sam-ckpt", "sam.pth"))
    assert sorted(p.name for p in (tmp_path / "tracks").iterdir()) == ["VID01_s15_80_crop", "VID02_s15_80_crop"]


def test_main_refuses_a_mistyped_clip_before_the_models_load(tmp_path, monkeypatch):
    _clip(tmp_path)
    with pytest.raises(SystemExit, match="no clip with depth .* at: VID99"):
        _main(monkeypatch, *_argv(tmp_path, "--sam-ckpt", "sam.pth", "--clips", "VID99"))


def test_main_refuses_to_cut_seeds_without_the_sam_weights(tmp_path, monkeypatch):
    _clip(tmp_path)
    with pytest.raises(SystemExit, match="--sam-ckpt is needed"):
        _main(monkeypatch, *_argv(tmp_path))


def test_main_exits_with_an_error_when_a_clip_fails(tmp_path, monkeypatch):
    _clip(tmp_path)
    bad = _clip(tmp_path, name="VID02_s15_80_crop")
    (bad / "input_images" / "000003.png").unlink()
    with pytest.raises(SystemExit, match="1 of 2 clip.* failed: VID02_s15_80_crop"):
        _main(monkeypatch, *_argv(tmp_path, "--sam-ckpt", "sam.pth"))
    assert (tmp_path / "tracks" / "VID01_s15_80_crop" / "track_rgb_t").is_dir()
