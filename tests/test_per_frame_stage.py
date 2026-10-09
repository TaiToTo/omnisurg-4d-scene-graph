"""Run `pipeline.per_frame` through a stand-in segmenter, and check what it writes and what it refuses.

The stand-in cuts every image into fixed regions, so these tests need no
torch and no weights. The segmenter inputs need the `render` extra.
"""

import json
import os
import sys

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("matplotlib")

import pipeline.per_frame as per_frame  # noqa: E402
from pipeline.per_frame import run_per_frame  # noqa: E402
from pipeline.track import SEED_SAM_KWARGS, read_clip  # noqa: E402
from surgical_core.geometry.render import global_depth01, sam_input_image  # noqa: E402

H, W, N = 12, 16, 4

# Settings off every default, so that a setting lost on its way to the input or the record shows.
SETTINGS = dict(sam_input="normal_edge", depth_source="pi3", points_per_side=8, seed_min_area=5, edge_gain=0.85,
                mask_ring=False)
ARGV = ["--sam-input", "normal_edge", "--depth-source", "pi3", "--points-per-side", "8", "--seed-min-area", "5",
        "--edge-gain", "0.85", "--keep-edge-ring"]
# The record those settings give, with the workbench's keys in its order.
RECORD = dict(clip="VID01_s15_80_crop", tag="t", seed_source="per_frame", track_base="rgb", sam_input="normal_edge",
              frames="all", depth_source="pi3", seed_min_area=5, seed_topk=0, point_grids=None,
              seed_input=dict(produced_by="sam", cache_path=None, points_per_side=8, seed_sam_kwargs=SEED_SAM_KWARGS,
                              seed_edge_gain=0.85, seed_smooth=False, edge_ring_masked=False))


class StandInSegmenter:
    """Cut every image into a left half (region 0), a right half (1) and a 2x2 corner (2)."""

    device = "cpu"

    def __init__(self):
        self.images, self.grids = [], []

    def label_map(self, image, points_per_side):
        self.images.append(image.copy())
        self.grids.append(points_per_side)
        labels = np.zeros(image.shape[:2], int)
        labels[:, W // 2:] = 1
        labels[:2, :2] = 2
        return labels


class FailingSegmenter(StandInSegmenter):
    """Cut two frames, then fail on the third, as a run that runs out of memory does."""

    def label_map(self, image, points_per_side):
        if len(self.images) == 2:
            raise RuntimeError("out of memory")
        return super().label_map(image, points_per_side)


def _clip(tmp_path, name="VID01_s15_80_crop"):
    """Write a clip of `N` frames with DA3 depth and Pi3X depth, and return its directory.

    The DA3 depth is a slope with a hole, 0.05 farther in each frame; the Pi3X depth is a curve without one.
    """
    d = tmp_path / "clips" / name
    (d / "input_images").mkdir(parents=True)
    (d / "exports" / "mini_npz").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(N):
        Image.fromarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8)).save(d / "input_images" / f"{i:06d}.png")
    yy, xx = np.mgrid[0:H, 0:W]
    depth = np.stack([0.3 + 0.01 * xx + 0.005 * yy + 0.05 * i for i in range(N)]).astype(np.float32)
    depth[:, 5:8, 6:10] = 0.0
    K = np.tile(np.array([[20.0, 0, W / 2], [0, 20.0, H / 2], [0, 0, 1]]), (N, 1, 1))
    np.savez(d / "exports" / "mini_npz" / "results.npz", depth=depth, intrinsics=K)
    pi3 = np.stack([1.0 + 0.04 * yy * yy for _ in range(N)]).astype(np.float32)
    np.savez(d / "exports" / "mini_npz" / "results__pi3x.npz", depth=pi3, intrinsics=K)
    return d


def _files(d):
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


def test_every_frame_is_cut_and_written_under_the_trackers_ids(tmp_path):
    seg = StandInSegmenter()
    lab_dir = run_per_frame(_clip(tmp_path), tmp_path / "tracks", "t", seg, "rgb", seed_min_area=5)
    assert lab_dir.name == "track_rgb_t" and len(seg.images) == N
    assert sorted(p.name for p in lab_dir.iterdir()) == [f"label_{i:04d}.npy" for i in range(N)] + ["seed_info.json"]
    labels = np.load(lab_dir / "label_0003.npy")
    assert labels.dtype == np.int16
    assert set(np.unique(labels)) == {-1, 1, 2}
    assert (labels[:2, :2] == -1).all()
    info = json.loads((lab_dir / "seed_info.json").read_text())
    assert (info["seed_source"], info["frames"], info["seed_min_area"]) == ("per_frame", "all", 5)
    assert info["seed_input"]["seed_smooth"] is False and info["seed_input"]["edge_ring_masked"] is None


def test_the_record_holds_every_setting_the_run_was_given(tmp_path):
    seg = StandInSegmenter()
    lab_dir = run_per_frame(_clip(tmp_path), tmp_path / "tracks", "t", seg, **SETTINGS)
    info = json.loads((lab_dir / "seed_info.json").read_text())
    assert info == RECORD
    assert list(info) == list(RECORD) and list(info["seed_input"]) == list(RECORD["seed_input"])
    assert seg.grids == [8] * N


def test_the_input_comes_from_the_chosen_depth_with_the_given_settings(tmp_path):
    clip = _clip(tmp_path)
    seg = StandInSegmenter()
    run_per_frame(clip, tmp_path / "tracks", "t", seg, **SETTINGS)
    depth, K = read_clip(clip, "pi3")
    for i in range(N):
        want = sam_input_image("normal_edge", depth[i], K[i], None, None, edge_gain=0.85, smooth=False,
                               mask_ring=False)
        assert np.array_equal(seg.images[i], want)
    da3, K3 = read_clip(clip, "da3")
    assert not np.array_equal(seg.images[0], sam_input_image("normal_edge", da3[0], K3[0], None, None,
                                                             edge_gain=0.85, smooth=False, mask_ring=False))


def test_the_edges_are_dark_and_their_ring_masked_by_default(tmp_path):
    clip = _clip(tmp_path)
    seg = StandInSegmenter()
    lab_dir = run_per_frame(clip, tmp_path / "tracks", "t", seg, "normal_edge")
    depth, K = read_clip(clip, "da3")
    want = sam_input_image("normal_edge", depth[1], K[1], None, None, edge_gain=1.0, smooth=False, mask_ring=True)
    assert np.array_equal(seg.images[1], want)
    seed_input = json.loads((lab_dir / "seed_info.json").read_text())["seed_input"]
    assert (seed_input["seed_edge_gain"], seed_input["edge_ring_masked"]) == (1.0, True)


def test_the_depth_input_is_normalised_over_the_whole_clip(tmp_path):
    clip = _clip(tmp_path)
    seg = StandInSegmenter()
    run_per_frame(clip, tmp_path / "tracks", "t", seg, "depth")
    depth, K = read_clip(clip, "da3")
    depth01 = global_depth01(depth)
    for i in range(N):
        assert np.array_equal(seg.images[i], sam_input_image("depth", depth[i], K[i], depth01[i], None,
                                                             mask_ring=True))
    alone = global_depth01(depth[-1:])[0]
    assert not np.array_equal(seg.images[-1], sam_input_image("depth", depth[-1], K[-1], alone, None, mask_ring=True))


def test_labels_left_by_an_earlier_run_are_refused_unless_replaced(tmp_path):
    clip = _clip(tmp_path)
    lab_dir = run_per_frame(clip, tmp_path / "tracks", "t", StandInSegmenter(), "rgb")
    stale = lab_dir / "label_0099.npy"
    np.save(stale, np.zeros((H, W), np.int16))
    with pytest.raises(ValueError, match="pass --overwrite"):
        run_per_frame(clip, tmp_path / "tracks", "t", StandInSegmenter(), "rgb")
    run_per_frame(clip, tmp_path / "tracks", "t", StandInSegmenter(), "rgb", overwrite=True)
    assert not stale.exists()


def test_a_run_that_fails_keeps_the_labels_an_earlier_run_left(tmp_path):
    clip = _clip(tmp_path)
    lab_dir = run_per_frame(clip, tmp_path / "tracks", "t", StandInSegmenter(), "rgb", seed_min_area=5)
    before = _files(lab_dir)
    with pytest.raises(RuntimeError, match="out of memory"):
        run_per_frame(clip, tmp_path / "tracks", "t", FailingSegmenter(), "normal", overwrite=True)
    assert _files(lab_dir) == before
    assert sorted(p.name for p in (tmp_path / "tracks" / clip.name).iterdir()) == ["track_rgb_t"]


def test_a_run_that_fails_leaves_no_labels(tmp_path):
    clip = _clip(tmp_path)
    (clip / "input_images" / "000002.png").unlink()
    with pytest.raises(FileNotFoundError, match="no image"):
        run_per_frame(clip, tmp_path / "tracks", "t", StandInSegmenter(), "rgb")
    assert list((tmp_path / "tracks" / clip.name).iterdir()) == []


def test_a_partial_directory_a_killed_run_left_is_not_carried_over(tmp_path):
    clip = _clip(tmp_path)
    partial = tmp_path / "tracks" / clip.name / ".track_rgb_t.partial"
    partial.mkdir(parents=True)
    np.save(partial / "label_0099.npy", np.zeros((H, W), np.int16))
    lab_dir = run_per_frame(clip, tmp_path / "tracks", "t", StandInSegmenter(), "rgb")
    assert "label_0099.npy" not in _files(lab_dir) and not partial.exists()


@pytest.mark.parametrize("record", [dict(seed_source="sam", seed_frame=2), None])
def test_labels_another_stage_wrote_are_not_replaced(tmp_path, record):
    clip = _clip(tmp_path)
    lab_dir = tmp_path / "tracks" / clip.name / "track_rgb_t"
    lab_dir.mkdir(parents=True)
    np.save(lab_dir / "label_0000.npy", np.zeros((H, W), np.int16))
    if record is not None:
        (lab_dir / "seed_info.json").write_text(json.dumps(record))
    before = _files(lab_dir)
    with pytest.raises(ValueError, match="did not write it; it is not replaced"):
        run_per_frame(clip, tmp_path / "tracks", "t", StandInSegmenter(), "rgb", overwrite=True)
    assert _files(lab_dir) == before


@pytest.mark.parametrize("kw, match", [(dict(sam_input="rgb_flat_l0"), "unknown input mode"),
                                       (dict(sam_input="rgb", depth_source="da2"), "unknown depth source")])
def test_an_unknown_setting_is_refused_before_the_depth_is_read(tmp_path, kw, match):
    clip = tmp_path / "clips" / "VID01_s15_80_crop"
    clip.mkdir(parents=True)
    with pytest.raises(ValueError, match=match):
        run_per_frame(clip, tmp_path / "tracks", "t", StandInSegmenter(), **kw)


def _main(monkeypatch, argv, segmenter=None):
    """Run `main` with `segmenter` in place of SAM; without one, loading the model fails the test."""
    def load(ckpt, device):
        if segmenter is None:
            raise AssertionError("the model loaded")
        segmenter.loaded = (ckpt, device, os.environ.get("CUDA_VISIBLE_DEVICES"))
        return segmenter

    monkeypatch.setattr(per_frame, "SeedSegmenter", load)
    monkeypatch.setattr(sys, "argv", ["per_frame", *argv])
    per_frame.main()


def _argv(tmp_path, *extra):
    return ["--input-dir", str(tmp_path / "clips"), "--tracks-root", str(tmp_path / "tracks"), "--tag", "t",
            "--sam-ckpt", "sam.pth", "--sam-input", "rgb", "--seed-min-area", "1", *extra]


def test_main_passes_every_option_to_the_stage(tmp_path, monkeypatch):
    clip = _clip(tmp_path)
    _main(monkeypatch, _argv(tmp_path), StandInSegmenter())
    seg = StandInSegmenter()
    _main(monkeypatch, _argv(tmp_path, *ARGV, "--device", "cpu", "--overwrite"), seg)
    assert seg.loaded[:2] == ("sam.pth", "cpu") and seg.grids == [8] * N
    info = json.loads((tmp_path / "tracks" / clip.name / "track_rgb_t" / "seed_info.json").read_text())
    assert info == RECORD


def test_main_runs_every_clip_with_depth(tmp_path, monkeypatch):
    _clip(tmp_path)
    _clip(tmp_path, name="VID02_s15_80_crop")
    (tmp_path / "clips" / "notes").mkdir()
    _main(monkeypatch, _argv(tmp_path), StandInSegmenter())
    assert sorted(p.name for p in (tmp_path / "tracks").iterdir()) == ["VID01_s15_80_crop", "VID02_s15_80_crop"]


def test_main_refuses_a_mistyped_clip_before_the_model_loads(tmp_path, monkeypatch):
    _clip(tmp_path)
    with pytest.raises(SystemExit, match="no clip with depth .* at: VID99"):
        _main(monkeypatch, _argv(tmp_path, "--clips", "VID99"))


def test_main_refuses_a_directory_without_clips_before_the_model_loads(tmp_path, monkeypatch):
    (tmp_path / "clips").mkdir()
    with pytest.raises(SystemExit, match="no clip with depth under"):
        _main(monkeypatch, _argv(tmp_path))


def test_main_exits_with_an_error_when_a_clip_fails(tmp_path, monkeypatch):
    _clip(tmp_path)
    bad = _clip(tmp_path, name="VID02_s15_80_crop")
    (bad / "input_images" / "000003.png").unlink()
    with pytest.raises(SystemExit, match="1 of 2 clip.* failed: VID02_s15_80_crop"):
        _main(monkeypatch, _argv(tmp_path), StandInSegmenter())
    assert (tmp_path / "tracks" / "VID01_s15_80_crop" / "track_rgb_t").is_dir()


def test_main_selects_the_gpu_before_the_model_loads(tmp_path, monkeypatch):
    _clip(tmp_path)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    seg = StandInSegmenter()
    _main(monkeypatch, _argv(tmp_path, "--gpu", "2"), seg)
    assert seg.loaded == ("sam.pth", "auto", "2")


def test_main_refuses_a_gpu_once_torch_is_imported(tmp_path, monkeypatch):
    _clip(tmp_path)
    monkeypatch.setitem(sys.modules, "torch", object())
    with pytest.raises(SystemExit, match="already imported"):
        _main(monkeypatch, _argv(tmp_path, "--gpu", "2"))
