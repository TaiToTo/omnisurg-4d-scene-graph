"""Run `pipeline.per_frame` through a stand-in segmenter, and check what it writes and what it refuses.

The stand-in cuts every image into fixed regions, so these tests need no
torch and no weights. The segmenter inputs need the `render` extra.
"""

import json

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("matplotlib")

from pipeline.per_frame import run_per_frame  # noqa: E402
from pipeline.track import read_clip  # noqa: E402
from surgical_core.geometry.render import sam_input_image  # noqa: E402

H, W, N = 12, 16, 4


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


def _clip(tmp_path, name="VID01_s15_80_crop"):
    d = tmp_path / "clips" / name
    (d / "input_images").mkdir(parents=True)
    (d / "exports" / "mini_npz").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(N):
        Image.fromarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8)).save(d / "input_images" / f"{i:06d}.png")
    yy, xx = np.mgrid[0:H, 0:W]
    depth = np.stack([0.3 + 0.01 * xx + 0.005 * yy for _ in range(N)]).astype(np.float32)
    depth[:, 5:8, 6:10] = 0.0
    K = np.tile(np.array([[20.0, 0, W / 2], [0, 20.0, H / 2], [0, 0, 1]]), (N, 1, 1))
    np.savez(d / "exports" / "mini_npz" / "results.npz", depth=depth, intrinsics=K)
    return d


def test_every_frame_is_cut_and_written_under_the_trackers_ids(tmp_path):
    seg = StandInSegmenter()
    lab_dir = run_per_frame(_clip(tmp_path), tmp_path / "tracks", "t", seg, "rgb", seed_min_area=5)
    assert lab_dir.name == "track_rgb_t" and len(seg.images) == N
    labels = np.load(lab_dir / "label_0003.npy")
    assert labels.dtype == np.int16
    assert set(np.unique(labels)) == {-1, 1, 2}
    assert (labels[:2, :2] == -1).all()
    info = json.loads((lab_dir / "seed_info.json").read_text())
    assert (info["seed_source"], info["frames"], info["seed_min_area"]) == ("per_frame", "all", 5)
    assert info["seed_input"]["seed_smooth"] is False and info["seed_input"]["edge_ring_masked"] is None


def test_the_input_is_not_smoothed_and_carries_the_ring_setting(tmp_path):
    clip = _clip(tmp_path)
    seg = StandInSegmenter()
    lab_dir = run_per_frame(clip, tmp_path / "tracks", "t", seg, "normal_edge", mask_ring=False)
    depth, K = read_clip(clip, "da3")
    want = sam_input_image("normal_edge", depth[1], K[1], None, None, edge_gain=1.0, smooth=False, mask_ring=False)
    assert np.array_equal(seg.images[1], want)
    assert json.loads((lab_dir / "seed_info.json").read_text())["seed_input"]["edge_ring_masked"] is False


def test_labels_left_by_an_earlier_run_are_refused_unless_replaced(tmp_path):
    clip = _clip(tmp_path)
    lab_dir = run_per_frame(clip, tmp_path / "tracks", "t", StandInSegmenter(), "rgb")
    stale = lab_dir / "label_0099.npy"
    np.save(stale, np.zeros((H, W), np.int16))
    with pytest.raises(ValueError, match="pass --overwrite"):
        run_per_frame(clip, tmp_path / "tracks", "t", StandInSegmenter(), "rgb")
    run_per_frame(clip, tmp_path / "tracks", "t", StandInSegmenter(), "rgb", overwrite=True)
    assert not stale.exists()


@pytest.mark.parametrize("kw, match", [(dict(sam_input="rgb_flat_l0"), "unknown input mode"),
                                       (dict(sam_input="rgb", depth_source="da2"), "unknown depth source")])
def test_an_unknown_setting_is_refused(tmp_path, kw, match):
    with pytest.raises(ValueError, match=match):
        run_per_frame(_clip(tmp_path), tmp_path / "tracks", "t", StandInSegmenter(), **kw)
