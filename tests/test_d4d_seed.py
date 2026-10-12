"""Run `pipeline.d4d_seed` through a stand-in segmenter, and check what it writes and what it refuses.

The stand-in cuts every image into a left half (region 0) and a right half (1), and keeps the images it is
given. Each refusal has a test that plants its fault:

- a side's record made from another depth file, or with another ring setting;
- a frame that cannot be read;
- a depth directory with no side's file.
"""

import json

import cv2
import numpy as np
import pytest

pytest.importorskip("matplotlib")

from pipeline import d4d_seed  # noqa: E402
from surgical_core.geometry.render import sam_input_image  # noqa: E402

H, W = 12, 16
STEM = "specimen_1__s_a__Clip_1__end"


class StandInSegmenter:
    def __init__(self):
        self.images = []

    def label_map(self, image, points_per_side):
        self.images.append((image.copy(), points_per_side))
        labels = np.zeros(image.shape[:2], int)
        labels[:, W // 2:] = 1
        return labels


def write_side(tmp_path, frame_size=(32, 24)) -> tuple:
    """Write a frame and a side's depth file that names it; return the depth file and the frame."""
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (frame_size[1], frame_size[0], 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "frame.png"), frame)
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir(exist_ok=True)
    depth = (1.0 + rng.random((H, W))).astype(np.float32)
    K = np.array([[20.0, 0, W / 2], [0, 20.0, H / 2], [0, 0, 1]], np.float32)
    path = depth_dir / f"{STEM}.npz"
    np.savez_compressed(path, depth=depth, conf=depth, intrinsics=K, extrinsics=np.eye(4, dtype=np.float32)[:3],
                        frame_index=np.int32(7), frame_minus_gt_s=np.float32(-3.0), window=np.int32(30),
                        process_res=np.int32(504), image_path=np.str_(str(tmp_path / "frame.png")))
    return path, frame


def test_a_side_is_cut_from_the_normal_edge_input_and_recorded(tmp_path):
    path, frame = write_side(tmp_path)
    seg = StandInSegmenter()
    out = tmp_path / "seed"
    out.mkdir()
    assert d4d_seed.run_side(path, out, seg, True, "/w/sam_vit_h_4b8939.pth") == 2

    # The segmenter saw the normal_edge input of the side's depth and frame, at full edge darkness, unsmoothed.
    with np.load(path) as z:
        rgb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (W, H), interpolation=cv2.INTER_AREA)
        want = sam_input_image("normal_edge", z["depth"], z["intrinsics"], np.zeros((H, W), np.float32), rgb,
                               edge_gain=1.0, smooth=False, mask_ring=True)
    (image, pps), = seg.images
    assert pps == 24 and np.array_equal(image, want)

    labels = np.load(out / f"{STEM}.npy")
    assert labels.dtype == np.int32 and labels[0, 0] == 0 and labels[0, -1] == 1
    record = json.loads((out / f"{STEM}.json").read_text())
    assert record == {"depth_file": f"{STEM}.npz", "depth_sha256": d4d_seed.file_sha256(path), "window": 30,
                      "process_res": 504, "frame_index": 7, "sam_input": "normal_edge", "points_per_side": 24,
                      "edge_gain": 1.0, "smooth": False, "edge_ring_masked": True,
                      "sam_ckpt": "sam_vit_h_4b8939.pth", "n_regions": 2}
    assert sorted(p.name for p in out.iterdir()) == [f"{STEM}.json", f"{STEM}.npy"]


def test_a_frame_that_cannot_be_read_is_refused(tmp_path):
    path, _ = write_side(tmp_path)
    (tmp_path / "frame.png").unlink()
    with pytest.raises(FileNotFoundError, match="cannot read the frame"):
        d4d_seed.run_side(path, tmp_path, StandInSegmenter(), True, "sam.pth")


def run_main(tmp_path, monkeypatch, extra=()):
    """Run `main` with the stand-in segmenter, and return it."""
    seg = StandInSegmenter()
    monkeypatch.setattr(d4d_seed, "SeedSegmenter", lambda ckpt, device: seg)
    d4d_seed.main(["--depth-dir", str(tmp_path / "depth"), "--out", str(tmp_path / "seed"), "--sam-ckpt", "sam.pth",
                   *extra])
    return seg


def test_main_skips_a_side_already_made_with_its_depth_and_ring(tmp_path, monkeypatch):
    write_side(tmp_path)
    assert len(run_main(tmp_path, monkeypatch).images) == 1
    assert run_main(tmp_path, monkeypatch).images == []


def test_main_refuses_a_side_made_with_another_ring_setting(tmp_path, monkeypatch):
    write_side(tmp_path)
    run_main(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="edge_ring_masked is True, where this run makes False"):
        run_main(tmp_path, monkeypatch, extra=["--keep-edge-ring"])


def test_main_refuses_a_side_made_from_another_depth(tmp_path, monkeypatch):
    path, _ = write_side(tmp_path)
    run_main(tmp_path, monkeypatch)
    with np.load(path) as z:
        fields = {k: z[k] for k in z.files}
    fields["depth"] = fields["depth"] + 1
    np.savez_compressed(path, **fields)
    with pytest.raises(ValueError, match="made from another depth"):
        run_main(tmp_path, monkeypatch)


def test_main_refuses_a_depth_directory_with_no_side(tmp_path, monkeypatch):
    (tmp_path / "depth").mkdir()
    with pytest.raises(SystemExit, match="no side's depth"):
        run_main(tmp_path, monkeypatch)
