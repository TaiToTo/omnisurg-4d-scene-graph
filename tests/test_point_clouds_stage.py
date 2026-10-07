"""Run `pipeline.point_clouds` on a synthetic depth bundle, and check what it writes and what it refuses.

Writing the point clouds needs trimesh, the `render` extra; those tests
skip without it.
"""

import json

import numpy as np
import pytest
from PIL import Image

from pipeline.point_clouds import run_point_clouds, update_manifest


def make_clip(tmp_path, n=3, zero_frame=None, seq_idx=True):
    c = tmp_path / "clip"
    (c / "input_images").mkdir(parents=True)
    for i in range(n):
        Image.fromarray(np.full((12, 16, 3), 40 * i, np.uint8)).save(c / "input_images" / f"{i:06d}.png")
    frames = [{"seq_idx": i, "native_frame": 100 + i} if seq_idx else {"native_frame": 100 + i} for i in range(n)]
    (c / "frame_manifest.json").write_text(json.dumps({"n_frames": n, "frames": frames}, indent=2))
    depth = np.stack([np.full((6, 8), 1.0 + i, np.float32) for i in range(n)])
    if zero_frame is not None:
        depth[zero_frame] = 0.0
    K = np.tile(np.array([[10.0, 0, 4], [0, 10.0, 3], [0, 0, 1]], np.float32), (n, 1, 1))
    E = np.tile(np.eye(4, dtype=np.float32)[:3], (n, 1, 1))
    E[:, 0, 3] = np.arange(n)
    (c / "exports" / "mini_npz").mkdir(parents=True)
    np.savez(c / "exports" / "mini_npz" / "results.npz", depth=depth, conf=np.ones_like(depth), extrinsics=E,
             intrinsics=K)
    return c


def test_each_frame_gets_a_cloud_and_its_placement_in_the_manifest(tmp_path):
    pytest.importorskip("trimesh")
    clip = make_clip(tmp_path)
    assert run_point_clouds(clip) == 3
    assert sorted(p.name for p in (clip / "pc_vis").iterdir()) == [f"frame_{i:04d}.glb" for i in range(3)]
    m = json.loads((clip / "frame_manifest.json").read_text())
    assert list(m["frames"][1]) == ["seq_idx", "native_frame", "glb_centroid", "camera_pos_glb",
                                    "camera_forward_glb", "camera_up_glb"]
    # The camera moves along x with an identity rotation, so it sits at -t.
    assert m["frames"][2]["camera_pos_glb"] == pytest.approx([-2.0, 0.0, 0.0])


def test_a_frame_with_no_usable_depth_gets_no_cloud_and_the_rest_do(tmp_path):
    pytest.importorskip("trimesh")
    clip = make_clip(tmp_path, zero_frame=1)
    assert run_point_clouds(clip) == 2
    m = json.loads((clip / "frame_manifest.json").read_text())
    assert "glb_centroid" not in m["frames"][1] and "glb_centroid" in m["frames"][2]


def test_a_clip_without_a_depth_bundle_or_an_image_is_refused(tmp_path):
    clip = make_clip(tmp_path)
    (clip / "input_images" / "000002.png").unlink()
    with pytest.raises(FileNotFoundError, match="000002.png"):
        run_point_clouds(clip)
    assert not (clip / "pc_vis").exists(), "refused before any cloud is written"
    (clip / "exports" / "mini_npz" / "results.npz").unlink()
    with pytest.raises(FileNotFoundError, match="depth stage"):
        run_point_clouds(clip)


def test_a_manifest_without_seq_idx_or_none_at_all_is_refused(tmp_path):
    clip = make_clip(tmp_path, seq_idx=False)
    with pytest.raises(ValueError, match="seq_idx"):
        update_manifest(clip, {0: {"glb_centroid": [0.0, 0.0, 0.0]}})
    (clip / "frame_manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="frame_manifest.json"):
        update_manifest(clip, {0: {"glb_centroid": [0.0, 0.0, 0.0]}})
