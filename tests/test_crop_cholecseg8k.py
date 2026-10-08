"""Test the CholecSeg8k crop on a clip made up for the test, and the rectangles it reads.

Each refusal has a test that plants its fault:

- a malformed rectangle;
- a clip without a rectangle;
- an image of another size;
- a cut clip left by an earlier run.
"""

import json

import numpy as np
import pytest
from PIL import Image

from pipeline.crop_cholecseg8k import crop_clip, load_rects

RECT = {"y0": 5, "y1": 25, "x0": 10, "x1": 30, "src_h": 30, "src_w": 40}


@pytest.fixture
def clip(tmp_path):
    """Write a clip of two frames, one with a colour mask; return its directory."""
    d = tmp_path / "VID07_s15_80"
    (d / "input_images").mkdir(parents=True)
    (d / "seg_masks").mkdir()
    rng = np.random.default_rng(0)
    for i in range(2):
        Image.fromarray(rng.integers(0, 256, (30, 40, 3), dtype=np.uint8)).save(d / "input_images" / f"{i:06d}.png")
    Image.fromarray(rng.integers(0, 256, (30, 40, 3), dtype=np.uint8)).save(d / "seg_masks" / "000000_color_mask.png")
    (d / "frame_manifest.json").write_text(json.dumps(
        {"video_id": "VID07", "video_num": 7, "frames": [{"seq_idx": 0}, {"seq_idx": 1}]}))
    return d


def test_images_and_masks_are_cut_to_the_rectangle(clip):
    dst = crop_clip(clip, {"VID07_s15_80": RECT})
    assert dst.name == "VID07_s15_80_crop"
    for sub, name in (("input_images", "000001.png"), ("seg_masks", "000000_color_mask.png")):
        src = np.asarray(Image.open(clip / sub / name))
        assert np.array_equal(np.asarray(Image.open(dst / sub / name)), src[5:25, 10:30])
    man = json.loads((dst / "frame_manifest.json").read_text())
    assert man["crop_info"] == RECT == json.loads((dst / "crop_info.json").read_text())
    assert [f["image_size"] for f in man["frames"]] == [[20, 20], [20, 20]]


def test_a_clip_without_a_rectangle_is_refused(clip):
    with pytest.raises(KeyError, match="VID07_s15_80 has no rectangle"):
        crop_clip(clip, {"VID07_s15_240": RECT})


def test_an_image_of_another_size_is_refused(clip):
    Image.fromarray(np.zeros((20, 40, 3), np.uint8)).save(clip / "input_images" / "000001.png")
    with pytest.raises(ValueError, match="not the frame's 40x30"):
        crop_clip(clip, {"VID07_s15_80": RECT})


def test_an_earlier_cut_is_refused_unless_replaced(clip):
    dst = crop_clip(clip, {"VID07_s15_80": RECT})
    stale = dst / "input_images" / "000099.png"
    stale.write_bytes(b"left by a longer run")
    with pytest.raises(ValueError, match="pass --overwrite"):
        crop_clip(clip, {"VID07_s15_80": RECT})
    crop_clip(clip, {"VID07_s15_80": RECT}, overwrite=True)
    assert not stale.exists()


@pytest.mark.parametrize("bad", [
    {k: v for k, v in RECT.items() if k != "x1"},
    dict(RECT, y0=5.0),
    dict(RECT, y1=31),
    dict(RECT, x0=30),
])
def test_a_rectangle_that_is_not_one_is_refused(tmp_path, bad):
    path = tmp_path / "crop_rects.json"
    path.write_text(json.dumps({"VID07": bad}))
    with pytest.raises(ValueError, match="VID07"):
        load_rects(str(path))
