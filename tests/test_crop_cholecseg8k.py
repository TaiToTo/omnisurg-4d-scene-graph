"""Test the CholecSeg8k crop on a clip made up for the test, and the rectangles it reads.

The clip's frames are wider than they are tall, and its rectangle is too, so that a width and a height
swapped are told apart. Each refusal has a test that plants its fault:

- a malformed rectangle;
- a clip without a rectangle, without a manifest, or without an image;
- a manifest that does not list one frame per image;
- an image or a mask of another size;
- a cropped clip left by an earlier run.

The command's own checks are run through `main`.
"""

import json
import sys

import numpy as np
import pytest
from PIL import Image

from pipeline.crop_cholecseg8k import RECT_KEYS, crop_clip, load_rects, main

W, H = 40, 30
RECT = {"y0": 5, "y1": 25, "x0": 10, "x1": 40, "src_h": H, "src_w": W}
FRAMES = [{"seq_idx": 0, "native_frame": 80, "timestamp_sec": 3.2, "has_seg_mask": True},
          {"seq_idx": 1, "native_frame": 95, "timestamp_sec": 3.8, "has_seg_mask": True},
          {"seq_idx": 2, "native_frame": 110, "timestamp_sec": 4.4, "has_seg_mask": False}]


def write_image(path, w=W, h=H, seed=0):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(path)


def make_clip(root, name="VID07_s15_80", manifest=None):
    """Write a clip of three frames, two with a colour mask; return its directory."""
    d = root / name
    (d / "input_images").mkdir(parents=True)
    (d / "seg_masks").mkdir()
    for i in range(3):
        write_image(d / "input_images" / f"{i:06d}.png", seed=i)
    for i in range(2):
        write_image(d / "seg_masks" / f"{i:06d}_color_mask.png", seed=10 + i)
    if manifest is None:
        manifest = {"video_id": "VID07", "video_num": 7, "track": {"start_native": 80, "stride": 15, "count": 3},
                    "n_frames": 3, "frames": json.loads(json.dumps(FRAMES))}
    (d / "frame_manifest.json").write_text(json.dumps(manifest, indent=2))
    return d


@pytest.fixture
def clip(tmp_path):
    return make_clip(tmp_path)


def rects_file(tmp_path, rects):
    path = tmp_path / "crop_rects.json"
    path.write_text(json.dumps(rects))
    return path


def test_images_and_masks_are_cropped_to_the_rectangle(clip):
    dst = crop_clip(clip, {"VID07_s15_80": RECT})
    assert dst.name == "VID07_s15_80_crop"
    assert sorted(p.name for p in (dst / "input_images").iterdir()) == ["000000.png", "000001.png", "000002.png"]
    assert sorted(p.name for p in (dst / "seg_masks").iterdir()) == ["000000_color_mask.png", "000001_color_mask.png"]
    for sub in ("input_images", "seg_masks"):
        for p in (clip / sub).iterdir():
            src = np.asarray(Image.open(p))
            assert np.array_equal(np.asarray(Image.open(dst / sub / p.name)), src[5:25, 10:40]), p.name


def test_the_manifest_keeps_its_keys_and_records_the_new_size_and_the_rectangle(clip):
    dst = crop_clip(clip, {"VID07_s15_80": RECT})
    man = json.loads((dst / "frame_manifest.json").read_text())
    assert list(man) == ["video_id", "video_num", "track", "n_frames", "frames", "crop_info"]
    assert man["video_id"] == "VID07" and man["track"]["stride"] == 15 and man["n_frames"] == 3
    assert [f["image_size"] for f in man["frames"]] == [[30, 20]] * 3
    assert [{k: v for k, v in f.items() if k != "image_size"} for f in man["frames"]] == FRAMES
    assert man["crop_info"] == RECT and list(man["crop_info"]) == list(RECT_KEYS)
    # The workbench writes the manifest indented by two and crop_info.json on one line, keys in this order.
    # A clip cropped here is compared with one cropped there byte for byte.
    assert (dst / "frame_manifest.json").read_text() == json.dumps(man, indent=2)
    assert (dst / "crop_info.json").read_text() == json.dumps({k: RECT[k] for k in RECT_KEYS})


def test_a_clip_without_masks_gets_no_seg_masks_directory(clip):
    for p in (clip / "seg_masks").iterdir():
        p.unlink()
    dst = crop_clip(clip, {"VID07_s15_80": RECT})
    assert not (dst / "seg_masks").exists()
    assert len(list((dst / "input_images").iterdir())) == 3


def test_a_clip_without_a_rectangle_is_refused(clip):
    with pytest.raises(KeyError, match="VID07_s15_80 has no rectangle"):
        crop_clip(clip, {"VID07_s15_240": RECT})
    assert not (clip.parent / "VID07_s15_80_crop").exists()


def test_a_clip_without_a_manifest_is_refused(clip):
    (clip / "frame_manifest.json").unlink()
    with pytest.raises(FileNotFoundError, match="has no frame_manifest.json"):
        crop_clip(clip, {"VID07_s15_80": RECT})


def test_a_clip_without_an_image_is_refused(clip):
    for p in (clip / "input_images").iterdir():
        p.unlink()
    with pytest.raises(FileNotFoundError, match=r"has no input_images/\*.png"):
        crop_clip(clip, {"VID07_s15_80": RECT})


@pytest.mark.parametrize("manifest, said", [
    ({"video_id": "VID07", "video_num": 7}, "lists no frames; VID07_s15_80 holds 3 images"),
    ({"video_id": "VID07", "video_num": 7, "frames": FRAMES[:2]}, "lists 2 frames; VID07_s15_80 holds 3 images"),
    ({"video_id": "VID07", "video_num": 7, "frames": "three"}, "lists no frames"),
])
def test_a_manifest_that_does_not_list_one_frame_per_image_is_refused(tmp_path, manifest, said):
    clip = make_clip(tmp_path, manifest=manifest)
    with pytest.raises(ValueError, match=said):
        crop_clip(clip, {"VID07_s15_80": RECT})
    assert not (tmp_path / "VID07_s15_80_crop").exists()


@pytest.mark.parametrize("sub, name, w, h", [
    ("input_images", "000001.png", W, 20),
    ("input_images", "000001.png", 30, H),
    ("input_images", "000002.png", H, W),
    ("seg_masks", "000001_color_mask.png", W, 20),
    ("seg_masks", "000000_color_mask.png", 30, H),
])
def test_an_image_or_a_mask_of_another_size_is_refused_before_anything_is_written(clip, sub, name, w, h):
    write_image(clip / sub / name, w=w, h=h)
    with pytest.raises(ValueError, match=rf"{name} is {w}x{h}; the rectangle was found on frames of 40x30"):
        crop_clip(clip, {"VID07_s15_80": RECT})
    assert not (clip.parent / "VID07_s15_80_crop").exists()


def test_an_earlier_crop_is_refused_unless_replaced(clip):
    dst = crop_clip(clip, {"VID07_s15_80": RECT})
    stale = dst / "input_images" / "000099.png"
    stale.write_bytes(b"left by a longer run")
    with pytest.raises(ValueError, match="pass --overwrite"):
        crop_clip(clip, {"VID07_s15_80": RECT})
    assert stale.exists()
    crop_clip(clip, {"VID07_s15_80": RECT}, overwrite=True)
    assert not stale.exists()
    assert len(list((dst / "input_images").iterdir())) == 3


def test_a_run_that_fails_after_overwrite_was_asked_leaves_the_earlier_crop(clip):
    dst = crop_clip(clip, {"VID07_s15_80": RECT})
    before = {p.relative_to(dst): p.read_bytes() for p in dst.rglob("*") if p.is_file()}
    write_image(clip / "input_images" / "000002.png", w=W, h=20)
    with pytest.raises(ValueError, match="000002.png is 40x20"):
        crop_clip(clip, {"VID07_s15_80": RECT}, overwrite=True)
    assert {p.relative_to(dst): p.read_bytes() for p in dst.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("bad", [
    {k: v for k, v in RECT.items() if k != "x1"},
    dict(RECT, note=1),
    dict(RECT, y0=5.0),
    dict(RECT, y0=True),
    dict(RECT, x1="40"),
    dict(RECT, y0=-1),
    dict(RECT, x0=-1),
    dict(RECT, y0=25),
    dict(RECT, x0=40),
    dict(RECT, y1=31),
    dict(RECT, x1=41),
])
def test_a_rectangle_that_is_not_one_is_refused(tmp_path, bad):
    with pytest.raises(ValueError, match="VID07"):
        load_rects(str(rects_file(tmp_path, {"VID07": bad})))


def test_a_rectangle_that_fills_the_frame_is_accepted(tmp_path):
    whole = {"y0": 0, "y1": H, "x0": 0, "x1": W, "src_h": H, "src_w": W}
    assert load_rects(str(rects_file(tmp_path, {"VID07": whole}))) == {"VID07": whole}


def run_main(monkeypatch, root, rects, *clips, overwrite=False):
    argv = ["pipeline.crop_cholecseg8k", "--input-dir", str(root), "--rects", str(rects), "--clips", *clips]
    monkeypatch.setattr(sys, "argv", argv + (["--overwrite"] if overwrite else []))
    main()


def test_a_missing_input_dir_stops_the_run(tmp_path, monkeypatch):
    rects = rects_file(tmp_path, {"VID07_s15_80": RECT})
    with pytest.raises(SystemExit, match="no such directory"):
        run_main(monkeypatch, tmp_path / "nowhere", rects, "VID07_s15_80")


def test_a_mistyped_clip_name_stops_the_run_before_any_clip_is_cropped(tmp_path, monkeypatch):
    make_clip(tmp_path)
    rects = rects_file(tmp_path, {"VID07_s15_80": RECT, "VID08_s15_80": RECT})
    with pytest.raises(SystemExit, match=r"no clip \(a directory with frame_manifest.json\) at: VID08_s15_80"):
        run_main(monkeypatch, tmp_path, rects, "VID07_s15_80", "VID08_s15_80")
    assert not (tmp_path / "VID07_s15_80_crop").exists()


def test_a_clip_without_a_rectangle_stops_the_run_before_any_clip_is_cropped(tmp_path, monkeypatch):
    make_clip(tmp_path, "VID07_s15_80")
    make_clip(tmp_path, "VID08_s15_80")
    rects = rects_file(tmp_path, {"VID07_s15_80": RECT})
    with pytest.raises(SystemExit, match="no rectangle in .*crop_rects.json for: VID08_s15_80"):
        run_main(monkeypatch, tmp_path, rects, "VID07_s15_80", "VID08_s15_80")
    assert not (tmp_path / "VID07_s15_80_crop").exists()


def test_a_clip_that_fails_does_not_stop_the_others_and_the_run_exits_non_zero(tmp_path, monkeypatch, capsys):
    for name in ("VID07_s15_80", "VID08_s15_80", "VID09_s15_80"):
        make_clip(tmp_path, name)
    write_image(tmp_path / "VID08_s15_80" / "input_images" / "000000.png", w=W, h=20)
    rects = rects_file(tmp_path, {n: RECT for n in ("VID07_s15_80", "VID08_s15_80", "VID09_s15_80")})
    with pytest.raises(SystemExit, match=r"1 of 3 clip\(s\) failed: VID08_s15_80"):
        run_main(monkeypatch, tmp_path, rects, "VID07_s15_80", "VID08_s15_80", "VID09_s15_80")
    assert "failed: ValueError" in capsys.readouterr().out
    for name in ("VID07_s15_80_crop", "VID09_s15_80_crop"):
        assert len(list((tmp_path / name / "input_images").iterdir())) == 3
    assert not (tmp_path / "VID08_s15_80_crop").exists()


def test_every_clip_is_cropped_to_its_own_rectangle_and_the_command_exits_clean(tmp_path, monkeypatch):
    make_clip(tmp_path, "VID07_s15_80")
    make_clip(tmp_path, "VID08_s15_80")
    other = dict(RECT, y0=0, x1=20)
    rects = rects_file(tmp_path, {"VID07_s15_80": RECT, "VID08_s15_80": other})
    run_main(monkeypatch, tmp_path, rects, "VID07_s15_80", "VID08_s15_80")
    assert json.loads((tmp_path / "VID07_s15_80_crop" / "crop_info.json").read_text()) == RECT
    assert json.loads((tmp_path / "VID08_s15_80_crop" / "crop_info.json").read_text()) == other
    assert Image.open(tmp_path / "VID08_s15_80_crop" / "input_images" / "000000.png").size == (10, 25)


def test_the_command_refuses_an_earlier_crop_unless_overwrite_is_given(tmp_path, monkeypatch, capsys):
    make_clip(tmp_path, "VID07_s15_80")
    rects = rects_file(tmp_path, {"VID07_s15_80": RECT})
    run_main(monkeypatch, tmp_path, rects, "VID07_s15_80")
    with pytest.raises(SystemExit, match=r"1 of 1 clip\(s\) failed: VID07_s15_80"):
        run_main(monkeypatch, tmp_path, rects, "VID07_s15_80")
    assert "pass --overwrite" in capsys.readouterr().out
    run_main(monkeypatch, tmp_path, rects, "VID07_s15_80", overwrite=True)
    assert (tmp_path / "VID07_s15_80_crop" / "crop_info.json").is_file()
