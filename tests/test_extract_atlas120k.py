"""Test the ATLAS-120k extraction on a release made up for the test.

The release holds one video: a synthetic mp4 whose frames all look
different, its JPEGs, and one clip for each outcome a clip of the index can
have: kept, split at a gap, a duplicate, too short, without a mask
directory, without masks. Each refusal has a test that plants its fault.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from evalkit.classes import load_table
from pipeline.extract_atlas120k import check_population, extract_video, stride_for
from surgical_core.atlas120k.clip_rects import load_clip_rects
from surgical_core.atlas120k.frame_ratio import FrameRatios

PROC, VID = "proc", "vid"
FPS, N_FRAMES, SIZE = 25, 200, (160, 120)
RECT = [10, 5, 140, 110]

# One step of 0.2 s is a stride of 5 at 25 fps, and a run of 20 frames thins to 4.
STEP, MIN_FRAMES = 0.2, 4

CLIPS = {
    "clip_0001": list(range(40, 70)),
    "clip_0003": list(range(80, 100)) + list(range(110, 140)),
    "clip_0004": list(range(40, 70)),
    "clip_0005": list(range(150, 160)),
    "clip_0006": list(range(160, 170)),
    "clip_0007": list(range(170, 180)),
}
COLOUR_MASKS = {"clip_0003"}
NO_MASK_DIR = {"clip_0006"}
NO_MASKS = {"clip_0007"}


def _ids() -> np.ndarray:
    ids = np.full((SIZE[1], SIZE[0]), 5, np.uint8)
    ids[:, :80] = 1
    ids[:30] = 12
    return ids


@pytest.fixture
def release(tmp_path):
    """Write the release, the crop rectangles and the frame ratio; return what `extract_video` takes."""
    root = tmp_path / "ATLAS"
    (root / "raw_data" / PROC).mkdir(parents=True)
    writer = cv2.VideoWriter(str(root / "raw_data" / PROC / f"{VID}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS,
                             SIZE)
    assert writer.isOpened(), "this OpenCV cannot write mp4v"
    rng = np.random.default_rng(0)
    frames = []
    for i in range(N_FRAMES):
        frame = np.empty((SIZE[1], SIZE[0], 3), np.uint8)
        frame[:] = rng.integers(0, 256, 3)
        cv2.putText(frame, str(i), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        writer.write(frame)
        frames.append(frame)
    writer.release()

    table = load_table("atlas120k")
    ids = _ids()
    colours = np.zeros((*ids.shape, 3), np.uint8)
    for i in np.unique(ids):
        colours[ids == i] = table.entries[int(i)].colour
    vdir = root / "atlas120k" / "train" / PROC / VID
    for clip, nums in CLIPS.items():
        (vdir / clip / "images").mkdir(parents=True)
        if clip not in NO_MASK_DIR:
            (vdir / clip / "masks").mkdir()
        for f in nums:
            cv2.imwrite(str(vdir / clip / "images" / f"frame_{f:06d}.jpg"), frames[f])
            if clip in NO_MASK_DIR | NO_MASKS:
                continue
            if clip in COLOUR_MASKS:
                Image.fromarray(colours).save(vdir / clip / "masks" / f"frame_{f:06d}.png")
            else:
                # A palette that disagrees with the class table: the ids are read by value.
                im = Image.fromarray(ids)
                im.putpalette([255 - v for v in range(256) for _ in range(3)])
                im.save(vdir / clip / "masks" / f"frame_{f:06d}.png")
    (vdir / "clip_index.json").write_text(json.dumps({"is_robot": False, "frame_digits": 6, "clips": CLIPS}))

    rects = tmp_path / "crop_rects.json"
    rects.write_text(json.dumps([{"procedure": PROC, "video": VID, "clip": c, "rect": RECT,
                                  "src_size": list(SIZE), "verdict": "ok"} for c in CLIPS]))
    ratios = tmp_path / "frame_ratio.json"
    ratios.write_text(json.dumps({"match_tol": 8.0, "videos": [{"procedure": PROC, "video": VID, "ratio": 1}]}))
    out = tmp_path / "out"
    out.mkdir()
    return dict(atlas_root=root, procedure=PROC, video=VID, out=out, rects=load_clip_rects(str(rects)),
                ratios=FrameRatios.load(str(ratios)), table=table, step_sec=STEP, min_frames=MIN_FRAMES,
                long_side=854, step_tol=0.2)


def _clip(out: Path, n: str) -> Path:
    return out / f"{PROC}__{VID}__gt_{n}"


def test_every_clip_of_the_index_is_written_or_recorded(release):
    written = extract_video(**release)
    assert written == [f"{PROC}__{VID}__gt_{n}" for n in ("0001", "0003s1", "0003s2")]
    report = json.loads((release["out"] / f"{PROC}__{VID}__extract_report.json").read_text())
    reasons = {r["source_clip"]: r["reason"] for r in report["clips"]}
    assert reasons == {"clip_0001": "kept", "clip_0003": "kept_after_split", "clip_0004": "duplicate",
                       "clip_0005": "too_short", "clip_0006": "no_mask_dir", "clip_0007": "no_masks"}
    assert report["clips"][2]["duplicate_of"] == [f"{PROC}__{VID}__gt_0001"]
    assert report["gt_stride"] == 5 and report["n_written"] == 3


def test_a_run_is_thinned_and_never_spans_a_gap(release):
    extract_video(**release)
    natives = {n: [f["native_frame"] for f in json.loads((_clip(release["out"], n) / "frame_manifest.json")
                                                         .read_text())["frames"]]
               for n in ("0001", "0003s1", "0003s2")}
    assert natives == {"0001": [40, 45, 50, 55, 60, 65], "0003s1": [80, 85, 90, 95],
                       "0003s2": [110, 115, 120, 125, 130, 135]}


def test_frames_and_masks_are_cut_to_the_confirmed_rectangle(release):
    extract_video(**release)
    x, y, w, h = RECT
    jpg = release["atlas_root"] / "atlas120k" / "train" / PROC / VID / "clip_0001" / "images" / "frame_000045.jpg"
    clip = _clip(release["out"], "0001")
    assert np.array_equal(cv2.imread(str(clip / "input_images" / "000001.png")), cv2.imread(str(jpg))[y:y + h, x:x + w])
    for n in ("0001", "0003s1"):
        mask = np.array(Image.open(_clip(release["out"], n) / "seg_masks" / "000001_class.png"))
        assert np.array_equal(mask, _ids()[y:y + h, x:x + w]), n
    man = json.loads((clip / "frame_manifest.json").read_text())
    assert man["crop"] == dict(zip("xywh", RECT)) and man["out_size"] == [w, h]
    assert all(f["has_gt"] and f["is_anchor"] for f in man["frames"])


def test_a_frame_is_resized_to_the_long_side(release):
    extract_video(**dict(release, long_side=70))
    img = cv2.imread(str(_clip(release["out"], "0001") / "input_images" / "000000.png"))
    mask = np.array(Image.open(_clip(release["out"], "0001") / "seg_masks" / "000000_class.png"))
    assert img.shape[:2] == mask.shape == (55, 70)
    assert set(np.unique(mask)) == {1, 5, 12}


def test_a_clip_without_a_confirmed_rectangle_is_refused(release):
    rects = {k: v for k, v in release["rects"].items() if k[2] != "clip_0003"}
    with pytest.raises(KeyError, match="clip_0003 has no confirmed crop rectangle"):
        extract_video(**dict(release, rects=rects))


def test_a_wrong_frame_ratio_is_refused(release):
    with pytest.raises(RuntimeError, match="do not correspond"):
        extract_video(**dict(release, ratios=FrameRatios({(PROC, VID): 2}, 8.0)))


def test_a_step_far_from_the_one_asked_for_is_refused(release):
    # 0.06 s at 25 fps rounds to a stride of 2, 0.08 s.
    assert stride_for(0.06, FPS, 1) == 2
    with pytest.raises(ValueError, match="suspect the frame ratio"):
        extract_video(**dict(release, step_sec=0.06))


def test_a_mask_of_another_size_is_refused(release):
    png = release["atlas_root"] / "atlas120k" / "train" / PROC / VID / "clip_0001" / "masks" / "frame_000050.png"
    Image.fromarray(np.ones((60, 80), np.uint8)).save(png)
    with pytest.raises(ValueError, match="not the mp4's 160x120"):
        extract_video(**release)


def test_a_kept_clip_without_jpegs_is_refused(release):
    images = release["atlas_root"] / "atlas120k" / "train" / PROC / VID / "clip_0001" / "images"
    for p in images.iterdir():
        p.unlink()
    with pytest.raises(RuntimeError, match="clip_0001: the release holds no JPEG"):
        extract_video(**release)


def test_a_short_clip_without_jpegs_is_still_recorded(release):
    images = release["atlas_root"] / "atlas120k" / "train" / PROC / VID / "clip_0005" / "images"
    for p in images.iterdir():
        p.unlink()
    extract_video(**release)
    report = json.loads((release["out"] / f"{PROC}__{VID}__extract_report.json").read_text())
    assert report["clips"][3]["reason"] == "too_short"


def test_earlier_output_is_refused_unless_replaced(release):
    extract_video(**release)
    stale = _clip(release["out"], "0001") / "input_images" / "000099.png"
    stale.write_bytes(b"left by a longer run")
    with pytest.raises(ValueError, match="pass --overwrite"):
        extract_video(**release)
    extract_video(**dict(release, overwrite=True))
    assert not stale.exists()


def test_clips_off_the_population_are_refused():
    names = [f"{PROC}__{VID}__gt_0001", f"{PROC}__{VID}__gt_0003s1"]
    population = names + ["other__vid__gt_0001"]
    check_population(names, population, PROC, VID)
    with pytest.raises(ValueError, match="not written"):
        check_population(names[:1], population, PROC, VID)
    with pytest.raises(ValueError, match="not in the population"):
        check_population(names + [f"{PROC}__{VID}__gt_0004"], population, PROC, VID)
