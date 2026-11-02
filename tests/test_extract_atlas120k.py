"""Test the ATLAS-120k extraction on a release made up for the test.

The release holds one video: a synthetic mp4 whose frames all look
different, its JPEGs, and one clip for each outcome a clip of the index can
have: kept, split at a gap, a duplicate, too short, without a mask
directory, without masks. Cuts are planted where a test needs them. Each
refusal has a test that plants its fault, and the command's own checks are
run through `main`.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from evalkit.classes import load_table
from pipeline.extract_atlas120k import (check_cuts_inside, check_population, extract_video, main, split_at_cuts,
                                        stride_for)
from pipeline.point_clouds import read_manifest
from surgical_core.atlas120k.clip_rects import load_clip_rects
from surgical_core.atlas120k.cuts import COLUMNS
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

# The clips the release yields under the rule above.
WRITTEN = [f"{PROC}__{VID}__gt_{n}" for n in ("0001", "0003s1", "0003s2")]


def _ids() -> np.ndarray:
    ids = np.full((SIZE[1], SIZE[0]), 5, np.uint8)
    ids[:, :80] = 1
    ids[:30] = 12
    return ids


@pytest.fixture
def release(tmp_path):
    """Write the release, the crop rectangles and the frame ratio; return what `extract_video` takes.

    The tables are `crop_rects.json`, `frame_ratio.json` and a `cuts.csv` with no row beside `out`, for the
    tests that run the command.
    """
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
    (tmp_path / "cuts.csv").write_text(",".join(COLUMNS) + "\n")
    out = tmp_path / "out"
    out.mkdir()
    return dict(atlas_root=root, procedure=PROC, video=VID, out=out, rects=load_clip_rects(str(rects)),
                ratios=FrameRatios.load(str(ratios)), table=table, step_sec=STEP, min_frames=MIN_FRAMES,
                long_side=854, step_tol=0.2, cuts={})


def _clip(out: Path, n: str) -> Path:
    return out / f"{PROC}__{VID}__gt_{n}"


def _source(release, clip: str) -> Path:
    return release["atlas_root"] / "atlas120k" / "train" / PROC / VID / clip


def _report(release) -> dict:
    return json.loads((release["out"] / f"{PROC}__{VID}__extract_report.json").read_text())


def test_every_clip_of_the_index_is_written_or_recorded(release):
    written = extract_video(**release)
    assert written == WRITTEN
    report = _report(release)
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
    jpg = _source(release, "clip_0001") / "images" / "frame_000045.jpg"
    clip = _clip(release["out"], "0001")
    assert np.array_equal(cv2.imread(str(clip / "input_images" / "000001.png")), cv2.imread(str(jpg))[y:y + h, x:x + w])
    for n in ("0001", "0003s1"):
        mask = np.array(Image.open(_clip(release["out"], n) / "seg_masks" / "000001_class.png"))
        assert np.array_equal(mask, _ids()[y:y + h, x:x + w]), n
    man = json.loads((clip / "frame_manifest.json").read_text())
    assert man["crop"] == dict(zip("xywh", RECT)) and man["out_size"] == [w, h]
    assert all(f["has_gt"] and f["is_anchor"] for f in man["frames"])
    # The point cloud and Pi3X stages refuse a manifest whose `n_frames` is not its number of frames.
    assert man["n_frames"] == 6 and read_manifest(clip, 6) == man


def test_a_frame_is_resized_to_the_long_side(release):
    extract_video(**dict(release, long_side=70))
    img = cv2.imread(str(_clip(release["out"], "0001") / "input_images" / "000000.png"))
    mask = np.array(Image.open(_clip(release["out"], "0001") / "seg_masks" / "000000_class.png"))
    assert img.shape[:2] == mask.shape == (55, 70)
    assert set(np.unique(mask)) == {1, 5, 12}


def test_masks_under_machine_masks_are_read(release):
    (_source(release, "clip_0001") / "masks").rename(_source(release, "clip_0001") / "machine_masks")
    assert extract_video(**release) == WRITTEN


def test_a_clip_without_a_confirmed_rectangle_is_refused(release):
    rects = {k: v for k, v in release["rects"].items() if k[2] != "clip_0003"}
    with pytest.raises(KeyError, match="clip_0003 has no confirmed crop rectangle"):
        extract_video(**dict(release, rects=rects))


def test_an_unmeasured_video_is_refused(release):
    with pytest.raises(KeyError, match="has not been measured"):
        extract_video(**dict(release, ratios=FrameRatios({}, 8.0)))


def test_a_wrong_frame_ratio_is_refused(release):
    with pytest.raises(RuntimeError, match="do not correspond"):
        extract_video(**dict(release, ratios=FrameRatios({(PROC, VID): 2}, 8.0)))


def test_a_step_far_from_the_one_asked_for_is_refused(release):
    # 0.06 s at 25 fps rounds to a stride of 2, 0.08 s.
    assert stride_for(0.06, FPS, 1) == 2
    with pytest.raises(ValueError, match="suspect the frame ratio"):
        extract_video(**dict(release, step_sec=0.06))


def test_a_mask_of_another_size_is_refused(release):
    png = _source(release, "clip_0001") / "masks" / "frame_000050.png"
    Image.fromarray(np.ones((60, 80), np.uint8)).save(png)
    with pytest.raises(ValueError, match="not the mp4's 160x120"):
        extract_video(**release)


def test_a_frame_of_another_size_is_refused(release):
    jpg = _source(release, "clip_0001") / "images" / "frame_000050.jpg"
    cv2.imwrite(str(jpg), np.zeros((60, 80, 3), np.uint8))
    with pytest.raises(ValueError, match="frame_000050.jpg is 80x60, not the mp4's 160x120"):
        extract_video(**release)


def test_a_kept_clip_without_jpegs_is_refused(release):
    for p in (_source(release, "clip_0001") / "images").iterdir():
        p.unlink()
    with pytest.raises(RuntimeError, match="clip_0001: the release holds no JPEG"):
        extract_video(**release)


def test_a_frame_opencv_cannot_write_is_refused(release, monkeypatch):
    monkeypatch.setattr("pipeline.extract_atlas120k.cv2.imwrite", lambda path, img: False)
    with pytest.raises(RuntimeError, match="cannot write .*input_images"):
        extract_video(**release)


def test_a_short_clip_without_jpegs_is_still_recorded(release):
    for p in (_source(release, "clip_0005") / "images").iterdir():
        p.unlink()
    extract_video(**release)
    assert _report(release)["clips"][3]["reason"] == "too_short"


def test_earlier_output_is_refused_unless_replaced(release):
    extract_video(**release)
    stale = _clip(release["out"], "0001") / "input_images" / "000099.png"
    stale.write_bytes(b"left by a longer run")
    with pytest.raises(ValueError, match="pass --overwrite"):
        extract_video(**release)
    assert stale.exists()
    extract_video(**dict(release, overwrite=True))
    assert not stale.exists()


def test_a_refusal_writes_and_removes_nothing(release):
    # The last frame of the last clip written is the fault, so a stage that checked while writing would have
    # written two clips and removed the earlier output before finding it.
    extract_video(**release)
    out = release["out"]
    before = {p.relative_to(out): p.read_bytes() if p.is_file() else None for p in out.rglob("*")}
    png = _source(release, "clip_0003") / "masks" / "frame_000135.png"
    Image.fromarray(np.ones((60, 80), np.uint8)).save(png)
    with pytest.raises(ValueError, match="not the mp4's"):
        extract_video(**dict(release, overwrite=True))
    assert {p.relative_to(out): p.read_bytes() if p.is_file() else None for p in out.rglob("*")} == before
    empty = release["out"].parent / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="not the mp4's"):
        extract_video(**dict(release, out=empty))
    assert list(empty.iterdir()) == []


def _natives(release, n: str) -> list[int]:
    manifest = json.loads((_clip(release["out"], n) / "frame_manifest.json").read_text())
    return [f["native_frame"] for f in manifest["frames"]]


def test_a_run_is_split_between_the_two_frames_a_cut_falls_between():
    run = list(range(10))
    assert split_at_cuts([run], (4,), 1) == [run[:5], run[5:]]
    # At ratio 2, index frame 4 is mp4 frame 8 and frame 5 is mp4 frame 10: a cut after 8 or 9 falls between them.
    assert split_at_cuts([run], (8,), 2) == split_at_cuts([run], (9,), 2) == [run[:5], run[5:]]
    assert split_at_cuts([run], (7,), 2) == [run[:4], run[4:]]
    assert split_at_cuts([run], (8, 9), 2) == [run[:5], run[5:]]
    # A cut in a gap, or outside the runs, splits nothing.
    runs = [list(range(5)), list(range(10, 15))]
    assert split_at_cuts(runs, (6, 20), 1) == runs


def test_a_cut_outside_the_clip_it_is_listed_for_is_refused():
    usable = list(range(10, 21))
    check_cuts_inside("c", (10, 19), usable, 1)
    check_cuts_inside("c", (39,), usable, 2)
    for cut, ratio in ((9, 1), (20, 1), (40, 2)):
        with pytest.raises(ValueError, match=rf"c: the cut\(s\) after mp4 frame\(s\) \[{cut}\] fall outside"):
            check_cuts_inside("c", (cut,), usable, ratio)


def test_each_piece_is_thinned_from_its_first_frame_and_a_short_one_dropped(release):
    # gt_0001 holds frames 40 to 69. The cut after 51 leaves 40 to 51, thinned to 3 frames and dropped, and 52 to
    # 69, thinned from 52. clip_0004 holds the same frames: a cut is the mp4's, so it is split too and is still
    # the same run.
    assert extract_video(**dict(release, cuts={f"{PROC}__{VID}__gt_0001": (51,)})) == WRITTEN
    assert _natives(release, "0001") == [52, 57, 62, 67]
    rows = {r["source_clip"]: r for r in _report(release)["clips"]}
    assert (rows["clip_0001"]["n_cuts"], rows["clip_0001"]["n_segments"]) == (1, 2)
    assert rows["clip_0004"]["n_cuts"] == 1 and rows["clip_0004"]["duplicate_of"] == [f"{PROC}__{VID}__gt_0001"]


def test_a_cut_can_make_two_clips_of_one_run(release):
    written = extract_video(**dict(release, min_frames=2, cuts={f"{PROC}__{VID}__gt_0001": (54,)}))
    assert {f"{PROC}__{VID}__gt_0001s1", f"{PROC}__{VID}__gt_0001s2"} <= set(written)
    assert (_natives(release, "0001s1"), _natives(release, "0001s2")) == ([40, 45, 50], [55, 60, 65])


@pytest.mark.parametrize("cuts, match", [
    ({f"{PROC}__{VID}__gt_0001": (75,)}, r"gt_0001: the cut\(s\) after mp4 frame\(s\) \[75\] fall outside"),
    ({f"{PROC}__{VID}__gt_0006": (165,)}, r"have cuts listed but no frame to check them against"),
    ({f"{PROC}__{VID}__gt_0099": (50,)}, r"have cuts listed but no frame to check them against"),
])
def test_a_cut_that_cannot_be_checked_against_its_clip_is_refused_before_writing(release, cuts, match):
    with pytest.raises(ValueError, match=match):
        extract_video(**dict(release, cuts=cuts))
    assert list(release["out"].iterdir()) == []


def test_the_command_splits_at_the_cuts_of_its_table(release, monkeypatch):
    row = dict(clip=f"{PROC}__{VID}__gt_0001", spot_first="50", spot_last="55", d_pix="40.0", boundary_before="51",
               boundary_after="52", cut="yes", cut_before="51", cut_after="52", rechecked="0")
    with open(release["out"].parent / "cuts.csv", "a") as f:
        f.write(",".join(row[c] for c in COLUMNS) + "\n")
    run_main(monkeypatch, release)
    assert _natives(release, "0001") == [52, 57, 62, 67]


def test_clips_off_the_population_are_refused():
    names = [f"{PROC}__{VID}__gt_0001", f"{PROC}__{VID}__gt_0003s1"]
    population = names + ["other__vid__gt_0001"]
    check_population(names, population, PROC, VID)
    with pytest.raises(ValueError, match=r"in the population, not to be written: \['proc__vid__gt_0003s1'\]"):
        check_population(names[:1], population, PROC, VID)
    with pytest.raises(ValueError, match=r"to be written, not in the population: \['proc__vid__gt_0004'\]"):
        check_population(names + [f"{PROC}__{VID}__gt_0004"], population, PROC, VID)


def run_main(monkeypatch, release, *args):
    """Run the command on the release with the test's step and run length."""
    root = release["out"].parent
    monkeypatch.setattr(sys, "argv", [
        "pipeline.extract_atlas120k", "--atlas-root", str(release["atlas_root"]), "--out", str(release["out"]),
        "--clip-rects", str(root / "crop_rects.json"), "--frame-ratios", str(root / "frame_ratio.json"),
        "--cuts", str(root / "cuts.csv"), "--step-sec", str(STEP), "--min-frames", str(MIN_FRAMES), *args])
    main()


def test_the_command_refuses_a_video_not_named_procedure_slash_video(release, monkeypatch):
    with pytest.raises(SystemExit, match="not <procedure>/<video>: vid"):
        run_main(monkeypatch, release, "--videos", f"{PROC}/{VID}", VID)
    assert list(release["out"].iterdir()) == []


def test_the_command_extracts_every_measured_video_and_checks_the_population(release, monkeypatch):
    population = release["out"].parent / "clips.txt"
    population.write_text("\n".join(WRITTEN + ["other__vid__gt_0001"]) + "\n")
    run_main(monkeypatch, release, "--population", str(population))
    assert sorted(p.name for p in release["out"].iterdir() if p.is_dir()) == WRITTEN


def test_the_command_fails_a_video_whose_clips_are_not_the_populations(release, monkeypatch, capsys):
    population = release["out"].parent / "clips.txt"
    population.write_text("\n".join(WRITTEN[:2]) + "\n")
    with pytest.raises(SystemExit, match=rf"1 of 1 video\(s\) failed: {PROC}/{VID}"):
        run_main(monkeypatch, release, "--videos", f"{PROC}/{VID}", "--population", str(population))
    assert "failed: ValueError" in capsys.readouterr().out
    assert list(release["out"].iterdir()) == []
    population.write_text("\n".join(WRITTEN + WRITTEN[:1]) + "\n")
    with pytest.raises(ValueError, match="more than once"):
        run_main(monkeypatch, release, "--population", str(population))


def test_a_population_mismatch_keeps_the_earlier_output(release, monkeypatch):
    # The depth stage's file stands for any output a later stage wrote into a clip.
    extract_video(**release)
    out = release["out"]
    (_clip(out, "0001") / "depth_raw").mkdir()
    (_clip(out, "0001") / "depth_raw" / "depth_000000.npy").write_bytes(b"written by the depth stage")
    before = {p.relative_to(out): p.read_bytes() if p.is_file() else None for p in out.rglob("*")}
    population = out.parent / "clips.txt"
    population.write_text("\n".join(WRITTEN[:2]) + "\n")
    with pytest.raises(SystemExit, match=rf"1 of 1 video\(s\) failed: {PROC}/{VID}"):
        run_main(monkeypatch, release, "--videos", f"{PROC}/{VID}", "--overwrite", "--population", str(population))
    assert {p.relative_to(out): p.read_bytes() if p.is_file() else None for p in out.rglob("*")} == before
