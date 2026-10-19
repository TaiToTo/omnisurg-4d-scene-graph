"""Test the StereoMIS extraction on clips and frames made up for the test.

The clips are a clip list's, and the reader's `iter_frame_set` and `video_info`
are replaced by stand-ins, so no video is decoded; the masks are real files,
read by the reader. Each refusal has a test that plants its fault.
"""

import json
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

import cv2  # noqa: E402

import pipeline.extract_stereomis as E  # noqa: E402
import surgical_core.stereomis as stereomis  # noqa: E402

W, H = stereomis.HALF_SIZE
FPS = 60000 / 1001


def make_clip(seq: str, k: int, frames: list[int], usable: bool = True, masks=None) -> dict:
    return dict(name=f"{seq}__clip_{k:04d}", seq=seq, usable=usable, drop="" if usable else "gt_static",
                stratum="moving", span_mm=12.345, depth_ok=1.0, stride=12, gt_row_offset=-3, frames=frames,
                times=[f / FPS for f in frames], masks=masks or [None] * len(frames))


# Clip 2 of P1 gives its first frame the mask of its own number and its second the one two frames later; its
# third frame has none.
CLIP_LIST = (make_clip("P1", 0, [4, 16]), make_clip("P1", 1, [1348, 1360], usable=False),
             make_clip("P1", 2, [2692, 2704, 2716], masks=[2693, 2707, None]),
             make_clip("P2_0", 0, [4, 16]), make_clip("P2_3", 0, [4, 16], usable=False))


def left_view(f: int) -> np.ndarray:
    """A left view that names its frame: red is the frame modulo 256, green the frame over 256, blue 7."""
    a = np.empty((H, W, 3), np.uint8)
    a[:] = (f % 256, f // 256, 7)
    return a


@pytest.fixture
def decoded(monkeypatch, tmp_path):
    """Replace the reader's decoding and probe; return the dataset's root and the frames decoded."""
    calls = []

    def iter_frame_set(root, seq, frames):
        calls.append((seq, list(frames)))
        for f in sorted(set(frames)):
            yield f, left_view(f), np.zeros((H, W, 3), np.uint8)

    monkeypatch.setattr(stereomis, "iter_frame_set", iter_frame_set)
    monkeypatch.setattr(stereomis, "video_info", lambda root, seq: dict(width=1280, height=2048, n_frames=9000,
                                                                       fps=FPS))
    return tmp_path / "StereoMIS", calls


def read_rgb(path) -> np.ndarray:
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[:, :, ::-1]


def test_each_usable_clip_gets_its_left_views_and_the_workbench_manifest(decoded, tmp_path):
    root, calls = decoded
    out = tmp_path / "clips"
    made = E.extract_sequence(root, CLIP_LIST, "P1", out, mask_instruments=False)
    assert made == [out / "P1__clip_0000", out / "P1__clip_0002"]
    assert calls == [("P1", [4, 16, 2692, 2704, 2716])]
    assert sorted(p.name for p in out.iterdir()) == ["P1__clip_0000", "P1__clip_0002"]
    images = sorted((out / "P1__clip_0002" / "input_images").iterdir())
    assert [p.name for p in images] == ["000000.png", "000001.png", "000002.png"]
    for p, f in zip(images, [2692, 2704, 2716]):
        np.testing.assert_array_equal(read_rgb(p), left_view(f))
    text = (out / "P1__clip_0000" / "frame_manifest.json").read_text()
    assert list(json.loads(text)) == ["dataset", "sequence", "clip_kind", "fps_native", "stride", "out_size",
                                      "n_frames", "gt_row_offset", "instruments_masked", "span_mm", "stratum",
                                      "frames"]
    assert json.loads(text) == {
        "dataset": "stereomis", "sequence": "P1", "clip_kind": "fixed_grid", "fps_native": FPS, "stride": 12,
        "out_size": [640, 512], "n_frames": 2, "gt_row_offset": -3, "instruments_masked": 0, "span_mm": 12.345,
        "stratum": "moving",
        "frames": [{"seq_idx": 0, "native_frame": 4, "gt_row": 1, "time_sec": 0.0667},
                   {"seq_idx": 1, "native_frame": 16, "gt_row": 13, "time_sec": 0.2669}]}
    assert text.startswith('{\n "dataset": "stereomis",') and not text.endswith("\n")


def test_the_manifest_takes_the_rows_and_times_the_clip_list_gives(decoded, tmp_path):
    """The planted fault: rows and times that the tables and the frame rate would not give."""
    root, _ = decoded
    clip = dict(CLIP_LIST[0], gt_row_offset=-7, times=[1.5, 1.75])
    E.extract_sequence(root, [clip], "P1", tmp_path / "clips", mask_instruments=False)
    manifest = json.loads((tmp_path / "clips" / "P1__clip_0000" / "frame_manifest.json").read_text())
    assert manifest["gt_row_offset"] == -7
    assert [(f["gt_row"], f["time_sec"]) for f in manifest["frames"]] == [(-3, 1.5), (9, 1.75)]


def write_mask(root, seq: str, number: int) -> None:
    """Write a mask whose left half is tissue and whose right half is instrument."""
    (root / seq / "masks").mkdir(parents=True, exist_ok=True)
    a = np.zeros((H, W, 3), np.uint8)
    a[:, : W // 2] = 255
    cv2.imwrite(str(root / seq / "masks" / ("%06dl.png" % number)), a)


def test_instruments_are_painted_black_outside_the_mask_the_list_gives_and_counted(decoded, tmp_path):
    """Mask 2719 lies within two frames of frame 2716, but the list gives that frame none."""
    root, _ = decoded
    for number in (2693, 2707, 2719):
        write_mask(root, "P1", number)
    out = tmp_path / "clips"
    E.extract_sequence(root, CLIP_LIST, "P1", out, mask_instruments=True)
    images = sorted((out / "P1__clip_0002" / "input_images").iterdir())
    for p, f, masked in zip(images, [2692, 2704, 2716], [True, True, False]):
        got, want = read_rgb(p), left_view(f)
        np.testing.assert_array_equal(got[:, : W // 2], want[:, : W // 2])
        np.testing.assert_array_equal(got[:, W // 2:], 0 if masked else want[:, W // 2:])
    manifest = json.loads((out / "P1__clip_0002" / "frame_manifest.json").read_text())
    assert manifest["instruments_masked"] == 2
    assert json.loads((out / "P1__clip_0000" / "frame_manifest.json").read_text())["instruments_masked"] == 0


def test_a_mask_the_list_gives_that_is_not_there_leaves_no_clip(decoded, tmp_path):
    root, _ = decoded
    write_mask(root, "P1", 2693)
    out = tmp_path / "clips"
    with pytest.raises(FileNotFoundError, match="002707l.png"):
        E.extract_sequence(root, CLIP_LIST, "P1", out, mask_instruments=True)
    assert list(out.iterdir()) == []


def test_a_sequence_without_a_usable_clip_writes_nothing(decoded, tmp_path):
    root, calls = decoded
    out = tmp_path / "clips"
    out.mkdir()
    assert E.extract_sequence(root, CLIP_LIST, "P2_3", out, mask_instruments=False) == []
    assert calls == [] and list(out.iterdir()) == []


@pytest.mark.parametrize("left_behind", ["P1__clip_0002", "P1__clip_0002.partial"])
def test_a_clip_that_exists_is_refused_before_anything_is_decoded(decoded, tmp_path, left_behind):
    root, calls = decoded
    out = tmp_path / "clips"
    (out / left_behind).mkdir(parents=True)
    with pytest.raises(FileExistsError, match=left_behind):
        E.extract_sequence(root, CLIP_LIST, "P1", out, mask_instruments=False)
    assert calls == [] and [p.name for p in out.iterdir()] == [left_behind]


def test_a_decode_that_fails_leaves_no_clip(decoded, tmp_path, monkeypatch):
    root, _ = decoded

    def failing(root, seq, frames):
        yield 4, left_view(4), left_view(4)
        raise RuntimeError("5 frames wanted, 1 decoded")

    monkeypatch.setattr(stereomis, "iter_frame_set", failing)
    out = tmp_path / "clips"
    with pytest.raises(RuntimeError, match="1 decoded"):
        E.extract_sequence(root, CLIP_LIST, "P1", out, mask_instruments=False)
    assert list(out.iterdir()) == []


@pytest.fixture
def clip_list_file(monkeypatch, tmp_path):
    """Write `CLIP_LIST` as a clip list, and read it back without the checks, whose clips are 112 frames long."""
    path = tmp_path / "clips.json"
    stereomis.write_clip_list(CLIP_LIST, path)
    monkeypatch.setattr(stereomis, "read_clip_list", lambda p: tuple(json.loads(p.read_text())))
    return path


def test_the_command_refuses_a_clip_of_any_sequence_before_decoding_the_first(decoded, tmp_path, monkeypatch,
                                                                              clip_list_file):
    root, calls = decoded
    out = tmp_path / "clips"
    (out / "P2_0__clip_0000").mkdir(parents=True)
    argv = ["extract_stereomis", "--root", str(root), "--clips", str(clip_list_file), "--out", str(out),
            "--sequences", "P1", "P2_0"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(FileExistsError, match="P2_0__clip_0000"):
        E.main()
    assert calls == []


def test_the_command_refuses_a_sequence_named_twice_before_decoding_the_first(decoded, tmp_path, monkeypatch,
                                                                             clip_list_file):
    root, calls = decoded
    out = tmp_path / "clips"
    argv = ["extract_stereomis", "--root", str(root), "--clips", str(clip_list_file), "--out", str(out),
            "--sequences", "P1", "P1", "P2_0"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="names P1 twice"):
        E.main()
    assert calls == [] and not out.exists()


def test_the_command_extracts_the_sequences_named(decoded, tmp_path, monkeypatch, capsys, clip_list_file):
    root, calls = decoded
    for number in (2693, 2707):
        write_mask(root, "P1", number)
    out = tmp_path / "clips"
    argv = ["extract_stereomis", "--root", str(root), "--clips", str(clip_list_file), "--out", str(out),
            "--sequences", "P1", "P2_0", "--mask-instruments"]
    monkeypatch.setattr(sys, "argv", argv)
    E.main()
    assert [seq for seq, _ in calls] == ["P1", "P2_0"]
    assert sorted(p.name for p in out.iterdir()) == ["P1__clip_0000", "P1__clip_0002", "P2_0__clip_0000"]
    assert capsys.readouterr().out.splitlines()[-1] == f"3 clips in {out}"


def test_two_clips_that_share_a_frame_are_refused_before_anything_is_decoded(decoded, tmp_path):
    root, calls = decoded
    shared = (make_clip("P1", 0, [4, 16]), make_clip("P1", 1, [16, 28]))
    out = tmp_path / "clips"
    out.mkdir()
    with pytest.raises(ValueError, match="hold the same frame"):
        E.extract_sequence(root, shared, "P1", out, mask_instruments=False)
    assert calls == [] and list(out.iterdir()) == []


def test_an_image_that_cannot_be_written_leaves_no_clip(decoded, tmp_path, monkeypatch):
    root, _ = decoded
    monkeypatch.setattr(E.cv2, "imwrite", lambda path, img: False)
    out = tmp_path / "clips"
    with pytest.raises(RuntimeError, match="cannot write"):
        E.extract_sequence(root, CLIP_LIST, "P1", out, mask_instruments=False)
    assert list(out.iterdir()) == []
