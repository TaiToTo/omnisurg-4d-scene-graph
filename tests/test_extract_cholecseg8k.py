"""Test the CholecSeg8k extraction on videos and annotations made up for the test.

Two synthetic videos stand for the two ways CholecSeg8k numbers frames: one
at the video's own rate, one at 30 frames a second against its 25. Their
annotated images are decoded frames of the video, so each matches one frame.
A third video changes slowly, to put a match on the edge of the search
window. Each refusal has a test that plants its fault.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import pipeline.extract_cholecseg8k as extract_cholecseg8k
from pipeline.extract_cholecseg8k import extract_clip, parse_clip, unannotated_natives

N_FRAMES = 200
SIZE = (64, 48)

# The capture class as OpenCV ships it; a test replaces `cv2.VideoCapture` itself.
VideoCapture = cv2.VideoCapture

# Video 1 is numbered at its own rate. Video 25 is numbered at 30 frames a second, so CholecSeg8k number s is
# native frame 60 + (s - 60) * 5 / 6. A stride of 6 numbers is 5 native frames, so no frame falls between two.
ANNOTATED = {
    1: {s: s for s in (60, 66, 72, 78, 84, 102, 108)},
    25: {s: 60 + (s - 60) * 5 // 6 for s in (60, 66, 72, 78, 84, 102, 108)},
}


def _write_video(path: Path, frames: list[np.ndarray]) -> list[np.ndarray]:
    """Write the frames as an mp4 and return them as decoded."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, SIZE)
    assert writer.isOpened(), "this OpenCV cannot write mp4v"
    for f in frames:
        writer.write(f)
    writer.release()
    cap = cv2.VideoCapture(str(path))
    decoded = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        decoded.append(f)
    cap.release()
    return decoded


def _distinct_frames() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    frames = []
    for i in range(N_FRAMES):
        f = np.empty((SIZE[1], SIZE[0], 3), np.uint8)
        f[:] = rng.integers(0, 256, 3)
        cv2.putText(f, str(i), (2, 36), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        frames.append(f)
    return frames


def _annotate(seg8k: Path, num: int, decoded: list[np.ndarray], natives: dict[int, int]) -> None:
    """Write each annotated frame's image, a decoded frame, and a colour mask, in one chunk directory."""
    chunk = seg8k / f"video{num:02d}" / f"video{num:02d}_{min(natives):05d}"
    chunk.mkdir(parents=True)
    for s, n in natives.items():
        cv2.imwrite(str(chunk / f"frame_{s}_endo.png"), decoded[n])
        mask = np.zeros_like(decoded[n])
        mask[:, : SIZE[0] // 2] = (255, 0, s % 256)
        cv2.imwrite(str(chunk / f"frame_{s}_endo_color_mask.png"), mask)


@pytest.fixture
def source(tmp_path):
    """Write videos 1 and 25 with their annotations; return the roots and the decoded frames."""
    seg8k, videos = tmp_path / "CholecSeg8k", tmp_path / "videos"
    videos.mkdir()
    decoded = {}
    for num, natives in ANNOTATED.items():
        decoded[num] = _write_video(videos / f"video{num:02d}.mp4", _distinct_frames())
        _annotate(seg8k, num, decoded[num], natives)
    out = tmp_path / "out"
    out.mkdir()
    return dict(seg8k_root=seg8k, videos_root=videos, out=out), decoded


def _manifest(out: Path, clip: str) -> dict:
    return json.loads((out / clip / "frame_manifest.json").read_text())


class _FaultyCapture:
    """A video capture whose `read` fails at one frame, and whose frame count can be overridden."""

    fail_at: int | None = None
    only_after_seek: bool = False
    frame_count: int | None = None

    def __init__(self, path: str):
        self._cap = VideoCapture(path)
        self._sought = None

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._sought = int(value)
        return self._cap.set(prop, value)

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT and self.frame_count is not None:
            return float(self.frame_count)
        return self._cap.get(prop)

    def read(self):
        at = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        sought, self._sought = self._sought, None
        if at == self.fail_at and (not self.only_after_seek or sought == at):
            return False, None
        return self._cap.read()

    def __getattr__(self, name):
        return getattr(self._cap, name)


@pytest.fixture
def faulty_capture(monkeypatch):
    """Replace the extractor's video capture with one whose faults the test sets; return the class."""
    class Capture(_FaultyCapture):
        pass
    monkeypatch.setattr(extract_cholecseg8k.cv2, "VideoCapture", Capture)
    return Capture


def _grey_source(tmp_path: Path, natives: dict[int, int]) -> dict:
    """Write a video whose frame i is grey level i, annotate it as video 2, and return the roots."""
    roots = dict(seg8k_root=tmp_path / "CholecSeg8k", videos_root=tmp_path / "videos", out=tmp_path / "out")
    roots["videos_root"].mkdir()
    grey = [np.full((SIZE[1], SIZE[0], 3), i, np.uint8) for i in range(N_FRAMES)]
    decoded = _write_video(roots["videos_root"] / "video02.mp4", grey)
    _annotate(roots["seg8k_root"], 2, decoded, natives)
    return roots


def test_a_clip_numbered_at_the_video_rate_keeps_its_numbers(source):
    roots, decoded = source
    extract_clip(**roots, clip="VID01_s6_60", count=10)
    frames = _manifest(roots["out"], "VID01_s6_60")["frames"]
    assert [f["native_frame"] for f in frames] == list(range(60, 120, 6))
    gap = [f for f in frames if not f["has_seg_mask"]]
    assert [f["seq_idx"] for f in gap] == [5, 6, 9]
    for f in gap:
        img = cv2.imread(str(roots["out"] / "VID01_s6_60" / "input_images" / f"{f['seq_idx']:06d}.png"))
        assert np.array_equal(img, decoded[1][f["native_frame"]])
        assert f["seg_frame"] is None and not f["is_anchor"]


def test_a_clip_numbered_at_30_fps_takes_the_native_frame_of_each_number(source):
    roots, decoded = source
    extract_clip(**roots, clip="VID25_s6_60", count=10)
    frames = _manifest(roots["out"], "VID25_s6_60")["frames"]
    assert [f["native_frame"] for f in frames] == list(range(60, 110, 5))
    assert [f["timestamp_sec"] for f in frames] == [round(n / 25, 2) for n in range(60, 110, 5)]
    for f in frames:
        img = cv2.imread(str(roots["out"] / "VID25_s6_60" / "input_images" / f"{f['seq_idx']:06d}.png"))
        assert np.array_equal(img, decoded[25][f["native_frame"]]), f
    assert [f["seg_frame"] for f in frames if f["has_seg_mask"]] == sorted(ANNOTATED[25])


def test_masks_are_written_only_on_annotated_frames(source):
    roots, _ = source
    extract_clip(**roots, clip="VID25_s6_60", count=10)
    names = sorted(p.name for p in (roots["out"] / "VID25_s6_60" / "seg_masks").iterdir())
    assert names == [f"{i:06d}_color_mask.png" for i in (0, 1, 2, 3, 4, 7, 8)]


def test_the_rate_is_carried_between_and_past_the_annotated_frames():
    # Between 30 and 60 the frames run at 24/30, not at the clip's 50/60. Number 45 is native frame 38 at the
    # first rate and 38.5 at the second.
    annotated = {0: 0, 30: 26, 60: 50}
    assert unannotated_natives([0, 15, 30, 45, 60, 75, 90], annotated) == {15: 13, 45: 38, 75: 63, 90: 75}


def test_a_half_rounds_up_however_the_rate_divides():
    # 123 frames over 150 numbers put number 75 at frame 61.5; in floating point 75 * (123 / 150) falls short of
    # it, and would round down.
    assert unannotated_natives([75], {0: 0, 150: 123}) == {75: 62}


def test_a_rate_neither_numbering_gives_is_refused():
    with pytest.raises(ValueError, match="not one of"):
        unannotated_natives([0, 10, 20], {0: 0, 30: 20})


def test_a_clip_with_one_annotated_frame_is_refused():
    with pytest.raises(ValueError, match="the rate needs two"):
        unannotated_natives([0, 10], {0: 0})


def test_native_frames_that_do_not_rise_are_refused(source):
    roots, decoded = source
    chunk = roots["seg8k_root"] / "video01" / "video01_00060"
    cv2.imwrite(str(chunk / "frame_72_endo.png"), decoded[1][64])
    with pytest.raises(ValueError, match="do not rise"):
        extract_clip(**roots, clip="VID01_s6_60", count=10)


def test_an_image_that_matches_no_frame_is_refused(source):
    roots, _ = source
    chunk = roots["seg8k_root"] / "video01" / "video01_00060"
    cv2.imwrite(str(chunk / "frame_66_endo.png"), np.full((SIZE[1], SIZE[0], 3), 7, np.uint8))
    with pytest.raises(ValueError, match="matches no video frame"):
        extract_clip(**roots, clip="VID01_s6_60", count=10)


def test_an_image_of_another_size_is_refused(source):
    roots, decoded = source
    chunk = roots["seg8k_root"] / "video01" / "video01_00060"
    cv2.imwrite(str(chunk / "frame_66_endo.png"), cv2.resize(decoded[1][66], (32, 24)))
    with pytest.raises(ValueError, match="the annotated image"):
        extract_clip(**roots, clip="VID01_s6_60", count=10)


def test_a_mask_without_its_image_is_refused(source):
    roots, _ = source
    (roots["seg8k_root"] / "video01" / "video01_00060" / "frame_66_endo.png").unlink()
    with pytest.raises(FileNotFoundError, match="has no image"):
        extract_clip(**roots, clip="VID01_s6_60", count=10)


def test_a_match_on_the_upper_edge_of_the_window_is_refused(tmp_path):
    # Frame i is grey level i, so the image of frame 75 differs by 5 from frame 70, the last frame the search
    # for number 10 reads.
    roots = _grey_source(tmp_path, {10: 75, 16: 81})
    with pytest.raises(ValueError, match="matched frame 70, on the edge of"):
        extract_clip(**roots, clip="VID02_s6_10", count=2)


def test_a_match_on_the_lower_edge_of_the_window_is_refused(tmp_path):
    # The search for number 190 reads frames 2 to 199, so the image of frame 0 differs by 2 from frame 2, the
    # first frame read.
    roots = _grey_source(tmp_path, {190: 0, 196: 196})
    with pytest.raises(ValueError, match="matched frame 2, on the edge of"):
        extract_clip(**roots, clip="VID02_s6_190", count=2)


def test_a_frame_that_cannot_be_decoded_inside_the_window_is_refused(source, faulty_capture):
    # The search for number 60 reads frames 0 to 120. Were frame 100 skipped, the search would go on and the
    # image of number 102 would still match frame 102.
    roots, _ = source
    faulty_capture.fail_at = 100
    with pytest.raises(RuntimeError, match="cannot read frame 100 while matching"):
        extract_clip(**roots, clip="VID01_s6_60", count=10)


def test_a_video_without_a_frame_count_is_refused(source, faulty_capture):
    roots, _ = source
    faulty_capture.frame_count = 0
    with pytest.raises(RuntimeError, match="reports 0 frames"):
        extract_clip(**roots, clip="VID01_s6_60", count=10)


def test_a_gap_frame_that_cannot_be_decoded_is_refused_and_leaves_nothing(source, faulty_capture, monkeypatch):
    # Number 90 of video 1 has no mask, so frame 90 is read once, after a seek to it, when the clip is written.
    roots, _ = source
    faulty_capture.fail_at, faulty_capture.only_after_seek = 90, True
    with pytest.raises(RuntimeError, match="cannot read frame 90 of"):
        extract_clip(**roots, clip="VID01_s6_60", count=10)
    assert list(roots["out"].iterdir()) == []
    monkeypatch.undo()
    extract_clip(**roots, clip="VID01_s6_60", count=10)


def test_an_earlier_clip_is_refused_unless_replaced(source):
    roots, _ = source
    extract_clip(**roots, clip="VID01_s6_60", count=10)
    stale = roots["out"] / "VID01_s6_60" / "input_images" / "000099.png"
    stale.write_bytes(b"left by a longer run")
    with pytest.raises(ValueError, match="pass --overwrite"):
        extract_clip(**roots, clip="VID01_s6_60", count=10)
    extract_clip(**roots, clip="VID01_s6_60", count=10, overwrite=True)
    assert not stale.exists()


def test_a_refused_clip_leaves_the_earlier_clip_in_place(source):
    roots, _ = source
    extract_clip(**roots, clip="VID01_s6_60", count=10)
    before = sorted(p.name for p in (roots["out"] / "VID01_s6_60").rglob("*"))
    chunk = roots["seg8k_root"] / "video01" / "video01_00060"
    cv2.imwrite(str(chunk / "frame_66_endo.png"), np.full((SIZE[1], SIZE[0], 3), 7, np.uint8))
    with pytest.raises(ValueError, match="matches no video frame"):
        extract_clip(**roots, clip="VID01_s6_60", count=10, overwrite=True)
    assert sorted(p.name for p in (roots["out"] / "VID01_s6_60").rglob("*")) == before
    assert [p.name for p in roots["out"].iterdir()] == ["VID01_s6_60"]


def test_a_name_that_is_not_a_clip_is_refused():
    assert parse_clip("VID25_s15_162") == (25, 15, 162)
    for name in ("VID25_s15_162_crop", "VID1_s6_60"):
        with pytest.raises(ValueError, match="expected VID"):
            parse_clip(name)
