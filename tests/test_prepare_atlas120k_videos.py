"""Test the video root on a release of two small videos made with FFmpeg: one H.264, one MPEG-4 part 2.

The tests skip where FFmpeg is not installed. Each refusal has a test that plants its fault.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import pipeline.prepare_atlas120k_videos as module
from pipeline.prepare_atlas120k_videos import check_converted, convert, main, prepare, probe

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="FFmpeg is not installed")

N_FRAMES, SIZE, RATE = 12, "64x48", 10


def make_video(path: Path, codec: str, n_frames: int = N_FRAMES, rate: int = RATE) -> None:
    """Write a test-pattern video of `n_frames` frames at `rate` fps in `codec` (`libx264` or `mpeg4`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", f"testsrc=size={SIZE}:rate={rate}", "-frames:v", str(n_frames), "-c:v", codec,
                    "-pix_fmt", "yuv420p", str(path)], check=True)


@pytest.fixture
def release(tmp_path):
    """A release with an H.264 video and an MPEG-4 video, and an empty `atlas120k/`."""
    src = tmp_path / "ATLAS"
    (src / "atlas120k" / "train").mkdir(parents=True)
    make_video(src / "raw_data" / "cholecystectomy" / "h264vid.mp4", "libx264")
    make_video(src / "raw_data" / "rarp" / "oddvid.mp4", "mpeg4")
    return src


def test_a_readable_video_is_linked_and_another_is_converted_and_recorded(release, tmp_path):
    dst = tmp_path / "ATLAS_h264"
    rows = prepare(release, dst)
    assert (dst / "atlas120k").is_symlink() and (dst / "atlas120k" / "train").is_dir()
    linked = dst / "raw_data" / "cholecystectomy" / "h264vid.mp4"
    assert linked.is_symlink() and linked.resolve() == (release / "raw_data" / "cholecystectomy" / "h264vid.mp4")
    converted = dst / "raw_data" / "rarp" / "oddvid.mp4"
    assert converted.is_file() and not converted.is_symlink()
    assert not list((dst / "raw_data" / "rarp").glob("*.part.*"))
    got = probe(converted)
    assert (got.codec, got.n_frames, got.frame_rate) == ("h264", N_FRAMES, f"{RATE}/1")
    assert [(r["procedure"], r["action"]) for r in rows] == [("cholecystectomy", "symlink"), ("rarp", "transcode")]
    assert rows[1]["src_codec"] == "mpeg4" and rows[1]["crf"] == 18 and rows[1]["frame_rate"] == f"{RATE}/1"
    record = json.loads((dst / "video_root.json").read_text())
    assert record["n_videos"] == 2 and record["n_transcoded"] == 1 and record["videos"] == rows


def test_a_converted_video_is_checked_and_not_made_again(release, tmp_path):
    dst = tmp_path / "ATLAS_h264"
    prepare(release, dst)
    converted = dst / "raw_data" / "rarp" / "oddvid.mp4"
    before = converted.stat().st_mtime_ns
    os.utime(converted, ns=(before - 10**9, before - 10**9))
    prepare(release, dst)
    assert converted.stat().st_mtime_ns == before - 10**9


def test_a_converted_video_at_another_crf_or_without_a_record_is_refused(release, tmp_path):
    dst = tmp_path / "ATLAS_h264"
    prepare(release, dst)
    with pytest.raises(ValueError, match="rarp/oddvid is already converted at crf 18, not 40"):
        prepare(release, dst, crf=40)
    (dst / "video_root.json").unlink()
    with pytest.raises(ValueError, match="rarp/oddvid is already converted, but .* does not record at what crf"):
        prepare(release, dst)


def test_a_conversion_that_changed_the_frame_count_is_refused(release, tmp_path):
    dst = tmp_path / "ATLAS_h264"
    # A shorter H.264 file stands where the conversion would be written, with the record a conversion leaves.
    make_video(dst / "raw_data" / "rarp" / "oddvid.mp4", "libx264", n_frames=N_FRAMES - 1)
    record_conversion(dst, "rarp", "oddvid", crf=18)
    with pytest.raises(ValueError, match=f"{N_FRAMES} frames to 64x48, {N_FRAMES - 1} frames"):
        prepare(release, dst)


def test_a_conversion_that_changed_the_frame_rate_is_refused(release, tmp_path):
    dst = tmp_path / "ATLAS_h264"
    # An H.264 file of the same frames at half the rate stands where the conversion would be written.
    make_video(dst / "raw_data" / "rarp" / "oddvid.mp4", "libx264", rate=RATE // 2)
    record_conversion(dst, "rarp", "oddvid", crf=18)
    with pytest.raises(ValueError, match=f"changed the frame rate from {RATE}/1 to {RATE // 2}/1"):
        prepare(release, dst)


def record_conversion(dst: Path, procedure: str, video: str, crf: int) -> None:
    """Write the `video_root.json` a conversion of one video at `crf` leaves, so that a planted file is checked."""
    rows = [dict(procedure=procedure, video=video, action="transcode", crf=crf)]
    (dst / "video_root.json").write_text(json.dumps(dict(videos=rows)), encoding="utf-8")


def test_a_converted_file_that_is_not_h264_is_refused(release, tmp_path):
    src = release / "raw_data" / "rarp" / "oddvid.mp4"
    with pytest.raises(ValueError, match="is mpeg4, not h264"):
        check_converted(src, src, probe(src))


def test_a_converted_file_opencv_cannot_read_is_refused(release, tmp_path, monkeypatch):
    src = release / "raw_data" / "cholecystectomy" / "h264vid.mp4"

    class Unreadable:
        def __init__(self, path):
            pass

        def isOpened(self):
            return True

        def read(self):
            return False, None

        def release(self):
            pass

    monkeypatch.setattr(module.cv2, "VideoCapture", Unreadable)
    with pytest.raises(RuntimeError, match="OpenCV cannot read a frame from the converted"):
        check_converted(src, src, probe(src))


def test_a_path_that_is_already_something_else_is_refused(release, tmp_path):
    dst = tmp_path / "ATLAS_h264"
    (dst / "atlas120k").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="is not a symlink to"):
        prepare(release, dst)
    shutil.rmtree(dst)
    (dst / "raw_data" / "rarp").mkdir(parents=True)
    (dst / "raw_data" / "rarp" / "oddvid.mp4").symlink_to(release / "raw_data" / "rarp" / "oddvid.mp4")
    with pytest.raises(FileExistsError, match="is a symlink, where a conversion"):
        prepare(release, dst)


def test_a_release_without_videos_is_refused(tmp_path):
    (tmp_path / "ATLAS" / "atlas120k").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no raw_data"):
        prepare(tmp_path / "ATLAS", tmp_path / "out")
    with pytest.raises(FileNotFoundError, match="no atlas120k/"):
        prepare(tmp_path / "nowhere", tmp_path / "out")


def test_a_file_without_a_video_stream_or_a_frame_count_is_refused(tmp_path):
    # An mp4 that holds only audio has no video stream.
    audio = tmp_path / "audio.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=d=1",
                    "-c:a", "aac", str(audio)], check=True)
    with pytest.raises(ValueError, match="has no video stream"):
        probe(audio)
    # A Matroska stream carries no frame count, whatever the file is called.
    make_video(tmp_path / "video.mkv", "libx264")
    matroska = tmp_path / "matroska.mp4"
    shutil.copy(tmp_path / "video.mkv", matroska)
    with pytest.raises(ValueError, match="does not say how many frames it has"):
        probe(matroska)
    # A file that is not a video at all is refused by ffprobe itself.
    text = tmp_path / "text.mp4"
    text.write_text("not a video")
    with pytest.raises(subprocess.CalledProcessError):
        probe(text)


def test_a_conversion_that_stops_leaves_no_file_at_the_destination(release, tmp_path, monkeypatch):
    src = release / "raw_data" / "rarp" / "oddvid.mp4"
    dst = tmp_path / "ATLAS_h264" / "raw_data" / "rarp" / "oddvid.mp4"
    dst.parent.mkdir(parents=True)

    def stops_half_way(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"half a video")
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(module.subprocess, "run", stops_half_way)
    with pytest.raises(subprocess.CalledProcessError):
        convert(src, dst, crf=18)
    assert not dst.exists()


def test_the_command_reports_a_refusal_and_exits_non_zero(release, tmp_path, monkeypatch):
    dst = tmp_path / "ATLAS_h264"
    (dst / "atlas120k").mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", ["pipeline.prepare_atlas120k_videos", "--src", str(release), "--dst", str(dst)])
    with pytest.raises(SystemExit, match="FileExistsError"):
        main()
