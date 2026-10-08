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

from pipeline.prepare_atlas120k_videos import check_converted, main, prepare, probe

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="FFmpeg is not installed")

N_FRAMES, SIZE = 12, "64x48"


def make_video(path: Path, codec: str, n_frames: int = N_FRAMES) -> None:
    """Write a test-pattern video of `n_frames` frames in `codec` (`libx264` or `mpeg4`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                    "-i", f"testsrc=size={SIZE}:rate=10", "-frames:v", str(n_frames), "-c:v", codec,
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
    assert probe(converted).codec == "h264" and probe(converted).n_frames == N_FRAMES
    assert [(r["procedure"], r["action"]) for r in rows] == [("cholecystectomy", "symlink"), ("rarp", "transcode")]
    assert rows[1]["src_codec"] == "mpeg4" and rows[1]["crf"] == 18
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


def test_a_conversion_that_changed_the_frame_count_is_refused(release, tmp_path):
    dst = tmp_path / "ATLAS_h264"
    # A shorter H.264 file stands where the conversion would be written.
    make_video(dst / "raw_data" / "rarp" / "oddvid.mp4", "libx264", n_frames=N_FRAMES - 1)
    with pytest.raises(ValueError, match=f"{N_FRAMES} frames to 64x48, {N_FRAMES - 1} frames"):
        prepare(release, dst)


def test_a_converted_file_that_is_not_h264_is_refused(release, tmp_path):
    src = release / "raw_data" / "rarp" / "oddvid.mp4"
    with pytest.raises(ValueError, match="is mpeg4, not h264"):
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


def test_a_file_without_a_frame_count_is_refused(tmp_path):
    path = tmp_path / "text.mp4"
    path.write_text("not a video")
    with pytest.raises(subprocess.CalledProcessError):
        probe(path)


def test_the_command_reports_a_refusal_and_exits_non_zero(release, tmp_path, monkeypatch):
    dst = tmp_path / "ATLAS_h264"
    (dst / "atlas120k").mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", ["pipeline.prepare_atlas120k_videos", "--src", str(release), "--dst", str(dst)])
    with pytest.raises(SystemExit, match="FileExistsError"):
        main()
