"""The clip-index to mp4 frame ratio: reading the table, refusing what is not
in it, and checking a ratio against pixels.

The table is tested for being used safely: an unmeasured video is refused
rather than given a ratio of 1, a malformed table is refused as a whole, and
the committed table covers every video the paper measures. The pixel check is
tested on a synthetic mp4 whose ratio is known, by planting a wrong ratio and
watching it refused, and by planting each way the check could blame the table
for something else.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from surgical_core.atlas120k.frame_ratio import MIN_NATIVE_FRAME, MISS_FLOOR, FrameRatios

META = Path(__file__).resolve().parent.parent / "atlas120k_meta"

# The synthetic video: long enough for frame `MIN_NATIVE_FRAME` at the true
# ratio, and short enough that frame 60 at that ratio is past its end.
N_FRAMES = 200
TRUE_RATIO = 4


@pytest.fixture(scope="module")
def table():
    return FrameRatios.load(str(META / "frame_ratio.json"))


def _write(tmp_path, videos, match_tol=8.0):
    p = tmp_path / "frame_ratio.json"
    p.write_text(json.dumps({"match_tol": match_tol, "videos": videos}), encoding="utf-8")
    return str(p)


def test_measured_ratios_are_looked_up(table):
    assert table.ratio("appendectomy", "41RKDh3INiU") == 1
    assert table.mp4_index("appendectomy", "41RKDh3INiU", 1234) == 1234
    assert table.ratio("cholecystectomy", "1ud3syYKD3A") == 4
    assert table.mp4_index("cholecystectomy", "1ud3syYKD3A", 30) == 120
    assert table.mp4_index("liver_resection", "wAnvytUuDkA", 1152) == 2304


def test_frame_numbers_may_be_strings(table):
    """`clip_index.json` sometimes writes the number as a string."""
    assert table.mp4_index("cholecystectomy", "Va3QomCEaTE", "30") == 120


def test_unmeasured_video_is_refused(table):
    """A video not in the table is not given ratio 1. That silence is how a
    frame from another moment gets in."""
    with pytest.raises(KeyError, match="not been measured"):
        table.ratio("newprocedure", "newvideo")
    with pytest.raises(KeyError, match="not been measured"):
        table.mp4_index("newprocedure", "newvideo", 0)


def test_missing_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        FrameRatios.load(str(tmp_path / "no_such.json"))


def test_malformed_ratio_is_refused(tmp_path):
    for bad in (0, -1, 2.5, "2", True):
        p = _write(tmp_path, [{"procedure": "p", "video": "v", "ratio": bad}])
        with pytest.raises(ValueError, match="positive integer"):
            FrameRatios.load(p)


@pytest.mark.parametrize("bad", [True, "8", 0, -1, MISS_FLOOR, 80, float("nan")])
def test_malformed_match_tol_is_refused(tmp_path, bad):
    """A tolerance at or above the smallest miss passes a frame from another
    moment as a match, so the check could no longer fail."""
    p = _write(tmp_path, [{"procedure": "p", "video": "v", "ratio": 1}], match_tol=bad)
    with pytest.raises(ValueError, match="does not separate"):
        FrameRatios.load(p)


def test_video_measured_twice_is_refused(tmp_path):
    p = _write(tmp_path, [{"procedure": "p", "video": "v", "ratio": 1},
                          {"procedure": "p", "video": "v", "ratio": 2}])
    with pytest.raises(ValueError, match="twice"):
        FrameRatios.load(p)


def test_committed_table_covers_the_population(table):
    """Every video the paper measures was measured, and the ratios are the
    three that were observed."""
    assert len(table) == 97
    videos = set()
    for line in (META / "clips.txt").read_text(encoding="utf-8").split():
        procedure, rest = line.split("__", 1)
        videos.add((procedure, rest.rsplit("__gt_", 1)[0]))
    assert all(v in table for v in videos)
    ratios = {table.ratio(*v) for v in videos}
    assert ratios <= {1, 2, 3, 4}


def test_committed_table_matches_within_tolerance():
    """Each listed match is a real match: well inside the tolerance that
    separates a match from a miss."""
    data = json.loads((META / "frame_ratio.json").read_text(encoding="utf-8"))
    assert data["n_videos"] == len(data["videos"]) == 97
    for row in data["videos"]:
        assert row["mp4_frame"] == row["native_frame"] * row["ratio"], row
        assert row["diff"] < data["match_tol"] / 4, row


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """A synthetic mp4 whose frames all look different, and a JPEG of any frame.

    Each frame is a flat colour drawn at random with its number written on
    it, so two frames differ by far more than `match_tol` while a frame and
    its own JPEG differ only by the two compressions. mp4v seeks back to the
    requested frame, which the check relies on.
    """
    d = tmp_path_factory.mktemp("video")
    path = str(d / "v.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 60, (160, 120))
    assert writer.isOpened(), "this OpenCV cannot write mp4v"
    rng = np.random.default_rng(0)
    frames = []
    for i in range(N_FRAMES):
        frame = np.empty((120, 160, 3), np.uint8)
        frame[:] = rng.integers(0, 256, 3)
        cv2.putText(frame, str(i), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        writer.write(frame)
        frames.append(frame)
    writer.release()

    def jpg(mp4_frame: int) -> str:
        out = str(d / f"frame_{mp4_frame:06d}.jpg")
        cv2.imwrite(out, frames[mp4_frame])
        return out

    return path, jpg


def _one(ratio: int) -> FrameRatios:
    return FrameRatios({("p", "v"): ratio}, 8.0)


def test_verify_passes_the_right_ratio(video):
    path, jpg = video
    native = MIN_NATIVE_FRAME
    assert _one(TRUE_RATIO).verify_against_bundled(
        path, jpg(native * TRUE_RATIO), "p", "v", native) < 8.0


@pytest.mark.parametrize("wrong", [1, 2, 3])
def test_verify_refuses_a_wrong_ratio(video, wrong):
    """The failing case must fail: a stale ratio fetches a frame from another
    moment, and the check says so."""
    path, jpg = video
    native = MIN_NATIVE_FRAME
    with pytest.raises(RuntimeError, match="do not correspond"):
        _one(wrong).verify_against_bundled(path, jpg(native * TRUE_RATIO), "p", "v", native)


@pytest.mark.parametrize("native", [0, 2, MIN_NATIVE_FRAME - 1])
def test_verify_refuses_a_frame_too_early_to_tell_ratios_apart(video, native):
    """At frame 2, ratio 3 maps to mp4 frame 6 and the window reaches the
    true frame 8, so a wrong ratio would pass; at frame 0 every ratio maps to
    the same frame. Such a frame is refused, not checked."""
    path, jpg = video
    with pytest.raises(ValueError, match="too close to the start"):
        _one(3).verify_against_bundled(path, jpg(native * TRUE_RATIO), "p", "v", native)


def test_verify_reports_an_unopenable_video(video, tmp_path):
    """A missing video is not reported as a stale table."""
    _, jpg = video
    native = MIN_NATIVE_FRAME
    with pytest.raises(RuntimeError, match="cannot open the video"):
        _one(TRUE_RATIO).verify_against_bundled(
            str(tmp_path / "no_such.mp4"), jpg(native * TRUE_RATIO), "p", "v", native)


def test_verify_reports_a_position_past_the_end(video):
    """Frame 60 at ratio 4 is mp4 frame 240 of 200. Nothing is read there,
    and the message names both causes rather than the video alone."""
    path, jpg = video
    with pytest.raises(RuntimeError, match="no frame could be read.*ratio is too large"):
        _one(TRUE_RATIO).verify_against_bundled(path, jpg(0), "p", "v", 60)
