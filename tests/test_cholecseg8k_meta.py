"""Check that the CholecSeg8k metadata files agree with one another and carry nothing private.

- The population is 25 windows of 16 videos.
- Every window's first frame and centre frame lie in an annotated chunk of `chunks.json`, and a window that
  breaks the rule is caught.
- Every clip of the population has a crop rectangle, and no other clip has one.
- The crop stage's own reader accepts the rectangles.
- No file carries a machine path, an email address or untranslated text, and the pattern that looks for
  them finds each one.
"""

import json
import re
import shutil
from pathlib import Path

import pytest

from pipeline.crop_cholecseg8k import load_rects

META = Path(__file__).resolve().parent.parent / "cholecseg8k_meta"
CLIP = re.compile(r"(VID\d{2})_s15_(\d+)_crop")
# A window: 30 frames, 15 CholecSeg8k frame numbers apart; its centre is frame 15.
WINDOW, STEP = 30, 15
PRIVATE = re.compile(r"/(home|var/autofs|mnt|Users)/|[぀-ヿ一-鿿！-｠]|[\w.+-]+@[\w-]+\.[\w.]+")


def _clips() -> list[str]:
    return (META / "clips.txt").read_text(encoding="utf-8").split()


def _check_windows(meta: Path) -> None:
    """Refuse a window whose first or centre frame lies in no annotated chunk, or whose video has no chunks."""
    chunks = json.loads((meta / "chunks.json").read_text(encoding="utf-8"))
    length, starts = chunks["chunk_frames"], chunks["chunk_starts"]
    for clip in (meta / "clips.txt").read_text(encoding="utf-8").split():
        video, start = CLIP.fullmatch(clip).group(1), int(CLIP.fullmatch(clip).group(2))
        assert video in starts, f"{clip}: {video} has no chunks in chunks.json"
        for which, frame in (("first", start), ("centre", start + STEP * (WINDOW // 2))):
            assert any(c <= frame < c + length for c in starts[video]), \
                f"{clip}: its {which} frame, {frame}, lies in no annotated chunk of {video}"


def test_the_population_is_25_windows_of_16_videos():
    clips = _clips()
    assert len(clips) == len(set(clips)) == 25
    assert all(CLIP.fullmatch(c) for c in clips), [c for c in clips if not CLIP.fullmatch(c)]
    assert len({CLIP.fullmatch(c).group(1) for c in clips}) == 16


def test_every_window_starts_on_a_gt_frame_and_has_one_at_its_centre():
    _check_windows(META)


def test_the_chunks_are_the_release_s_101_chunks_of_80_frames_in_17_videos():
    chunks = json.loads((META / "chunks.json").read_text(encoding="utf-8"))
    assert chunks["chunk_frames"] == 80 and len(chunks["chunk_starts"]) == 17
    assert sum(len(v) for v in chunks["chunk_starts"].values()) == 101


@pytest.mark.parametrize("clip, why", [
    ("VID20_s15_3260_crop", "centre frame, 3485"),  # the pilot's window: its centre lies between two chunks
    ("VID01_s15_60_crop", "first frame, 60"),  # starts before the video's first chunk
    ("VID02_s15_80_crop", "VID02 has no chunks"),
])
def test_a_window_that_breaks_the_rule_is_caught(tmp_path, clip, why):
    for name in ("clips.txt", "chunks.json"):
        shutil.copy(META / name, tmp_path / name)
    with open(tmp_path / "clips.txt", "a", encoding="utf-8") as f:
        f.write(clip + "\n")
    with pytest.raises(AssertionError, match=why):
        _check_windows(tmp_path)


def test_every_clip_of_the_population_has_a_rectangle_and_no_other_does():
    rects = load_rects(str(META / "crop_rects.json"))
    assert set(rects) == {c.removesuffix("_crop") for c in _clips()}


def test_no_file_carries_a_path_an_email_address_or_untranslated_text():
    for p in sorted(META.iterdir()):
        hits = PRIVATE.findall(p.read_text(encoding="utf-8"))
        assert not hits, (p.name, hits)


@pytest.mark.parametrize("text", [
    "root: /Users/someone/outputs/cholec_gt",
    "/home/someone/data", "/mnt/data", "/var/autofs/data",
    "someone@example.com",
    "crop は全クリップ均一",
    "ＶＩＤ０１",
])
def test_the_private_pattern_finds_what_it_is_for(text):
    assert PRIVATE.search(text), text


def test_the_private_pattern_passes_the_metadata_s_own_words():
    assert not PRIVATE.search("VID01_s15_80_crop  y0 y1 x0 x1 src_h src_w  cholec80 video01.mp4")
