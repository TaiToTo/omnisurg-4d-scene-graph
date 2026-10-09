"""Check that the CholecSeg8k metadata files agree with one another and carry nothing private.

- The population is 27 windows of 17 videos.
- Every clip of the population has a crop rectangle, and no other clip has one.
- The crop stage's own reader accepts the rectangles.
- No file carries a machine path, an email address or untranslated text, and the pattern that looks for
  them finds each one.
"""

import re
from pathlib import Path

import pytest

from pipeline.crop_cholecseg8k import load_rects

META = Path(__file__).resolve().parent.parent / "cholecseg8k_meta"
CLIP = re.compile(r"(VID\d{2})_s15_\d+_crop")
PRIVATE = re.compile(r"/(home|var/autofs|mnt|Users)/|[぀-ヿ一-鿿！-｠]|[\w.+-]+@[\w-]+\.[\w.]+")


def _clips() -> list[str]:
    return (META / "clips.txt").read_text(encoding="utf-8").split()


def test_the_population_is_27_windows_of_17_videos():
    clips = _clips()
    assert len(clips) == len(set(clips)) == 27
    assert all(CLIP.fullmatch(c) for c in clips), [c for c in clips if not CLIP.fullmatch(c)]
    assert len({CLIP.fullmatch(c).group(1) for c in clips}) == 17


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
