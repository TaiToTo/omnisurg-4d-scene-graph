"""The ATLAS-120k metadata files agree with one another, and carry nothing private.

The files under `atlas120k_meta/` are copies of data made in the private
workbench, and nothing in this repository reads them yet. Until the readers
are ported, these checks are what keeps an edit to one file from quietly
contradicting another: the paper's population is one list, every clip in it
has a crop rectangle and a depth fingerprint, and none of the files carries a
path from the machine they were made on.
"""

import json
import re
import shutil
from pathlib import Path

import pytest

META = Path(__file__).resolve().parent.parent / "atlas120k_meta"

# Anything that looks like a filesystem path on a development machine. The
# workbench's manifests carried one such path when they were copied.
MACHINE_PATH = re.compile(r"/(home|var/autofs|mnt|Users)/")
# Any CJK character: the notes were translated on the way in, and a fresh copy
# from the workbench would bring Japanese back.
CJK = re.compile(r"[぀-ヿ一-鿿]")

CLIP_NAME = re.compile(r"[a-z_]+__[A-Za-z0-9_-]{11}__gt_\d{4}")


def _clips(meta: Path) -> list[str]:
    return meta.joinpath("clips.txt").read_text(encoding="utf-8").split()


def _key(clip: str) -> str:
    """`<procedure>__<video>__gt_<n>` → `<procedure>/<video>/clip_<n>`.

    A video id may itself start with `_`, so the split is on the first `__`
    and then on `__gt_`, never on every `__`.
    """
    procedure, rest = clip.split("__", 1)
    video, n = rest.rsplit("__gt_", 1)
    return f"{procedure}/{video}/clip_{n}"


def _check_depth_fingerprints(meta: Path) -> None:
    manifest = json.loads(meta.joinpath("depth_manifest.json").read_text())
    assert set(manifest["clips"]) == set(_clips(meta))
    assert manifest["n_clips"] == len(manifest["clips"])
    assert manifest["n_frames"] == sum(c["n_frames"] for c in manifest["clips"].values())
    for clip, entry in manifest["clips"].items():
        assert entry["shape"][0] == entry["n_frames"], clip
        assert re.fullmatch(r"[0-9a-f]{64}", entry["depth_sha256"]), clip
        assert re.fullmatch(r"[0-9a-f]{64}", entry["intrinsics_sha256"]), clip


def _check_crop_rectangles(meta: Path) -> None:
    rects = json.loads(meta.joinpath("crop_rects.json").read_text(encoding="utf-8"))
    by_key = {e["key"]: e for e in rects}
    assert len(by_key) == len(rects), "a clip was judged twice"
    missing = [c for c in _clips(meta) if _key(c) not in by_key]
    assert not missing, missing
    for e in rects:
        assert e["verdict"] in ("ok", "ng"), e["key"]
        x, y, w, h = e["rect"]
        sw, sh = e["src_size"]
        assert 0 <= x and 0 <= y and w > 0 and h > 0, e["key"]
        assert x + w <= sw and y + h <= sh, e["key"]


def test_population_is_one_list_of_distinct_clips():
    clips = _clips(META)
    assert len(clips) == len(set(clips))
    assert all(CLIP_NAME.fullmatch(c) for c in clips)


def test_every_clip_in_the_population_has_a_depth_fingerprint():
    _check_depth_fingerprints(META)


def test_every_clip_in_the_population_has_a_crop_rectangle():
    _check_crop_rectangles(META)


def test_video_manifest_and_its_clip_list_agree():
    """`videos.json` is the inventory before extraction and `clips.json` what
    came out, so clip numbers are not compared (one video's outputs were
    numbered by position at the time); videos and counts are."""
    videos = json.loads(META.joinpath("videos", "videos.json").read_text())
    clips = json.loads(META.joinpath("videos", "clips.json").read_text())
    population = [p.split() for p in META.joinpath("videos", "population.txt").read_text().split("\n") if p]
    assert len(population) == videos["n_videos"] == clips["n_videos"]
    kept = {f"{v['procedure']}__{v['video']}" for v in videos["videos"] if not v["excluded"]}
    assert kept == {f"{p}__{v}" for p, v in population}
    assert clips["n_clips"] == len(clips["clips"]) == videos["n_clips"]
    assert all(CLIP_NAME.fullmatch(c) for c in clips["clips"])
    assert {c.rsplit("__gt_", 1)[0] for c in clips["clips"]} <= kept


@pytest.mark.parametrize(
    "path", sorted(str(p.relative_to(META)) for p in META.rglob("*") if p.is_file()))
def test_no_machine_path_and_no_japanese(path):
    text = META.joinpath(path).read_text(encoding="utf-8")
    assert not MACHINE_PATH.search(text), path
    assert not CJK.search(text), path


def test_the_checks_fail_on_a_planted_clip_without_metadata(tmp_path):
    """A check earns its place by failing: a clip added to the population
    without a rectangle or a depth fingerprint must be caught by both."""
    fake = tmp_path / "atlas120k_meta"
    shutil.copytree(META, fake)
    with open(fake / "clips.txt", "a", encoding="utf-8") as f:
        f.write("cholecystectomy__00000000000__gt_9999\n")
    with pytest.raises(AssertionError):
        _check_crop_rectangles(fake)
    with pytest.raises(AssertionError):
        _check_depth_fingerprints(fake)
