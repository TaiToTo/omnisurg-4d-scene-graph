"""The ATLAS-120k metadata files agree with one another, and carry nothing private.

The files under `atlas120k_meta/` are copies of data made in the private
workbench. The readers in `surgical_core/atlas120k/` check each file on its
own; these checks are what keeps an edit to one file from quietly
contradicting another: the paper's population is one list, every clip in it
has a crop rectangle and a depth fingerprint, no two of its clips share a
frame of the release, and none of the files carries a path from the machine
they were made on.
"""

import itertools
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path

import pytest

META = Path(__file__).resolve().parent.parent / "atlas120k_meta"

# Anything that looks like a filesystem path on a development machine. The
# workbench's manifests carried one such path when they were copied.
MACHINE_PATH = re.compile(r"/(home|var/autofs|mnt|Users)/")
# Any CJK character, including the full-width punctuation the workbench's notes
# used around dates: the notes were translated on the way in, and a fresh copy
# from the workbench would bring Japanese back.
CJK = re.compile(r"[぀-ヿ一-鿿！-｠]")
# An e-mail address. None of the files should name a person.
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")

CLIP_NAME = re.compile(r"[a-z_]+__[A-Za-z0-9_-]{11}__gt_\d{4}")

# The pairs of clips of one video whose `native_range`s intersect, with the
# length of the intersection. Two clips share at most that many frames, fewer
# when one skips frames inside its range. A new pair fails until looked at.
INTERSECTING_RANGES = {
    ("cholecystectomy/_-aytJndMV4", "clip_0005", "clip_0006"): 150,
    ("cholecystectomy/_-aytJndMV4", "clip_0006", "clip_0007"): 27,
    ("cholecystectomy/_-aytJndMV4", "clip_0007", "clip_0008"): 5,
    ("cholecystectomy/_-aytJndMV4", "clip_0008", "clip_0009"): 49,
    ("cholecystectomy/_-aytJndMV4", "clip_0009", "clip_0010"): 69,
    ("cholecystectomy/_-aytJndMV4", "clip_0010", "clip_0011"): 141,
    ("hemicolectomy/5YDMlxTl0k8", "clip_0016", "clip_0017"): 122,
    ("hemicolectomy/5YDMlxTl0k8", "clip_0016", "clip_0018"): 59,
}

# The clips of the release that do not hold every frame of their
# `native_range`, with how many they skip. A new one fails until looked at.
SKIPPED_FRAMES = {
    ("cholecystectomy/_-aytJndMV4", "clip_0005"): 329,
    ("cholecystectomy/_-aytJndMV4", "clip_0006"): 1355,
    ("cholecystectomy/_-aytJndMV4", "clip_0007"): 4,
    ("cholecystectomy/_-aytJndMV4", "clip_0008"): 4679,
    ("cholecystectomy/_-aytJndMV4", "clip_0009"): 840,
    ("cholecystectomy/_-aytJndMV4", "clip_0010"): 2265,
    ("hemicolectomy/5YDMlxTl0k8", "clip_0016"): 122,
}


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


def _release(meta: Path) -> dict[tuple[str, str], dict]:
    """Each clip of the release, `(procedure/video, clip_NNNN)`, to its entry in `videos/videos.json`."""
    videos = json.loads(meta.joinpath("videos", "videos.json").read_text())
    return {(f"{v['procedure']}/{v['video']}", c["clip"]): c for v in videos["videos"] for c in v["clips"]}


def _release_ranges(meta: Path) -> dict[tuple[str, str], tuple[int, int]]:
    return {k: tuple(c["native_range"]) for k, c in _release(meta).items()}


def _skipped_frames(meta: Path) -> dict[tuple[str, str], int]:
    """Each clip of the release whose `n_native` is not its range's length, to the difference."""
    out = {}
    for k, c in _release(meta).items():
        first, last = c["native_range"]
        if last - first + 1 != c["n_native"]:
            out[k] = last - first + 1 - c["n_native"]
    return out


def _intersections(ranges: dict[tuple[str, str], tuple[int, int]]) -> dict[tuple[str, str, str], int]:
    """Every pair of clips of one video whose ranges intersect, to the length of the intersection."""
    by_video = defaultdict(list)
    for (video, clip), r in ranges.items():
        by_video[video].append((clip, r))
    out = {}
    for video, clips in by_video.items():
        for (a, ra), (b, rb) in itertools.combinations(sorted(clips), 2):
            shared = min(ra[1], rb[1]) - max(ra[0], rb[0]) + 1
            if shared > 0:
                out[(video, a, b)] = shared
    return out


def _check_population_shares_no_frame(meta: Path) -> None:
    # A clip of the population is a run of frames inside one clip of the
    # release, so two whose release ranges are apart share no frame.
    ranges = _release_ranges(meta)
    keys = [tuple(_key(clip).rsplit("/", 1)) for clip in _clips(meta)]
    missing = [k for k in keys if k not in ranges]
    assert not missing, f"population clips with no range in videos/videos.json: {missing}"
    intersecting = _intersections({k: ranges[k] for k in keys})
    assert not intersecting, f"population clips whose release ranges intersect: {intersecting}"


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


def test_cut_marks_are_judgements_on_boundaries_of_known_videos():
    """The log is append-only, so a key may repeat; what must hold is that
    every line is a complete judgement and that the last one per key, the
    verdict, is a value a reader can act on."""
    population = {
        " ".join(p.split()) for p in META.joinpath("videos", "population.txt").read_text().split("\n") if p}
    last = {}
    for line in META.joinpath("cut_marks.jsonl").read_text(encoding="utf-8").splitlines():
        mark = json.loads(line)
        assert mark["key"] == f"{mark['procedure']}/{mark['video']}#{mark['cand']}"
        assert mark["is_cut"] in ("yes", "no", "dk", None), mark["key"]
        assert mark["cut_between"][0] + 1 == mark["cut_between"][1], mark["key"]
        last[mark["key"]] = mark
    assert all(m["is_cut"] is not None for m in last.values()), "a boundary was never judged"
    cuts = {f"{m['procedure']} {m['video']}" for m in last.values() if m["is_cut"] == "yes"}
    assert cuts <= population, cuts - population


def test_audits_cover_the_same_videos_and_clips_as_the_manifests():
    """Each audit was run on the earlier extraction's output, so it must list
    exactly the clips (or videos) that extraction wrote."""
    clips = json.loads(META.joinpath("videos", "clips.json").read_text())
    coverage = json.loads(META.joinpath("audit", "gt_coverage.json").read_text())
    assert {c["clip"] for c in coverage["clips"]} == set(clips["clips"])
    assert coverage["n_clips"] == len(coverage["clips"])
    population = {"__".join(p.split()) for p in META.joinpath("videos", "population.txt").read_text().split("\n") if p}
    residue = json.loads(META.joinpath("audit", "ui_residue.json").read_text())
    assert {v["video"] for v in residue["videos"]} == population
    rects = json.loads(META.joinpath("crop_rects.json").read_text(encoding="utf-8"))
    scope = json.loads(META.joinpath("audit", "crop_scope_table.json").read_text())
    assert {v["video"] for v in scope["videos"]} == {f"{e['procedure']}/{e['video']}" for e in rects}


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
    assert not EMAIL.search(text), path


def test_the_private_material_patterns_match_what_they_are_for():
    """The patterns above are only worth keeping if they catch what the
    workbench's files actually carried: a mounted home path, a Japanese note
    with full-width punctuation, and an address."""
    assert MACHINE_PATH.search("/var/autofs/nfs/disk0/usrs/someone/atlas_root")
    assert MACHINE_PATH.search("/home/someone/ws")
    assert CJK.search("矩形を一番下まで伸ばす（指示）")
    assert CJK.search("（2026-09-13）")
    assert EMAIL.search("someone@example.com")


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


def test_the_release_s_intersecting_ranges_are_the_known_ones():
    assert _intersections(_release_ranges(META)) == INTERSECTING_RANGES


def test_the_release_s_clips_that_skip_frames_are_the_known_ones():
    assert _skipped_frames(META) == SKIPPED_FRAMES


def test_no_two_clips_of_the_population_share_a_frame():
    _check_population_shares_no_frame(META)


def test_an_intersection_is_counted_and_a_population_clip_that_shares_frames_is_caught(tmp_path):
    """A range inside another intersects it by its own length, not by the distance to the other's end."""
    assert _intersections({("p/v", "clip_0001"): (0, 99), ("p/v", "clip_0002"): (10, 19),
                           ("p/v", "clip_0003"): (99, 120), ("p/w", "clip_0001"): (0, 99)}) == {
        ("p/v", "clip_0001", "clip_0002"): 10, ("p/v", "clip_0001", "clip_0003"): 1}
    fake = tmp_path / "atlas120k_meta"
    shutil.copytree(META, fake)
    with open(fake / "clips.txt", "a", encoding="utf-8") as f:
        f.write("cholecystectomy___-aytJndMV4__gt_0006\n")
    with pytest.raises(AssertionError, match="ranges intersect"):
        _check_population_shares_no_frame(fake)


def test_a_population_clip_missing_from_the_release_is_caught(tmp_path):
    fake = tmp_path / "atlas120k_meta"
    shutil.copytree(META, fake)
    with open(fake / "clips.txt", "a", encoding="utf-8") as f:
        f.write("cholecystectomy__00000000000__gt_9999\n")
    with pytest.raises(AssertionError, match="no range"):
        _check_population_shares_no_frame(fake)


def test_a_clip_that_loses_a_frame_inside_its_range_is_counted(tmp_path):
    fake = tmp_path / "atlas120k_meta"
    shutil.copytree(META, fake)
    path = fake / "videos" / "videos.json"
    videos = json.loads(path.read_text())
    video, clip = videos["videos"][0], videos["videos"][0]["clips"][0]
    clip["n_native"] -= 1
    path.write_text(json.dumps(videos))
    assert _skipped_frames(fake) == {
        **SKIPPED_FRAMES, (f"{video['procedure']}/{video['video']}", clip["clip"]): 1}
