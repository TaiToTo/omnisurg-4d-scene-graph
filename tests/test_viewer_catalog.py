"""The viewer's catalog lists a clip only when the clip holds every file the viewer reads.

Each refusal is tested by planting its fault in an otherwise complete clip.
"""
import json
import re
from pathlib import Path

import pytest

from pipeline.viewer_catalog import CATALOG_FORMAT, build_catalog, main, parse_geometry, write_staged

REPO = Path(__file__).resolve().parent.parent
POSE = {"glb_centroid": [0, 0, -1], "camera_pos_glb": [0, 0, 0], "camera_forward_glb": [0, 0, -1], "camera_up_glb": [0, 1, 0]}
DEFAULT = ("da3", {})


def make_clip(root: Path, rel: str, n: int = 3, tracks=("cholecseg8k", "sam3d"), dataset=None, pi3x=False,
              hierarchy=True) -> Path:
    """Write a complete clip of `n` frames under `root/rel`, every file the viewer reads present."""
    clip = root / rel
    (clip / "pc_vis").mkdir(parents=True)
    (clip / "input_images").mkdir()
    frames = []
    for i in range(n):
        fr = {"seq_idx": i, **POSE}
        if pi3x:
            fr["geometry_sources"] = {"pi3x": dict(POSE)}
            (clip / f"pc_vis/frame_{i:04d}__pi3x.glb").write_bytes(b"glb")
        (clip / f"pc_vis/frame_{i:04d}.glb").write_bytes(b"glb")
        (clip / f"input_images/{i:06d}.png").write_bytes(b"png")
        for t in tracks:
            for kind in ("seg_frame", "graph_frame"):
                (clip / f"pc_vis/{kind}_{i:04d}__{t}.json").write_text("{}")
        if hierarchy:
            (clip / f"pc_vis/hierarchy_frame_{i:04d}.json").write_text("{}")
        frames.append(fr)
    for t in tracks:
        (clip / f"pc_vis/temporal_graph__{t}.json").write_text("{}")
    manifest = {"video_id": "VID01", "n_frames": n, "frames": frames}
    if dataset:
        manifest.update(dataset=dataset, procedure="adrenalectomy", youtube_id="abc")
    if pi3x:
        manifest["geometry_sources"] = {"pi3x": {"resolution": [4, 3]}}
    (clip / "frame_manifest.json").write_text(json.dumps(manifest))
    return clip


def test_a_complete_clip_is_listed_with_its_tracks_coverage_and_hierarchy(tmp_path):
    make_clip(tmp_path, "cholec/VID01_a")
    catalog, files = build_catalog(tmp_path, ["cholec/VID01_a"], DEFAULT)
    assert catalog["format"] == CATALOG_FORMAT
    assert catalog["clips"] == [{
        "path": "cholec/VID01_a", "dataset": "CholecSeg8k", "group": "VID01", "label": "VID01_a", "frames": 3,
        "geometry": "da3", "hierarchy": True, "tracks": ["cholecseg8k", "sam3d"],
        "coverage": {"cholecseg8k": 3, "sam3d": 3},
    }]
    assert set(catalog["tracks"]) == {"cholecseg8k", "sam3d"}
    assert "pc_vis/temporal_graph__sam3d.json" in files["cholec/VID01_a"]
    assert "pc_vis/frame_0002.glb" in files["cholec/VID01_a"]


def test_an_atlas_clip_is_grouped_by_procedure_and_video_and_drawn_from_its_dataset_s_source(tmp_path):
    make_clip(tmp_path, "atlas/x", tracks=("atlas_gt", "sam3d_edge"), dataset="atlas120k", pi3x=True, hierarchy=False)
    make_clip(tmp_path, "cholec/y", pi3x=True)
    catalog, files = build_catalog(tmp_path, ["atlas/x", "cholec/y"], parse_geometry(["pi3x", "cholecseg8k=da3"]))
    atlas, cholec = catalog["clips"]
    assert (atlas["dataset"], atlas["group"], atlas["geometry"], atlas["hierarchy"]) == ("ATLAS-120k", "adrenalectomy / abc", "pi3x", False)
    assert cholec["geometry"] == "da3"
    assert "pc_vis/frame_0000__pi3x.glb" in files["atlas/x"]


def test_a_track_with_overlays_on_some_frames_has_its_coverage_counted(tmp_path):
    clip = make_clip(tmp_path, "c")
    for kind in ("seg_frame", "graph_frame"):
        (clip / f"pc_vis/{kind}_0001__cholecseg8k.json").unlink()
    catalog, _ = build_catalog(tmp_path, ["c"], DEFAULT)
    assert catalog["clips"][0]["coverage"] == {"cholecseg8k": 2, "sam3d": 3}


def remove(name):
    return lambda clip: (clip / name).unlink()


def edit_manifest(fn):
    def plant(clip):
        m = json.loads((clip / "frame_manifest.json").read_text())
        fn(m)
        (clip / "frame_manifest.json").write_text(json.dumps(m))
    return plant


FAULTS = {
    "a frame without its point cloud": (remove("pc_vis/frame_0001.glb"), "no pc_vis/frame_0001.glb"),
    "a frame without its image": (remove("input_images/000002.png"), "no input_images/000002.png"),
    "a segmentation without its graph": (remove("pc_vis/graph_frame_0001__sam3d.json"), "not both"),
    "a track without its temporal graph": (remove("pc_vis/temporal_graph__sam3d.json"), "no pc_vis/temporal_graph__sam3d.json"),
    "a track the viewer does not name": (
        lambda c: [(c / f"pc_vis/{k}_0000__mystery.json").write_text("{}") for k in ("seg_frame", "graph_frame")],
        "track 'mystery' is not one the viewer names"),
    "an overlay past the last frame": (
        lambda c: [(c / f"pc_vis/{k}_0007__sam3d.json").write_text("{}") for k in ("seg_frame", "graph_frame")],
        "past the last"),
    "a frame without a pose": (edit_manifest(lambda m: m["frames"][1].pop("camera_up_glb")), "frame 1 has no da3 pose"),
    "a frame count that disagrees with the frames": (edit_manifest(lambda m: m.update(n_frames=4)), "n_frames is 4"),
    "a dataset the viewer does not name": (edit_manifest(lambda m: m.update(dataset="lapex")), "dataset 'lapex'"),
    "a hierarchy frame past the last frame": (
        lambda c: (c / "pc_vis/hierarchy_frame_0009.json").write_text("{}"), "a hierarchy frame past the last frame"),
    "a clip without any track": (
        lambda c: [p.unlink() for p in (c / "pc_vis").glob("*_frame_*__*.json")], "no track"),
}


@pytest.mark.parametrize("name", FAULTS)
def test_a_clip_missing_what_the_viewer_reads_is_refused(tmp_path, name):
    plant, message = FAULTS[name]
    plant(make_clip(tmp_path, "c"))
    with pytest.raises(ValueError, match=re.escape(message)):
        build_catalog(tmp_path, ["c"], DEFAULT)


def test_another_source_needs_its_pose_on_every_frame_and_its_resolution(tmp_path):
    clip = make_clip(tmp_path, "c", pi3x=True)
    m = json.loads((clip / "frame_manifest.json").read_text())
    del m["frames"][2]["geometry_sources"]
    del m["geometry_sources"]
    (clip / "frame_manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError) as e:
        build_catalog(tmp_path, ["c"], ("pi3x", {}))
    assert "frame 2 has no pi3x pose" in str(e.value) and "no resolution for pi3x" in str(e.value)


@pytest.mark.parametrize("rel", ["../c", "c/../d", "/c", "c/./d", "c?x=1", "c d"])
def test_a_clip_path_the_viewer_would_refuse_is_refused_here(tmp_path, rel):
    with pytest.raises(ValueError, match="relative path of plain names"):
        build_catalog(tmp_path, [rel], DEFAULT)


def test_every_refused_clip_is_named_and_a_clip_listed_twice_is_refused(tmp_path):
    make_clip(tmp_path, "good")
    (make_clip(tmp_path, "bad1") / "pc_vis/frame_0000.glb").unlink()
    (make_clip(tmp_path, "bad2") / "input_images/000000.png").unlink()
    with pytest.raises(ValueError) as e:
        build_catalog(tmp_path, ["good", "bad1", "bad2"], DEFAULT)
    assert "bad1:" in str(e.value) and "bad2:" in str(e.value) and "good:" not in str(e.value)
    with pytest.raises(ValueError, match="listed twice"):
        build_catalog(tmp_path, ["good", "good"], DEFAULT)


@pytest.mark.parametrize("specs, message", [
    (["pi3x", "da3"], "two sources"),
    (["lapex=pi3x"], "no dataset 'lapex'"),
    (["pi3x/.."], "is not SOURCE"),
])
def test_a_geometry_option_that_cannot_be_read_is_refused(specs, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        parse_geometry(specs)


def test_staging_copies_exactly_the_files_the_viewer_reads(tmp_path):
    root = tmp_path / "outputs"
    clip = make_clip(root, "cholec/VID01_a")
    (clip / "depth_raw").mkdir()
    (clip / "depth_raw/0000.npy").write_bytes(b"private")
    (clip / "pc_vis/cleanup_frame_0000__sam3d.json").write_text("{}")
    catalog, files = build_catalog(root, ["cholec/VID01_a"], DEFAULT)
    out = tmp_path / "site" / "data"
    count = write_staged(root, out, catalog, files)
    staged = sorted(str(p.relative_to(out / "cholec/VID01_a")) for p in (out / "cholec/VID01_a").rglob("*") if p.is_file())
    assert staged == sorted(files["cholec/VID01_a"]) and count == len(staged)
    assert json.loads((out / "catalog.json").read_text()) == catalog
    assert not (tmp_path / "site" / "data.partial").exists()


def test_staging_refuses_a_directory_that_exists(tmp_path):
    root = tmp_path / "outputs"
    make_clip(root, "c")
    catalog, files = build_catalog(root, ["c"], DEFAULT)
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        write_staged(root, tmp_path / "out", catalog, files)


def test_a_refused_clip_writes_nothing(tmp_path, monkeypatch):
    root = tmp_path / "outputs"
    (make_clip(root, "c") / "pc_vis/frame_0000.glb").unlink()
    monkeypatch.setattr("sys.argv", ["viewer_catalog", "--root", str(root), "--clips", "c", "--in-place"])
    with pytest.raises(SystemExit, match="nothing written"):
        main()
    assert not (root / "catalog.json").exists()


def test_the_format_string_is_the_one_the_viewer_reads():
    source = (REPO / "viewer" / "src" / "lib" / "catalog.js").read_text()
    assert f"export const CATALOG_FORMAT = '{CATALOG_FORMAT}';" in source
