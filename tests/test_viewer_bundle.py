"""Run `pipeline.viewer_bundle` on synthetic clips laid out as the export stage writes them, and check what it writes and refuses.

The clips' point clouds are written by `glb_bytes`, which lays a GLB out as
`write_point_cloud_glb` does, so these tests need no trimesh. The test that
reads a file the writer itself wrote skips without the `render` extra.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pipeline import viewer_bundle as vb
from surgical_core.atlas120k.frame_ratio import FrameRatios
from surgical_core.viewer.glb import read_point_cloud_glb

W, H = 4, 3            # DA3's grid in these clips
PW, PH = 6, 4          # Pi3X's grid
CLIP = "lap__vid__gt_0001"


def glb_bytes(points: np.ndarray, primitives: int = 1) -> bytes:
    """A point-cloud GLB as trimesh writes one: one mesh, `primitives` primitives, float32 POSITION, uint8 COLOR_0."""
    pts = np.asarray(points, "<f4")
    cols = np.full((len(pts), 4), 200, np.uint8)
    blob = pts.tobytes() + cols.tobytes()
    gltf = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "mode": 0}] * primitives}],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(pts), "type": "VEC3"},
                          {"bufferView": 1, "componentType": 5121, "count": len(pts), "type": "VEC4",
                           "normalized": True}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": pts.nbytes},
                            {"buffer": 0, "byteOffset": pts.nbytes, "byteLength": cols.nbytes}],
            "buffers": [{"byteLength": len(blob)}]}
    js = json.dumps(gltf).encode()
    js += b" " * (-len(js) % 4)
    blob += b"\0" * (-len(blob) % 4)
    body = struct.pack("<I4s", len(js), b"JSON") + js + struct.pack("<I4s", len(blob), b"BIN\0") + blob
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def grid_points(w: int, h: int, scale: float = 1.0) -> np.ndarray:
    """One point per pixel of a w x h grid, row-major, x along the columns and y down the rows."""
    v, u = np.mgrid[0:h, 0:w]
    return np.stack([u * scale, -v * scale, np.zeros_like(u, float)], -1).reshape(-1, 3)


def two_regions(w: int, h: int) -> np.ndarray:
    """Region 1 on the left half, region 2 on the right half, row-major."""
    lab = np.ones((h, w), np.int64)
    lab[:, w // 2:] = 2
    return lab.ravel()


def seg(labels, track="sam3d", stage="anchor", w=W, h=H, classes=(1, 2)) -> dict:
    return {"width": w, "height": h, "src_colors": [[0, 0, 0]] * (w * h), "seg_colors": [[0, 0, 0]] * (w * h),
            "classes": [{"id": c, "name": f"obj {c}", "color": [0.1 * c, 0.2, 0.3], "centroid": [0, 0]}
                        for c in classes],
            "vertex_seg": [int(v) for v in labels], "track": track,
            "provenance": {"track": track, "stage": stage, "source": "normal_map+sam", "anchor_frame": None}}


def graph(track="sam3d") -> dict:
    node = {"label": "x", "pos": [9.0, 9.0, 9.0], "color": [0.1, 0.2, 0.3], "area_frac": 0.5,
            "axes": np.eye(3).tolist(), "axes_radius": [1, 1, 1], "extent_3d": [1, 1, 1]}
    return {"nodes": [{"id": 1, **node}, {"id": 2, **node}],
            "edges": [{"src": 1, "dst": 2, "relation": "left", "edge_type": "spatial"},
                      {"src": 2, "dst": 1, "relation": "right", "edge_type": "spatial"}],
            "track": track, "provenance": {"track": track, "stage": "anchor"}}


def make_clip(root, n=3, dataset=None, tracks=("sam3d",), pi3x=True, name=CLIP, frame_ratio=1,
              video=("lar", "vid")):
    """A clip of `n` frames with DA3 and Pi3X clouds, and every track's regions and graphs on every frame.

    Frame `i` is native frame `30 * i` at 30 fps. An ATLAS-120k clip records `frame_ratio`, as the extraction
    stage writes it; a CholecSeg8k clip records none.
    """
    c = root / name
    (c / "input_images").mkdir(parents=True)
    (c / "pc_vis").mkdir()
    frames = []
    for i in range(n):
        Image.fromarray(np.full((H, W, 3), 40 * i, np.uint8)).save(c / "input_images" / f"{i:06d}.png")
        (c / "pc_vis" / f"frame_{i:04d}.glb").write_bytes(glb_bytes(grid_points(W, H)))
        place = {"glb_centroid": [i, 0, 0], "camera_pos_glb": [0, 0, 1], "camera_forward_glb": [0, 0, -1],
                 "camera_up_glb": [0, 1, 0]}
        frame = {"seq_idx": i, "native_frame": 30 * i, **place}
        if pi3x:
            (c / "pc_vis" / f"frame_{i:04d}__pi3x.glb").write_bytes(glb_bytes(grid_points(PW, PH, 2.0)))
            frame["geometry_sources"] = {"pi3x": {**place, "glb_centroid": [2 * i, 0, 0], "n_vertices": PW * PH}}
        frames.append(frame)
        for t in tracks:
            (c / "pc_vis" / f"seg_frame_{i:04d}__{t}.json").write_text(json.dumps(seg(two_regions(W, H), t)))
            (c / "pc_vis" / f"graph_frame_{i:04d}__{t}.json").write_text(json.dumps(graph(t)))
    for t in tracks:
        (c / "pc_vis" / f"temporal_graph__{t}.json").write_text(json.dumps(
            {"track": t, "nodes": [{"id": 1, "label": "obj 1", "present_frames": list(range(n))}],
             "relations": [{"src": 1, "dst": 2, "relation": "left", "frames": [0, 1]}]}))
    manifest = {"n_frames": n, "fps_native": 30.0, "frames": frames,
                "depth_info": {"depth_shape": [n, H, W]},
                "geometry_sources": {"pi3x": {"resolution": [PW, PH]}} if pi3x else {}}
    if dataset:
        manifest.update(dataset=dataset, procedure=video[0], youtube_id=video[1], frame_ratio=frame_ratio)
    else:
        manifest["video_id"] = "VID01"
    (c / "frame_manifest.json").write_text(json.dumps(manifest))
    return c


def read(path):
    return json.loads(path.read_text())


def test_runs_round_trip_and_hold_value_count_pairs():
    labels = np.array([0, 0, 3, 3, 3, 0, 7])
    assert vb.encode_runs(labels) == [0, 2, 3, 3, 0, 1, 7, 1]
    assert vb.decode_runs(vb.encode_runs(labels)).tolist() == labels.tolist()
    assert vb.encode_runs(np.array([], int)) == []


def test_the_grid_remap_takes_each_pixel_from_the_nearest_source_pixel():
    assert vb.grid_remap_index(3, 2, 3, 2).tolist() == list(range(6))
    # Pixel centres, not corners: the right half of a 3-pixel row is nearest its last pixel.
    assert vb.grid_remap_index(3, 1, 2, 1).tolist() == [0, 2]
    assert vb.grid_remap_index(1, 3, 1, 2).tolist() == [0, 2]
    # Each pixel of a 2x2 grid spreads over a 2x2 block of the 4x4 one.
    assert vb.grid_remap_index(2, 2, 4, 4).reshape(4, 4).tolist() == [
        [0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]]
    # The viewer's earlier JavaScript mapping, written out for one row and one column of a real pair of grids.
    m = vb.grid_remap_index(504, 350, 602, 420).reshape(420, 602)
    assert m[0, 601] == 503 and m[419, 0] == 349 * 504
    assert m[210, 301] == int((210.5 * 350) // 420) * 504 + int((301.5 * 504) // 602)


def test_region_geometry_gives_the_centroid_and_the_longest_axis_first():
    pts = np.array([[0, 0, 0], [2, 0, 0], [4, 0, 0], [6, 0, 0], [0, 5, 0], [9, 9, 9]], float)
    labels = np.array([1, 1, 1, 1, 0, 2])
    g = vb.region_geometry(pts, labels)
    assert set(g) == {1, 2}
    assert g[1]["pos"].tolist() == [3, 0, 0]
    assert abs(g[1]["axes"][0][0]) == pytest.approx(1)
    assert g[1]["axes_radius"][0] == pytest.approx(2 * np.std([0, 2, 4, 6]))
    assert g[2]["axes"].tolist() == np.eye(3).tolist() and g[2]["axes_radius"].tolist() == [0, 0, 0]


def test_region_axes_are_a_rotation_not_a_reflection():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(600, 3)) * [3, 2, 1]
    labels = rng.integers(1, 40, size=600)
    for g in vb.region_geometry(pts, labels).values():
        assert np.linalg.det(g["axes"]) == pytest.approx(1.0)


def test_a_node_without_points_on_the_new_cloud_is_left_out_with_its_edges():
    g = vb.place_graph(graph(), {1: {"pos": np.array([1.0, 2, 3]), "axes": np.eye(3), "axes_radius": np.ones(3)}})
    assert [n["id"] for n in g["nodes"]] == [1] and g["edges"] == []
    assert g["nodes"][0]["pos"] == [1, 2, 3] and g["nodes"][0]["label"] == "x"
    assert g["nodes"][0]["graph_pos"] == [9.0, 9.0, 9.0]


def test_instrument_ids_are_the_tool_classes_of_the_class_tables():
    assert vb.instrument_ids("cholecseg8k") == [5, 9]
    assert vb.instrument_ids("atlas120k") == [1, 30, 41]


def test_frame_times_come_from_clip_time_with_the_atlas_frame_ratio(tmp_path):
    # Frame i is native frame 30 i at 30 fps; with a ratio of 2 the video runs twice as far per frame.
    clip = make_clip(tmp_path / "in", dataset="atlas120k", tracks=("atlas_gt",), frame_ratio=2)
    rec = vb.bundle_clip(clip, tmp_path / "out", ["atlas_gt"])
    assert [f["time_s"] for f in rec["frames"]] == [0.0, 2.0, 4.0]
    # A frame with a timestamp is taken as it is.
    m = read(clip / "frame_manifest.json")
    for i, f in enumerate(m["frames"]):
        f["timestamp_sec"] = 7.25 + i
    (clip / "frame_manifest.json").write_text(json.dumps(m))
    rec = vb.bundle_clip(clip, tmp_path / "out", ["atlas_gt"], overwrite=True)
    assert [f["time_s"] for f in rec["frames"]] == [7.25, 8.25, 9.25]


def test_an_atlas_clip_without_a_recorded_ratio_is_timed_from_the_table_of_measured_ratios(tmp_path, monkeypatch):
    # The table measures this video at 1; `frame_times` refuses the clip without the table.
    clip = make_clip(tmp_path / "in", dataset="atlas120k", tracks=("atlas_gt",),
                     video=("adrenalectomy", "16GPCUPkXYQ"))
    m = read(clip / "frame_manifest.json")
    m.pop("frame_ratio")
    (clip / "frame_manifest.json").write_text(json.dumps(m))
    table = str(Path(vb.__file__).resolve().parent.parent / "atlas120k_meta" / "frame_ratio.json")
    rec = vb.bundle_clip(clip, tmp_path / "out", ["atlas_gt"], ratios=FrameRatios.load(table))
    assert [f["time_s"] for f in rec["frames"]] == [0.0, 1.0, 2.0]
    monkeypatch.setattr(sys, "argv", ["viewer_bundle", "--input-dir", str(tmp_path / "in"), "--out",
                                      str(tmp_path / "out2"), "--clips", CLIP, "--tracks", "atlas_gt",
                                      "--frame-ratios", table])
    vb.main()
    assert read(tmp_path / "out2" / CLIP / "clip.json")["frames"][2]["time_s"] == 2.0


@pytest.mark.parametrize("fault,match", [
    (lambda m: m.pop("fps_native"), "cannot make frame times"),
    (lambda m: m.pop("frame_ratio"), "no `frame_ratio`"),
    (lambda m: m["frames"][1].update(timestamp_sec=2.0), "1 of 3 frames carry"),
])
def test_a_clip_whose_frame_times_cannot_be_made_is_refused(tmp_path, fault, match):
    clip = make_clip(tmp_path / "in", dataset="atlas120k", tracks=("atlas_gt",), frame_ratio=2)
    (tmp_path / "out").mkdir()
    m = read(clip / "frame_manifest.json")
    fault(m)
    (clip / "frame_manifest.json").write_text(json.dumps(m))
    _fails(tmp_path, match, tracks=("atlas_gt",))


def test_a_da3_clip_gets_its_frames_clouds_regions_graphs_and_a_catalog_entry(tmp_path):
    clip = make_clip(tmp_path / "in", tracks=("cholecseg8k", "sam3d"))
    for i in range(3):
        (clip / "pc_vis" / f"hierarchy_frame_{i:04d}.json").write_text(
            json.dumps({"tracks": ["cholecseg8k", "sam3d"], "nodes": [], "edges": []}))
    out = tmp_path / "out"
    rec = vb.bundle_clip(clip, out, ["sam3d", "cholecseg8k"])
    d = out / CLIP
    assert sorted(p.name for p in d.iterdir()) == ["clip.json", "clouds", "frames", "graphs", "hierarchy", "regions"]
    assert read(d / "clip.json") == rec
    assert rec["geometry"] == {"source": "da3", "name": "Depth Anything 3", "grid": [W, H], "graphs_built_on": "da3"}
    assert [t["id"] for t in rec["tracks"]] == ["sam3d", "cholecseg8k"]
    assert rec["tracks"][1]["instrument_ids"] == [5, 9] and rec["tracks"][0]["instrument_ids"] == []
    assert rec["tracks"][0]["region_frames"] == [0, 1, 2] and rec["tracks"][0]["temporal"]
    assert rec["hierarchy"] == {"tracks": ["sam3d", "cholecseg8k"], "frames": [0, 1, 2]}
    assert rec["frames"][1]["centroid"] == [1, 0, 0] and rec["frames"][2]["time_s"] == 2.0
    assert (rec["procedure"], rec["video"], rec["source_url"]) == ("Cholecystectomy", "VID01", None)
    # The clouds are the export's files; the regions decode to its labels; the graphs are untouched on DA3.
    assert (d / "clouds" / "0001.glb").read_bytes() == (clip / "pc_vis" / "frame_0001.glb").read_bytes()
    r = read(d / "regions" / "sam3d" / "0002.json")
    assert vb.decode_runs(r["runs"]).tolist() == two_regions(W, H).tolist() and r["grid"] == [W, H]
    assert read(d / "graphs" / "sam3d" / "0002.json") == graph()
    assert Image.open(d / "frames" / "0002.jpg").size == (W, H)
    entries = vb.write_catalog(out)
    assert read(out / "catalog.json") == {"format": vb.FORMAT, "clips": entries}
    assert entries[0]["tracks"] == [{"id": "sam3d", "short": "Zero-shot"},
                                    {"id": "cholecseg8k", "short": "Manual annotation"}]


def test_pi3x_resamples_the_labels_and_places_the_nodes_on_its_cloud(tmp_path):
    clip = make_clip(tmp_path / "in")
    rec = vb.bundle_clip(clip, tmp_path / "out", ["sam3d"], geometry="pi3x")
    d = tmp_path / "out" / CLIP
    assert rec["geometry"]["grid"] == [PW, PH] and rec["frames"][1]["centroid"] == [2, 0, 0]
    assert (d / "clouds" / "0000.glb").read_bytes() == (clip / "pc_vis" / "frame_0000__pi3x.glb").read_bytes()
    labels = vb.decode_runs(read(d / "regions" / "sam3d" / "0000.json")["runs"])
    assert labels.tolist() == two_regions(PW, PH).tolist()
    # Region 1 is the left three columns of the 6x4 Pi3X grid, two units apart: its centroid is at x = 2.
    nodes = {n["id"]: n for n in read(d / "graphs" / "sam3d" / "0000.json")["nodes"]}
    assert nodes[1]["pos"] == pytest.approx([2.0, -3.0, 0.0])
    assert nodes[2]["pos"] == pytest.approx([8.0, -3.0, 0.0])
    assert nodes[1]["extent_3d"] == [1, 1, 1] and nodes[1]["graph_pos"] == [9.0, 9.0, 9.0]
    # The relations were chosen in DA3's camera, so its axes travel with Pi3X's.
    assert rec["frames"][0]["graph_camera_forward"] == [0, 0, -1]
    assert "graph_camera_forward" not in vb.bundle_clip(clip, tmp_path / "da3", ["sam3d"])["frames"][0]


def test_an_atlas_clip_names_its_procedure_and_links_its_video(tmp_path):
    clip = make_clip(tmp_path / "in", dataset="atlas120k", tracks=("atlas_gt",))
    for i in range(3):
        (clip / "pc_vis" / f"seg_frame_{i:04d}__atlas_gt.json").write_text(
            json.dumps(seg(two_regions(W, H), "atlas_gt", classes=(1, 2))))
    rec = vb.bundle_clip(clip, tmp_path / "out", ["atlas_gt"])
    assert (rec["dataset_name"], rec["procedure"]) == ("ATLAS-120k", "Low anterior resection")
    assert rec["source_url"] == "https://www.youtube.com/watch?v=vid"
    assert rec["tracks"][0]["instrument_ids"] == [1, 30, 41]


def test_a_track_covers_only_the_frames_it_has_files_for(tmp_path):
    clip = make_clip(tmp_path / "in")
    (clip / "pc_vis" / "seg_frame_0001__sam3d.json").unlink()
    (clip / "pc_vis" / "graph_frame_0001__sam3d.json").unlink()
    (clip / "pc_vis" / "graph_frame_0002__sam3d.json").unlink()
    (clip / "pc_vis" / "temporal_graph__sam3d.json").unlink()
    rec = vb.bundle_clip(clip, tmp_path / "out", ["sam3d"])
    t = rec["tracks"][0]
    assert (t["region_frames"], t["graph_frames"], t["temporal"]) == ([0, 2], [0], False)
    assert sorted(p.name for p in (tmp_path / "out" / CLIP / "regions" / "sam3d").iterdir()) == ["0000.json", "0002.json"]


def test_a_hierarchy_with_a_track_left_out_is_not_copied(tmp_path):
    clip = make_clip(tmp_path / "in", tracks=("cholecseg8k", "sam3d"))
    (clip / "pc_vis" / "hierarchy_frame_0000.json").write_text(
        json.dumps({"tracks": ["cholecseg8k", "sam3d"], "nodes": [], "edges": []}))
    rec = vb.bundle_clip(clip, tmp_path / "out", ["sam3d"])
    assert rec["hierarchy"] is None and not (tmp_path / "out" / CLIP / "hierarchy").exists()


def _fails(tmp_path, match, tracks=("sam3d",), geometry="da3", **kw):
    clip = tmp_path / "in" / CLIP
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        vb.bundle_clip(clip, tmp_path / "out", list(tracks), geometry, **kw)
    # A refused clip leaves nothing behind.
    assert not (tmp_path / "out" / CLIP).exists() and not list((tmp_path / "out").glob(".*"))


def test_unknown_tracks_and_tracks_of_another_dataset_are_refused(tmp_path):
    make_clip(tmp_path / "in")
    (tmp_path / "out").mkdir()
    _fails(tmp_path, "unknown track", tracks=("sam4d",))
    _fails(tmp_path, "labels atlas120k classes", tracks=("atlas_gt",))
    _fails(tmp_path, "unknown geometry", geometry="da2")


def test_a_track_without_regions_or_with_a_graph_but_no_regions_is_refused(tmp_path):
    clip = make_clip(tmp_path / "in")
    (tmp_path / "out").mkdir()
    _fails(tmp_path, r"no pc_vis/seg_frame_NNNN__sam3d_edge", tracks=("sam3d_edge",))
    (clip / "pc_vis" / "seg_frame_0001__sam3d.json").unlink()
    _fails(tmp_path, r"without regions on frames \[1\]")


@pytest.mark.parametrize("fault,match", [
    (lambda s: s.update(width=W + 1), "one label per pixel"),
    (lambda s: s.update(width=H, height=W), "one label per pixel"),
    (lambda s: s.update(vertex_seg=s["vertex_seg"][:-1]), "one label per pixel"),
    (lambda s: s["vertex_seg"].__setitem__(0, 5), r"labels \[5\] are no class"),
    (lambda s: s["vertex_seg"].__setitem__(0, -1), r"labels \[-1\] are no class"),
    (lambda s: s["provenance"].update(stage="seed"), "neither anchor nor propagated"),
    (lambda s: s["vertex_seg"].__setitem__(0, 1.5), "not integers"),
    (lambda s: s["classes"][0].update(color="#ff0000"), "class 1 is not"),
    (lambda s: s["classes"][0].update(color=[0.1, 0.2]), "class 1 is not"),
    (lambda s: s["classes"][0].update(name=["Liver"]), "class 1 is not"),
    (lambda s: s["classes"][0].update(id="1"), "class '1' is not"),
    (lambda s: s["classes"][0].update(id=0), "class 0 is not"),
])
def test_regions_that_do_not_fit_the_clip_are_refused(tmp_path, fault, match):
    clip = make_clip(tmp_path / "in")
    (tmp_path / "out").mkdir()
    path = clip / "pc_vis" / "seg_frame_0002__sam3d.json"
    s = read(path)
    fault(s)
    path.write_text(json.dumps(s))
    _fails(tmp_path, match)


@pytest.mark.parametrize("fault,match", [
    (lambda g: g["nodes"][0].update(id=3), "node 3 is not a region"),
    (lambda g: g["nodes"][0].update(id="1"), "node '1' is not a region"),
    (lambda g: g["nodes"][0].update(label=["x"]), "node 1 is not a region"),
    (lambda g: g["nodes"][0].update(pos=[1.0, 2.0]), "node 1 is not a region"),
    (lambda g: g["nodes"][0].update(pos=[1.0, 2.0, "3"]), "node 1 is not a region"),
    (lambda g: g["edges"][0].update(dst=7), "edge of 1 and 7"),
    (lambda g: g["edges"][0].update(relation=5), "edge of 1 and 2"),
])
def test_a_frame_graph_of_another_shape_or_off_the_frames_regions_is_refused(tmp_path, fault, match):
    clip = make_clip(tmp_path / "in")
    (tmp_path / "out").mkdir()
    path = clip / "pc_vis" / "graph_frame_0001__sam3d.json"
    g = read(path)
    fault(g)
    path.write_text(json.dumps(g))
    _fails(tmp_path, match)
    _fails(tmp_path, match, geometry="pi3x")


@pytest.mark.parametrize("fault,match", [
    (lambda tg: tg["relations"][0].update(frames=["<img src=x onerror=alert(1)>"]), "relation of 1 and 2"),
    (lambda tg: tg["relations"][0].update(frames=[3]), "frames 0 to 2"),
    (lambda tg: tg["relations"][0].update(src="1"), "relation of"),
    (lambda tg: tg["relations"][0].update(relation=["left"]), "relation of 1 and 2"),
    (lambda tg: tg["nodes"][0].update(present_frames=[0.5]), "node 1"),
    (lambda tg: tg["nodes"][0].update(label=["x"]), "node 1"),
])
def test_a_graph_through_time_of_another_shape_is_refused(tmp_path, fault, match):
    clip = make_clip(tmp_path / "in")
    (tmp_path / "out").mkdir()
    path = clip / "pc_vis" / "temporal_graph__sam3d.json"
    tg = read(path)
    fault(tg)
    path.write_text(json.dumps(tg))
    _fails(tmp_path, match)


HIERARCHY = {"tracks": ["cholecseg8k", "sam3d"],
             "nodes": [{"key": "cholecseg8k:1", "track": "cholecseg8k", "id": 1, "label": "Liver"},
                       {"key": "sam3d:2", "track": "sam3d", "id": 2, "label": "obj 2"}],
             "edges": [{"src": "cholecseg8k:1", "dst": "sam3d:2", "relation": "contains", "edge_type": "hierarchy"}]}


@pytest.mark.parametrize("fault,match", [
    (lambda h: h.update(tracks="cholecseg8k"), "tracks 'cholecseg8k' are not two of"),
    (lambda h: h.update(tracks=["sam3d", "sam3d"]), "are not two of"),
    (lambda h: h["nodes"][0].update(key=1), "node 1 is not a string key"),
    (lambda h: h["nodes"][0].update(track="atlas_gt"), "node 'cholecseg8k:1' is not"),
    (lambda h: h["nodes"][0].update(id="1"), "node 'cholecseg8k:1' is not"),
    (lambda h: h["nodes"][0].update(label=None), "node 'cholecseg8k:1' is not"),
    (lambda h: h["edges"][0].update(dst="sam3d:9"), "edge of 'cholecseg8k:1' and 'sam3d:9'"),
    (lambda h: h["edges"][0].update(relation=["contains"]), "edge of 'cholecseg8k:1' and 'sam3d:2'"),
])
def test_a_hierarchy_of_another_shape_is_refused(tmp_path, fault, match):
    clip = make_clip(tmp_path / "in", tracks=("cholecseg8k", "sam3d"))
    (tmp_path / "out").mkdir()
    h = json.loads(json.dumps(HIERARCHY))
    fault(h)
    (clip / "pc_vis" / "hierarchy_frame_0002.json").write_text(json.dumps(h))
    _fails(tmp_path, match, tracks=("cholecseg8k", "sam3d"))


@pytest.mark.parametrize("fault,match,geometry", [
    (lambda m: m.update(n_frames=4), "frames 0 to 2 once each", "da3"),
    (lambda m: m.update(frames=[], n_frames=0), "frames 0 to -1 once each", "da3"),
    (lambda m: m["depth_info"].pop("depth_shape"), "depth_shape does not give", "da3"),
    (lambda m: m["depth_info"].update(depth_shape=[2, H, W]), "depth_shape does not give", "da3"),
    (lambda m: m["geometry_sources"]["pi3x"].pop("resolution"), "pi3x.resolution is missing", "pi3x"),
    (lambda m: m.update(dataset="lapex"), "unknown dataset 'lapex'", "da3"),
    (lambda m: m["frames"][1].pop("camera_forward_glb"), "frame 1 has no DA3 camera axes", "pi3x"),
    (lambda m: m["frames"][1].pop("camera_forward_glb"), "frame 1 has no da3 placement", "da3"),
])
def test_a_manifest_that_does_not_describe_the_clip_is_refused(tmp_path, fault, match, geometry):
    clip = make_clip(tmp_path / "in")
    (tmp_path / "out").mkdir()
    m = read(clip / "frame_manifest.json")
    fault(m)
    (clip / "frame_manifest.json").write_text(json.dumps(m))
    _fails(tmp_path, match, geometry=geometry)


def test_a_cloud_without_a_point_per_pixel_is_refused(tmp_path):
    clip = make_clip(tmp_path / "in")
    (tmp_path / "out").mkdir()
    (clip / "pc_vis" / "frame_0001.glb").write_bytes(glb_bytes(grid_points(W, H)[:-1]))
    _fails(tmp_path, r"frame_0001.glb holds 11 points for a 4x3 grid")
    (clip / "pc_vis" / "frame_0001__pi3x.glb").write_bytes(glb_bytes(grid_points(W, H)))
    _fails(tmp_path, r"frame_0001__pi3x.glb holds 12 points for a 6x4 grid", geometry="pi3x")


def test_clips_that_lack_a_frame_record_an_image_or_a_cloud_are_refused(tmp_path):
    clip = make_clip(tmp_path / "in")
    (tmp_path / "out").mkdir()
    m = read(clip / "frame_manifest.json")
    del m["frames"][1]["geometry_sources"]["pi3x"]["camera_up_glb"]
    (clip / "frame_manifest.json").write_text(json.dumps(m))
    _fails(tmp_path, "frame 1 has no pi3x placement", geometry="pi3x")
    m["frames"][1]["geometry_sources"]["pi3x"]["camera_up_glb"] = [0, 1, 0]
    del m["frames"][1]["camera_up_glb"]
    (clip / "frame_manifest.json").write_text(json.dumps(m))
    _fails(tmp_path, "frame 1 has no DA3 camera axes", geometry="pi3x")
    m["frames"][1]["camera_up_glb"] = [0, 1, 0]
    (clip / "frame_manifest.json").write_text(json.dumps(m))
    (clip / "pc_vis" / "frame_0002.glb").unlink()
    _fails(tmp_path, "frame 2 has no da3 cloud")
    (clip / "input_images" / "000000.png").unlink()
    _fails(tmp_path, "2 images in input_images/ for 3 frames")
    m["frames"][1]["seq_idx"] = 5
    (clip / "frame_manifest.json").write_text(json.dumps(m))
    _fails(tmp_path, "does not list frames 0 to 2 once each")


def test_a_clip_already_published_is_replaced_only_with_overwrite_and_only_once_complete(tmp_path):
    clip = make_clip(tmp_path / "in")
    out = tmp_path / "out"
    vb.bundle_clip(clip, out, ["sam3d"])
    with pytest.raises(ValueError, match="pass --overwrite"):
        vb.bundle_clip(clip, out, ["sam3d"])
    # A run that fails on its last frame leaves the earlier copy as it was.
    before = read(out / CLIP / "clip.json")
    (clip / "pc_vis" / "seg_frame_0002__sam3d.json").write_text(json.dumps(seg(two_regions(W, H), stage="seed")))
    with pytest.raises(ValueError, match="neither anchor"):
        vb.bundle_clip(clip, out, ["sam3d"], overwrite=True)
    assert read(out / CLIP / "clip.json") == before and not list(out.glob(".*"))
    # A run that completes replaces the copy and leaves nothing aside, even after an interrupted one.
    (clip / "pc_vis" / "seg_frame_0002__sam3d.json").write_text(json.dumps(seg(two_regions(W, H))))
    (out / f".{CLIP}.old").mkdir()
    (out / f".{CLIP}.old" / "clip.json").write_text("{}")
    (out / f".{CLIP}.partial" / "frames").mkdir(parents=True)
    (out / f".{CLIP}.partial" / "frames" / "0009.jpg").write_bytes(b"left by an interrupted run")
    vb.bundle_clip(clip, out, ["sam3d"], overwrite=True)
    assert (out / CLIP / "regions" / "sam3d" / "0002.json").is_file() and not list(out.glob(".*"))
    assert sorted(p.name for p in (out / CLIP / "frames").iterdir()) == ["0000.jpg", "0001.jpg", "0002.jpg"]


def test_the_catalog_lists_every_clip_but_hidden_directories_in_reading_order(tmp_path):
    out = tmp_path / "out"
    for name, ds in (("b", "cholecseg8k"), ("a", "cholecseg8k"), ("c", "atlas120k")):
        (out / name).mkdir(parents=True)
        (out / name / "clip.json").write_text(json.dumps(
            {"id": name, "dataset": ds, "dataset_name": vb.DATASET_NAMES[ds], "procedure": "P", "video": "v",
             "n_frames": 2, "geometry": {"name": "Pi3X"}, "tracks": []}))
    (out / ".d.partial").mkdir()
    (out / ".d.partial" / "clip.json").write_text("{}")
    assert [e["id"] for e in vb.write_catalog(out)] == ["c", "a", "b"]


def test_main_runs_every_clip_writes_the_catalog_and_exits_non_zero_on_a_failure(tmp_path, monkeypatch, capsys):
    make_clip(tmp_path / "in", name="good")
    bad = make_clip(tmp_path / "in", name="bad")
    (bad / "pc_vis" / "frame_0000.glb").unlink()
    argv = ["viewer_bundle", "--input-dir", str(tmp_path / "in"), "--out", str(tmp_path / "out"),
            "--clips", "bad", "good", "--tracks", "sam3d"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="1 of 2 clip"):
        vb.main()
    assert [e["id"] for e in read(tmp_path / "out" / "catalog.json")["clips"]] == ["good"]
    monkeypatch.setattr(sys, "argv", argv[:5] + ["--clips", "nosuch", "--tracks", "sam3d"])
    with pytest.raises(SystemExit, match="no clip .* at: nosuch"):
        vb.main()
    monkeypatch.setattr(sys, "argv", argv[:5] + ["--clips", "good", "--tracks", "sam3d", "sam3d"])
    with pytest.raises(SystemExit, match="names a track twice"):
        vb.main()


def test_the_reader_returns_the_points_in_file_order(tmp_path):
    pts = grid_points(5, 2, 0.5)
    (tmp_path / "c.glb").write_bytes(glb_bytes(pts))
    assert read_point_cloud_glb(tmp_path / "c.glb").tolist() == pts.astype("f4").tolist()


def _edit_gltf(data: bytes, edit) -> bytes:
    """Apply `edit` to the glTF JSON of a GLB and rebuild the file, keeping its binary chunk."""
    json_len = struct.unpack_from("<I", data, 12)[0]
    gltf = json.loads(data[20:20 + json_len])
    edit(gltf)
    js = json.dumps(gltf).encode()
    js += b" " * (-len(js) % 4)
    body = struct.pack("<I4s", len(js), b"JSON") + js + data[20 + json_len:]
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def _patched(data: bytes, at: int, value: bytes) -> bytes:
    out = bytearray(data)
    out[at:at + len(value)] = value
    return bytes(out)


def _cut(data: bytes, end: int) -> bytes:
    """The first `end` bytes, with the header's length set to match, so that the cut is found inside a chunk."""
    return _patched(data[:end], 8, struct.pack("<I", end))


@pytest.mark.parametrize("fault,match", [
    (lambda d: b"not a glb at all, but long enough", "not a GLB"),
    (lambda d: _patched(d, 4, struct.pack("<I", 3)), "GLB version 3"),
    (lambda d: _patched(d, 8, struct.pack("<I", len(d) + 4)), "in a file of"),
    (lambda d: _patched(d, 16, b"JSOM"), "first chunk is not JSON"),
    (lambda d: _patched(d, 20 + struct.unpack_from("<I", d, 12)[0] + 4, b"BIM\0"), "second chunk is not binary"),
    (lambda d: _cut(d, len(d) - 16), "runs past the end of the file"),
    (lambda d: _cut(d, 30), "not a point-cloud GLB"),
    (lambda d: _edit_gltf(d, lambda g: g.pop("accessors")), "not a point-cloud GLB"),
    (lambda d: _edit_gltf(d, lambda g: g["meshes"][0]["primitives"].append({})), "one mesh of one primitive"),
    (lambda d: _edit_gltf(d, lambda g: g["accessors"][0].update(componentType=5123)), "not tightly packed float32"),
    (lambda d: _edit_gltf(d, lambda g: g["accessors"][0].update(type="VEC2")), "not tightly packed float32"),
    (lambda d: _edit_gltf(d, lambda g: g["bufferViews"][0].update(byteStride=16)), "not tightly packed float32"),
    (lambda d: _edit_gltf(d, lambda g: g["accessors"][0].update(count=g["accessors"][0]["count"] + 1)),
     "runs past the end of its buffer view"),
    (lambda d: _edit_gltf(d, lambda g: g["bufferViews"][0].update(byteLength=10 ** 6)),
     "runs past the end of its buffer view"),
])
def test_the_reader_refuses_a_file_of_another_layout_or_cut_short(tmp_path, fault, match):
    data = fault(glb_bytes(grid_points(5, 2, 0.5)))
    (tmp_path / "x.glb").write_bytes(data)
    with pytest.raises(ValueError, match=match):
        read_point_cloud_glb(tmp_path / "x.glb")


def test_the_reader_reads_what_the_writer_wrote(tmp_path):
    pytest.importorskip("trimesh")
    from surgical_core.viewer.glb import write_point_cloud_glb
    pts = np.random.default_rng(0).normal(size=(50, 3))
    centroid = write_point_cloud_glb(tmp_path / "w.glb", pts, np.zeros((50, 3), np.uint8))
    assert read_point_cloud_glb(tmp_path / "w.glb") == pytest.approx((pts - centroid).astype("f4"), abs=1e-6)
