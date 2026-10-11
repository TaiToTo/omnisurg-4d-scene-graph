"""Write the static files the web viewer reads. The clips' overlays come from the export stage.

Each clip goes to `<out>/<clip>/`: `clip.json`, each frame as a JPEG, each frame's point cloud of one
geometry source, and per track the regions and the scene graphs of the frames the track covers. The
regions are run-length coded labels, one per point of the cloud, in the cloud's order. The export stage
builds the overlays on DA3's cloud; with `--geometry pi3x` the labels are resampled onto Pi3X's pixel
grid and each node is moved to its region's centroid on the Pi3X cloud. Every value the viewer draws
into its page is checked for its kind before it is written. A clip is written to a hidden directory
and replaces an earlier one only once every file is written. The run then rewrites
`<out>/catalog.json`, the list of every clip under `<out>`. `docs/pipeline.md` describes each file.

Usage:
    python -m pipeline.viewer_bundle --input-dir /path/to/clips --out /path/to/site/data \\
        --clips <clip> ... --tracks <track> ... [--geometry da3] [--overwrite] \\
        [--frame-ratios atlas120k_meta/frame_ratio.json]
"""

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from evalkit.classes import ClassType, load_table
from surgical_core.atlas120k.frame_ratio import FrameRatios
from surgical_core.clip_time import frame_times
from surgical_core.viewer.glb import read_point_cloud_glb

FORMAT = 1
MANIFEST = "frame_manifest.json"
CATALOG = "catalog.json"
JPEG_QUALITY = 90
# Node geometry is written to six decimals: a micrometre on these clouds, and a third of the file.
DECIMALS = 6
# A drawn axis reaches two standard deviations each way; the viewer uses only the ratio of the three.
SIGMA_TO_HALF = 2.0
PLACEMENT_KEYS = ("glb_centroid", "camera_pos_glb", "camera_forward_glb", "camera_up_glb")
GEOMETRY_NAMES = {"da3": "Depth Anything 3", "pi3x": "Pi3X"}
DATASET_NAMES = {"atlas120k": "ATLAS-120k", "cholecseg8k": "CholecSeg8k"}
# Procedure names as a reader writes them; ATLAS-120k names its directories in snake case.
PROCEDURE_NAMES = {
    "adrenalectomy": "Adrenalectomy",
    "appendectomy": "Appendectomy",
    "cholecystectomy": "Cholecystectomy",
    "colectomy": "Colectomy",
    "esophagectomy": "Esophagectomy",
    "gastric_surgery": "Gastric surgery",
    "gastrojejunostomy": "Gastrojejunostomy",
    "hemicolectomy": "Hemicolectomy",
    "lar": "Low anterior resection",
    "liver_resection": "Liver resection",
    "rarp": "Robot-assisted radical prostatectomy",
    "rectopexy": "Rectopexy",
    "sigmoidcolectomy": "Sigmoid colectomy",
    "splenectomy": "Splenectomy",
}


@dataclass(frozen=True)
class Track:
    """How the viewer names a track, and what its region ids are.

    Attributes:
        name: the full name, in a legend.
        short: the name on a button.
        semantic: a region is a class with a name, not a numbered object.
        seeded: the regions of one frame were carried to the others.
        anchor_badge: the mark of a frame the regions were made on.
        tracked_badge: the mark of a frame the regions were carried to.
        dataset: the dataset whose class ids the region ids are; None for numbered objects.
    """
    name: str
    short: str
    semantic: bool
    seeded: bool
    anchor_badge: str
    tracked_badge: str
    dataset: str | None


TRACKS = {
    "cholecseg8k": Track("Manual annotation (CholecSeg8k), tracked between annotated frames", "Manual annotation",
                         True, False, "ANNOTATED FRAME", "tracked from annotation", "cholecseg8k"),
    "atlas_gt": Track("Manual annotation (ATLAS-120k)", "Manual annotation", True, False,
                      "ANNOTATED FRAME", "annotated region", "atlas120k"),
    "gt_tracked": Track("Manual annotation of one frame, tracked", "Annotation, tracked", True, True,
                        "ANNOTATED SEED", "tracked region", None),
    "sam3d": Track("Zero-shot from surface normals, tracked", "Zero-shot", False, True,
                   "ZERO-SHOT SEED", "tracked region", None),
    "sam3d_edge": Track("Zero-shot from surface normals and edges, tracked", "Zero-shot", False, True,
                        "ZERO-SHOT SEED", "tracked region", None),
}


def encode_runs(labels: np.ndarray) -> list[int]:
    """Run-length code a label array as `[value, count, value, count, ...]`."""
    labels = np.asarray(labels).ravel()
    if labels.size == 0:
        return []
    starts = np.concatenate([[0], np.flatnonzero(np.diff(labels)) + 1])
    counts = np.diff(np.concatenate([starts, [labels.size]]))
    runs = np.empty(2 * starts.size, dtype=np.int64)
    runs[0::2] = labels[starts]
    runs[1::2] = counts
    return runs.tolist()


def decode_runs(runs: list[int]) -> np.ndarray:
    """Expand `encode_runs`' output back into the label array."""
    values, counts = np.asarray(runs[0::2], np.int64), np.asarray(runs[1::2], np.int64)
    return np.repeat(values, counts)


def grid_remap_index(src_w: int, src_h: int, dst_w: int, dst_h: int) -> np.ndarray:
    """Map each pixel of a dst_w x dst_h grid to the nearest pixel of a src_w x src_h grid of the same image.

    Both grids are row-major rasters of one frame, so pixel centres map through coordinates normalised to
    the frame. Returns the source index of every destination pixel, row-major.
    """
    rows = np.minimum(src_h - 1, np.floor(((np.arange(dst_h) + 0.5) * src_h) / dst_h)).astype(np.int64)
    cols = np.minimum(src_w - 1, np.floor(((np.arange(dst_w) + 0.5) * src_w) / dst_w)).astype(np.int64)
    return (rows[:, None] * src_w + cols[None, :]).ravel()


def region_geometry(points: np.ndarray, labels: np.ndarray) -> dict[int, dict]:
    """Compute each region's centroid and principal axes on a cloud, from the label of each point.

    Returns region id to `pos`, `axes` (one unit axis per row, the longest first, right-handed) and
    `axes_radius`. A region of fewer than three points keeps the coordinate axes, with zero radii.
    """
    pts = np.asarray(points, np.float64)
    out = {}
    for rid in np.unique(labels[labels > 0]).tolist():
        p = pts[labels == rid]
        pos = p.mean(axis=0)
        if len(p) < 3:
            axes, radius = np.eye(3), np.zeros(3)
        else:
            values, vectors = np.linalg.eigh(np.cov(p.T, bias=True))
            order = np.argsort(values)[::-1]
            axes = vectors[:, order].T
            # A left-handed basis is a reflection, which the viewer cannot turn into a rotation.
            if np.linalg.det(axes) < 0:
                axes[2] = -axes[2]
            radius = SIGMA_TO_HALF * np.sqrt(np.clip(values[order], 0.0, None))
        out[rid] = {"pos": pos, "axes": axes, "axes_radius": radius}
    return out


def _rounded(a) -> list:
    return np.round(np.asarray(a, np.float64), DECIMALS).tolist()


def place_graph(graph: dict, geometry: dict[int, dict]) -> dict:
    """Return the graph with each node at its region's place on another cloud, from `region_geometry`.

    The position the graph was built at stays, as `graph_pos`: the spatial relations were chosen from it.
    A node whose region kept no point on the resampled grid is left out, with every edge that touches it:
    drawn at the other cloud's coordinates, it would sit away from its region. `check_graph` has already
    refused a node that is no region of the frame at all.
    """
    nodes = []
    for n in graph.get("nodes", []):
        g = geometry.get(n["id"])
        if g is None:
            continue
        nodes.append({**n, "pos": _rounded(g["pos"]), "graph_pos": n["pos"], "axes": _rounded(g["axes"]),
                      "axes_radius": _rounded(g["axes_radius"])})
    kept = {n["id"] for n in nodes}
    edges = [e for e in graph.get("edges", []) if e["src"] in kept and e["dst"] in kept]
    return {**graph, "nodes": nodes, "edges": edges}


def instrument_ids(dataset: str) -> list[int]:
    """Return the class ids of a dataset's class table that are tools, which the viewer can hide from the cloud."""
    table = load_table(dataset)
    return sorted(int(cid) for cid, e in table.entries.items() if e.type == ClassType.TOOL)


def clip_times(clip_dir: Path, ratios: FrameRatios | None = None) -> list[float]:
    """Return the time of each frame in its video, in seconds, through `surgical_core.clip_time.frame_times`.

    That function holds the one rule that turns a manifest into seconds, with ATLAS-120k's `frame_ratio`.
    A second copy here would drift from it, and a wrong time would show on the page as a plausible number.
    `ratios`, the measured ratios, is asked only for an ATLAS-120k manifest that records no `frame_ratio`.

    Raises:
        ValueError: the manifest gives no way to make seconds, or `frame_times` refuses it.
    """
    try:
        times = frame_times(str(clip_dir.parent), clip_dir.name, ratios)
    except RuntimeError as e:
        raise ValueError(str(e)) from e
    return [round(float(t), 3) for t in times]


def _overlay(clip_dir: Path, kind: str, i: int, track: str | None = None) -> Path:
    name = f"{kind}_{i:04d}.json" if track is None else f"{kind}_{i:04d}__{track}.json"
    return clip_dir / "pc_vis" / name


def _cloud(clip_dir: Path, i: int, geometry: str) -> Path:
    suffix = "" if geometry == "da3" else f"__{geometry}"
    return clip_dir / "pc_vis" / f"frame_{i:04d}{suffix}.glb"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")


def read_clip(clip_dir: Path, geometry: str, ratios: FrameRatios | None = None) -> dict:
    """Read and check what every frame of a clip needs: the manifest, the images, the grids and the placements.

    Returns the manifest, the image paths, the overlays' grid and the cloud's grid (width, height), and
    each frame's placement under the viewer's names. `ratios` is passed to `clip_times`.

    Raises:
        FileNotFoundError: no manifest, or a frame without its cloud.
        ValueError: one of these:
            - `seq_idx` does not run from 0 to N - 1, `n_frames` is not N, or there are not N images
            - the frames' times cannot be made (`clip_times`)
            - the manifest gives no grid for DA3 or for the geometry source
            - a frame has no placement for the geometry source, or no DA3 camera axes under Pi3X
    """
    manifest_path = clip_dir / MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{clip_dir.name} has no {MANIFEST}")
    manifest = _read_json(manifest_path)
    frames = manifest.get("frames", [])
    n = len(frames)
    if [f.get("seq_idx") for f in frames] != list(range(n)) or manifest.get("n_frames", n) != n or n == 0:
        raise ValueError(f"{clip_dir.name}: the manifest does not list frames 0 to {n - 1} once each")
    images = sorted((clip_dir / "input_images").glob("*.png"))
    if len(images) != n:
        raise ValueError(f"{clip_dir.name}: {len(images)} images in input_images/ for {n} frames")
    times = clip_times(clip_dir, ratios)

    shape = manifest.get("depth_info", {}).get("depth_shape")
    if not (isinstance(shape, list) and len(shape) == 3 and shape[0] == n):
        raise ValueError(f"{clip_dir.name}: depth_info.depth_shape does not give DA3's grid for {n} frames")
    overlay_grid = (shape[2], shape[1])
    if geometry == "da3":
        cloud_grid = overlay_grid
    else:
        res = manifest.get("geometry_sources", {}).get(geometry, {}).get("resolution")
        if not (isinstance(res, list) and len(res) == 2):
            raise ValueError(f"{clip_dir.name}: geometry_sources.{geometry}.resolution is missing")
        cloud_grid = tuple(res)

    placements = []
    for i, f in enumerate(frames):
        entry = f if geometry == "da3" else f.get("geometry_sources", {}).get(geometry, {})
        if any(entry.get(k) is None for k in PLACEMENT_KEYS):
            raise ValueError(f"{clip_dir.name}: frame {f['seq_idx']} has no {geometry} placement "
                             f"({', '.join(PLACEMENT_KEYS)})")
        if not _cloud(clip_dir, f["seq_idx"], geometry).is_file():
            raise FileNotFoundError(f"{clip_dir.name}: frame {f['seq_idx']} has no {geometry} cloud")
        placement = {"time_s": times[i],
                     "centroid": entry["glb_centroid"], "camera_position": entry["camera_pos_glb"],
                     "camera_forward": entry["camera_forward_glb"], "camera_up": entry["camera_up_glb"]}
        if geometry != "da3":
            # The graph's relations were chosen in DA3's camera; the viewer reads an edge in it.
            if f.get("camera_forward_glb") is None or f.get("camera_up_glb") is None:
                raise ValueError(f"{clip_dir.name}: frame {f['seq_idx']} has no DA3 camera axes")
            placement.update(graph_camera_forward=f["camera_forward_glb"], graph_camera_up=f["camera_up_glb"])
        placements.append(placement)
    return {"manifest": manifest, "images": images, "overlay_grid": overlay_grid, "cloud_grid": cloud_grid,
            "placements": placements}


def regions_of(seg: dict, overlay_grid: tuple[int, int], where: str) -> tuple[np.ndarray, dict]:
    """Check one frame's regions, as the export stage wrote them, and return its labels and the rest of the record.

    Raises:
        ValueError: one of these:
            - the regions are not on DA3's grid, or do not label every pixel of it
            - a label is not an integer, or is no class the record lists
            - a class has no integer id above 0, no string name, or no colour of three numbers
            - the stage is neither `anchor` nor `propagated`
    """
    w, h = seg.get("width"), seg.get("height")
    labels = np.asarray(seg.get("vertex_seg", []))
    if labels.size and labels.dtype.kind not in "iu":
        raise ValueError(f"{where}: labels are not integers")
    labels = labels.astype(np.int64)
    if (w, h) != overlay_grid or labels.size != w * h:
        raise ValueError(f"{where}: {labels.size} labels on a {w}x{h} grid; the viewer needs one label per pixel "
                         f"of DA3's {overlay_grid[0]}x{overlay_grid[1]} grid")
    # The viewer writes the name into the legend and reads the colour as three channels.
    classes = []
    for c in seg.get("classes", []):
        if (type(c.get("id")) is not int or c["id"] <= 0 or not isinstance(c.get("name"), str)
                or not _numbers(c.get("color"), 3)):
            raise ValueError(f"{where}: class {c.get('id')!r} is not an integer id above 0 with a name and a "
                             "colour of three numbers")
        classes.append({"id": c["id"], "name": c["name"], "color": c["color"]})
    unknown = set(np.unique(labels).tolist()) - {0} - {c["id"] for c in classes}
    if unknown:
        raise ValueError(f"{where}: labels {sorted(unknown)} are no class of the record")
    prov = seg.get("provenance", {})
    if prov.get("stage") not in ("anchor", "propagated"):
        raise ValueError(f"{where}: provenance stage {prov.get('stage')!r} is neither anchor nor propagated")
    anchor = prov.get("anchor_frame")
    return labels, {"classes": classes, "stage": prov["stage"], "anchor_frame": anchor}


def _numbers(value, n: int) -> bool:
    """Return whether `value` is a list of `n` finite numbers, not booleans."""
    return (isinstance(value, list) and len(value) == n
            and all(type(v) in (int, float) and np.isfinite(v) for v in value))


def check_graph(graph: dict, labels: np.ndarray, where: str) -> None:
    """Check that a frame's scene graph names regions of the frame, with a string label and a position.

    The viewer draws the ids, the labels and the relations into the page and places each node at `pos`,
    so a value of another kind is refused here. A node that is no region of the frame is refused too:
    `place_graph` can only leave out a node whose region kept no point, and the two must not be confused.

    Raises:
        ValueError: a node has no integer id among the frame's labels, no string label or no position of
            three numbers; or an edge has no node ids of the graph at its ends or no string relation.
    """
    regions = set(np.unique(labels[labels > 0]).tolist())
    ids = set()
    for node in graph.get("nodes", []):
        if (type(node.get("id")) is not int or node["id"] not in regions or not isinstance(node.get("label"), str)
                or not _numbers(node.get("pos"), 3)):
            raise ValueError(f"{where}: node {node.get('id')!r} is not a region of the frame with a string label "
                             "and a position of three numbers")
        ids.add(node["id"])
    for e in graph.get("edges", []):
        if e.get("src") not in ids or e.get("dst") not in ids or not isinstance(e.get("relation"), str):
            raise ValueError(f"{where}: an edge of {e.get('src')!r} and {e.get('dst')!r} does not join two nodes "
                             "of the graph with a string relation")


def check_hierarchy(h: dict, tracks: list[str], where: str) -> None:
    """Check that a hierarchy relates two of the bundle's tracks, with string keys and labels.

    The viewer draws the labels and the relations into the page and joins the nodes by their keys.

    Raises:
        ValueError: `tracks` is not two of the bundle's tracks; a node has no string key, no track of the two,
            no integer id or no string label; or an edge has no node keys of the hierarchy at its ends or no
            string relation.
    """
    pair = h.get("tracks")
    if not (isinstance(pair, list) and len(pair) == 2 and set(pair) <= set(tracks) and pair[0] != pair[1]):
        raise ValueError(f"{where}: tracks {pair!r} are not two of {', '.join(tracks)}")
    keys = set()
    for node in h.get("nodes", []):
        if (not isinstance(node.get("key"), str) or node.get("track") not in pair or type(node.get("id")) is not int
                or not isinstance(node.get("label"), str)):
            raise ValueError(f"{where}: node {node.get('key')!r} is not a string key of one of the two tracks "
                             "with an integer id and a string label")
        keys.add(node["key"])
    for e in h.get("edges", []):
        if e.get("src") not in keys or e.get("dst") not in keys or not isinstance(e.get("relation"), str):
            raise ValueError(f"{where}: an edge of {e.get('src')!r} and {e.get('dst')!r} does not join two nodes "
                             "of the hierarchy with a string relation")


def check_temporal(tg: dict, n: int, where: str) -> None:
    """Check that a graph through time names its nodes and frames by integers, frames from 0 to n - 1.

    The viewer draws these numbers into the page, so a value of another kind is refused here.

    Raises:
        ValueError: a node id, a relation's ends, or a frame is not such an integer, or a label not a string.
    """
    def frames_ok(fs):
        return isinstance(fs, list) and all(type(f) is int and 0 <= f < n for f in fs)
    for node in tg.get("nodes", []):
        if (type(node.get("id")) is not int or not frames_ok(node.get("present_frames", []))
                or not isinstance(node.get("label", ""), str)):
            raise ValueError(f"{where}: node {node.get('id')!r} is not an integer id with frames 0 to {n - 1}")
    for r in tg.get("relations", []):
        if (type(r.get("src")) is not int or type(r.get("dst")) is not int or not isinstance(r.get("relation"), str)
                or not frames_ok(r.get("frames", []))):
            raise ValueError(f"{where}: a relation of {r.get('src')!r} and {r.get('dst')!r} is not integer ids "
                             f"with frames 0 to {n - 1}")


def track_record(clip_dir: Path, track: str, n: int, dataset: str) -> dict:
    """Describe one track of a clip for `clip.json`: its words and the frames it has files for.

    Raises:
        ValueError: the track is unknown, belongs to another dataset, has no regions on any frame, or has a
            scene graph on a frame without regions.
    """
    if track not in TRACKS:
        raise ValueError(f"unknown track {track!r}; one of {', '.join(TRACKS)}")
    words = TRACKS[track]
    if words.dataset is not None and words.dataset != dataset:
        raise ValueError(f"{clip_dir.name}: track {track} labels {words.dataset} classes, but the clip is {dataset}")
    region_frames = [i for i in range(n) if _overlay(clip_dir, "seg_frame", i, track).is_file()]
    graph_frames = [i for i in range(n) if _overlay(clip_dir, "graph_frame", i, track).is_file()]
    if not region_frames:
        raise ValueError(f"{clip_dir.name}: no pc_vis/seg_frame_NNNN__{track}.json")
    orphans = sorted(set(graph_frames) - set(region_frames))
    if orphans:
        raise ValueError(f"{clip_dir.name}: {track} has a scene graph without regions on frames {orphans}")
    return {"id": track, "name": words.name, "short": words.short, "semantic": words.semantic,
            "seeded": words.seeded, "anchor_badge": words.anchor_badge, "tracked_badge": words.tracked_badge,
            "instrument_ids": instrument_ids(words.dataset) if words.dataset else [],
            "region_frames": region_frames, "graph_frames": graph_frames,
            "temporal": (clip_dir / "pc_vis" / f"temporal_graph__{track}.json").is_file()}


def bundle_clip(clip_dir: Path, out: Path, tracks: list[str], geometry: str = "da3",
                overwrite: bool = False, ratios: FrameRatios | None = None) -> dict:
    """Write one clip's files under `out` and return its `clip.json`. `ratios` is passed to `clip_times`.

    Raises:
        FileNotFoundError: a file the clip needs is missing.
        ValueError: the geometry source or the dataset is unknown, the clip is already under `out` and
            `overwrite` is false, or a check of `read_clip`, `regions_of`, `check_graph`, `check_hierarchy`,
            `check_temporal`, `track_record` or of a cloud fails. A cloud must hold one point per pixel of its
            grid, since the labels and the viewer's depth filter are read by pixel.
    """
    # Check the settings and every file that does not need the overlays read.
    if geometry not in GEOMETRY_NAMES:
        raise ValueError(f"unknown geometry source {geometry!r}; one of {', '.join(GEOMETRY_NAMES)}")
    target = out / clip_dir.name
    if target.exists() and not overwrite:
        raise ValueError(f"{target} exists; pass --overwrite to replace it")
    clip = read_clip(clip_dir, geometry, ratios)
    manifest, n = clip["manifest"], len(clip["placements"])
    dataset = manifest.get("dataset", "cholecseg8k")
    if dataset not in DATASET_NAMES:
        raise ValueError(f"{clip_dir.name}: unknown dataset {dataset!r}")
    records = [track_record(clip_dir, t, n, dataset) for t in tracks]
    remap = (None if clip["cloud_grid"] == clip["overlay_grid"]
             else grid_remap_index(*clip["overlay_grid"], *clip["cloud_grid"]))
    # A hierarchy relates two tracks; one that relates a track left out of the bundle cannot be drawn.
    hierarchy = {}
    for i in range(n):
        if not _overlay(clip_dir, "hierarchy_frame", i).is_file():
            continue
        h = _read_json(_overlay(clip_dir, "hierarchy_frame", i))
        if isinstance(h.get("tracks"), list) and not set(h["tracks"]) <= set(tracks):
            continue
        check_hierarchy(h, tracks, f"{clip_dir.name} frame {i} hierarchy")
        hierarchy[i] = h["tracks"]

    # Write every frame's files into a hidden directory.
    partial = out / f".{clip_dir.name}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    try:
        for i in range(n):
            with Image.open(clip["images"][i]) as im:
                im.convert("RGB").save(_mkdir(partial / "frames") / f"{i:04d}.jpg", quality=JPEG_QUALITY)
            cloud = _cloud(clip_dir, i, geometry)
            points = read_point_cloud_glb(cloud)
            if len(points) != clip["cloud_grid"][0] * clip["cloud_grid"][1]:
                raise ValueError(f"{clip_dir.name}: {cloud.name} holds {len(points)} points for a "
                                 f"{clip['cloud_grid'][0]}x{clip['cloud_grid'][1]} grid")
            shutil.copyfile(cloud, _mkdir(partial / "clouds") / f"{i:04d}.glb")
            for rec in records:
                _write_track_frame(clip_dir, partial, rec, i, clip, remap, points, geometry)
        for rec in records:
            if rec["temporal"]:
                tg = _read_json(clip_dir / "pc_vis" / f"temporal_graph__{rec['id']}.json")
                check_temporal(tg, n, f"{clip_dir.name} {rec['id']} temporal graph")
                _write_json(partial / "graphs" / rec["id"] / "temporal.json", tg)
        for i in hierarchy:
            shutil.copyfile(_overlay(clip_dir, "hierarchy_frame", i), _mkdir(partial / "hierarchy") / f"{i:04d}.json")

        record = {
            "format": FORMAT, "id": clip_dir.name, "dataset": dataset, "dataset_name": DATASET_NAMES[dataset],
            **_source_of(manifest, dataset), "n_frames": n,
            "geometry": {"source": geometry, "name": GEOMETRY_NAMES[geometry], "grid": list(clip["cloud_grid"]),
                         "graphs_built_on": "da3"},
            "frames": clip["placements"], "tracks": records,
            "hierarchy": ({"tracks": sorted({t for ts in hierarchy.values() for t in ts}, key=tracks.index),
                           "frames": sorted(hierarchy)} if hierarchy else None),
        }
        _write_json(partial / "clip.json", record)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise

    # Replace the earlier copy only now that every file is written: move it aside, move the new one in,
    # then delete the old one, so an interruption leaves one of the two whole.
    old = out / f".{clip_dir.name}.old"
    if old.exists():
        shutil.rmtree(old)
    if target.exists():
        target.rename(old)
    partial.rename(target)
    shutil.rmtree(old, ignore_errors=True)
    return record


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_track_frame(clip_dir: Path, partial: Path, rec: dict, i: int, clip: dict, remap, points, geometry):
    """Write one track's regions and scene graph of frame `i`, on the cloud's grid."""
    track = rec["id"]
    if i not in rec["region_frames"]:
        return
    where = f"{clip_dir.name} frame {i} {track}"
    labels, rest = regions_of(_read_json(_overlay(clip_dir, "seg_frame", i, track)), clip["overlay_grid"], where)
    # The graph names regions of DA3's grid, so it is checked against the labels before they are resampled.
    graph = _read_json(_overlay(clip_dir, "graph_frame", i, track)) if i in rec["graph_frames"] else None
    if graph is not None:
        check_graph(graph, labels, f"{where} graph")
    if remap is not None:
        labels = labels[remap]
    _write_json(partial / "regions" / track / f"{i:04d}.json",
                {"grid": list(clip["cloud_grid"]), "runs": encode_runs(labels), **rest})
    if graph is not None:
        if geometry != "da3":
            graph = place_graph(graph, region_geometry(points, labels))
        _write_json(partial / "graphs" / track / f"{i:04d}.json", graph)


def _source_of(manifest: dict, dataset: str) -> dict:
    """Return the procedure, the video and a link to it, as the manifest records them."""
    if dataset == "atlas120k":
        proc, video = manifest.get("procedure", ""), manifest.get("youtube_id", "")
        url = f"https://www.youtube.com/watch?v={video}" if video else None
    else:
        proc, video, url = "cholecystectomy", manifest.get("video_id", ""), None
    return {"procedure": PROCEDURE_NAMES.get(proc, proc.replace("_", " ").capitalize()), "video": video,
            "source_url": url}


def write_catalog(out: Path) -> list[dict]:
    """Rewrite `out/catalog.json` from the `clip.json` of every clip under `out`, and return its entries."""
    entries = []
    for path in sorted(out.glob("*/clip.json")):
        if path.parent.name.startswith("."):
            continue
        c = _read_json(path)
        entries.append({"id": c["id"], "dataset": c["dataset"], "dataset_name": c["dataset_name"],
                        "procedure": c["procedure"], "video": c["video"], "n_frames": c["n_frames"],
                        "geometry": c["geometry"]["name"],
                        "tracks": [{"id": t["id"], "short": t["short"]} for t in c["tracks"]]})
    entries.sort(key=lambda e: (e["dataset_name"], e["procedure"], e["id"]))
    tmp = out / f".{CATALOG}.partial"
    _write_json(tmp, {"format": FORMAT, "clips": entries})
    tmp.replace(out / CATALOG)
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="The directory that holds the clips.")
    ap.add_argument("--out", required=True, help="The viewer's data directory; catalog.json is written here.")
    ap.add_argument("--clips", nargs="+", required=True, help="Clip directories under --input-dir.")
    ap.add_argument("--tracks", nargs="+", required=True, choices=sorted(TRACKS),
                    help="The tracks to publish, in the order the viewer offers them.")
    ap.add_argument("--geometry", default="da3", choices=sorted(GEOMETRY_NAMES),
                    help="The reconstruction whose point clouds the viewer draws.")
    ap.add_argument("--overwrite", action="store_true", help="Replace a clip already under --out.")
    ap.add_argument("--frame-ratios", help="The measured frame ratios, frame_ratio.json. Needed only for an "
                    "ATLAS-120k clip whose manifest records no frame_ratio.")
    args = ap.parse_args()
    if len(set(args.tracks)) != len(args.tracks):
        raise SystemExit(f"--tracks names a track twice: {' '.join(args.tracks)}")
    ratios = FrameRatios.load(args.frame_ratios) if args.frame_ratios else None
    # Find the clips first: a mistyped clip name fails here, not halfway through the run.
    root, out = Path(args.input_dir), Path(args.out)
    missing = [c for c in args.clips if not (root / c / MANIFEST).is_file()]
    if missing:
        raise SystemExit(f"no clip (a directory with {MANIFEST}) at: {', '.join(missing)}")
    out.mkdir(parents=True, exist_ok=True)
    # Run every clip, even after one fails, list the failures and exit non-zero.
    failed = []
    for name in args.clips:
        print(name)
        try:
            rec = bundle_clip(root / name, out, args.tracks, args.geometry, args.overwrite, ratios)
            print(f"  {rec['n_frames']} frame(s), tracks {', '.join(t['id'] for t in rec['tracks'])}")
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")
            failed.append(name)
    print(f"{len(write_catalog(out))} clip(s) in {out / CATALOG}")
    if failed:
        raise SystemExit(f"{len(failed)} of {len(args.clips)} clip(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
