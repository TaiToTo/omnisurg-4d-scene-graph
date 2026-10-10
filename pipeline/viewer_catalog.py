"""Write the catalog the viewer opens, beside the clips or in a copy of the files the viewer reads.

The catalog lists each clip with its dataset, its group, its frame count,
the depth model whose point clouds it shows, and its tracks, and names each
track the way the viewer shows it. A clip enters only when every frame has
its point cloud, its camera pose and its image, and every track has a
segmentation and a graph on the same frames and a temporal graph. One clip
that fails these checks stops the run, and nothing is written.
`viewer/README.md` lists the files a clip holds.

Usage:
    python -m pipeline.viewer_catalog --root /path/to/outputs --clips cholec_gt/VID01_s15_80_crop ... \\
        [--tracks <track> ...] [--geometry pi3x] [--geometry cholecseg8k=da3] (--in-place | --out /path/to/site/data)
"""

import argparse
import json
import math
import re
import shutil
from pathlib import Path

# The format string `viewer/src/lib/catalog.js` reads; any other is refused there.
CATALOG_FORMAT = "omnisurg-viewer-catalog/1"

# How the viewer names each track the export writes. A track id not listed
# here is refused: the viewer would not know whether its regions are annotation.
TRACKS = {
    "cholecseg8k": {"label": "Manual annotation (CholecSeg8k)", "short": "Manual annotation",
                    "semantic": True, "seeded": False,
                    "anchor_badge": "ANNOTATED FRAME", "tracked_badge": "tracked region"},
    "atlas_gt": {"label": "Manual annotation (ATLAS-120k)", "short": "Manual annotation",
                 "semantic": True, "seeded": False,
                 "anchor_badge": "ANNOTATED FRAME", "tracked_badge": "annotated region"},
    "gt_tracked": {"label": "Manual annotation of one frame, tracked", "short": "Annotation, tracked",
                   "semantic": True, "seeded": True,
                   "anchor_badge": "ANNOTATED FRAME", "tracked_badge": "tracked region"},
    "sam3d": {"label": "Automatic, zero-shot (surface normals)", "short": "Automatic",
              "semantic": False, "seeded": True,
              "anchor_badge": "ZERO-SHOT SEED", "tracked_badge": "tracked region"},
    "sam3d_edge": {"label": "Automatic, zero-shot (surface normals and edges)", "short": "Automatic",
                   "semantic": False, "seeded": True,
                   "anchor_badge": "ZERO-SHOT SEED", "tracked_badge": "tracked region"},
}

# A manifest without a `dataset` key belongs to CholecSeg8k (docs/pipeline.md, "A clip").
DATASETS = {"cholecseg8k": "CholecSeg8k", "atlas120k": "ATLAS-120k"}

# The depth model whose files carry no `__<source>` suffix.
DEFAULT_GEOMETRY = "da3"
POSE_KEYS = ("glb_centroid", "camera_pos_glb", "camera_forward_glb", "camera_up_glb")

# The rule `viewer/src/lib/catalog.js` applies to a clip path: names only, no `.` or `..`.
PATH_RE = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9._-]*(/[A-Za-z0-9_-][A-Za-z0-9._-]*)*$")
ID_RE = re.compile(r"^[a-z0-9_]+$")
OVERLAY_RE = re.compile(r"^(seg_frame|graph_frame)_(\d{4})__([a-z0-9_]+)\.json$")
HIERARCHY_RE = re.compile(r"^hierarchy_frame_(\d{4})\.json$")


def parse_geometry(specs: list[str]) -> tuple[str, dict[str, str]]:
    """Read the `--geometry` values: one source for every clip, and `DATASET=SOURCE` for a dataset of its own.

    Raises:
        ValueError: a source or a dataset that is not a plain id, an unknown dataset, or two defaults.
    """
    default, by_dataset = None, {}
    for spec in specs:
        dataset, sep, source = spec.rpartition("=")
        if not ID_RE.match(source) or (sep and not ID_RE.match(dataset)):
            raise ValueError(f"--geometry {spec!r} is not SOURCE or DATASET=SOURCE")
        if sep:
            if dataset not in DATASETS:
                raise ValueError(f"--geometry {spec!r}: no dataset {dataset!r}; known: {', '.join(DATASETS)}")
            by_dataset[dataset] = source
        elif default is not None:
            raise ValueError(f"--geometry gives two sources for every clip: {default} and {source}")
        else:
            default = source
    return default or DEFAULT_GEOMETRY, by_dataset


def is_vector(v, n: int) -> bool:
    """Return whether `v` is a list of `n` finite numbers."""
    return (isinstance(v, list) and len(v) == n
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v))


def glb_name(i: int, source: str) -> str:
    """Return the file name of frame `i`'s point cloud from `source`."""
    return f"frame_{i:04d}.glb" if source == DEFAULT_GEOMETRY else f"frame_{i:04d}__{source}.glb"


def clip_entry(root: Path, rel: str, geometry: tuple[str, dict[str, str]],
               tracks: set[str] | None = None) -> tuple[dict, list[str]]:
    """Check one clip and return its catalog entry and the files the viewer reads, relative to the clip.

    `tracks`, when given, names the tracks to list; a clip's other tracks are
    left out, and neither checked nor copied.

    Raises:
        ValueError: the clip lacks something the viewer reads; the message names it.
    """
    if not PATH_RE.match(rel):
        raise ValueError(f"{rel!r} is not a relative path of plain names")
    clip = root / rel
    manifest_path = clip / "frame_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{rel}: no frame_manifest.json")
    manifest = json.loads(manifest_path.read_text())

    # The frames, the dataset and the group.
    frames = manifest.get("frames") or []
    n = manifest.get("n_frames", len(frames))
    if n < 1 or len(frames) != n:
        raise ValueError(f"{rel}: n_frames is {n} and the manifest lists {len(frames)} frames")
    dataset = manifest.get("dataset", "cholecseg8k")
    if dataset not in DATASETS:
        raise ValueError(f"{rel}: dataset {dataset!r} is not one the viewer names; known: {', '.join(DATASETS)}")
    if dataset == "atlas120k":
        group = f"{manifest.get('procedure', '?')} / {manifest.get('youtube_id', '?')}"
    else:
        group = manifest.get("video_id") or rel.rsplit("/", 1)[-1].split("_")[0]

    # Every frame's point cloud, pose and image.
    default, by_dataset = geometry
    source = by_dataset.get(dataset, default)
    files = ["frame_manifest.json"]
    problems = []
    for i, fr in enumerate(frames):
        pose = fr if source == DEFAULT_GEOMETRY else (fr.get("geometry_sources") or {}).get(source) or {}
        if not all(is_vector(pose.get(k), 3) for k in POSE_KEYS):
            problems.append(f"frame {i} has no {source} pose")
        for name in (f"pc_vis/{glb_name(i, source)}", f"input_images/{i:06d}.png"):
            if not (clip / name).is_file():
                problems.append(f"no {name}")
            files.append(name)
    resolution = (manifest.get("geometry_sources") or {}).get(source, {}).get("resolution")
    if source != DEFAULT_GEOMETRY and not (is_vector(resolution, 2) and all(isinstance(x, int) and x > 0 for x in resolution)):
        problems.append(f"the manifest has no resolution [W, H] for {source}")

    # Every track's segmentation and graph frames, and its temporal graph.
    found: dict[str, dict[str, set[int]]] = {}
    hierarchy = set()
    for p in (clip / "pc_vis").iterdir() if (clip / "pc_vis").is_dir() else []:
        if (m := OVERLAY_RE.match(p.name)) and (tracks is None or m.group(3) in tracks):
            found.setdefault(m.group(3), {"seg_frame": set(), "graph_frame": set()})[m.group(1)].add(int(m.group(2)))
        elif m := HIERARCHY_RE.match(p.name):
            hierarchy.add(int(m.group(1)))
    if not found:
        problems.append("no track: no pc_vis/seg_frame_NNNN__<track>.json")
    coverage = {}
    for track, kinds in sorted(found.items()):
        if track not in TRACKS:
            problems.append(f"track {track!r} is not one the viewer names; known: {', '.join(TRACKS)}")
            continue
        if kinds["seg_frame"] != kinds["graph_frame"]:
            odd = sorted(kinds["seg_frame"] ^ kinds["graph_frame"])
            problems.append(f"track {track}: frames {odd[:5]} have a segmentation or a graph, not both")
        if max(kinds["seg_frame"] | kinds["graph_frame"]) >= n:
            problems.append(f"track {track}: an overlay of a frame past the last, {n - 1}")
        if not (clip / f"pc_vis/temporal_graph__{track}.json").is_file():
            problems.append(f"track {track}: no pc_vis/temporal_graph__{track}.json")
        coverage[track] = len(kinds["seg_frame"])
        files += [f"pc_vis/{kind}_{i:04d}__{track}.json" for kind in ("seg_frame", "graph_frame") for i in sorted(kinds[kind])]
        files.append(f"pc_vis/temporal_graph__{track}.json")
    if hierarchy and max(hierarchy) >= n:
        problems.append(f"a hierarchy frame past the last frame, {n - 1}")
    files += [f"pc_vis/hierarchy_frame_{i:04d}.json" for i in sorted(hierarchy)]
    if problems:
        raise ValueError(f"{rel}: " + "; ".join(problems))
    entry = {
        "path": rel, "dataset": DATASETS[dataset], "group": group, "label": rel.rsplit("/", 1)[-1],
        "frames": n, "geometry": source, "hierarchy": bool(hierarchy),
        "tracks": sorted(coverage, key=lambda t: (not TRACKS[t]["semantic"], t)), "coverage": coverage,
    }
    return entry, files


def build_catalog(root: Path, clips: list[str], geometry: tuple[str, dict[str, str]],
                  tracks: list[str] | None = None) -> tuple[dict, dict[str, list[str]]]:
    """Check every clip and return the catalog and each clip's files; `tracks` limits the tracks listed.

    Raises:
        ValueError: one line per clip refused, all of them; a clip listed twice; or a track the viewer does not name.
    """
    if len(set(clips)) != len(clips):
        raise ValueError("a clip is listed twice")
    unknown = sorted(set(tracks or []) - set(TRACKS))
    if unknown:
        raise ValueError(f"--tracks {', '.join(unknown)}: not tracks the viewer names; known: {', '.join(TRACKS)}")
    entries, files, problems = [], {}, []
    for rel in clips:
        try:
            entry, clip_files = clip_entry(root, rel, geometry, set(tracks) if tracks else None)
        except ValueError as e:
            problems.append(str(e))
            continue
        entries.append(entry)
        files[rel] = clip_files
    if problems:
        raise ValueError("\n".join(problems))
    used = {t for e in entries for t in e["tracks"]}
    catalog = {"format": CATALOG_FORMAT, "tracks": {t: TRACKS[t] for t in sorted(used)}, "clips": entries}
    return catalog, files


def write_staged(root: Path, out: Path, catalog: dict, files: dict[str, list[str]]) -> int:
    """Copy each clip's files from `root` into `out` and write `out/catalog.json`; return the number of files.

    The copy is made in a directory beside `out` and renamed only once it is
    complete, so a failed copy leaves no half-written site.

    Raises:
        FileExistsError: `out` exists.
    """
    if out.exists():
        raise FileExistsError(f"{out} exists; give a new directory")
    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    count = 0
    for rel, names in files.items():
        for name in names:
            dst = partial / rel / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / rel / name, dst)
            count += 1
    (partial / "catalog.json").write_text(json.dumps(catalog, indent=1) + "\n")
    partial.rename(out)
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 fromfile_prefix_chars="@")
    ap.add_argument("--root", required=True, help="The directory the clip paths are relative to.")
    ap.add_argument("--clips", nargs="+", required=True,
                    help="Clip paths under --root, in the order the viewer lists them. @FILE reads one per line.")
    ap.add_argument("--tracks", nargs="+",
                    help="The tracks to list, of those a clip has. Default: every track a clip has.")
    ap.add_argument("--geometry", action="append", default=[],
                    help=f"The depth model whose clouds the viewer shows: SOURCE for every clip, DATASET=SOURCE for "
                         f"one dataset. Default: {DEFAULT_GEOMETRY}.")
    where = ap.add_mutually_exclusive_group(required=True)
    where.add_argument("--in-place", action="store_true", help="Write catalog.json into --root and copy nothing.")
    where.add_argument("--out", help="A new directory to copy the clips' files and catalog.json into.")
    args = ap.parse_args()
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"no such directory: {root}")
    try:
        geometry = parse_geometry(args.geometry)
        catalog, files = build_catalog(root, args.clips, geometry, args.tracks)
    except ValueError as e:
        raise SystemExit(f"nothing written:\n{e}")
    if args.in_place:
        (root / "catalog.json").write_text(json.dumps(catalog, indent=1) + "\n")
        print(f"{root / 'catalog.json'}: {len(catalog['clips'])} clip(s)")
        return
    out = Path(args.out)
    if out.resolve() == root.resolve():
        raise SystemExit("--out is --root; give --in-place to write the catalog there")
    try:
        count = write_staged(root, out, catalog, files)
    except FileExistsError as e:
        raise SystemExit(str(e))
    print(f"{out}: {len(catalog['clips'])} clip(s), {count} file(s) and catalog.json")


if __name__ == "__main__":
    main()
