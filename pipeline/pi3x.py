"""Run the Pi3X stage: reconstruct each clip with Pi3X, a geometry source in addition to DA3.

The stage writes `exports/mini_npz/results__pi3x.npz`, `depth_vis/NNNN__pi3x.jpg`
and `pc_vis/frame_NNNN__pi3x.glb` into the clip, and leaves DA3's files alone.
It adds `geometry_sources.pi3x` to the manifest: per frame the point cloud's
centroid, size and camera axes, and for the run its settings, runtime and
round-trip check. The check back-projects the stage's depth through its poses
and compares the points with the ones Pi3X predicted. The stage refuses:

- a clip that fails the round-trip check, as an inverted pose convention does;
- a clip whose manifest does not list each frame of `input_images/` once;
- a clip that already holds the stage's output; `--overwrite` removes it first.

Usage:
    python -m pipeline.pi3x --input-dir /path/to/clips [--clips <clip> ...] [--device auto] [--gpu N] [--overwrite]
"""

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from recon3d_wrapper import Reconstruction, Reconstructor
from recon3d_wrapper.pi3x import DEFAULT_PIXEL_LIMIT, DEVICES, Pi3X
from surgical_core.geometry.camera import backproject_depth, cam_to_world, world_to_gltf
from surgical_core.geometry.valid import valid_depth_mask
from surgical_core.viewer.camera_axes import camera_axes_in_gltf
from surgical_core.viewer.depth_vis import depth_to_colormap
from surgical_core.viewer.glb import write_point_cloud_glb

# The name this geometry source writes under, in file names and in the manifest.
SOURCE = "pi3x"

# The stage's own files in a clip, by the directory or file that holds them. The DA3 stage writes into the same
# directories without a suffix, so the stage names its files, never a directory.
OWN_FILES = {"depth_vis": rf"\d+__{SOURCE}\.jpg", "pc_vis": rf"frame_\d+__{SOURCE}\.glb"}
BUNDLE = f"exports/mini_npz/results__{SOURCE}.npz"

# How many times the inverted pose reading's error must exceed the stage's own. The right reading's error is 100
# to 300 times smaller on every clip measured, so 10 leaves an order of magnitude of slack.
ROUNDTRIP_MIN_RATIO = 10.0

# The largest round-trip error allowed at the 99.9th percentile, relative to the median depth. Pi3X's rays are
# not quite a pinhole: the fit leaves p99.9 up to 1.1e-2. An inverted pose gives about 1 to 3.
DEFAULT_ROUNDTRIP_TOL = 3e-2


def verify_roundtrip(rec: Reconstruction, n_sample: int = 20000, seed: int = 0) -> dict:
    """Compare the world points that depth and poses give with the ones Pi3X predicted.

    The error is taken as a percentile, not a maximum: the maximum grows with the number of frames even when the
    geometry is equally good. The control reads the same pose as camera to world, and its error is reduced the
    same way, so that the ratio of the two depends only on the convention.

    Args:
        rec: a reconstruction with `points`.
        n_sample: the points sampled per frame.
        seed: the sampling seed.

    Returns:
        The median depth, the error at p50, p99.9 and its maximum, the control at p99.9, all relative to the
        median depth, and the frames checked.

    Raises:
        ValueError: no world points, no pixel with usable depth, or an error that is not finite: a pose, the
            intrinsics or a world point holds NaN or infinity.
    """
    if rec.points is None:
        raise ValueError("the reconstruction has no world points to check the poses against")
    rng = np.random.default_rng(seed)
    valid_all = valid_depth_mask(rec.depth)
    if not valid_all.any():
        raise ValueError("no pixel has usable depth; nothing to verify")
    depth_med = float(np.median(rec.depth[valid_all]))
    err, err_inverted = [], []
    for i in range(len(rec.depth)):
        if not valid_all[i].any():
            continue
        idx = np.flatnonzero(valid_all[i].reshape(-1))
        if len(idx) > n_sample:
            idx = rng.choice(idx, n_sample, replace=False)
        R, t = rec.extrinsics[i][:3, :3], rec.extrinsics[i][:3, 3]
        # Compared in the world's frame, where Pi3X's points are. The glTF flip changes no distance.
        ref = rec.points[i].reshape(-1, 3)[idx]
        cam = backproject_depth(rec.depth[i], rec.intrinsics[i])[idx]
        err.append(np.linalg.norm(cam_to_world(cam, R, t) - ref, axis=-1))
        # The control reads [R|t] as camera to world: world = R @ cam + t.
        err_inverted.append(np.linalg.norm(cam @ R.T + t - ref, axis=-1))
    pooled = np.concatenate(err) / depth_med
    # A NaN makes every percentile NaN. Both of the stage's comparisons are false for NaN, so the clip would pass.
    if not np.isfinite(pooled).all():
        raise ValueError(f"the round-trip error is not finite at {np.count_nonzero(~np.isfinite(pooled))} of "
                         f"{len(pooled)} sampled pixels; a pose, the intrinsics or a world point holds NaN or infinity")
    pooled_inverted = np.concatenate(err_inverted) / depth_med
    return {
        "median_depth": depth_med,
        "p50_rel": float(np.percentile(pooled, 50)),
        "p999_rel": float(np.percentile(pooled, 99.9)),
        "max_rel": float(pooled.max()),
        "p999_rel_if_pose_inverted": float(np.percentile(pooled_inverted, 99.9)),
        "frames_checked": len(err),
    }


def read_manifest(clip_dir: Path, n_input: int) -> dict:
    """Read the clip's manifest, and check that it lists each frame of `input_images/` once.

    Frames are matched by `seq_idx`, their position in `input_images/`. A duplicate value would drop a frame's
    entry, and a missing value would pair a frame's record with another frame's point cloud.

    Raises:
        FileNotFoundError: the clip has no manifest.
        ValueError: a frame without `seq_idx`, `seq_idx` values other than 0 to `n_input` - 1 once each, or an
            `n_frames` other than `n_input`.
    """
    manifest_path = clip_dir / "frame_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    frames = manifest.get("frames", [])
    if any("seq_idx" not in fr for fr in frames):
        raise ValueError(f"{manifest_path}: some frames have no seq_idx")
    seq_idx = sorted(fr["seq_idx"] for fr in frames)
    if seq_idx != list(range(n_input)):
        raise ValueError(f"{manifest_path}: input_images/ holds {n_input} frames, but the manifest's {len(seq_idx)} "
                         f"seq_idx values are not 0 to {n_input - 1} once each: {seq_idx[:6]}")
    if manifest.get("n_frames") != n_input:
        raise ValueError(f"{manifest_path}: n_frames is {manifest.get('n_frames')}, but input_images/ holds "
                         f"{n_input} frames")
    return manifest


def write_manifest(clip_dir: Path, manifest: dict) -> None:
    """Write the manifest to the clip."""
    (clip_dir / "frame_manifest.json").write_text(json.dumps(manifest, indent=2))


def update_manifest(clip_dir: Path, manifest: dict, source: str, per_frame: dict[int, dict],
                    provenance: dict) -> None:
    """Merge a geometry source's per-frame records and its run record into the manifest, and write it to the clip.

    A frame's records go under `frames[i].geometry_sources[source]`, one level below DA3's keys. Written at the
    frame's top level, they would overwrite DA3's centroid and move every DA3 point cloud.

    Args:
        clip_dir: the clip.
        manifest: the clip's manifest, as `read_manifest` returned it.
        source: the geometry source's name.
        per_frame: the records, by `seq_idx`.
        provenance: the run's record, stored at `geometry_sources[source]`.
    """
    by_seq_idx = {fr["seq_idx"]: fr for fr in manifest["frames"]}
    for seq_idx, meta in per_frame.items():
        by_seq_idx[seq_idx].setdefault("geometry_sources", {})[source] = meta
    manifest["frames"] = [by_seq_idx[k] for k in sorted(by_seq_idx)]
    manifest.setdefault("geometry_sources", {})[source] = provenance
    write_manifest(clip_dir, manifest)


def own_files(clip_dir: Path) -> dict[str, list[Path]]:
    """Return the stage's own files in the clip, under the directory or file that holds them."""
    found = {}
    for sub, pattern in OWN_FILES.items():
        d = clip_dir / sub
        found[sub] = sorted(p for p in d.iterdir() if re.fullmatch(pattern, p.name)) if d.is_dir() else []
    found[BUNDLE] = [clip_dir / BUNDLE] if (clip_dir / BUNDLE).is_file() else []
    return found


def existing_output(clip_dir: Path, manifest: dict) -> list[str]:
    """List what this stage, in any version, already wrote into the clip.

    Each entry names a directory with its count of the stage's files, the bundle with its keys, or the manifest's
    records. The run's record is listed with its `vo`: true marks the workbench's chunked mode, which this stage
    does not run.
    """
    held = []
    for name, files in own_files(clip_dir).items():
        if not files:
            continue
        if name == BUNDLE:
            with np.load(files[0]) as z:
                held.append(f"{name} with keys {z.files}")
        else:
            held.append(f"{name} ({len(files)} file{'s' if len(files) > 1 else ''})")
    record = manifest.get("geometry_sources", {}).get(SOURCE)
    if record is not None:
        held.append(f"geometry_sources.{SOURCE} with keys {list(record)} and vo={record.get('vo')}")
    n = sum(SOURCE in fr.get("geometry_sources", {}) for fr in manifest.get("frames", []))
    if n:
        held.append(f"geometry_sources.{SOURCE} on {n} frame{'s' if n > 1 else ''}")
    return held


def remove_output(clip_dir: Path, manifest: dict) -> None:
    """Remove the stage's own files from the clip and its records from the manifest, and leave another source's.

    The records are removed from `manifest` itself, which is then written to the clip. A frame that gets no point
    cloud in the new run would otherwise keep the earlier run's cloud and record, and the viewer would draw them.
    """
    for files in own_files(clip_dir).values():
        for path in files:
            path.unlink()
    for entry in [manifest, *manifest.get("frames", [])]:
        sources = entry.get("geometry_sources")
        if sources is not None:
            sources.pop(SOURCE, None)
            if not sources:
                del entry["geometry_sources"]
    write_manifest(clip_dir, manifest)


def read_colors(image_paths: list[Path], size: tuple[int, int]) -> list[np.ndarray]:
    """Read every frame's image as (H * W, 3) RGB at the depth's `size`, (W, H), in the order of the depth's pixels.

    Raises:
        ValueError: an image that cannot be read.
    """
    colors = []
    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(f"cannot read {img_path}")
        colors.append(cv2.cvtColor(cv2.resize(img_bgr, size, interpolation=cv2.INTER_LINEAR),
                                   cv2.COLOR_BGR2RGB).reshape(-1, 3))
    return colors


def run_pi3x(clip_dir: Path, model: Reconstructor, pixel_limit: int, conf_thre: float = 0.0,
             roundtrip_tol: float = DEFAULT_ROUNDTRIP_TOL, max_points: int = 0, overwrite: bool = False) -> None:
    """Run the stage on one clip.

    The stage checks every input and every frame's points before it writes or removes anything, so that a refused
    clip is left as it was.

    Args:
        clip_dir: the clip.
        model: the reconstruction model. It must return world points.
        pixel_limit: the pixel budget the model was built with. The manifest records it.
        conf_thre: drop pixels at or below this confidence from the point clouds. 0 keeps every pixel with usable
            depth, as the DA3 stage does, so that the two sources differ in geometry and not in masking.
        roundtrip_tol: the largest round-trip error allowed at p99.9, relative to the median depth.
        max_points: the most points a cloud keeps, sampled with a fixed seed; 0 keeps all.
        overwrite: replace the stage's output when the clip already holds it.

    Raises:
        FileNotFoundError: no image or no manifest.
        ValueError: a manifest `read_manifest` refuses, the clip already holds the stage's output and `overwrite`
            is false, the model returned another number of frames, the round-trip check failed or `verify_roundtrip`
            refused the reconstruction, an image could not be read, or no frame keeps a pixel for its point cloud.
    """
    # Find the clip's images and check its manifest before the model runs, so that a refused clip costs no
    # inference.
    image_paths = sorted((clip_dir / "input_images").glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"{clip_dir} has no input_images/*.png")
    n_input = len(image_paths)
    manifest = read_manifest(clip_dir, n_input)

    # Refuse a clip that already holds the stage's output, unless the caller asked to replace it.
    held = existing_output(clip_dir, manifest)
    if held and not overwrite:
        raise ValueError(f"{clip_dir.name} already holds the Pi3X stage's output ({'; '.join(held)}); "
                         "pass --overwrite to replace it")

    # Reconstruct the images.
    print(f"  {n_input} frames, model={model.model_id}, pixel_limit={pixel_limit}")
    t0 = time.perf_counter()
    rec = model.reconstruct(image_paths)
    runtime = time.perf_counter() - t0
    n_frames, H, W = rec.depth.shape
    if n_frames != n_input:
        raise ValueError(f"{clip_dir.name}: the model returned {n_frames} of {n_input} frames")
    print(f"  inference {runtime:.1f}s, depth ({n_frames}, {H}, {W})")

    # Refuse a clip whose poses fail the round trip. A wrong pose convention places every point cloud, graph
    # node and camera wrong, and nothing later detects it. The ratio to the inverted reading is the sharp test;
    # the absolute tolerance is a backstop.
    check = verify_roundtrip(rec)
    ratio = check["p999_rel_if_pose_inverted"] / max(check["p999_rel"], 1e-12)
    print(f"  round trip: p50 {check['p50_rel']:.2e}, p99.9 {check['p999_rel']:.2e}, inverted {ratio:.0f}x worse")
    if ratio < ROUNDTRIP_MIN_RATIO:
        raise ValueError(f"{clip_dir.name}: the inverted pose reading is nearly as good ({ratio:.1f}x, need "
                         f"{ROUNDTRIP_MIN_RATIO}x); c2w_to_extrinsics is likely inverted")
    if check["p999_rel"] > roundtrip_tol:
        raise ValueError(f"{clip_dir.name}: round-trip p99.9 {check['p999_rel']:.3e} exceeds {roundtrip_tol:.1e}")

    # Read every frame's colours, choose the pixels each cloud keeps, and refuse a clip in which no frame keeps one.
    colors = read_colors(image_paths, (W, H))
    keep = valid_depth_mask(rec.depth).reshape(n_frames, -1)
    if conf_thre > 0:
        keep &= rec.conf.reshape(n_frames, -1) > conf_thre
    if not keep.any():
        raise ValueError(f"{clip_dir.name}: no frame keeps a pixel for its point cloud (conf_thre {conf_thre})")

    # Remove the earlier output, now that every check has passed.
    if held:
        remove_output(clip_dir, manifest)

    # Write the bundle and the depth images, each under the source's suffix.
    exports = clip_dir / "exports" / "mini_npz"
    exports.mkdir(parents=True, exist_ok=True)
    np.savez(str(exports / f"results__{SOURCE}.npz"), depth=rec.depth.astype(np.float32),
             conf=rec.conf.astype(np.float32), extrinsics=rec.extrinsics.astype(np.float32),
             intrinsics=rec.intrinsics.astype(np.float32))
    depth_vis = clip_dir / "depth_vis"
    depth_vis.mkdir(exist_ok=True)
    for i in range(n_frames):
        Image.fromarray(depth_to_colormap(rec.depth[i])).save(str(depth_vis / f"{i:04d}__{SOURCE}.jpg"))

    # Write a point cloud per frame, and record its centroid, its size and the camera's axes.
    pc_vis = clip_dir / "pc_vis"
    pc_vis.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    per_frame: dict[int, dict] = {}
    n_verts = []
    for i in range(n_frames):
        if not keep[i].any():
            # A frame can keep no pixel. The viewer shows no cloud for it; the other frames still get theirs.
            print(f"  frame {i:04d}: no pixel kept, no point cloud")
            continue
        R, t = rec.extrinsics[i][:3, :3], rec.extrinsics[i][:3, 3]
        gltf_pts = world_to_gltf(cam_to_world(backproject_depth(rec.depth[i], rec.intrinsics[i]), R, t))[keep[i]]
        cols = colors[i][keep[i]]
        if max_points and len(gltf_pts) > max_points:
            sel = rng.choice(len(gltf_pts), max_points, replace=False)
            gltf_pts, cols = gltf_pts[sel], cols[sel]
        centroid = write_point_cloud_glb(pc_vis / f"frame_{i:04d}__{SOURCE}.glb", gltf_pts, cols, recenter=True)
        per_frame[i] = {"glb_centroid": centroid.tolist(), "n_vertices": int(len(gltf_pts)),
                        **camera_axes_in_gltf(R, t)}
        n_verts.append(len(gltf_pts))

    # Record the run in the manifest. The summaries cover every pixel with usable depth, whatever `conf_thre` drops.
    # `vo` and `vo_fixed_scale` keep the workbench's record of its chunked mode, which this stage does not run.
    valid = valid_depth_mask(rec.depth)
    depth_valid, conf_valid = rec.depth[valid], rec.conf[valid]
    update_manifest(clip_dir, manifest, SOURCE, per_frame, {
        "model": model.model_id,
        "vo": False,
        "vo_fixed_scale": None,
        "pixel_limit": pixel_limit,
        "conf_thre": conf_thre,
        "resolution": [int(W), int(H)],
        "n_frames": len(per_frame),
        "median_vertices": int(np.median(n_verts)),
        "runtime_sec": round(runtime, 2),
        "conf_coverage": {str(th): float((conf_valid > th).mean()) for th in (0.1, 0.2, 0.3, 0.5)},
        "depth_range": [float(depth_valid.min()), float(depth_valid.max())],
        "roundtrip": check,
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="The directory that holds the clips.")
    ap.add_argument("--clips", nargs="*", help="Clip directories under --input-dir. Default: every clip there.")
    ap.add_argument("--device", default="auto", choices=DEVICES, help="auto takes CUDA when there is one.")
    ap.add_argument("--gpu", type=int, default=None,
                    help="The index of the GPU to use. Default: what CUDA_VISIBLE_DEVICES says, or the first GPU.")
    ap.add_argument("--pixel-limit", type=int, default=DEFAULT_PIXEL_LIMIT,
                    help="The most pixels a frame keeps after resizing.")
    ap.add_argument("--conf-thre", type=float, default=0.0,
                    help="Drop pixels at or below this confidence from the point clouds. 0 keeps them all.")
    ap.add_argument("--max-points", type=int, default=0, help="The most points per cloud. 0 keeps all.")
    ap.add_argument("--roundtrip-tol", type=float, default=DEFAULT_ROUNDTRIP_TOL,
                    help="The largest round-trip error at p99.9, relative to the median depth.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Replace the stage's output in a clip that already holds it.")
    args = ap.parse_args()
    # Find the clips before the model loads: its weights take 5.1 GB, and a mistyped clip name should fail first.
    root = Path(args.input_dir)
    if not root.is_dir():
        raise SystemExit(f"no such directory: {root}")
    clips = ([root / c for c in args.clips] if args.clips
             else sorted(d for d in root.iterdir() if (d / "frame_manifest.json").is_file()))
    missing = [c.name for c in clips if not (c / "frame_manifest.json").is_file()]
    if missing:
        raise SystemExit(f"no clip (a directory with frame_manifest.json) at: {', '.join(missing)}")
    if not clips:
        raise SystemExit(f"no clip under {root}")
    model = Pi3X(device=args.device, gpu=args.gpu, pixel_limit=args.pixel_limit)
    print(f"{model.model_id} on {model.device}, {len(clips)} clip(s)")
    # Run every clip, even after one fails. Any error, the GPU running out of memory included, counts as the
    # failure of that clip alone. The run lists each failure and exits non-zero.
    failed = []
    for clip_dir in clips:
        print(clip_dir.name)
        try:
            run_pi3x(clip_dir, model, args.pixel_limit, args.conf_thre, args.roundtrip_tol, args.max_points,
                     overwrite=args.overwrite)
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")
            failed.append(clip_dir.name)
    if failed:
        raise SystemExit(f"{len(failed)} of {len(clips)} clip(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
