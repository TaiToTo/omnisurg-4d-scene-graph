"""Write each frame's point cloud from a clip's depth bundle, and record where the cloud is in the world.

The stage reads `exports/mini_npz/results.npz`, which the depth stage wrote,
and needs no model. It writes `pc_vis/frame_NNNN.glb` for every frame with
usable depth. Each cloud is centred on its centroid. The stage records that
centroid, `glb_centroid`, and the camera's position and axes in glTF's
coordinate system, `camera_*_glb`, on the frame's manifest entry. The
viewer's world mode reads these keys to place every frame's cloud and camera
in one scene.

Usage:
    python -m pipeline.point_clouds --input-dir /path/to/clips [--clips <clip> ...]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from surgical_core.geometry.camera import backproject_depth, cam_to_world, world_to_gltf
from surgical_core.geometry.valid import valid_depth_mask
from surgical_core.viewer.camera_axes import camera_axes_in_gltf
from surgical_core.viewer.glb import write_point_cloud_glb


def read_bundle(clip_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the clip's depth bundle: the depth maps, the intrinsics and the extrinsics, one of each per frame.

    Raises:
        FileNotFoundError: the clip has no depth bundle.
        ValueError: the three arrays do not hold one entry per frame. The stage would otherwise write some clouds
            and then stop on the frame that has no extrinsics.
    """
    npz_path = clip_dir / "exports" / "mini_npz" / "results.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"{clip_dir} has no {npz_path.relative_to(clip_dir)}; run the depth stage first")
    with np.load(str(npz_path)) as npz:
        depth, K_all, E_all = npz["depth"], npz["intrinsics"], npz["extrinsics"]
    if not (len(K_all) == len(E_all) == len(depth)):
        raise ValueError(f"{npz_path}: {len(depth)} depth maps, {len(K_all)} intrinsics and {len(E_all)} extrinsics; "
                         "the bundle must hold one of each per frame")
    return depth, K_all, E_all


def read_manifest(clip_dir: Path, n_frames: int) -> dict:
    """Read the clip's manifest and check that its frames are the bundle's: `seq_idx` 0 to `n_frames` - 1, each once.

    Raises:
        FileNotFoundError: the clip has no manifest.
        ValueError: a frame without `seq_idx`; `seq_idx` values that are not 0 to `n_frames` - 1 once each, where a
            duplicate would drop an entry and an index past the bundle would add one, both without a word; or an
            `n_frames` that is not the number of frames.
    """
    manifest_path = clip_dir / "frame_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    frames = manifest.get("frames", [])
    if any("seq_idx" not in fr for fr in frames):
        raise ValueError(f"{manifest_path}: some frames have no seq_idx")
    seq_idx = sorted(fr["seq_idx"] for fr in frames)
    if seq_idx != list(range(n_frames)):
        raise ValueError(f"{manifest_path}: the bundle has {n_frames} frames, but the manifest's {len(seq_idx)} "
                         f"seq_idx values are not 0 to {n_frames - 1} once each: {seq_idx[:6]}")
    if manifest.get("n_frames") != n_frames:
        raise ValueError(f"{manifest_path}: n_frames is {manifest.get('n_frames')}, but the manifest holds "
                         f"{n_frames} frames")
    return manifest


def read_colors(clip_dir: Path, n_frames: int, size: tuple[int, int]) -> list[np.ndarray]:
    """Read every frame's image as (H * W, 3) RGB at the depth's `size`, (W, H), in the order of the depth's pixels.

    Raises:
        FileNotFoundError: a frame of the bundle has no image.
        ValueError: an image that cannot be read.
    """
    img_paths = [clip_dir / "input_images" / f"{i:06d}.png" for i in range(n_frames)]
    missing = [p.name for p in img_paths if not p.is_file()]
    if missing:
        shown = ", ".join(missing[:5]) + (", ..." if len(missing) > 5 else "")
        raise FileNotFoundError(f"{clip_dir.name}: the bundle has {n_frames} frames, but {len(missing)} of them "
                                f"have no image: {shown}")
    colors = []
    for img_path in img_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(f"{clip_dir.name}: cannot read {img_path}")
        colors.append(cv2.cvtColor(cv2.resize(img_bgr, size, interpolation=cv2.INTER_LINEAR),
                                   cv2.COLOR_BGR2RGB).reshape(-1, 3))
    return colors


def write_placements(manifest_path: Path, manifest: dict, per_frame: dict[int, dict]) -> None:
    """Write each frame's records onto its manifest entry, matched by `seq_idx`, and the manifest to its file."""
    by_seq_idx = {fr["seq_idx"]: fr for fr in manifest["frames"]}
    for seq_idx, meta in per_frame.items():
        by_seq_idx[seq_idx].update(meta)
    manifest["frames"] = [by_seq_idx[k] for k in sorted(by_seq_idx)]
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def run_point_clouds(clip_dir: Path) -> int:
    """Run the stage on one clip and return the number of point clouds written.

    The stage reads and checks every input before it writes the first cloud, so that a refused clip is left as it was.

    Raises:
        FileNotFoundError: no depth bundle, no manifest, or no image for a frame of the bundle.
        ValueError: a bundle `read_bundle` refuses, a manifest `read_manifest` refuses, or an image that cannot
            be read.
    """
    # Read every input, and refuse the clip before anything is written.
    depth, K_all, E_all = read_bundle(clip_dir)
    n_frames, H, W = depth.shape
    manifest = read_manifest(clip_dir, n_frames)
    colors = read_colors(clip_dir, n_frames, (W, H))

    # Write a cloud per frame, coloured by the frame's image, and keep where the cloud is.
    pc_vis = clip_dir / "pc_vis"
    pc_vis.mkdir(exist_ok=True)
    per_frame: dict[int, dict] = {}
    for i in range(n_frames):
        R, t = E_all[i][:3, :3], E_all[i][:3, 3]
        gltf_pts = world_to_gltf(cam_to_world(backproject_depth(depth[i], K_all[i]), R, t))
        valid = valid_depth_mask(depth[i].reshape(-1))
        if not valid.any():
            # A frame can hold no usable depth. The viewer shows no cloud for it; the other frames still get theirs.
            print(f"  frame {i:04d}: no usable depth, no point cloud")
            continue
        centroid = write_point_cloud_glb(pc_vis / f"frame_{i:04d}.glb", gltf_pts[valid], colors[i][valid],
                                         recenter=True)
        per_frame[i] = {"glb_centroid": centroid.tolist(), **camera_axes_in_gltf(R, t)}

    # Record where each cloud is.
    if per_frame:
        write_placements(clip_dir / "frame_manifest.json", manifest, per_frame)
    return len(per_frame)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="The directory that holds the clips.")
    ap.add_argument("--clips", nargs="*", help="Clip directories under --input-dir. Default: every clip there.")
    args = ap.parse_args()
    # Find the clips first: a mistyped clip name fails here, not as a clip without a depth bundle.
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
    # Run every clip, even after one fails. Any error counts as the failure of that clip alone; on a run of 40
    # clips the rest would otherwise stay undone. The run lists each failure and exits non-zero.
    failed = []
    for clip_dir in clips:
        print(clip_dir.name)
        try:
            print(f"  {run_point_clouds(clip_dir)} point cloud(s)")
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")
            failed.append(clip_dir.name)
    if failed:
        raise SystemExit(f"{len(failed)} of {len(clips)} clip(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
