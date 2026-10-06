"""Write each frame's point cloud from a clip's depth bundle, and record where the cloud sits in the world.

The stage reads `exports/mini_npz/results.npz`, which the depth stage wrote,
and needs no model. It writes `pc_vis/frame_NNNN.glb` for every frame with
usable depth. Each cloud is centred on its centroid. The stage records that
centroid, `glb_centroid`, and the camera's position and axes in glTF's frame,
`camera_*_glb`, on the frame's manifest entry. The viewer's world mode reads
these keys to place every frame's cloud and camera in one scene.

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


def update_manifest(clip_dir: Path, per_frame: dict[int, dict]) -> None:
    """Write each frame's records onto its manifest entry, matched by `seq_idx`.

    Raises:
        FileNotFoundError: the clip has no manifest.
        ValueError: a frame without `seq_idx`. Its records could not be matched to it.
    """
    manifest_path = clip_dir / "frame_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{clip_dir} has no frame_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    frames_in = manifest.get("frames", [])
    if any("seq_idx" not in fr for fr in frames_in):
        raise ValueError(f"{manifest_path}: some frames have no seq_idx")
    existing = {fr["seq_idx"]: fr for fr in frames_in}
    for seq_idx, meta in per_frame.items():
        existing.setdefault(seq_idx, {"seq_idx": seq_idx}).update(meta)
    manifest["frames"] = [existing[k] for k in sorted(existing)]
    manifest["n_frames"] = len(manifest["frames"])
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def run_point_clouds(clip_dir: Path) -> int:
    """Run the stage on one clip and return the number of point clouds written.

    Raises:
        FileNotFoundError: no depth bundle, or no image for a frame of the bundle.
        ValueError: an image that cannot be read, or a manifest `update_manifest` refuses.
    """
    # Read the depth bundle.
    npz_path = clip_dir / "exports" / "mini_npz" / "results.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"{clip_dir} has no {npz_path.relative_to(clip_dir)}; run the depth stage first")
    with np.load(str(npz_path)) as npz:
        depth, K_all, E_all = npz["depth"], npz["intrinsics"], npz["extrinsics"]
    n_frames, H, W = depth.shape

    # Write a cloud per frame, coloured by the frame, and keep where it sits.
    pc_vis = clip_dir / "pc_vis"
    pc_vis.mkdir(exist_ok=True)
    per_frame: dict[int, dict] = {}
    for i in range(n_frames):
        R, t = E_all[i][:3, :3], E_all[i][:3, 3]
        img_path = clip_dir / "input_images" / f"{i:06d}.png"
        if not img_path.is_file():
            raise FileNotFoundError(f"{clip_dir.name}: the bundle has frame {i}, but there is no {img_path.name}")
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise ValueError(f"{clip_dir.name}: cannot read {img_path}")
        colors = cv2.cvtColor(cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_LINEAR),
                              cv2.COLOR_BGR2RGB).reshape(-1, 3)
        gltf_pts = world_to_gltf(cam_to_world(backproject_depth(depth[i], K_all[i]), R, t))
        valid = valid_depth_mask(depth[i].reshape(-1))
        if not valid.any():
            # A frame can hold no usable depth. The viewer shows no cloud for it; the other frames stand.
            print(f"  frame {i:04d}: no usable depth, no point cloud")
            continue
        centroid = write_point_cloud_glb(pc_vis / f"frame_{i:04d}.glb", gltf_pts[valid], colors[valid],
                                         recenter=True)
        per_frame[i] = {"glb_centroid": centroid.tolist(), **camera_axes_in_gltf(R, t)}

    # Record where each cloud sits.
    if per_frame:
        update_manifest(clip_dir, per_frame)
    return len(per_frame)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="The directory that holds the clips.")
    ap.add_argument("--clips", nargs="*", help="Clip directories under --input-dir. Default: every clip there.")
    args = ap.parse_args()
    root = Path(args.input_dir)
    if not root.is_dir():
        raise SystemExit(f"no such directory: {root}")
    clips = ([root / c for c in args.clips] if args.clips
             else sorted(d for d in root.iterdir() if (d / "frame_manifest.json").is_file()))
    # Run every clip, even after one fails. The run lists each failure and exits non-zero.
    failed = []
    for clip_dir in clips:
        print(clip_dir.name)
        try:
            print(f"  {run_point_clouds(clip_dir)} point cloud(s)")
        except (OSError, ValueError) as e:
            print(f"  failed: {e}")
            failed.append(clip_dir.name)
    if failed:
        raise SystemExit(f"{len(failed)} of {len(clips)} clip(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
