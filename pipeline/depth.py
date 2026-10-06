"""Run the depth stage: estimate depth for every frame of each clip, and write it with its images and point clouds.

A clip is a directory that holds `frame_manifest.json` and `input_images/`.
The stage writes `depth_raw/depth_NNNNNN.npy`, `depth_vis/NNNN.jpg`,
`exports/mini_npz/results.npz` and `pc_vis/frame_NNNN.glb`, and adds
`depth_info` to the manifest. The bundle `results.npz` holds depth,
confidence, intrinsics and world-to-camera extrinsics. A CholecSeg8k clip
must already be cut to the endoscope's view; the stage refuses one that has
no `crop_info.json`.

Usage:
    python -m pipeline.depth --input-dir /path/to/clips [--clips <clip> ...] [--device auto] [--gpu 0] [--no-glb]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from recon3d_wrapper import Reconstructor
from recon3d_wrapper.da3 import DA3, DEFAULT_PROCESS_RES
from surgical_core.geometry.camera import backproject_depth, cam_to_world, world_to_gltf
from surgical_core.geometry.valid import valid_depth_mask
from surgical_core.viewer.depth_vis import depth_to_colormap
from surgical_core.viewer.glb import write_point_cloud_glb

# The dataset of a manifest that has no `dataset` key. CholecSeg8k's manifests were written before the key existed.
DEFAULT_DATASET = "cholec_gt"

# The datasets whose clips need no crop. ATLAS-120k frames have no endoscope border; their extraction removes any
# letterbox and records it in the manifest's `crop`.
UNCROPPED_DATASETS = frozenset({"atlas120k"})


def check_cropped(clip_dir: Path, manifest: dict) -> None:
    """Refuse an endoscope clip that was not cut to the endoscope's view.

    The model would otherwise see the black border around the view.

    Raises:
        ValueError: the clip needs `crop_info.json` and has none.
    """
    dataset = manifest.get("dataset", DEFAULT_DATASET)
    if dataset not in UNCROPPED_DATASETS and not (clip_dir / "crop_info.json").is_file():
        raise ValueError(f"{clip_dir.name}: a {dataset} clip without crop_info.json is not cut to the endoscope's "
                         "view; the depth stage takes cropped clips only")


def run_depth(clip_dir: Path, model: Reconstructor, process_res: int, write_glb: bool = True) -> None:
    """Run the stage on one clip.

    Args:
        clip_dir: the clip.
        model: the reconstruction model. It is loaded once for every clip.
        process_res: the resolution the model was built with. `depth_info` records it.
        write_glb: write the point clouds. Only the viewer reads them, and they take about 20 MB a frame.

    Raises:
        FileNotFoundError: the clip has no manifest or no image.
        ValueError: `check_cropped` refuses the clip.
    """
    # Read the clip's manifest and images, and refuse an uncropped endoscope clip.
    manifest_path = clip_dir / "frame_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{clip_dir} has no frame_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    check_cropped(clip_dir, manifest)
    image_paths = sorted((clip_dir / "input_images").glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"{clip_dir} has no input_images/*.png")
    n_frames = len(image_paths)
    print(f"  {n_frames} frames, process_res={process_res}")

    # Estimate depth and poses for every frame, in one call.
    rec = model.reconstruct(image_paths)
    depth, conf = rec.depth, rec.conf
    H, W = depth.shape[1], depth.shape[2]
    print(f"  depth shape ({n_frames}, {H}, {W})")

    # Write the raw depth and its images.
    (clip_dir / "depth_raw").mkdir(exist_ok=True)
    for i in range(n_frames):
        np.save(str(clip_dir / "depth_raw" / f"depth_{i:06d}.npy"), depth[i])
    (clip_dir / "depth_vis").mkdir(exist_ok=True)
    for i in range(n_frames):
        Image.fromarray(depth_to_colormap(depth[i])).save(str(clip_dir / "depth_vis" / f"{i:04d}.jpg"))

    # Write the bundle that the later stages read.
    exports = clip_dir / "exports" / "mini_npz"
    exports.mkdir(parents=True, exist_ok=True)
    np.savez(str(exports / "results.npz"), depth=depth.astype(np.float32), conf=conf.astype(np.float32),
             extrinsics=rec.extrinsics.astype(np.float32), intrinsics=rec.intrinsics.astype(np.float32))

    # Write a point cloud per frame, coloured by the frame. Every pixel with usable depth is kept.
    if write_glb:
        (clip_dir / "pc_vis").mkdir(exist_ok=True)
        for i in range(n_frames):
            R, t = rec.extrinsics[i][:3, :3], rec.extrinsics[i][:3, 3]
            img = cv2.cvtColor(cv2.resize(cv2.imread(str(image_paths[i])), (W, H), interpolation=cv2.INTER_LINEAR),
                               cv2.COLOR_BGR2RGB)
            gltf_pts = world_to_gltf(cam_to_world(backproject_depth(depth[i], rec.intrinsics[i]), R, t))
            valid = valid_depth_mask(depth[i].reshape(-1))
            if not valid.any():
                # A frame can hold no usable depth. The viewer shows no cloud for it; the other frames stand.
                print(f"  frame {i:04d}: no usable depth, no point cloud")
                continue
            write_point_cloud_glb(clip_dir / "pc_vis" / f"frame_{i:04d}.glb", gltf_pts[valid],
                                  img.reshape(-1, 3)[valid], recenter=True)

    # Record what was made in the manifest. `border_inpaint` stays false: the stage fills in no border.
    manifest["depth_info"] = {
        "model": model.model_id,
        "process_res": process_res,
        "depth_shape": [n_frames, H, W],
        "depth_range": [float(np.nanmin(depth)), float(np.nanmax(depth))],
        "conf_range": [float(np.nanmin(conf)), float(np.nanmax(conf))],
        "border_inpaint": False,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="The directory that holds the clips.")
    ap.add_argument("--clips", nargs="*", help="Clip directories under --input-dir. Default: every clip there.")
    ap.add_argument("--device", default="auto", help="auto, cuda or cpu. auto takes CUDA when there is one.")
    ap.add_argument("--gpu", type=int, default=0, help="The index of the GPU to use.")
    ap.add_argument("--process-res", type=int, default=DEFAULT_PROCESS_RES,
                    help="The longest side each frame is resized to.")
    ap.add_argument("--no-glb", action="store_true", help="Skip the point clouds. Only the viewer reads them.")
    args = ap.parse_args()
    root = Path(args.input_dir)
    if not root.is_dir():
        raise SystemExit(f"no such directory: {root}")
    clips = ([root / c for c in args.clips] if args.clips
             else sorted(d for d in root.iterdir() if (d / "frame_manifest.json").is_file()))
    model = DA3(device=args.device, gpu=args.gpu, process_res=args.process_res)
    print(f"{model.model_id} on {model.device}, {len(clips)} clip(s)")
    # Run every clip, even after one fails. The run lists each failure and exits non-zero.
    failed = []
    for clip_dir in clips:
        print(clip_dir.name)
        try:
            run_depth(clip_dir, model, args.process_res, write_glb=not args.no_glb)
        except (OSError, ValueError) as e:
            print(f"  failed: {e}")
            failed.append(clip_dir.name)
    if failed:
        raise SystemExit(f"{len(failed)} of {len(clips)} clip(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
