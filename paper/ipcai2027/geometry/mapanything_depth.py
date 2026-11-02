"""Run MapAnything on clips and write its depth and poses as `exports/mini_npz/results__mapanything.npz`.

The bundle holds the keys, dtypes and conventions of the Pi3X stage's bundle: `depth`, `conf`, world-to-camera
`extrinsics` and the `intrinsics` of the depth's own resolution. `MapAnything` implements the interface of
`recon3d_wrapper`, so that `d4d_wholeclip_depth.py` runs it too. It resizes every frame to a long side of 518
without cropping, runs all frames in one call on the images alone, and refuses poses that fail the convention
test. Before the stage writes a bundle, it also bounds the full round trip and the principal point. It writes
nothing else into a clip, and skips a clip that already holds the bundle. The file is not named after the
`mapanything` package, since a script's own directory comes first on the import path.

Usage:
    python paper/ipcai2027/geometry/mapanything_depth.py --input-dir /path/to/clips [--clips <clip> ...] --gpu N
"""

import argparse
import os
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.pi3x import read_manifest, verify_roundtrip
from recon3d_wrapper import Reconstruction, select_gpu
from recon3d_wrapper.pi3x import c2w_to_extrinsics
from surgical_core.geometry.camera import cam_to_world

MODEL_REPO = "facebook/map-anything"
# The Hugging Face commit the paper's runs loaded, pinned so that a later upload changes nothing here.
MODEL_REVISION = "a1d87e9086706fb9974f3be5a3e3a0ca5401c5aa"
BUNDLE = "exports/mini_npz/results__mapanything.npz"
# Every entry of upstream's 518 resolution mapping has 518 on its long side; the patch is DINOv2's.
LONG_SIDE = 518
PATCH = 14
NORM_TYPE = "dinov2"
# How many times the inverted pose reading's error must exceed the right one's, and the largest error of the
# right one at p99.9, relative to the median depth. The paper's clips gave at least 38 times and at most 1.8e-3.
POSE_MIN_RATIO = 10.0
POSE_TOL = 1e-2
# The largest full round-trip error at p99.9, relative to the median depth. MapAnything's rays are not a pinhole,
# and the fitted intrinsics leave about 5 px; the paper's clips gave at most 7.8e-2.
ROUNDTRIP_TOL = 1e-1
# How far the principal point may sit from the image centre, as a share of the side.
CENTRE_TOL = 0.05


def target_size(width: int, height: int) -> tuple[int, int]:
    """Return the size a frame is resized to: the long side `LONG_SIDE`, the short side the nearest multiple of
    `PATCH` to the aspect ratio. Upstream's loader would crop the centre instead."""
    if width >= height:
        return LONG_SIDE, max(PATCH, round(LONG_SIDE * height / width / PATCH) * PATCH)
    return max(PATCH, round(LONG_SIDE * width / height / PATCH) * PATCH), LONG_SIDE


def load_views(paths: Sequence[Path]) -> tuple[list[dict], tuple[int, int]]:
    """Return MapAnything's input views of the frames, built as upstream builds them but resized, never cropped.

    Raises:
        ValueError: no frame, or frames of different sizes.
    """
    import torchvision.transforms as tvf
    from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

    if not paths:
        raise ValueError("no frame to run")
    norm = IMAGE_NORMALIZATION_DICT[NORM_TYPE]
    to_input = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=norm.mean, std=norm.std)])
    with Image.open(paths[0]) as probe:
        src = probe.size
    w, h = target_size(*src)
    views = []
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            if im.size != src:
                raise ValueError(f"{p}: size {im.size}, but the first frame is {src}")
            # Upstream downsamples with Lanczos and upsamples with bicubic.
            resample = Image.Resampling.LANCZOS if w < src[0] else Image.Resampling.BICUBIC
            img = im.convert("RGB").resize((w, h), resample)
        views.append(dict(img=to_input(img)[None], true_shape=np.int32([[h, w]]), idx=i, instance=str(i),
                          data_norm_type=[NORM_TYPE]))
    return views, (w, h)


def pose_test(pts_cam: np.ndarray, pts_world: np.ndarray, E: np.ndarray, median_depth: float,
              n_sample: int = 20000, seed: int = 0) -> dict:
    """Compare the model's camera-frame points moved by the pose with its world points, under both pose readings.

    The intrinsics take no part, so the test reads the pose convention alone.

    Returns:
        The right reading's error at p99.9, relative to the median depth, and the inverted reading's ratio to it.

    Raises:
        ValueError: an error that is not finite.
    """
    rng = np.random.default_rng(seed)
    err, err_inv = [], []
    for i in range(len(E)):
        idx = rng.choice(pts_cam[i].shape[0] * pts_cam[i].shape[1], n_sample, replace=False)
        cam, ref = pts_cam[i].reshape(-1, 3)[idx], pts_world[i].reshape(-1, 3)[idx]
        R, t = E[i][:3, :3], E[i][:3, 3]
        err.append(np.linalg.norm(cam_to_world(cam, R, t) - ref, axis=-1))
        err_inv.append(np.linalg.norm(cam @ R.T + t - ref, axis=-1))
    e, e_inv = np.concatenate(err) / median_depth, np.concatenate(err_inv) / median_depth
    if not (np.isfinite(e).all() and np.isfinite(e_inv).all()):
        raise ValueError("the pose test's error is not finite")
    p999 = float(np.percentile(e, 99.9))
    return {"p999_rel": p999, "ratio_p999": float(np.percentile(e_inv, 99.9)) / max(p999, 1e-12)}


class MapAnything:
    """Load MapAnything once onto a GPU.

    Args:
        gpu: the index of the GPU to use, set through `CUDA_VISIBLE_DEVICES`.

    Raises:
        RuntimeError: `gpu` is given after torch was imported.
    """

    model_id = "mapanything"

    def __init__(self, gpu: int | None = None) -> None:
        select_gpu(gpu)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # Imported here, not at the top: CUDA_VISIBLE_DEVICES takes effect only before torch's first import.
        from mapanything.models import MapAnything as Net

        self.model = Net.from_pretrained(MODEL_REPO, revision=MODEL_REVISION).to("cuda").eval()

    def reconstruct(self, image_paths: Sequence[Path]) -> Reconstruction:
        """Return depth, poses and world points for the frames, in the order given.

        Raises:
            ValueError: another number of frames or another size comes back, a value is not finite, a pose is not a
                rotation, or the poses fail the convention test.
        """
        import torch

        # Run every frame at once, images only, with every pixel kept.
        views, (w, h) = load_views(image_paths)
        n = len(views)
        preds = self.model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16",
                                 apply_mask=False, mask_edges=False, apply_confidence_mask=False)
        if len(preds) != n:
            raise ValueError(f"{len(preds)} predictions for {n} frames")
        cat = lambda k: torch.cat([p[k].float() for p in preds]).cpu().numpy()  # noqa: E731
        depth, conf, K, c2w = cat("depth_z")[..., 0], cat("conf"), cat("intrinsics"), cat("camera_poses")
        pts_world, pts_cam = cat("pts3d"), cat("pts3d_cam")

        # Refuse another size, a value that is not finite, and a pose that is not a rotation.
        if depth.shape != (n, h, w):
            raise ValueError(f"depth {depth.shape}, not the input's {(n, h, w)}")
        if not all(np.isfinite(v).all() for v in (depth, conf, K, c2w)):
            raise ValueError("the depth, confidence, intrinsics or poses hold a value that is not finite")
        R = c2w[:, :3, :3].astype(np.float64)
        if (np.abs(np.einsum("nji,njk->nik", R, R) - np.eye(3)).max() > 1e-3
                or np.abs(np.linalg.det(R) - 1).max() > 1e-3):
            raise ValueError("a camera pose's rotation is not orthonormal with determinant +1")

        # Refuse poses whose inverted reading fits nearly as well, or whose right reading fits badly.
        E = c2w_to_extrinsics(c2w.astype(np.float64))
        test = pose_test(pts_cam, pts_world, E, float(np.median(depth[depth > 0])))
        if test["ratio_p999"] < POSE_MIN_RATIO or test["p999_rel"] > POSE_TOL:
            raise ValueError(f"the pose test fails: p99.9 {test['p999_rel']:.2e} (at most {POSE_TOL:.0e}), "
                             f"inverted {test['ratio_p999']:.1f} times worse (at least {POSE_MIN_RATIO:.0f})")
        return Reconstruction(depth=depth.astype(np.float32), conf=conf.astype(np.float32),
                              intrinsics=K.astype(np.float32), extrinsics=E, points=pts_world.astype(np.float32))


def run_clip(clip_dir: Path, model: MapAnything) -> str:
    """Run MapAnything on one clip, bound its geometry, and write the bundle; return a line on the run.

    Raises:
        FileNotFoundError: the clip has no image or no manifest.
        ValueError: `read_manifest` refuses the manifest, `reconstruct` refuses the output, or the full round trip
            or the principal point exceeds its bound. Nothing is written then.
    """
    # Read the frames in the manifest's order; the bundle's frame i is read downstream as the manifest's frame i.
    pngs = sorted((clip_dir / "input_images").glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"{clip_dir} has no input_images/*.png")
    order = [fr["seq_idx"] for fr in read_manifest(clip_dir, len(pngs))["frames"]]
    if order != list(range(len(pngs))):
        raise ValueError(f"{clip_dir.name}: the manifest lists its frames in another order than seq_idx")

    # Reconstruct, and bound the full round trip and the principal point.
    t0 = time.perf_counter()
    rec = model.reconstruct(pngs)
    seconds = time.perf_counter() - t0
    trip = verify_roundtrip(rec)["p999_rel"]
    _, h, w = rec.depth.shape
    off = max(np.abs(rec.intrinsics[:, 0, 2] - (w - 1) / 2).max() / w,
              np.abs(rec.intrinsics[:, 1, 2] - (h - 1) / 2).max() / h)
    if trip > ROUNDTRIP_TOL or off > CENTRE_TOL:
        raise ValueError(f"{clip_dir.name}: round trip p99.9 {trip:.3e} (at most {ROUNDTRIP_TOL:.0e}) or principal "
                         f"point {off:.3f} of the side from the centre (at most {CENTRE_TOL})")

    # Write the bundle under a temporary name, and move it into place.
    dest = clip_dir / BUNDLE
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.stem + ".partial.npz")
    np.savez(str(tmp), depth=rec.depth, conf=rec.conf, extrinsics=rec.extrinsics, intrinsics=rec.intrinsics)
    os.replace(tmp, dest)
    return f"{len(pngs)} frames at {w}x{h} in {seconds:.0f} s; round trip p99.9 {trip:.2e}, principal point {off:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="The directory that holds the clips.")
    ap.add_argument("--clips", nargs="*", help="Clip directories under --input-dir. Default: every clip there.")
    ap.add_argument("--gpu", type=int, default=None, help="The index of the GPU to use.")
    args = ap.parse_args()

    # Find the clips before the model loads, and skip a clip that already holds the bundle.
    root = Path(args.input_dir)
    if not root.is_dir():
        raise SystemExit(f"no such directory: {root}")
    clips = ([root / c for c in args.clips] if args.clips
             else sorted(d for d in root.iterdir() if (d / "frame_manifest.json").is_file()))
    missing = [c.name for c in clips if not (c / "frame_manifest.json").is_file()]
    if missing or not clips:
        raise SystemExit(f"no clip (a directory with frame_manifest.json) at: {missing or root}")
    todo = [c for c in clips if not (c / BUNDLE).exists()]
    print(f"{len(clips)} clip(s), {len(clips) - len(todo)} with the bundle already", flush=True)
    if not todo:
        return

    # Run every clip, even after one fails, and exit non-zero naming each failure.
    model = MapAnything(gpu=args.gpu)
    failed = []
    for clip_dir in todo:
        try:
            print(f"{clip_dir.name}: {run_clip(clip_dir, model)}", flush=True)
        except Exception as e:
            print(f"{clip_dir.name} failed: {type(e).__name__}: {e}", flush=True)
            failed.append(clip_dir.name)
    if failed:
        raise SystemExit(f"{len(failed)} of {len(todo)} clip(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
