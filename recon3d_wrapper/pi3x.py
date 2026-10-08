"""Run Pi3X behind the `Reconstructor` interface.

Pi3X predicts each frame's points and camera pose together. This backend
turns them into depth, intrinsics fitted to the predicted rays, and
world-to-camera extrinsics. It also keeps the world points, so that a stage
can check the pose convention against them. The module depends on the
upstream `pi3` package and on torch, which the `recon3d` extra installs. It
imports them only when a model is built, so importing the module needs
neither.
"""

import contextlib
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from recon3d_wrapper import Reconstruction

MODEL_REPO = "yyfz233/Pi3X"

# The most pixels a frame keeps after resizing. Upstream's default; a larger budget lost coverage on the clips measured.
DEFAULT_PIXEL_LIMIT = 255_000

# The side of the model's patch. A resized frame's sides are multiples of it.
PATCH = 14


def c2w_to_extrinsics(poses_c2w: np.ndarray) -> np.ndarray:
    """Invert camera-to-world poses into world-to-camera `[R|t]`.

    Pi3X reports a pose as `p_world = Rc2w @ p_cam + tc2w`, so `R = Rc2w.T` and `t = -Rc2w.T @ tc2w`.

    Args:
        poses_c2w: (N, 4, 4) camera-to-world matrices.

    Returns:
        (N, 3, 4) float32 world-to-camera matrices.

    Raises:
        ValueError: the poses are not (N, 4, 4). A (N, 3, 4) input would be inverted into a mirrored scene.
    """
    poses = np.asarray(poses_c2w)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses_c2w must be (N, 4, 4), got {poses.shape}")
    R = np.transpose(poses[:, :3, :3], (0, 2, 1))
    t = -np.einsum("nij,nj->ni", R, poses[:, :3, 3])
    return np.concatenate([R, t[:, :, None]], axis=2).astype(np.float32)


def target_size(width: int, height: int, pixel_limit: int) -> tuple[int, int]:
    """Return the size a frame is resized to: sides that are multiples of `PATCH`, at most `pixel_limit` pixels.

    The search is upstream's. It keeps the aspect ratio as closely as the patch grid allows.
    """
    scale = math.sqrt(pixel_limit / (width * height)) if width * height > 0 else 1.0
    w_target, h_target = width * scale, height * scale
    k, m = round(w_target / PATCH), round(h_target / PATCH)
    while (k * PATCH) * (m * PATCH) > pixel_limit:
        if k / m > w_target / h_target:
            k -= 1
        else:
            m -= 1
    return max(1, k) * PATCH, max(1, m) * PATCH


def load_images(image_paths: Sequence[Path], pixel_limit: int = DEFAULT_PIXEL_LIMIT) -> np.ndarray:
    """Load the frames in the order given, all resized to the first frame's target size.

    Upstream's loader takes a directory and sorts it again, which would drop the caller's order.

    Returns:
        (N, H, W, 3) float32 in [0, 1].

    Raises:
        ValueError: no path.
    """
    paths = [Path(p) for p in image_paths]
    if not paths:
        raise ValueError("image_paths is empty")
    with Image.open(paths[0]) as probe:
        w, h = target_size(*probe.size, pixel_limit)
    # Resize one frame at a time into one buffer, so a long clip is never held at full resolution.
    out = np.empty((len(paths), h, w, 3), dtype=np.float32)
    for i, p in enumerate(paths):
        with Image.open(p) as src:
            img = src.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
        out[i] = np.asarray(img, dtype=np.float32) / 255.0
    return out


def autocast_dtype(device: str, capability: tuple[int, int] | None = None) -> str | None:
    """Return the reduced precision Pi3X runs in on `device`, or None to run in float32.

    CUDA runs bfloat16 from compute capability 8 and float16 below, as the workbench did. The CPU runs
    float32: upstream's camera head takes a determinant, and the CPU has no bfloat16 kernel for it.

    Args:
        device: "cpu" or "cuda".
        capability: the GPU's compute capability, for "cuda".

    Raises:
        ValueError: another device, or "cuda" without a capability.
    """
    if device == "cpu":
        return None
    if device != "cuda" or capability is None:
        raise ValueError(f"no precision for device {device!r} with capability {capability!r}")
    return "bfloat16" if capability[0] >= 8 else "float16"


class Pi3X:
    """Load Pi3X once onto a device.

    Args:
        device: "auto", "cpu" or "cuda". "auto" takes CUDA when there is one.
        gpu: the index of the GPU to use, set through `CUDA_VISIBLE_DEVICES`.
        pixel_limit: the most pixels a frame keeps after resizing.

    Raises:
        RuntimeError: `gpu` is given after torch was imported. `CUDA_VISIBLE_DEVICES` would then have no effect.
    """

    model_id = "pi3x"

    def __init__(self, device: str = "auto", gpu: int | None = None,
                 pixel_limit: int = DEFAULT_PIXEL_LIMIT) -> None:
        if gpu is not None:
            if "torch" in sys.modules:
                raise RuntimeError("torch is already imported, so CUDA_VISIBLE_DEVICES cannot select the GPU")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # Imported here, not at the top: CUDA_VISIBLE_DEVICES takes effect only before torch's first import.
        import torch
        from pi3.models.pi3x import Pi3X as Net

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.pixel_limit = pixel_limit
        net = Net.from_pretrained(MODEL_REPO).eval()
        # The stage passes no conditions, so the multimodal branch only costs memory.
        net.disable_multimodal()
        self.model = net.to(device)

    def reconstruct(self, image_paths: Sequence[Path]) -> Reconstruction:
        """Return depth, poses and world points for the frames, in the order given."""
        import torch
        from pi3.utils.geometry import recover_intrinsic_from_rays_d

        imgs = torch.from_numpy(load_images(image_paths, self.pixel_limit)).permute(0, 3, 1, 2).contiguous()
        imgs = imgs.to(self.device)
        dtype = autocast_dtype(self.device, torch.cuda.get_device_capability() if self.device == "cuda" else None)
        precision = torch.amp.autocast("cuda", dtype=getattr(torch, dtype)) if dtype else contextlib.nullcontext()
        with torch.no_grad(), precision:
            res = self.model(imgs[None])

        # Depth is the local points' z; the intrinsics are fitted to the local rays.
        points = res["points"].float()
        local = res["local_points"].float()
        conf = torch.sigmoid(res["conf"][..., 0].float())
        K = recover_intrinsic_from_rays_d(torch.nn.functional.normalize(local, dim=-1),
                                          force_center_principal_point=True)
        return Reconstruction(
            depth=local[0, ..., 2].cpu().numpy().astype(np.float32),
            conf=conf[0].cpu().numpy().astype(np.float32),
            intrinsics=K[0].cpu().numpy().astype(np.float32),
            extrinsics=c2w_to_extrinsics(res["camera_poses"].float()[0].cpu().numpy()),
            points=points[0].cpu().numpy().astype(np.float32),
        )
