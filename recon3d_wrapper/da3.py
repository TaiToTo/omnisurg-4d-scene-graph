"""Run Depth Anything 3 behind the `Reconstructor` interface.

The model loads once and then reconstructs any number of sequences. The
module depends on the upstream `depth_anything_3` package and on torch,
which the `recon3d` extra installs. It imports them only when a model is
built, so importing the module needs neither.
"""

import os
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from recon3d_wrapper import Reconstruction

DEFAULT_MODEL_ID = "depth-anything/DA3-LARGE"

# The longest side each frame is resized to. 504 is DA3's training resolution and a multiple of its 14-pixel patch.
DEFAULT_PROCESS_RES = 504


def world_to_camera(extrinsics: np.ndarray) -> np.ndarray:
    """Return the (N, 3, 4) world-to-camera part of the extrinsics DA3 returns.

    The version the extra installs returns (N, 3, 4); other versions return (N, 4, 4) homogeneous matrices, whose
    last row is (0, 0, 0, 1) and carries nothing. `Reconstruction` holds the (N, 3, 4) part of either.

    Raises:
        ValueError: the matrices are neither (N, 4, 4) with that last row nor (N, 3, 4).
    """
    extrinsics = np.asarray(extrinsics)
    if extrinsics.ndim == 3 and extrinsics.shape[1:] == (3, 4):
        return extrinsics
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (4, 4):
        raise ValueError(f"extrinsics are {extrinsics.shape}, not (N, 4, 4) or (N, 3, 4)")
    if not np.allclose(extrinsics[:, 3, :], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("extrinsics are (N, 4, 4) but not homogeneous: the last row is not (0, 0, 0, 1)")
    return extrinsics[:, :3, :]


def select_gpu(gpu: int | None) -> None:
    """Select the GPU to use, through `CUDA_VISIBLE_DEVICES`. None leaves the environment as it is.

    Raises:
        RuntimeError: torch is already imported. The variable would then have no effect.
    """
    if gpu is None:
        return
    if "torch" in sys.modules:
        raise RuntimeError("torch is already imported, so CUDA_VISIBLE_DEVICES cannot select the GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


class DA3:
    """Load DA3 once onto a device.

    Args:
        model_id: the Hugging Face model id. The weights download on first use.
        device: "auto", "cpu" or "cuda". "auto" takes CUDA when there is one.
        gpu: the index of the GPU to use, set through `CUDA_VISIBLE_DEVICES`. None leaves the environment as it is.
        process_res: the longest side each frame is resized to before inference.

    Raises:
        RuntimeError: `gpu` is given after torch was imported. `CUDA_VISIBLE_DEVICES` would then have no effect.
    """

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "auto", gpu: int | None = None,
                 process_res: int = DEFAULT_PROCESS_RES) -> None:
        select_gpu(gpu)
        # Imported here, not at the top: CUDA_VISIBLE_DEVICES takes effect only before torch's first import.
        import torch
        from depth_anything_3.api import DepthAnything3

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.device = device
        self.process_res = process_res
        self.model = DepthAnything3.from_pretrained(model_id).to(device)

    def reconstruct(self, image_paths: Sequence[Path]) -> Reconstruction:
        """Return depth and poses for the frames, in the order given."""
        pred = self.model.inference([str(p) for p in image_paths], process_res=self.process_res)
        return Reconstruction(depth=np.asarray(pred.depth), conf=np.asarray(pred.conf),
                              intrinsics=np.asarray(pred.intrinsics), extrinsics=world_to_camera(pred.extrinsics))
