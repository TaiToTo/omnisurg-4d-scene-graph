"""Define the interface that every reconstruction model implements.

A backend is built with its settings. Its `reconstruct(image_paths)` returns
a `Reconstruction`: depth, confidence, intrinsics and world-to-camera
extrinsics for the frames, and world points from a model that predicts them.
The backends are DA3, in `recon3d_wrapper.da3`, and Pi3X, in
`recon3d_wrapper.pi3x`. Both select their GPU through `select_gpu`. This
package imports neither backend, so nothing here needs torch.
"""

import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Reconstruction:
    """Hold what a model returns for N frames, as numpy arrays.

    Attributes:
        depth: (N, H, W) depth.
        conf: (N, H, W) confidence.
        intrinsics: (N, 3, 3).
        extrinsics: (N, 3, 4), world to camera.
        points: (N, H, W, 3) world points, from a model that predicts them directly, as Pi3X does; None
            otherwise. A stage checks the pose convention against them.

    Raises:
        ValueError: the shapes do not describe one sequence of N frames, N above 0.
    """

    depth: np.ndarray
    conf: np.ndarray
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    points: np.ndarray | None = None

    def __post_init__(self) -> None:
        # The depth sets N, so its shape is checked first and the others are checked against it.
        if self.depth.ndim != 3 or len(self.depth) == 0:
            raise ValueError(f"not one sequence of frames: depth {self.depth.shape}, not (N, H, W) with N above 0")
        n = len(self.depth)
        expected = {"conf": self.depth.shape, "intrinsics": (n, 3, 3), "extrinsics": (n, 3, 4)}
        if self.points is not None:
            expected["points"] = (*self.depth.shape, 3)
        wrong = [f"{k} {getattr(self, k).shape}, not {v}" for k, v in expected.items() if getattr(self, k).shape != v]
        if wrong:
            raise ValueError(f"not one sequence of {n} frames: " + "; ".join(wrong))


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


class Reconstructor(Protocol):
    """Reconstruct a sequence of frames. `model_id` names the weights; the stages record it with their output."""

    model_id: str

    def reconstruct(self, image_paths: Sequence[Path]) -> Reconstruction:
        """Return depth and poses for the frames, in the order given."""
        ...
