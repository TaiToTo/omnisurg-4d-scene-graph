"""Stand-ins for the tracking stages' tests: a segmenter, a tracker and a clip with depth, none alike per frame.

The segmenter cuts the k-th image it is given one column farther right
than the one before; the tracker carries its seed masks one column to the
right per frame away from the frame it was prompted on; the clip's images
and depth differ per frame. A label written on another frame, a seed read
from another frame or a propagation that starts away from the prompt
therefore shows. No torch and no weights; the inputs need the `render` extra.
"""
import numpy as np
from PIL import Image

from sam3_wrapper import FrameResult

H, W = 12, 16


class StandInSegmenter:
    """Cut each image into a left part (region 0), a right part (1) and a 2x2 corner (2).

    The k-th image this segmenter is given, counted across clips, is cut at column `W // 2 + k`, so a frame
    written with another frame's labels shows. An image with `k` of `W // 2` or more has no right part.
    """

    device = "cpu"

    def __init__(self, *args, **kwargs):
        self.images, self.grids = [], []

    def label_map(self, image, points_per_side):
        labels = np.zeros(image.shape[:2], int)
        labels[:, W // 2 + len(self.images):] = 1
        self.images.append(image.copy())
        self.grids.append(points_per_side)
        labels[:2, :2] = 2
        return labels


class StandInTracker:
    """Carry the seed masks, rolled one column to the right per frame away from the prompted frame, in the direction asked.

    `calls` records each `propagate` call as `(start frame, reverse)`; `frames`, `seed_frame`, `masks` and `ids`
    hold what the stage handed in.
    """

    device = "cpu"

    def __init__(self, *args, **kwargs):
        self.calls = []

    def init_video(self, frames):
        self.frames = frames

    def add_masks(self, frame_idx, masks, obj_ids):
        self.seed_frame, self.masks, self.ids = frame_idx, masks, obj_ids

    def propagate(self, start_frame_idx, reverse=False):
        self.calls.append((start_frame_idx, reverse))
        order = range(start_frame_idx, -1, -1) if reverse else range(start_frame_idx, len(self.frames))
        for f in order:
            masks = np.stack([carried(m, f, self.seed_frame) for m in self.masks])
            yield FrameResult(f, list(self.ids), masks, np.linspace(0.9, 0.5, len(self.ids)))


def carried(seed: np.ndarray, frame: int, seed_frame: int) -> np.ndarray:
    """The seed mask or label map as the stand-in tracker carries it to `frame`."""
    return np.roll(seed, frame - seed_frame, axis=1)


def write_depth_clip(root, name="VID01_s15_80_crop", n=5, *, scale=1, hole=False, pi3x=False):
    """Write a clip of `n` frames with DA3's bundle, and Pi3X's when asked, and return its directory.

    The images are random, another draw per frame, and `scale` times the depth's size, as a clip's images
    are larger than its depth. The DA3 depth is a slope 0.01 farther per frame, with a hole when asked;
    the Pi3X depth is a curve without one.
    """
    d = root / name
    (d / "input_images").mkdir(parents=True)
    (d / "exports" / "mini_npz").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        Image.fromarray(rng.integers(0, 256, (H * scale, W * scale, 3), dtype=np.uint8)).save(
            d / "input_images" / f"{i:06d}.png")
    yy, xx = np.mgrid[0:H, 0:W]
    depth = np.stack([0.3 + 0.01 * xx + 0.005 * yy + 0.01 * i for i in range(n)]).astype(np.float32)
    if hole:
        depth[:, 5:8, 6:10] = 0.0
    K = np.tile(np.array([[20.0, 0, W / 2], [0, 20.0, H / 2], [0, 0, 1]]), (n, 1, 1))
    np.savez(d / "exports" / "mini_npz" / "results.npz", depth=depth, intrinsics=K, extrinsics=np.zeros((n, 3, 4)))
    if pi3x:
        pi3 = np.stack([1.0 + 0.04 * yy * yy for _ in range(n)]).astype(np.float32)
        np.savez(d / "exports" / "mini_npz" / "results__pi3x.npz", depth=pi3, intrinsics=K)
    return d
