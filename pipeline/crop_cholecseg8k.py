"""Crop CholecSeg8k clips to the rectangle inside the endoscope's view.

The rectangles are data, one per clip, in `cholecseg8k_meta/crop_rects.json`.
For a clip `<clip>`, the stage writes `<clip>_crop` in the same directory,
holding:

- every image of `input_images/` and every colour mask of `seg_masks/`,
  cropped to the clip's rectangle;
- the frame manifest, with each frame's `image_size` and the rectangle as
  `crop_info`;
- the rectangle again as `crop_info.json`.

Usage:
    python -m pipeline.crop_cholecseg8k --input-dir /path/to/clips --rects cholecseg8k_meta/crop_rects.json \\
        --clips VID01_s15_80 [VID25_s15_162 ...] [--overwrite]
"""

import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

SUFFIX = "_crop"
PART = ".part"
RECT_KEYS = ("y0", "y1", "x0", "x1", "src_h", "src_w")
OUTPUTS = ("input_images", "seg_masks", "frame_manifest.json", "crop_info.json")


def load_rects(path: str) -> dict[str, dict[str, int]]:
    """Read the rectangles, keyed by clip, and refuse one that does not lie inside its frame.

    Raises:
        ValueError: a rectangle's keys are not `RECT_KEYS`, a value is not an integer, or the rectangle is empty
            or reaches past its frame.
    """
    rects = json.loads(Path(path).read_text())
    for clip, r in rects.items():
        if set(r) != set(RECT_KEYS) or not all(isinstance(r[k], int) and not isinstance(r[k], bool) for k in r):
            raise ValueError(f"{clip}: a rectangle holds the integers {RECT_KEYS}, not {r}")
        if not (0 <= r["y0"] < r["y1"] <= r["src_h"] and 0 <= r["x0"] < r["x1"] <= r["src_w"]):
            raise ValueError(f"{clip}: the rectangle {r} is empty or reaches past its frame")
    return rects


def check_size(path: Path, rect: dict[str, int]) -> None:
    """Refuse an image file whose size is not the frame's. Only the file's header is read.

    On a smaller image numpy shortens a slice without an error, and the frame and its mask would come from
    different regions.

    Raises:
        ValueError: the sizes differ.
    """
    with Image.open(path) as im:
        size = im.size
    if size != (rect["src_w"], rect["src_h"]):
        raise ValueError(f"{path} is {size[0]}x{size[1]}; the rectangle was found on frames of "
                         f"{rect['src_w']}x{rect['src_h']}")


def crop_image(path: Path, dst: Path, rect: dict[str, int]) -> None:
    """Crop an image to the rectangle and write it."""
    img = np.asarray(Image.open(path).convert("RGB"))
    Image.fromarray(img[rect["y0"]:rect["y1"], rect["x0"]:rect["x1"]]).save(dst)


def cropped_manifest(manifest: dict, rect: dict[str, int]) -> dict:
    """Return the cropped clip's manifest: the clip's, with each frame's new `image_size` and the rectangle."""
    cropped = copy.deepcopy(manifest)
    for f in cropped["frames"]:
        f["image_size"] = [rect["x1"] - rect["x0"], rect["y1"] - rect["y0"]]
    cropped["crop_info"] = {k: rect[k] for k in RECT_KEYS}
    return cropped


def later_keys(held: dict, cropped: dict) -> list[str]:
    """List the keys of a cropped clip's manifest that this stage does not write, in the manifest and its frames.

    A later stage records its run in the manifest, as the depth stage adds `depth_info`. A key of `held` that
    `cropped` lacks is such a record.
    """
    def frame_keys(manifest: dict) -> set[str]:
        return {k for f in manifest.get("frames", []) for k in f}

    return ([f"{k} in frame_manifest.json" for k in sorted(set(held) - set(cropped))]
            + [f"{k} in a frame of frame_manifest.json" for k in sorted(frame_keys(held) - frame_keys(cropped))])


def write_cropped(dst: Path, images: list[Path], masks: list[Path], manifest: dict, rect: dict[str, int]) -> None:
    """Write the cropped images and masks into `dst`, then the cropped clip's manifest, then the rectangle."""
    (dst / "input_images").mkdir(parents=True)
    for p in images:
        crop_image(p, dst / "input_images" / p.name, rect)
    if masks:
        (dst / "seg_masks").mkdir()
    for p in masks:
        crop_image(p, dst / "seg_masks" / p.name, rect)
    with open(dst / "frame_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(dst / "crop_info.json", "w") as f:
        json.dump({k: rect[k] for k in RECT_KEYS}, f)


def crop_clip(clip_dir: Path, rects: dict[str, dict[str, int]], overwrite: bool = False) -> Path:
    """Crop one clip and return the directory it was written to.

    The cropped clip is written as `<clip>_crop.part` and renamed when every file is written. A run that fails
    leaves the clip, and a cropped clip an earlier run left, as they were.

    Raises:
        FileNotFoundError: the clip has no manifest or no image.
        KeyError: the clip has no rectangle.
        ValueError: the manifest does not list one frame per image, an image or a mask is not the frame's size,
            the cropped clip exists and `overwrite` is false, or the cropped clip holds a later stage's output.
    """
    # Read the clip and its rectangle.
    manifest_path = clip_dir / "frame_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{clip_dir} has no frame_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if clip_dir.name not in rects:
        raise KeyError(f"{clip_dir.name} has no rectangle")
    rect = rects[clip_dir.name]
    images = sorted((clip_dir / "input_images").glob("*.png"))
    if not images:
        raise FileNotFoundError(f"{clip_dir} has no input_images/*.png")
    masks = sorted((clip_dir / "seg_masks").glob("*_color_mask.png"))

    # Check that the manifest lists one frame per image, that every image and mask is the frame's size, and
    # that an earlier cropped clip may be replaced.
    frames = manifest.get("frames")
    if not isinstance(frames, list) or len(frames) != len(images):
        n = len(frames) if isinstance(frames, list) else "no"
        raise ValueError(f"{manifest_path} lists {n} frames; {clip_dir.name} holds {len(images)} images")
    for p in images + masks:
        check_size(p, rect)
    cropped = cropped_manifest(manifest, rect)
    dst = clip_dir.parent / f"{clip_dir.name}{SUFFIX}"
    if dst.exists() and not overwrite:
        raise ValueError(f"{dst} exists; pass --overwrite to replace it")
    # A later stage's output, such as depth, is not this stage's to remove, even when asked to overwrite. A later
    # stage writes files of its own, and may add keys to the manifest without leaving a file.
    later = sorted(p.name for p in dst.iterdir() if p.name not in OUTPUTS) if dst.exists() else []
    if (dst / "frame_manifest.json").is_file():
        later += later_keys(json.loads((dst / "frame_manifest.json").read_text()), cropped)
    if later:
        raise ValueError(f"{dst} holds {', '.join(later)}, which a later stage wrote; remove {dst} by hand to "
                         f"crop {clip_dir.name} again")

    # Write the cropped clip under a temporary name. A run that fails removes it.
    part = dst.with_name(dst.name + PART)
    if part.exists():
        shutil.rmtree(part)
    try:
        write_cropped(part, images, masks, cropped, rect)
    except BaseException:
        shutil.rmtree(part, ignore_errors=True)
        raise

    # Replace the earlier cropped clip with the new one.
    if dst.exists():
        shutil.rmtree(dst)
    part.rename(dst)
    print(f"  {dst.name}: {len(images)} images, {len(masks)} masks, "
          f"y[{rect['y0']}:{rect['y1']}] x[{rect['x0']}:{rect['x1']}]")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="The directory that holds the clips.")
    ap.add_argument("--rects", required=True, help="The rectangle of each clip, crop_rects.json.")
    ap.add_argument("--clips", nargs="+", required=True, help="Clip directories under --input-dir.")
    ap.add_argument("--overwrite", action="store_true", help="Replace a cropped clip an earlier run left.")
    args = ap.parse_args()

    # Read the rectangles, and find every clip and its rectangle, before any image is read.
    rects = load_rects(args.rects)
    root = Path(args.input_dir)
    if not root.is_dir():
        raise SystemExit(f"no such directory: {root}")
    missing = [c for c in args.clips if not (root / c / "frame_manifest.json").is_file()]
    if missing:
        raise SystemExit(f"no clip (a directory with frame_manifest.json) at: {', '.join(missing)}")
    without_rect = [c for c in args.clips if c not in rects]
    if without_rect:
        raise SystemExit(f"no rectangle in {args.rects} for: {', '.join(without_rect)}")

    # Crop every clip, even after one fails, and list the failures at the end.
    failed = []
    for clip in args.clips:
        print(clip)
        try:
            crop_clip(root / clip, rects, overwrite=args.overwrite)
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}")
            failed.append(clip)
    if failed:
        raise SystemExit(f"{len(failed)} of {len(args.clips)} clip(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
