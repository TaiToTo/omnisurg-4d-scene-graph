"""Cut CholecSeg8k clips to the rectangle inside the endoscope's view of their video.

For a clip `<clip>`, it writes `<clip>_crop` beside it: every image of
`input_images/` and every colour mask of `seg_masks/`, cut to the video's
rectangle; the frame manifest, with each frame's `image_size` and the
rectangle as `crop_info`; and the rectangle again as `crop_info.json`. The
rectangles are data, one per clip, in `cholecseg8k_meta/crop_rects.json`.

Usage:
    python -m pipeline.crop_cholecseg8k --input-dir /path/to/clips --rects cholecseg8k_meta/crop_rects.json \\
        --clips VID01_s15_80 [VID25_s15_162 ...] [--overwrite]
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

SUFFIX = "_crop"
RECT_KEYS = ("y0", "y1", "x0", "x1", "src_h", "src_w")


def load_rects(path: str) -> dict[str, dict[str, int]]:
    """Read the rectangles, keyed by clip, and refuse one that does not lie inside its frame.

    Raises:
        ValueError: a rectangle lacks a key, holds a value that is not an integer, or is empty or reaches past
            its frame.
    """
    rects = json.loads(Path(path).read_text())
    for clip, r in rects.items():
        if set(r) != set(RECT_KEYS) or not all(isinstance(r[k], int) and not isinstance(r[k], bool) for k in r):
            raise ValueError(f"{clip}: a rectangle holds the integers {RECT_KEYS}, not {r}")
        if not (0 <= r["y0"] < r["y1"] <= r["src_h"] and 0 <= r["x0"] < r["x1"] <= r["src_w"]):
            raise ValueError(f"{clip}: the rectangle {r} is empty or reaches past its frame")
    return rects


def cut(path: Path, dst: Path, rect: dict[str, int]) -> None:
    """Cut an image to the rectangle and write it, refusing one that is not the frame's size."""
    img = np.asarray(Image.open(path).convert("RGB"))
    if img.shape[:2] != (rect["src_h"], rect["src_w"]):
        raise ValueError(f"{path} is {img.shape[1]}x{img.shape[0]}, not the frame's "
                         f"{rect['src_w']}x{rect['src_h']} the rectangle was found on")
    Image.fromarray(img[rect["y0"]:rect["y1"], rect["x0"]:rect["x1"]]).save(dst)


def crop_clip(clip_dir: Path, rects: dict[str, dict[str, int]], overwrite: bool = False) -> Path:
    """Cut one clip and return the directory it was written to.

    Raises:
        FileNotFoundError: the clip has no manifest or no image.
        KeyError: the clip has no rectangle.
        ValueError: the cut clip exists and `overwrite` is false, or an image is not the frame's size.
    """
    # Read the clip and its rectangle, and refuse a cut clip an earlier run left.
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
    dst = clip_dir.parent / f"{clip_dir.name}{SUFFIX}"
    if dst.exists():
        if not overwrite:
            raise ValueError(f"{dst} exists; pass --overwrite to replace it")
        shutil.rmtree(dst)

    # Cut the images and the colour masks.
    (dst / "input_images").mkdir(parents=True)
    for p in images:
        cut(p, dst / "input_images" / p.name, rect)
    masks = sorted((clip_dir / "seg_masks").glob("*_color_mask.png"))
    if masks:
        (dst / "seg_masks").mkdir()
    for p in masks:
        cut(p, dst / "seg_masks" / p.name, rect)

    # Write the manifest with each frame's new size and the rectangle, and the rectangle on its own.
    for f in manifest.get("frames", []):
        f["image_size"] = [rect["x1"] - rect["x0"], rect["y1"] - rect["y0"]]
    manifest["crop_info"] = {k: rect[k] for k in RECT_KEYS}
    with open(dst / "frame_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    with open(dst / "crop_info.json", "w") as f:
        json.dump({k: rect[k] for k in RECT_KEYS}, f)
    print(f"  {dst.name}: {len(images)} images, {len(masks)} masks, "
          f"y[{rect['y0']}:{rect['y1']}] x[{rect['x0']}:{rect['x1']}]")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="The directory that holds the clips.")
    ap.add_argument("--rects", required=True, help="The rectangle of each clip, crop_rects.json.")
    ap.add_argument("--clips", nargs="+", required=True, help="Clip directories under --input-dir.")
    ap.add_argument("--overwrite", action="store_true", help="Replace a cut clip an earlier run left.")
    args = ap.parse_args()

    # Read the rectangles and find the clips before any image is read.
    rects = load_rects(args.rects)
    root = Path(args.input_dir)
    missing = [c for c in args.clips if not (root / c / "frame_manifest.json").is_file()]
    if missing:
        raise SystemExit(f"no clip (a directory with frame_manifest.json) at: {', '.join(missing)}")

    # Cut every clip, even after one fails, and list the failures at the end.
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
