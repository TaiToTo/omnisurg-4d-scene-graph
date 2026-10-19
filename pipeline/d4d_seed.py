"""Cut each D4D side's frame into seed regions, the regions the front/behind relation is scored on.

The stage reads each side's file of `pipeline.d4d_depth`, and the frame it names. It builds the `normal_edge`
input from the depth, with the edges at full darkness and the normals unsmoothed. The tracking stage's
segmenter then cuts the input at 24 points per side. The stage writes the regions, `<side>.npy`, numbered
0..K-1 and -1 for none, and their record, `<side>.json`, which holds the depth file's hash and the settings.
It skips a side whose two files exist once the record holds the depth file's hash and this run's settings, and
refuses any other record.

Usage:
    python -m pipeline.d4d_seed --depth-dir /path/to/d4d_depth --out /path/to/d4d_seed \\
        --sam-ckpt sam_vit_h_4b8939.pth [--keep-edge-ring] [--shard 0 --n-shards 1] [--gpu N]
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from pipeline.track import SeedSegmenter
from recon3d_wrapper import select_gpu
from surgical_core.geometry.render import sam_input_image

# The seed's input and the segmenter's grid, as the tracking stage's `op_edge_center` condition cuts its seed.
SAM_INPUT = "normal_edge"
EDGE_GAIN = 1.0
SMOOTH = False
POINTS_PER_SIDE = 24


def file_sha256(path: Path) -> str:
    """Return the sha256 hex digest of the file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def settings(mask_ring: bool, sam_ckpt: str) -> dict:
    """Return the settings a run with these arguments writes to a side's record."""
    return dict(sam_input=SAM_INPUT, points_per_side=POINTS_PER_SIDE, edge_gain=EDGE_GAIN, smooth=SMOOTH,
                edge_ring_masked=mask_ring, sam_ckpt=Path(sam_ckpt).name)


def check_done(record_path: Path, depth_path: Path, expected: dict) -> None:
    """Refuse a side's record that another depth file or other settings made.

    The regions hold only their shape, so the record alone tells which depth and which settings made them.

    Raises:
        ValueError: the record's `depth_sha256` is not the depth file's hash, or a field of `expected` differs
            from the record's, or is missing.
    """
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if record.get("depth_sha256") != file_sha256(depth_path):
        raise ValueError(f"{record_path.name}: made from another depth than {depth_path.name}; remove the side's "
                         "two files to make them again")
    differ = [f"{k} is {record.get(k)!r}, where this run makes {v!r}" for k, v in expected.items()
              if record.get(k) != v]
    if differ:
        raise ValueError(f"{record_path.name}: {'; '.join(differ)}; remove the side's two files, or run with the "
                         "record's settings")


def frame_rgb(path: str, shape: tuple[int, int]) -> np.ndarray:
    """Return the frame as (H, W, 3) uint8 RGB at `shape`, resized with area averaging.

    Raises:
        FileNotFoundError: the frame cannot be read.
    """
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"cannot read the frame {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != shape:
        rgb = cv2.resize(rgb, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return rgb


def run_side(depth_path: Path, out: Path, segmenter, mask_ring: bool, sam_ckpt: str) -> int:
    """Cut one side's frame into regions, write the regions and their record, and return the number of regions.

    The regions are written before the record, each under a temporary name and then renamed. A side is done
    only when both files exist, so a run stopped midway leaves no side that a later run would skip.
    """
    # Build the seed input from the side's depth and frame.
    with np.load(depth_path, allow_pickle=False) as z:
        depth, K = z["depth"].astype(np.float32), z["intrinsics"].astype(np.float32)
        image_path = str(z["image_path"])
        fields = {k: int(z[k]) for k in ("window", "process_res", "frame_index")}
    # `normal_edge` reads no grey image; zeros of the depth's shape stand in for it.
    image = sam_input_image(SAM_INPUT, depth, K, np.zeros_like(depth), frame_rgb(image_path, depth.shape),
                            edge_gain=EDGE_GAIN, smooth=SMOOTH, mask_ring=mask_ring)

    # Cut the input into regions, and write them and then their record.
    labels = segmenter.label_map(image, POINTS_PER_SIDE).astype(np.int32)
    stem = depth_path.name[:-len(".npz")]
    tmp = out / f"{stem}.npy.partial"
    with open(tmp, "wb") as f:
        np.save(f, labels)
    os.replace(tmp, out / f"{stem}.npy")
    record = dict(depth_file=depth_path.name, depth_sha256=file_sha256(depth_path), **fields,
                  **settings(mask_ring, sam_ckpt), n_regions=int(labels.max()) + 1)
    tmp = out / f"{stem}.json.partial"
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, out / f"{stem}.json")
    return record["n_regions"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--depth-dir", required=True, help="the directory pipeline.d4d_depth wrote")
    ap.add_argument("--out", required=True, help="the directory the regions and records go to")
    ap.add_argument("--sam-ckpt", required=True, help="the SAM ViT-H weights, sam_vit_h_4b8939.pth")
    ap.add_argument("--keep-edge-ring", action="store_true",
                    help="keep the edges' ring along the image border and around invalid depth, as the tracking "
                         "stage's option of that name does")
    ap.add_argument("--shard", type=int, default=0, help="run the sides whose position modulo --n-shards is this")
    ap.add_argument("--n-shards", type=int, default=1, help="the number of processes the sides are split over")
    ap.add_argument("--device", default="auto", help="auto, cuda or cpu; auto takes CUDA when there is one")
    ap.add_argument("--gpu", type=int, default=None, help="the index of the GPU to use")
    args = ap.parse_args(argv)
    if not 0 <= args.shard < args.n_shards:
        ap.error(f"--shard {args.shard} is not in 0..{args.n_shards - 1}")
    mask_ring = not args.keep_edge_ring
    expected = settings(mask_ring, args.sam_ckpt)

    # List this shard's sides, and skip those already made, once their records are checked.
    depth_files = sorted(Path(args.depth_dir).glob("*.npz"))[args.shard::args.n_shards]
    if not depth_files:
        raise SystemExit(f"no side's depth (*.npz) in {args.depth_dir} for shard {args.shard}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = []
    for path in depth_files:
        stem = path.name[:-len(".npz")]
        if (out / f"{stem}.npy").is_file() and (out / f"{stem}.json").is_file():
            check_done(out / f"{stem}.json", path, expected)
        else:
            todo.append(path)
    print(f"shard {args.shard}/{args.n_shards}: {len(depth_files)} sides, {len(todo)} to run")
    if not todo:
        return

    select_gpu(args.gpu)
    segmenter = SeedSegmenter(args.sam_ckpt, args.device)
    for i, path in enumerate(todo, 1):
        n = run_side(path, out, segmenter, mask_ring, args.sam_ckpt)
        print(f"  {i}/{len(todo)} {path.name}: {n} regions", flush=True)


if __name__ == "__main__":
    main()
