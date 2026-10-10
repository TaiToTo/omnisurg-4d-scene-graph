"""Extract the StereoMIS clips into the clip directories the depth stages read.

The clips are those `surgical_core.stereomis.clips` marks usable. Each clip
gets the rectified left view of its frames at half resolution, as
`input_images/NNNNNN.png` numbered from 0, and a `frame_manifest.json` that
gives each image's position in the video, its ground truth row and its time.
With `--mask-instruments`, each image is painted black outside the tissue
mask nearest its frame; a frame with no mask within two frames keeps its
image. A sequence's frames are decoded in one pass. A clip that already
exists is refused before anything is decoded.

Usage:
    python -m pipeline.extract_stereomis --root /path/to/StereoMIS --depth-root /path/to/StereoMIS_depth \\
        --out /path/to/clips [--sequences P1 P2_0 ...] [--mask-instruments]
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

import surgical_core.stereomis as stereomis


def manifest(clip: dict, fps: float, n_masked: int) -> dict:
    """Return the contents of a clip's `frame_manifest.json`.

    The keys and their order are the workbench's, so that a clip extracted here equals one extracted there.

    Args:
        clip: one of the dicts `stereomis.clips` returns.
        fps: the video's frame rate, as `stereomis.video_info` gives it.
        n_masked: how many of the clip's images were painted outside their tissue mask.
    """
    seq = clip["seq"]
    return dict(dataset="stereomis", sequence=seq, clip_kind="fixed_grid", fps_native=fps, stride=clip["stride"],
                out_size=list(stereomis.HALF_SIZE), n_frames=len(clip["frames"]),
                gt_row_offset=stereomis.GT_ROW_OFFSET[seq], instruments_masked=n_masked,
                span_mm=clip["span_mm"], stratum=clip["stratum"],
                frames=[dict(seq_idx=i, native_frame=f, gt_row=stereomis.gt_row(seq, f), time_sec=round(t, 4))
                        for i, (f, t) in enumerate(zip(clip["frames"], clip["times"]))])


def refuse_existing(clips: list[dict], out: Path) -> None:
    """Refuse a clip whose directory, or whose `<clip>.partial`, exists under `out`.

    Raises:
        FileExistsError: a clip's directory, or its `<clip>.partial`, exists. A clip is not replaced, because a
            later stage may have written into it.
    """
    for c in clips:
        for d in (out / c["name"], out / f"{c['name']}.partial"):
            if d.exists():
                raise FileExistsError(f"{d} exists; remove it to extract {c['name']} again")


def extract_sequence(root: Path, depth_root: Path, seq: str, out: Path, mask_instruments: bool) -> list[Path]:
    """Write every usable clip of a sequence under `out`, and return the clip directories.

    Each clip is written as `<clip>.partial` and renamed when its manifest is written.

    Args:
        root: the StereoMIS directory, which holds one directory per sequence.
        depth_root: the depth export, which holds `<sequence>/stats.npy`.
        seq: the sequence.
        out: the directory the clips are written to.
        mask_instruments: whether each image is painted black outside the tissue mask nearest its frame.

    Raises:
        FileExistsError: a clip's directory, or its `<clip>.partial`, exists already.
        ValueError: two clips, or two places of one clip, hold the same frame.
        RuntimeError: an image cannot be written.
    """
    # Find the clips, and refuse before decoding if a clip is there already.
    clips = [c for c in stereomis.clips(root, depth_root, seq) if c["usable"]]
    refuse_existing(clips, out)
    if not clips:
        return []

    # Map each frame to its clip and place. A frame held twice would be written to one place only.
    owner = {f: (c, i) for c in clips for i, f in enumerate(c["frames"])}
    if len(owner) != sum(len(c["frames"]) for c in clips):
        raise ValueError(f"{seq}: two clips, or two places of one clip, hold the same frame")

    # Decode the sequence's frames in one pass, and write each left view into its clip as it arrives.
    n_masked = {c["name"]: 0 for c in clips}
    try:
        for c in clips:
            (out / f"{c['name']}.partial" / "input_images").mkdir(parents=True)
        for f, left, _ in stereomis.iter_frame_set(root, seq, sorted(owner)):
            c, i = owner[f]
            if mask_instruments:
                m = stereomis.load_mask_nearest(root, seq, f)
                if m is not None:
                    left = np.where(m[:, :, None], left, 0)
                    n_masked[c["name"]] += 1
            path = out / f"{c['name']}.partial" / "input_images" / ("%06d.png" % i)
            if not cv2.imwrite(str(path), left[:, :, ::-1]):
                raise RuntimeError(f"cannot write {path}")

        # Write each clip's manifest, then move the clip into place.
        fps = stereomis.video_info(root, seq)["fps"]
        for c in clips:
            with open(out / f"{c['name']}.partial" / "frame_manifest.json", "w") as fh:
                json.dump(manifest(c, fps, n_masked[c["name"]]), fh, indent=1)
    except BaseException:
        for c in clips:
            shutil.rmtree(out / f"{c['name']}.partial", ignore_errors=True)
        raise
    for c in clips:
        (out / f"{c['name']}.partial").rename(out / c["name"])
    return [out / c["name"] for c in clips]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="The StereoMIS directory, which holds one directory per sequence.")
    ap.add_argument("--depth-root", required=True, help="The depth export, which holds <sequence>/stats.npy.")
    ap.add_argument("--out", required=True, help="The directory the clips are written to.")
    ap.add_argument("--sequences", nargs="+", default=list(stereomis.SEQUENCES), choices=stereomis.SEQUENCES,
                    help="The sequences to extract; every sequence by default.")
    ap.add_argument("--mask-instruments", action="store_true",
                    help="Paint each image black outside the tissue mask nearest its frame.")
    args = ap.parse_args()

    root, depth_root, out = Path(args.root), Path(args.depth_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Refuse a clip that exists before any sequence is decoded.
    for seq in args.sequences:
        refuse_existing([c for c in stereomis.clips(root, depth_root, seq) if c["usable"]], out)
    total = 0
    for seq in args.sequences:
        made = extract_sequence(root, depth_root, seq, out, args.mask_instruments)
        total += len(made)
        print(f"{seq}: {len(made)} clips", flush=True)
    print(f"{total} clips in {out}")


if __name__ == "__main__":
    main()
