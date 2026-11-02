"""Compute the stereo depth of each usable StereoMIS clip's frames, the reference depth of the geometry results.

Each sequence is decoded once, and each frame's depth is taken by the matcher of `pose_controls` (`sgbm`,
`sgbm_depth`), in mm at half resolution, 0 where unknown. Before a clip is written, each decoded left view is
compared with the clip's own image under `--clip-root`, so that the depth belongs to the frames the models saw.
Writes `<out>/<clip>.npz` with `depth_dmm` (uint16, tenths of a mm) and `K`, the rectified intrinsics at half
resolution. A clip whose file exists is skipped.

Usage:
    python paper/ipcai2027/geometry/stereo_depth.py --root /path/to/StereoMIS --clips stereomis_meta/clips.json \\
        --clip-root /path/to/clips --out /path/to/stereo_depth [--sequences P2_6 P2_7] [--jobs 4]
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

import surgical_core.stereomis as stereomis
import trajectory_eval.pose_controls as pose_controls

# The largest mean absolute difference, in grey levels, between a decoded view and the clip's image of it.
# Another FFmpeg build may convert colours a little differently; a frame off by one differs by far more.
SAME_IMAGE_TOL = 3.0


def sequence_depth(root: Path, seq: str, clips: list[dict], clip_root: Path, out: Path) -> list[str]:
    """Write the stereo depth of the clips of one sequence, decoding the sequence once; return a line per clip.

    Raises:
        FileNotFoundError: a clip's image is missing.
        ValueError: a decoded left view differs from the clip's image by more than `SAME_IMAGE_TOL`. Nothing of
            that clip is written.
    """
    # Map each frame to its clip and its place in the clip.
    c = stereomis.calib(root, seq)
    bf = float(c.K_half[0, 0] * c.baseline_mm)
    owner = {f: (clip["name"], i) for clip in clips for i, f in enumerate(clip["frames"])}
    depth = {clip["name"]: np.zeros((len(clip["frames"]), *stereomis.HALF_SIZE[::-1]), np.uint16) for clip in clips}
    worst = dict.fromkeys(depth, 0.0)

    # Decode the sequence once; compare each left view with the clip's image, and match the pair.
    matcher = pose_controls.sgbm()
    for f, left, right in stereomis.iter_frame_set(root, seq, sorted(owner)):
        name, i = owner[f]
        path = clip_root / name / "input_images" / f"{i:06d}.png"
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"no image {path}")
        worst[name] = max(worst[name], float(np.abs(image[:, :, ::-1].astype(float) - left).mean()))
        grey_l, grey_r = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY), cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
        depth[name][i] = np.round(pose_controls.sgbm_depth(matcher, grey_l, grey_r, bf) * 10).astype(np.uint16)

    # Write each clip whose views are its images.
    lines = []
    for name, d in depth.items():
        if worst[name] > SAME_IMAGE_TOL:
            raise ValueError(f"{name}: a decoded view differs from the clip's image by {worst[name]:.2f} grey levels")
        np.savez_compressed(out / f"{name}.npz", depth_dmm=d, K=c.K_half)
        lines.append(f"{name}: depth on {(d > 0).mean():.3f} of the pixels, median {np.median(d[d > 0]) / 10:.1f} mm")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="The StereoMIS directory, which holds one directory per sequence.")
    ap.add_argument("--clips", required=True, help="The clip list, stereomis_meta/clips.json.")
    ap.add_argument("--clip-root", required=True, help="The extracted clips, whose images are compared.")
    ap.add_argument("--out", required=True, help="The directory the depth files are written to.")
    ap.add_argument("--sequences", nargs="+", default=list(stereomis.SEQUENCES), choices=stereomis.SEQUENCES,
                    help="The sequences whose clips are computed; every sequence by default.")
    ap.add_argument("--jobs", type=int, default=1, help="Sequences decoded at once.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = [c for c in stereomis.usable_clips(stereomis.read_clip_list(Path(args.clips)), args.sequences)
             if not (out / f"{c['name']}.npz").is_file()]
    seqs = sorted({c["seq"] for c in clips})
    print(f"{len(clips)} clips without stereo depth, over {len(seqs)} sequences", flush=True)
    with ProcessPoolExecutor(args.jobs) as ex:
        jobs = [ex.submit(sequence_depth, Path(args.root), s, [c for c in clips if c["seq"] == s],
                          Path(args.clip_root), out) for s in seqs]
        for job in jobs:
            print("\n".join(job.result()), flush=True)


if __name__ == "__main__":
    main()
