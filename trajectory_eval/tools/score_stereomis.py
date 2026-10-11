"""Score the camera trajectories of the StereoMIS clips: the controls', or the depth stages'.

The clips are those the clip list marks usable, at the ground truth rows and
times it gives. `--controls` scores the controls of `pose_controls`: the static
floor, the line floor and the stereo ceiling, which reads the tissue masks the
list gives. `--methods` scores the trajectories the depth stages wrote into
each clip under `--clip-root`, and `--shuffled` each method's poses shuffled.
Every row records `trajectory_code_sha`, and the list's hash as
`clip_list_sha`. A clip without a method's bundle is refused before anything
is scored.

Usage:
    python -m trajectory_eval.tools.score_stereomis --root /path/to/StereoMIS --clips stereomis_meta/clips.json \\
        --controls --out controls.json
    python -m trajectory_eval.tools.score_stereomis --root /path/to/StereoMIS --clips stereomis_meta/clips.json \\
        --methods da3 pi3x --clip-root /path/to/clips [--shuffled] [--suffix _masked] --out methods.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import pipeline.depth
import pipeline.pi3x
import surgical_core.stereomis as stereomis
import trajectory_eval.pose_controls as pose_controls
from trajectory_eval.pose_metrics import centers_from_cam_to_world
from trajectory_eval.score import method_trajectory, score
from trajectory_eval.tools.scores import LIST_KEY

# The bundle each depth stage writes into a clip, as the stage names it.
METHOD_BUNDLES = {"da3": pipeline.depth.BUNDLE, "pi3x": pipeline.pi3x.BUNDLE}
# The conditions `--controls` scores, in the order each clip's rows are written.
CONTROLS = ("floor_static", "floor_line", "ceil_stereo")


def usable_clips(clip_list, sequences: list[str]) -> list[dict]:
    """Return the clips of the sequences that the clip list marks usable, sequence by sequence."""
    return [c for seq in sequences for c in clip_list if c["seq"] == seq and c["usable"]]


def ceiling(root: Path, clips: list[dict], seq: str) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """Run the stereo ceiling on clips of one sequence, decoding the sequence once.

    Each frame takes the tissue mask the clip list gives it, or none.

    Args:
        root: the StereoMIS directory, which holds one directory per sequence.
        clips: clips of the sequence, from the clip list.
        seq: the sequence.

    Returns:
        For each clip's name, its camera centres, its camera-to-world rotations and how many steps failed.

    Raises:
        ValueError: a clip is empty or its frames do not increase, or a clip does not end before the next
            one starts.
    """
    # Map each frame to its clip and its mask.
    if not clips:
        return {}
    owner = {f: c["name"] for c in clips for f in c["frames"]}
    mask_of = {f: m for c in clips for f, m in zip(c["frames"], c["masks"])}

    # Refuse clips one decode cannot feed in turn. The decode yields each frame once and in increasing order, so
    # each clip's frames must increase, and each clip must end before the next one starts.
    runs = sorted(list(c["frames"]) for c in clips)
    if (not all(len(f) and np.all(np.diff(f) > 0) for f in runs)
            or any(a[-1] >= b[0] for a, b in zip(runs, runs[1:]))):
        raise ValueError(f"{seq}: a clip is empty, its frames do not increase, or it does not end before the "
                         "next clip starts, so one decode cannot feed the clips in turn")

    # Decode the sequence once, and chain each clip's poses from its first frame.
    c = stereomis.calib(root, seq)
    vo, cur, out = pose_controls.StereoVO(c.K_half, c.baseline_mm), None, {}
    for f, left, right in stereomis.iter_frame_set(root, seq, sorted(owner)):
        if owner[f] != cur:
            if cur is not None:
                out[cur] = vo.result() + (vo.n_fail,)
            vo.reset()
            cur = owner[f]
        vo.add(left, right, None if mask_of[f] is None else stereomis.load_mask(root, seq, mask_of[f]))
    out[cur] = vo.result() + (vo.n_fail,)
    return out


def score_controls(root: Path, clip_list, sequences: list[str]) -> list[dict]:
    """Return the rows of the three controls on every usable clip; a ceiling row also records `vo_fail`.

    Raises:
        ValueError: the ceiling returns no trajectory for a usable clip.
    """
    rows = []
    for seq in sequences:
        clips = usable_clips(clip_list, [seq])
        if not clips:
            print(f"[{seq}] no usable clip", flush=True)
            continue
        trajectories = ceiling(root, clips, seq)
        lacking = [c["name"] for c in clips if c["name"] not in trajectories]
        if lacking:
            raise ValueError(f"{seq}: the ceiling returned no trajectory for {lacking}")
        for c in clips:
            gt_T = stereomis.gt_pose(root, seq, stereomis.gt_rows(c))
            c_est, R_est, n_fail = trajectories[c["name"]]
            by_cond = {"floor_static": pose_controls.static_floor(len(c["frames"])),
                       "floor_line": pose_controls.line_floor(centers_from_cam_to_world(gt_T), c["times"]),
                       "ceil_stereo": (c_est, R_est)}
            for cond in CONTROLS:
                row = score(c, cond, gt_T, *by_cond[cond])
                if cond == "ceil_stereo":
                    row["vo_fail"] = n_fail
                rows.append(row)
        print(f"[{seq}] {len(clips)} clips", flush=True)
    return rows


def score_methods(root: Path, clip_list, sequences: list[str], clip_root: Path, methods: list[str],
                  shuffled: bool = False, suffix: str = "") -> list[dict]:
    """Return the rows of each method on every usable clip, and of its shuffled poses when `shuffled` is set.

    A method's condition is its name followed by `suffix`, and the shuffled one adds `_shuf`.

    Raises:
        FileNotFoundError: a clip lacks a method's bundle. Every such clip is named, before anything is scored.
    """
    clips = usable_clips(clip_list, sequences)
    lacking = [f"{c['name']}:{m}" for c in clips for m in methods
               if not (clip_root / c["name"] / METHOD_BUNDLES[m]).is_file()]
    if lacking:
        raise FileNotFoundError(f"{len(lacking)} bundle(s) missing under {clip_root}: {lacking}")
    rows = []
    for c in clips:
        gt_T = stereomis.gt_pose(root, c["seq"], stereomis.gt_rows(c))
        for m in methods:
            est_c, est_R = method_trajectory(clip_root / c["name"] / METHOD_BUNDLES[m], len(c["frames"]))
            rows.append(score(c, f"{m}{suffix}", gt_T, est_c, est_R))
            if shuffled:
                rows.append(score(c, f"{m}{suffix}_shuf", gt_T, *pose_controls.shuffled(est_c, est_R)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="The StereoMIS directory, which holds one directory per sequence.")
    ap.add_argument("--clips", required=True, help="The clip list, stereomis_meta/clips.json.")
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--controls", action="store_true", help="Score the two floors and the ceiling.")
    what.add_argument("--methods", nargs="+", choices=sorted(METHOD_BUNDLES), help="Score these depth stages.")
    ap.add_argument("--clip-root", help="The clips the depth stages wrote into; --methods needs it.")
    ap.add_argument("--shuffled", action="store_true", help="Also score each method's poses in a shuffled order.")
    ap.add_argument("--suffix", default="", help="Appended to each method's name to name its condition.")
    ap.add_argument("--sequences", nargs="+", default=list(stereomis.SEQUENCES), choices=stereomis.SEQUENCES,
                    help="The sequences to score; every sequence by default.")
    ap.add_argument("--out", required=True, help="The JSON file the rows are written to.")
    args = ap.parse_args()
    if args.methods and not args.clip_root:
        ap.error("--methods needs --clip-root")
    if args.controls and (args.clip_root or args.shuffled or args.suffix):
        ap.error("--clip-root, --shuffled and --suffix apply to --methods only")

    root, list_path = Path(args.root), Path(args.clips)
    try:
        clip_list = stereomis.read_clip_list(list_path)
        rows = (score_controls(root, clip_list, args.sequences) if args.controls else
                score_methods(root, clip_list, args.sequences, Path(args.clip_root), args.methods, args.shuffled,
                              args.suffix))
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"{type(e).__name__}: {e}")
    list_sha = hashlib.sha256(list_path.read_bytes()).hexdigest()
    for r in rows:
        r[LIST_KEY] = list_sha
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"{len(rows)} rows -> {out}")
    for cond in dict.fromkeys(r["cond"] for r in rows):
        v = np.array([r["ate_rel"] for r in rows if r["cond"] == cond])
        print(f"  {cond:16s} n={len(v):3d}  ate_rel median {np.median(v):.4f}  mean {v.mean():.4f}")


if __name__ == "__main__":
    main()
