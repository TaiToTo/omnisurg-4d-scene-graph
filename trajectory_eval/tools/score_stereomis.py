"""Score the camera trajectory of each usable StereoMIS clip, for the controls or for the depth stages.

`--controls` scores the static floor, the line floor and the stereo ceiling
of `pose_controls`. `--methods` scores the trajectories the depth stages
wrote into each clip under `--clip-root`. `--shuffled` also scores each
method's poses in a shuffled order, which keeps the trajectory's shape and
loses its order. `--masked` names the conditions `<method>_masked`, for clips
extracted with `--mask-instruments`. Every row is made by
`trajectory_eval.score.score`, so it records `trajectory_code_sha`. Before
anything is scored, `check_clip_root` refuses clips the rows would misname or
score against the wrong frames.

Usage:
    python -m trajectory_eval.tools.score_stereomis --root /path/to/StereoMIS --depth-root /path/to/StereoMIS_depth \\
        --controls --out controls.json
    python -m trajectory_eval.tools.score_stereomis --root /path/to/StereoMIS --depth-root /path/to/StereoMIS_depth \\
        --methods da3 pi3x --clip-root /path/to/clips [--shuffled] [--masked] --out methods.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

import pipeline.depth
import pipeline.pi3x
import surgical_core.stereomis as stereomis
import trajectory_eval.tools.pose_controls as pose_controls
from trajectory_eval.pose_metrics import centers_from_cam_to_world
from trajectory_eval.score import method_trajectory, score

# The bundle each depth stage writes into a clip, as the stage names it.
METHOD_BUNDLES = {"da3": pipeline.depth.BUNDLE, "pi3x": pipeline.pi3x.BUNDLE}
# The file in which the extraction stage lists a clip's frames.
MANIFEST = "frame_manifest.json"
# The conditions `--controls` scores, in the order each clip's rows are written.
CONTROLS = ("floor_static", "floor_line", "ceil_stereo")
# The seed of the shuffled order. One generator is made for each clip and method, so every clip of one length
# is shuffled the same way.
SHUFFLE_SEED = 0


def usable_clips(root: Path, depth_root: Path, seq: str) -> list[dict]:
    """Return the clips of a sequence that `stereomis.clips` marks usable."""
    return [c for c in stereomis.clips(root, depth_root, seq) if c["usable"]]


def score_controls(root: Path, depth_root: Path, sequences: list[str]) -> list[dict]:
    """Return the rows of the three controls on every usable clip.

    A ceiling row also records `vo_fail`, the number of the odometry's steps that failed.

    Raises:
        ValueError: the ceiling returns no trajectory for a usable clip.
    """
    rows = []
    for seq in sequences:
        clips = usable_clips(root, depth_root, seq)
        if not clips:
            print(f"[{seq}] no usable clip", flush=True)
            continue
        ceiling = pose_controls.run_sequence(root, depth_root, seq)
        lacking = [c["name"] for c in clips if c["name"] not in ceiling]
        if lacking:
            raise ValueError(f"{seq}: the ceiling returned no trajectory for {lacking}")
        for c in clips:
            gt_T = stereomis.gt_pose(root, seq, c["frames"])
            c_est, R_est, n_fail = ceiling[c["name"]]
            trajectories = {"floor_static": pose_controls.static_floor(len(c["frames"])),
                            "floor_line": pose_controls.line_floor(centers_from_cam_to_world(gt_T), c["times"]),
                            "ceil_stereo": (c_est, R_est)}
            for cond in CONTROLS:
                row = score(c, cond, gt_T, *trajectories[cond])
                if cond == "ceil_stereo":
                    row["vo_fail"] = n_fail
                rows.append(row)
        print(f"[{seq}] {len(clips)} clips", flush=True)
    return rows


def check_clip_root(clips: list[dict], clip_root: Path, methods: list[str], masked: bool) -> None:
    """Refuse clips whose rows would be misnamed or scored against the wrong frames.

    Raises:
        FileNotFoundError: a clip lacks a method's bundle, or its manifest. Every clip that lacks a bundle is named.
        ValueError: a clip's manifest lists other frames than `stereomis.clips` gives it, or the painted images
            contradict `masked`.
    """
    # Refuse the clips that lack a method's bundle, naming every one.
    lacking = [f"{c['name']}:{m}" for c in clips for m in methods
               if not (clip_root / c["name"] / METHOD_BUNDLES[m]).is_file()]
    if lacking:
        raise FileNotFoundError(f"{len(lacking)} bundle(s) missing under {clip_root}: {lacking}")
    manifests = [json.loads((clip_root / c["name"] / MANIFEST).read_text()) for c in clips]

    # Refuse a clip cut from other frames. Its bundle holds as many poses as the truth, so only its manifest shows it.
    moved = [c["name"] for c, m in zip(clips, manifests)
             if [f["native_frame"] for f in m["frames"]] != list(c["frames"])]
    if moved:
        raise ValueError(f"{len(moved)} clip(s) under {clip_root} hold other frames than the ground truth is read "
                         f"at: {moved}")

    # Refuse `masked` when the painted images contradict it. The clips are counted together, since a masked clip
    # with no tissue mask near its frames paints no image. A clip extracted without masks may not record the count.
    painted = sum(m.get("instruments_masked", 0) for m in manifests)
    if masked and not painted:
        raise ValueError(f"--masked is given, but no clip under {clip_root} has a painted image")
    if painted and not masked:
        raise ValueError(f"{painted} images under {clip_root} are painted outside their tissue mask; pass --masked")


def score_methods(root: Path, depth_root: Path, sequences: list[str], clip_root: Path, methods: list[str],
                  shuffled: bool = False, masked: bool = False) -> list[dict]:
    """Return the rows of each method on every usable clip, and of its shuffled poses when `shuffled` is set.

    A method's condition is its name, followed by `_masked` when `masked` is set. The shuffled one adds `_shuf`.

    Raises:
        FileNotFoundError: `check_clip_root` finds a bundle or a manifest missing, before anything is scored.
        ValueError: `check_clip_root` refuses a clip's frames or the painted images, before anything is scored.
    """
    clips = [c for seq in sequences for c in usable_clips(root, depth_root, seq)]
    check_clip_root(clips, clip_root, methods, masked)
    rows = []
    for c in clips:
        gt_T = stereomis.gt_pose(root, c["seq"], c["frames"])
        for m in methods:
            cond = f"{m}_masked" if masked else m
            est_c, est_R = method_trajectory(clip_root / c["name"] / METHOD_BUNDLES[m], len(c["frames"]))
            rows.append(score(c, cond, gt_T, est_c, est_R))
            if shuffled:
                order = np.random.default_rng(SHUFFLE_SEED).permutation(len(est_c))
                rows.append(score(c, f"{cond}_shuf", gt_T, est_c[order], est_R[order]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="The StereoMIS directory, which holds one directory per sequence.")
    ap.add_argument("--depth-root", required=True, help="The depth export, which holds <sequence>/stats.npy.")
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--controls", action="store_true", help="Score the two floors and the ceiling.")
    what.add_argument("--methods", nargs="+", choices=sorted(METHOD_BUNDLES), help="Score these depth stages.")
    ap.add_argument("--clip-root", help="The clips the depth stages wrote into; --methods needs it.")
    ap.add_argument("--shuffled", action="store_true", help="Also score each method's poses in a shuffled order.")
    ap.add_argument("--masked", action="store_true",
                    help="The clips were extracted with --mask-instruments; name each condition <method>_masked.")
    ap.add_argument("--sequences", nargs="+", default=list(stereomis.SEQUENCES), choices=stereomis.SEQUENCES,
                    help="The sequences to score; every sequence by default.")
    ap.add_argument("--out", required=True, help="The JSON file the rows are written to.")
    args = ap.parse_args()
    if args.methods and not args.clip_root:
        ap.error("--methods needs --clip-root")
    if args.controls and (args.clip_root or args.shuffled or args.masked):
        ap.error("--clip-root, --shuffled and --masked apply to --methods only")
    # A sequence or a method named twice scores its clips twice, which `scores.read_rows` refuses only later.
    for flag, names in (("--sequences", args.sequences), ("--methods", args.methods or [])):
        twice = sorted({n for n in names if names.count(n) > 1})
        if twice:
            ap.error(f"{flag} names {twice} more than once")

    root, depth_root = Path(args.root), Path(args.depth_root)
    try:
        rows = (score_controls(root, depth_root, args.sequences) if args.controls else
                score_methods(root, depth_root, args.sequences, Path(args.clip_root), args.methods, args.shuffled,
                              args.masked))
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"{type(e).__name__}: {e}")
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
