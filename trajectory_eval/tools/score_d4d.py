"""Score the camera trajectories of the D4D clips in which the optical tracker sees the camera move.

D4D's optical tracker logs the camera's pose, the truth the trajectories are scored against. `census`
measures how far the tracker's camera moves in each clip of a population. A clip is scored when the RMS
radius of its true trajectory, `span_mm`, is at least `stereomis.CLIP_MIN_SPAN_MM`. `prepare` writes each
such clip for the depth stages, as `CLIP_FRAMES` frames spread over the whole clip. `score` scores the
floors of `pose_controls` and the depth stages' poses through `trajectory_eval.score.score`, on the frames
the tracker saw. Each row adds their number, `n_tracked`, and their share, `tracked_frac`. `score` leaves
out a clip whose tracker misses too many frames or jumps, and refuses clips the rows would misname or
score against the wrong frames.

Usage:
    python -m trajectory_eval.tools.score_d4d census --root /path/to/D4D/extracted --clips d4d_meta/clips.txt \\
        --out census.json
    python -m trajectory_eval.tools.score_d4d prepare --root /path/to/D4D/extracted --census census.json \\
        --out /path/to/clips
    python -m trajectory_eval.tools.score_d4d score --root /path/to/D4D/extracted --census census.json \\
        --clip-root /path/to/clips --methods da3 pi3x --out rows.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import surgical_core.d4d as d4d
import trajectory_eval.tools.pose_controls as pose_controls
# A D4D clip takes a StereoMIS clip's number of frames, least span and strata, so that the two results compare.
from surgical_core.stereomis import CLIP_FRAMES, CLIP_MIN_SPAN_MM, stratum_of
from trajectory_eval.pose_metrics import centers_from_cam_to_world, gt_span
from trajectory_eval.score import method_trajectory, score
from trajectory_eval.tools.score_stereomis import MANIFEST, METHOD_BUNDLES

# The tracker logs translations in m; the trajectories are scored in mm.
M_TO_MM = 1000.0
# The least number and share of a clip's frames that the tracker must see for the clip to be scored. The
# frames it misses are left out of the truth and of every estimate. The census asks for the number only.
MIN_TRACKED_FRAMES = 30
MIN_TRACKED_FRAC = 0.6
# The fastest true motion between two tracked frames, in mm/s and degrees/s, of a clip that is scored. These
# are StereoMIS's 5 mm and 5 degrees a step at its 60 steps a second. They are rates because a D4D clip's
# frames are spaced by the clip's length. A tracker that finds a lost marker again can jump hundreds of mm.
GT_JUMP_MM_S = 300.0
GT_JUMP_DEG_S = 300.0


def read_population(path: Path) -> list[str]:
    """Return the clips a population file lists, one `<specimen>/<session>/<clip>` a line.

    Raises:
        ValueError: the file lists no clip, or a clip twice.
    """
    keys = path.read_text().split()
    if not keys:
        raise ValueError(f"{path} lists no clip")
    twice = sorted({k for k in keys if keys.count(k) > 1})
    if twice:
        raise ValueError(f"{path} lists {twice} more than once")
    return keys


def loader_clips(root: str, keys: list[str]) -> list:
    """Return the loader's clips that `keys` names, in the loader's order.

    Raises:
        ValueError: the loader yields no clip of a key.
    """
    wanted = set(keys)
    found = [c for c in d4d.clips(root) if d4d.clip_key(c) in wanted]
    lacking = sorted(wanted - {d4d.clip_key(c) for c in found})
    if lacking:
        raise ValueError(f"the dataset at {root} holds no clip {lacking}")
    return found


def census_record(clip, log: tuple | None) -> dict:
    """Return how far the tracker's camera moves in a clip, or the reason it is not measured.

    The poses are read at `CLIP_FRAMES` times spread evenly from the clip's first frame to its last. `span_mm`
    is the RMS radius of the tracked camera centres, and `path_mm` the length of their path. The reasons are:

    - `no_log`: the session has no tracker log;
    - `log_not_covering`: the log starts after the clip's first frame or ends before its last;
    - `few_tracked`: the tracker sees fewer than `MIN_TRACKED_FRAMES` of the times.
    """
    times = d4d.frame_times(clip)
    key = d4d.clip_key(clip)
    rec = dict(key=key, specimen=key.split("/")[0], n_frames=len(times),
               duration_s=round(float(times[-1] - times[0]), 2), tracker_prefix=None if log is None else log[2])
    if log is None or times[0] < log[0][0] or times[-1] > log[0][-1]:
        return rec | dict(span_mm=None, reason="no_log" if log is None else "log_not_covering")
    want = np.linspace(times[0], times[-1], min(len(times), CLIP_FRAMES))
    poses, tracked = d4d.tracker_poses(log[0], log[1], want)
    centres = poses[tracked][:, :3, 3] * M_TO_MM
    rec["tracked_frac"] = round(float(tracked.mean()), 3)
    if len(centres) < MIN_TRACKED_FRAMES:
        return rec | dict(span_mm=None, reason="few_tracked")
    return rec | dict(span_mm=round(gt_span(centres), 3),
                      path_mm=round(float(np.linalg.norm(np.diff(centres, axis=0), axis=1).sum()), 2))


def census(clips: list) -> list[dict]:
    """Return each clip's census record, listing each session's tracker log once."""
    logs, out = {}, []
    for clip in clips:
        session = str(clip.session.path)
        if session not in logs:
            logs[session] = d4d.tracker_log(session)
        out.append(census_record(clip, logs[session]))
    return out


def selected(records: list[dict]) -> list[str]:
    """Return the keys of the census's clips whose `span_mm` is at least `CLIP_MIN_SPAN_MM`, in its order."""
    return [r["key"] for r in records if r["span_mm"] is not None and r["span_mm"] >= CLIP_MIN_SPAN_MM]


def trajectory_clip(clip) -> dict:
    """Return a clip cut to `CLIP_FRAMES` frames spread evenly over it.

    Returns:
        `key`, `name`, `specimen`, `session` (the session's directory), `frames` (each frame's position in
        the clip), `times` (s), `images` (the rectified left images), `interval_s` and `duration_s`.
    """
    all_times = d4d.frame_times(clip)
    frames = np.unique(np.linspace(0, len(all_times) - 1, CLIP_FRAMES).round().astype(int))
    key = d4d.clip_key(clip)
    duration = float(all_times[frames[-1]] - all_times[frames[0]])
    return dict(key=key, name=key.replace("/", "__"), specimen=key.split("/")[0], session=str(clip.session.path),
                frames=[int(i) for i in frames], times=all_times[frames],
                images=[clip.left_img_paths[i] for i in frames],
                interval_s=round(duration / max(1, len(frames) - 1), 4), duration_s=round(duration, 2))


def manifest(clip: dict) -> dict:
    """Return the contents of a trajectory clip's `frame_manifest.json`."""
    return dict(dataset="d4d", clip_key=clip["key"], n_frames=len(clip["frames"]), interval_sec=clip["interval_s"],
                duration_sec=clip["duration_s"],
                frames=[dict(seq_idx=i, native_frame=n) for i, n in enumerate(clip["frames"])])


def prepare(clips: list[dict], out: Path) -> None:
    """Write each clip under `out` as `input_images/NNNNNN.png` and `frame_manifest.json`.

    A clip is written as `<name>.partial` and renamed when its manifest is written. A clip is not replaced,
    since the depth stages write into it.

    Raises:
        FileExistsError: a clip's directory, or its `<name>.partial`, exists, before anything is written.
        ValueError: an image cannot be read or written.
    """
    # Refuse every clip that exists, before anything is written.
    held = [c["name"] for c in clips if (out / c["name"]).exists() or (out / f"{c['name']}.partial").exists()]
    if held:
        raise FileExistsError(f"{len(held)} clip(s) under {out} exist already, or as .partial: {held}")

    # Write each clip's images and manifest, then move the clip into place.
    for c in clips:
        part = out / f"{c['name']}.partial"
        (part / "input_images").mkdir(parents=True)
        for i, src in enumerate(c["images"]):
            image, dst = cv2.imread(src), part / "input_images" / f"{i:06d}.png"
            if image is None:
                raise ValueError(f"{src}: not an image")
            if not cv2.imwrite(str(dst), image):
                raise ValueError(f"{dst}: not written")
        with open(part / MANIFEST, "w") as fh:
            json.dump(manifest(c), fh, indent=1)
        part.rename(out / c["name"])
        print(f"  {c['name']}  {len(c['frames'])} frames over {c['duration_s']} s", flush=True)


def check_clip_root(clips: list[dict], clip_root: Path, methods: list[str]) -> None:
    """Refuse clips whose rows would be misnamed or scored against the wrong frames.

    Raises:
        FileNotFoundError: a clip lacks a method's bundle, or its manifest. Every clip that lacks a bundle is named.
        ValueError: a clip's manifest names another clip, or lists other frames than the truth is read at.
    """
    # Refuse the clips that lack a method's bundle, naming every one.
    lacking = [f"{c['name']}:{m}" for c in clips for m in methods
               if not (clip_root / c["name"] / METHOD_BUNDLES[m]).is_file()]
    if lacking:
        raise FileNotFoundError(f"{len(lacking)} bundle(s) missing under {clip_root}: {lacking}")

    # Refuse a clip cut from another clip or from other frames. Its bundle holds as many poses as the truth, so
    # only its manifest shows it.
    moved = []
    for c in clips:
        m = json.loads((clip_root / c["name"] / MANIFEST).read_text())
        if m.get("clip_key") != c["key"] or [f["native_frame"] for f in m["frames"]] != c["frames"]:
            moved.append(c["name"])
    if moved:
        raise ValueError(f"{len(moved)} clip(s) under {clip_root} hold another clip or other frames than the truth "
                         f"is read at: {moved}")


def true_trajectory(clip: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return the tracker's camera-to-tracker pose at each of a clip's frames, in mm, and whether it is tracked.

    Raises:
        ValueError: the clip's session has no tracker log, or a frame's time is outside it.
    """
    log = d4d.tracker_log(clip["session"])
    if log is None:
        raise ValueError(f"{clip['key']}: the session has no tracker log")
    poses, tracked = d4d.tracker_poses(log[0], log[1], clip["times"])
    poses[:, :3, 3] *= M_TO_MM
    return poses, tracked


def leave_out(clip: dict, gt_T: np.ndarray, tracked: np.ndarray) -> str:
    """Return why a clip is not scored, or "".

    The reasons, the first that holds:

    - `few_tracked`: the tracker sees fewer than `MIN_TRACKED_FRAMES` frames, or fewer than `MIN_TRACKED_FRAC`;
    - `gt_jump`: between two tracked frames, the truth moves faster than `GT_JUMP_MM_S` or turns faster than
      `GT_JUMP_DEG_S`;
    - `gt_static`: the tracked camera centres' RMS radius is below `CLIP_MIN_SPAN_MM`.
    """
    n = len(tracked)
    if tracked.sum() < max(MIN_TRACKED_FRAMES, MIN_TRACKED_FRAC * n):
        return "few_tracked"
    gt_c, gt_R = centers_from_cam_to_world(gt_T[tracked]), gt_T[tracked][:, :3, :3]
    # Two frames of one time would divide by zero.
    dt = np.maximum(np.diff(clip["times"][tracked]), 1e-6)
    speed = np.linalg.norm(np.diff(gt_c, axis=0), axis=1) / dt
    step = np.einsum("nji,njk->nik", gt_R[:-1], gt_R[1:])
    turn = np.degrees(np.arccos(np.clip((np.trace(step, axis1=1, axis2=2) - 1) / 2, -1, 1))) / dt
    if speed.max() > GT_JUMP_MM_S or turn.max() > GT_JUMP_DEG_S:
        return "gt_jump"
    return "gt_static" if gt_span(gt_c) < CLIP_MIN_SPAN_MM else ""


def score_clip(clip: dict, gt_T: np.ndarray, tracked: np.ndarray, clip_root: Path, methods: list[str]) -> list[dict]:
    """Return a clip's rows: the floors, then each method, on the frames the tracker saw.

    A row's times run from the clip's first frame, tracked or not.
    """
    # Keep the truth's tracked frames, and describe the clip as `score` reads it.
    gt_T = gt_T[tracked]
    gt_c = centers_from_cam_to_world(gt_T)
    span = gt_span(gt_c)
    times = (clip["times"] - clip["times"][0])[tracked]
    row_clip = dict(name=clip["name"], seq=clip["specimen"], stratum=stratum_of(span), span_mm=round(span, 3),
                    times=times)

    # Make the floors from the truth, and read each method's poses at the same frames.
    trajectories = {"floor_static": pose_controls.static_floor(len(gt_c)),
                    "floor_line": pose_controls.line_floor(gt_c, times)}
    for m in methods:
        est_c, est_R = method_trajectory(clip_root / clip["name"] / METHOD_BUNDLES[m], len(tracked))
        trajectories[m] = (est_c[tracked], est_R[tracked])

    # Score each condition, and add how many frames the tracker saw.
    seen = dict(n_tracked=int(tracked.sum()), tracked_frac=round(float(tracked.mean()), 3))
    return [score(row_clip, cond, gt_T, c, R) | seen for cond, (c, R) in trajectories.items()]


def score_clips(clips: list[dict], clip_root: Path, methods: list[str]) -> list[dict]:
    """Return the rows of every clip that is not left out, and print each clip left out with its reason.

    Raises:
        FileNotFoundError: `check_clip_root` finds a bundle or a manifest missing, before anything is scored.
        ValueError: `check_clip_root` refuses a clip, or `true_trajectory` its truth.
    """
    check_clip_root(clips, clip_root, methods)
    rows = []
    for c in clips:
        gt_T, tracked = true_trajectory(c)
        why = leave_out(c, gt_T, tracked)
        if why:
            print(f"  left out {c['name']}: {why} ({int(tracked.sum())} of {len(tracked)} frames tracked)")
            continue
        rows += score_clip(c, gt_T, tracked, clip_root, methods)
    return rows


def write_json(data: list[dict], out: Path) -> None:
    """Write `data` to `out` as indented JSON, making its directory."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(data, fh, indent=1)


def parser() -> argparse.ArgumentParser:
    """Return the parser of the three steps' arguments."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = ap.add_subparsers(dest="step", required=True)
    census_ = steps.add_parser("census", help="Measure the tracker's motion in each clip of a population.")
    census_.add_argument("--clips", required=True, help="The population file, such as d4d_meta/clips.txt.")
    census_.add_argument("--out", required=True, help="The JSON file the census is written to.")
    prepare_ = steps.add_parser("prepare", help="Write each selected clip for the depth stages.")
    prepare_.add_argument("--out", required=True, help="The directory the clips are written into.")
    score_ = steps.add_parser("score", help="Score each selected clip's floors and methods against the tracker.")
    score_.add_argument("--clip-root", required=True, help="The clips the depth stages wrote into.")
    score_.add_argument("--methods", nargs="+", required=True, choices=sorted(METHOD_BUNDLES),
                        help="The depth stages to score.")
    score_.add_argument("--out", required=True, help="The JSON file the rows are written to.")
    for p in (census_, prepare_, score_):
        p.add_argument("--root", required=True, help="The dataset's root, which holds the specimen_* directories.")
    for p in (prepare_, score_):
        p.add_argument("--census", required=True, help="The census the census step wrote.")
    return ap


def main(argv: list[str] | None = None) -> None:
    ap = parser()
    args = ap.parse_args(argv)
    if args.step == "score" and len(set(args.methods)) < len(args.methods):
        ap.error(f"--methods names a method more than once: {args.methods}")

    try:
        # Measure the census, and stop.
        if args.step == "census":
            records = census(loader_clips(args.root, read_population(Path(args.clips))))
            write_json(records, Path(args.out))
            print(f"{len(records)} clips, {len(selected(records))} with span_mm >= {CLIP_MIN_SPAN_MM} -> {args.out}")
            return

        # Cut the clips the census selects, then write them or score them.
        keys = selected(json.loads(Path(args.census).read_text()))
        clips = [trajectory_clip(c) for c in loader_clips(args.root, keys)]
        if args.step == "prepare":
            Path(args.out).mkdir(parents=True, exist_ok=True)
            prepare(clips, Path(args.out))
            return
        rows = score_clips(clips, Path(args.clip_root), args.methods)
    except (FileExistsError, FileNotFoundError, ValueError) as e:
        raise SystemExit(f"{type(e).__name__}: {e}")
    write_json(rows, Path(args.out))
    print(f"{len(rows)} rows of {len({r['clip'] for r in rows})} of {len(clips)} clips -> {args.out}")
    for cond in dict.fromkeys(r["cond"] for r in rows):
        v = np.array([r["ate_rel"] for r in rows if r["cond"] == cond])
        print(f"  {cond:14s} n={len(v):3d}  ate_rel median {np.median(v):.4f}  mean {v.mean():.4f}")


if __name__ == "__main__":
    main()
