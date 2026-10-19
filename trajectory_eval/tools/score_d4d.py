"""Cut the D4D clips for the camera trajectory result, and score their trajectories against the optical tracker.

D4D's optical tracker logs the camera's pose, the truth the trajectories are scored against. `select` cuts
each clip of a population to `CLIP_FRAMES` frames spread over the whole clip, reads the tracker at their
times, and writes the clip list: each clip, whether it is usable, and why not. `prepare` writes each usable
clip's frames for the depth stages. `score` scores the floors of `pose_controls` and the depth stages' poses
through `trajectory_eval.score.score`, on the frames the tracker saw. Every row records
`trajectory_code_sha`, and the list's hash as `clip_list_sha`. Before anything is scored, `check_clip_root`
refuses clips the rows would misname or score against the wrong frames.

Usage:
    python -m trajectory_eval.tools.score_d4d select --root /path/to/D4D/extracted \\
        --population d4d_meta/clips.txt --out d4d_meta/trajectory_clips.json
    python -m trajectory_eval.tools.score_d4d prepare --root /path/to/D4D/extracted \\
        --clips d4d_meta/trajectory_clips.json --out /path/to/clips
    python -m trajectory_eval.tools.score_d4d score --root /path/to/D4D/extracted \\
        --clips d4d_meta/trajectory_clips.json --clip-root /path/to/clips --methods da3 pi3x --out rows.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

import surgical_core.d4d as d4d
import trajectory_eval.pose_controls as pose_controls
# A D4D clip takes a StereoMIS clip's number of frames, least span and strata, so that the two results compare.
from surgical_core.stereomis import CLIP_FRAMES, CLIP_MIN_SPAN_MM, stratum_of
from trajectory_eval.pose_metrics import centers_from_cam_to_world, gt_span
from trajectory_eval.score import method_trajectory, score
from trajectory_eval.tools.score_stereomis import MANIFEST, METHOD_BUNDLES
from trajectory_eval.tools.scores import LIST_KEY

# The tracker logs translations in m; the trajectories are scored in mm.
M_TO_MM = 1000.0
# The least number and share of a clip's frames that the tracker must see for the clip to be usable. The frames
# it misses are left out of the truth and of every estimate.
MIN_TRACKED_FRAMES = 30
MIN_TRACKED_FRAC = 0.6
# The fastest true motion between two tracked frames, in mm/s and degrees/s, of a usable clip. These are
# StereoMIS's 5 mm and 5 degrees a step at its 60 steps a second. They are rates because a D4D clip's frames are
# spaced by the clip's length. A tracker that finds a lost marker again can jump hundreds of mm.
GT_JUMP_MM_S = 300.0
GT_JUMP_DEG_S = 300.0

# The reasons a clip is left out, in the order they are tested.
DROPS = ("no_log", "log_not_covering", "few_tracked", "gt_jump", "gt_static")
# The keys of a clip of the clip list, in the order the list writes them. A clip left out holds no frames.
CLIP_KEYS = ("name", "key", "specimen", "usable", "drop", "stratum", "span_mm", "n_tracked", "frames", "times",
             "tracked")


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


def trajectory_clip(clip) -> dict:
    """Return a clip cut to `CLIP_FRAMES` frames spread evenly over it; a shorter clip keeps every frame.

    Returns:
        `key`, `name`, `specimen`, `session` (the session's directory), `frames` (each frame's position in
        the clip), `times` (s) and `images` (the rectified left images).
    """
    all_times = d4d.frame_times(clip)
    frames = np.unique(np.linspace(0, len(all_times) - 1, CLIP_FRAMES).round().astype(int))
    key = d4d.clip_key(clip)
    return dict(key=key, name=key.replace("/", "__"), specimen=key.split("/")[0], session=str(clip.session.path),
                frames=[int(i) for i in frames], times=all_times[frames],
                images=[clip.left_img_paths[i] for i in frames])


def leave_out(times: np.ndarray, gt_T: np.ndarray, tracked: np.ndarray) -> str:
    """Return why a clip whose frames the tracker log covers is left out, or "".

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
    dt = np.maximum(np.diff(times[tracked]), 1e-6)
    speed = np.linalg.norm(np.diff(gt_c, axis=0), axis=1) / dt
    step = np.einsum("nji,njk->nik", gt_R[:-1], gt_R[1:])
    turn = np.degrees(np.arccos(np.clip((np.trace(step, axis1=1, axis2=2) - 1) / 2, -1, 1))) / dt
    if speed.max() > GT_JUMP_MM_S or turn.max() > GT_JUMP_DEG_S:
        return "gt_jump"
    return "gt_static" if gt_span(gt_c) < CLIP_MIN_SPAN_MM else ""


def list_entry(clip: dict, log: tuple | None) -> dict:
    """Return a trajectory clip's entry in the clip list, with the keys of `CLIP_KEYS`.

    A clip is left out by the first reason of `DROPS` that holds: `no_log`, the session has no tracker log;
    `log_not_covering`, the log starts after the clip's first frame or ends before its last; then those of
    `leave_out`. `span_mm` is the RMS radius of the tracked camera centres, None when no frame is tracked. A clip
    left out records None for its frames, times and tracked frames.
    """
    entry = dict(name=clip["name"], key=clip["key"], specimen=clip["specimen"], usable=False, drop="", stratum=None,
                 span_mm=None, n_tracked=None, frames=None, times=None, tracked=None)
    # Leave out a clip whose session has no tracker log, or whose log does not cover it.
    times = clip["times"]
    if log is None:
        return entry | dict(drop="no_log")
    log_times, log_files, _ = log
    if times[0] < log_times[0] or times[-1] > log_times[-1]:
        return entry | dict(drop="log_not_covering")

    # Read the truth at the clip's frames, and test the reasons of `leave_out`.
    gt_T, tracked = d4d.tracker_poses(log_times, log_files, times)
    gt_T[:, :3, 3] *= M_TO_MM
    drop = leave_out(times, gt_T, tracked)
    entry |= dict(drop=drop, usable=not drop, n_tracked=int(tracked.sum()))
    if tracked.any():
        span = gt_span(centers_from_cam_to_world(gt_T[tracked]))
        entry |= dict(stratum=stratum_of(span), span_mm=round(span, 3))
    if not drop:
        entry |= dict(frames=clip["frames"], times=[float(t) for t in times], tracked=[bool(k) for k in tracked])
    return entry


def select(clips: list[dict]) -> list[dict]:
    """Return each trajectory clip's entry in the clip list, listing each session's tracker log once."""
    logs, out = {}, []
    for clip in clips:
        if clip["session"] not in logs:
            logs[clip["session"]] = d4d.tracker_log(clip["session"])
        out.append(list_entry(clip, logs[clip["session"]]))
    return out


def write_clip_list(entries: list[dict], path: Path) -> None:
    """Write the clip list as JSON, one clip a line, and read it back before it replaces `path`.

    Raises:
        ValueError: `read_clip_list` refuses the list. `path` is then left as it was, and the list is at
            `<path>.partial`.
    """
    part = path.with_name(path.name + ".partial")
    part.write_text("[\n" + ",\n".join(json.dumps(e) for e in entries) + "\n]\n", encoding="utf-8")
    read_clip_list(part)
    part.rename(path)


def read_clip_list(path: Path) -> list[dict]:
    """Return the clips of a clip list, in its order.

    Raises:
        ValueError: the file is not a list of clips with the keys of `CLIP_KEYS`; a clip comes twice, or its name
            or specimen is not its key's; its reason is not one of `DROPS`, or it is usable while it records a
            reason, or the reverse; a usable clip's frames, times and tracked frames are not lists of one length
            at most `CLIP_FRAMES`, its frames or times do not increase, or `n_tracked` does not count its tracked
            frames; a clip left out records frames.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(c, dict) and tuple(c) == CLIP_KEYS for c in data):
        raise ValueError(f"{path} is not a list of clips with the keys {list(CLIP_KEYS)}, in that order")
    seen = set()
    for c in data:
        key = c["key"]
        if key in seen:
            raise ValueError(f"{path}: {key!r} comes twice")
        seen.add(key)
        if c["name"] != key.replace("/", "__") or c["specimen"] != key.split("/")[0]:
            raise ValueError(f"{path}: {key}'s name or specimen is not its key's")
        if c["drop"] not in ("",) + DROPS or c["usable"] is not (c["drop"] == ""):
            raise ValueError(f"{path}: {key} records usable {c['usable']!r} and the reason {c['drop']!r}; a clip is "
                             f"usable, true, exactly when it records no reason, and a reason is one of {DROPS}")
        lists = [c[k] for k in ("frames", "times", "tracked")]
        if not c["usable"]:
            if lists != [None] * 3:
                raise ValueError(f"{path}: {key} is left out but records frames")
            continue
        if not all(isinstance(v, list) and 0 < len(v) == len(c["frames"]) <= CLIP_FRAMES for v in lists):
            raise ValueError(f"{path}: {key}'s frames, times and tracked frames are not lists of one length, at "
                             f"most {CLIP_FRAMES}")
        if not (np.all(np.diff(c["frames"]) > 0) and np.all(np.diff(c["times"]) > 0)):
            raise ValueError(f"{path}: {key}'s frames or times do not increase")
        if c["n_tracked"] != sum(c["tracked"]):
            raise ValueError(f"{path}: {key} records n_tracked {c['n_tracked']}, but {sum(c['tracked'])} tracked "
                             "frames")
    return data


def usable_clips(clip_list: list[dict]) -> list[dict]:
    """Return the clips the clip list marks usable, in its order.

    Raises:
        ValueError: the list marks no clip usable, so there is nothing to write or score.
    """
    clips = [c for c in clip_list if c["usable"]]
    if not clips:
        raise ValueError("the clip list marks no clip usable")
    return clips


def prepare(root: str, clip_list: list[dict], out: Path) -> None:
    """Write each usable clip under `out` as `input_images/NNNNNN.png` and `frame_manifest.json`.

    A clip is written as `<name>.partial` and renamed when its manifest is written. A clip is not replaced,
    since the depth stages write into it.

    Raises:
        FileExistsError: a clip's directory, or its `<name>.partial`, exists, before anything is written.
        ValueError: the list marks no clip usable; the dataset gives a clip's frames other times than the list,
            before anything is written; or an image cannot be read or written.
    """
    # Refuse every clip that exists, and every clip whose frames the dataset times otherwise, before anything is
    # written.
    clips = usable_clips(clip_list)
    held = [c["name"] for c in clips if (out / c["name"]).exists() or (out / f"{c['name']}.partial").exists()]
    if held:
        raise FileExistsError(f"{len(held)} clip(s) under {out} exist already, or as .partial: {held}")
    by_key = {d4d.clip_key(c): c for c in loader_clips(root, [c["key"] for c in clips])}
    images = {c["name"]: [by_key[c["key"]].left_img_paths[i] for i in c["frames"]] for c in clips}
    moved = [c["name"] for c in clips if [d4d.name_time(p) for p in images[c["name"]]] != c["times"]]
    if moved:
        raise ValueError(f"the dataset at {root} gives {len(moved)} clip(s) other frame times than the clip list: "
                         f"{moved}")

    # Write each clip's images and manifest, then move the clip into place.
    out.mkdir(parents=True, exist_ok=True)
    for c in clips:
        part = out / f"{c['name']}.partial"
        (part / "input_images").mkdir(parents=True)
        for i, src in enumerate(images[c["name"]]):
            image, dst = cv2.imread(src), part / "input_images" / f"{i:06d}.png"
            if image is None:
                raise ValueError(f"{src}: not an image")
            if not cv2.imwrite(str(dst), image):
                raise ValueError(f"{dst}: not written")
        duration = c["times"][-1] - c["times"][0]
        manifest = dict(dataset="d4d", clip_key=c["key"], n_frames=len(c["frames"]),
                        interval_sec=round(duration / max(1, len(c["frames"]) - 1), 4),
                        duration_sec=round(duration, 2),
                        frames=[dict(seq_idx=i, native_frame=n) for i, n in enumerate(c["frames"])])
        with open(part / MANIFEST, "w") as fh:
            json.dump(manifest, fh, indent=1)
        part.rename(out / c["name"])
        print(f"  {c['name']}  {len(c['frames'])} frames over {round(duration, 2)} s", flush=True)


def check_clip_root(clips: list[dict], clip_root: Path, methods: list[str]) -> None:
    """Refuse clips whose rows would be misnamed or scored against the wrong frames.

    Raises:
        FileNotFoundError: a clip lacks a method's bundle, or its manifest. Every clip that lacks a bundle is named.
        ValueError: a clip's manifest names another clip, or lists other frames than the clip list gives it.
    """
    # Refuse the clips that lack a method's bundle, naming every one.
    lacking = [f"{c['name']}:{m}" for c in clips for m in methods
               if not (clip_root / c["name"] / METHOD_BUNDLES[m]).is_file()]
    if lacking:
        raise FileNotFoundError(f"{len(lacking)} bundle(s) missing under {clip_root}: {lacking}")

    # Refuse a clip cut from another clip or from other frames. Its bundle holds as many poses as the truth, so
    # only its manifest shows it. Clips of one length have the same frames, so the manifest's key tells them apart.
    moved = []
    for c in clips:
        m = json.loads((clip_root / c["name"] / MANIFEST).read_text())
        if m.get("clip_key") != c["key"] or [f["native_frame"] for f in m["frames"]] != c["frames"]:
            moved.append(c["name"])
    if moved:
        raise ValueError(f"{len(moved)} clip(s) under {clip_root} hold another clip or other frames than the clip "
                         f"list gives: {moved}")


def true_trajectory(root: str, clip: dict) -> np.ndarray:
    """Return the tracker's camera-to-tracker pose at each of a clip's frames, in mm.

    Raises:
        ValueError: the clip's session has no tracker log, a frame's time is outside it, or the log marks other
            frames tracked than the clip list.
    """
    log = d4d.tracker_log(str(Path(root, *clip["key"].split("/")[:2])))
    if log is None:
        raise ValueError(f"{clip['key']}: the session has no tracker log")
    poses, tracked = d4d.tracker_poses(log[0], log[1], np.asarray(clip["times"]))
    if tracked.tolist() != clip["tracked"]:
        raise ValueError(f"{clip['key']}: the tracker log marks other frames tracked than the clip list")
    poses[:, :3, 3] *= M_TO_MM
    return poses


def score_clip(clip: dict, gt_T: np.ndarray, clip_root: Path, methods: list[str]) -> list[dict]:
    """Return a clip's rows: the floors, then each method, on the frames the tracker saw.

    A row's times run from the clip's first frame, tracked or not.
    """
    # Keep the truth's tracked frames, and describe the clip as `score` reads it.
    tracked = np.array(clip["tracked"])
    gt_T = gt_T[tracked]
    gt_c = centers_from_cam_to_world(gt_T)
    times = (np.asarray(clip["times"]) - clip["times"][0])[tracked]
    row_clip = dict(name=clip["name"], seq=clip["specimen"], stratum=clip["stratum"], span_mm=clip["span_mm"],
                    times=times)

    # Make the floors from the truth, and read each method's poses at the same frames.
    trajectories = {"floor_static": pose_controls.static_floor(len(gt_c)),
                    "floor_line": pose_controls.line_floor(gt_c, times)}
    for m in methods:
        est_c, est_R = method_trajectory(clip_root / clip["name"] / METHOD_BUNDLES[m], len(tracked))
        trajectories[m] = (est_c[tracked], est_R[tracked])

    # Score each condition, and add how many frames the tracker saw.
    seen = dict(n_tracked=clip["n_tracked"], tracked_frac=round(clip["n_tracked"] / len(tracked), 3))
    return [score(row_clip, cond, gt_T, c, R) | seen for cond, (c, R) in trajectories.items()]


def score_clips(root: str, clip_list: list[dict], clip_root: Path, methods: list[str]) -> list[dict]:
    """Return the rows of every usable clip.

    Raises:
        FileNotFoundError: `check_clip_root` finds a bundle or a manifest missing, before anything is scored.
        ValueError: the list marks no clip usable, `check_clip_root` refuses a clip, or `true_trajectory` its truth.
    """
    clips = usable_clips(clip_list)
    check_clip_root(clips, clip_root, methods)
    return [row for c in clips for row in score_clip(c, true_trajectory(root, c), clip_root, methods)]


def parser() -> argparse.ArgumentParser:
    """Return the parser of the three steps' arguments."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = ap.add_subparsers(dest="step", required=True)
    select_ = steps.add_parser("select", help="Cut each clip of a population and write the clip list.")
    select_.add_argument("--population", required=True, help="The population file, such as d4d_meta/clips.txt.")
    select_.add_argument("--out", required=True, help="The clip list to write, such as d4d_meta/trajectory_clips.json.")
    prepare_ = steps.add_parser("prepare", help="Write each usable clip for the depth stages.")
    prepare_.add_argument("--out", required=True, help="The directory the clips are written into.")
    score_ = steps.add_parser("score", help="Score each usable clip's floors and methods against the tracker.")
    score_.add_argument("--clip-root", required=True, help="The clips the depth stages wrote into.")
    score_.add_argument("--methods", nargs="+", required=True, choices=sorted(METHOD_BUNDLES),
                        help="The depth stages to score.")
    score_.add_argument("--out", required=True, help="The JSON file the rows are written to.")
    for p in (select_, prepare_, score_):
        p.add_argument("--root", required=True, help="The dataset's root, which holds the specimen_* directories.")
    for p in (prepare_, score_):
        p.add_argument("--clips", required=True, help="The clip list, d4d_meta/trajectory_clips.json.")
    return ap


def main(argv: list[str] | None = None) -> None:
    ap = parser()
    args = ap.parse_args(argv)
    if args.step == "score" and len(set(args.methods)) < len(args.methods):
        ap.error(f"--methods names a method more than once: {args.methods}")

    try:
        # Cut the population's clips and write the clip list, and stop.
        if args.step == "select":
            clips = [trajectory_clip(c) for c in loader_clips(args.root, read_population(Path(args.population)))]
            entries = select(clips)
            write_clip_list(entries, Path(args.out))
            print(f"{len(entries)} clips, {sum(e['usable'] for e in entries)} usable -> {args.out}")
            return

        # Read the clip list, then write its clips or score them.
        list_path = Path(args.clips)
        list_sha = hashlib.sha256(list_path.read_bytes()).hexdigest()
        clip_list = read_clip_list(list_path)
        if args.step == "prepare":
            prepare(args.root, clip_list, Path(args.out))
            return
        rows = score_clips(args.root, clip_list, Path(args.clip_root), args.methods)
    except (FileExistsError, FileNotFoundError, ValueError) as e:
        raise SystemExit(f"{type(e).__name__}: {e}")
    for r in rows:
        r[LIST_KEY] = list_sha
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)
    print(f"{len(rows)} rows of {len({r['clip'] for r in rows})} clips -> {out}")
    for cond in dict.fromkeys(r["cond"] for r in rows):
        v = np.array([r["ate_rel"] for r in rows if r["cond"] == cond])
        print(f"  {cond:14s} n={len(v):3d}  ate_rel median {np.median(v):.4f}  mean {v.mean():.4f}")


if __name__ == "__main__":
    main()
