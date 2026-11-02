"""Score MapAnything's camera trajectory on each usable StereoMIS clip, as `score_stereomis --methods` scores DA3's.

Each row is `trajectory_eval.score.score`'s, under the condition `mapanything`, at the ground truth rows the clip
list gives. A row records `trajectory_code_sha` and the clip list's hash as `clip_list_sha`, so that
`trajectory_table.py` reads it with the rows of `score_stereomis`. Before anything is scored, the command refuses
a clip without the bundle and a clip whose manifest lists other frames than the clip list.

Usage:
    python paper/ipcai2027/geometry/mapanything_trajectories.py --root /path/to/StereoMIS \\
        --clips stereomis_meta/clips.json --clip-root /path/to/clips --out mapanything_rows.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import surgical_core.stereomis as stereomis
from mapanything_depth import BUNDLE
from trajectory_eval.score import method_trajectory, score
from trajectory_eval.tools.score_stereomis import MANIFEST
from trajectory_eval.tools.scores import LIST_KEY

CONDITION = "mapanything"


def check_clip_root(clips: list[dict], clip_root: Path) -> None:
    """Refuse clips whose rows would be scored against the wrong frames.

    Raises:
        FileNotFoundError: a clip lacks the bundle or its manifest. Every clip that lacks the bundle is named.
        ValueError: a clip's manifest lists other frames than the clip list gives it.
    """
    lacking = [c["name"] for c in clips if not (clip_root / c["name"] / BUNDLE).is_file()]
    if lacking:
        raise FileNotFoundError(f"{len(lacking)} clip(s) under {clip_root} lack {BUNDLE}: {lacking}")
    moved = [c["name"] for c in clips
             if [f["native_frame"] for f in json.loads((clip_root / c["name"] / MANIFEST).read_text())["frames"]]
             != list(c["frames"])]
    if moved:
        raise ValueError(f"{len(moved)} clip(s) under {clip_root} hold other frames than the clip list: {moved}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="The StereoMIS directory, which holds one directory per sequence.")
    ap.add_argument("--clips", required=True, help="The clip list, stereomis_meta/clips.json.")
    ap.add_argument("--clip-root", required=True, help="The clips `mapanything_depth.py` wrote into.")
    ap.add_argument("--out", required=True, help="The JSON file the rows are written to.")
    args = ap.parse_args()

    # Read the clip list, and refuse the clips before any is scored.
    list_path, clip_root = Path(args.clips), Path(args.clip_root)
    list_sha = hashlib.sha256(list_path.read_bytes()).hexdigest()
    clips = stereomis.usable_clips(stereomis.read_clip_list(list_path))
    try:
        check_clip_root(clips, clip_root)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"{type(e).__name__}: {e}")

    # Score each clip against its ground truth poses.
    rows = []
    for c in clips:
        gt_T = stereomis.gt_pose(Path(args.root), c["seq"], stereomis.gt_rows(c))
        est_c, est_R = method_trajectory(clip_root / c["name"] / BUNDLE, len(c["frames"]))
        rows.append(score(c, CONDITION, gt_T, est_c, est_R) | {LIST_KEY: list_sha})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
