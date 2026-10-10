"""Extract, crop and depth-estimate every enumerated window.

Each window goes through three stages, in one job on the next free GPU slot:
this repository's extractor (`pipeline.extract_cholecseg8k`, which places
the unannotated frames at their true time), then the tagged
`scripts/crop_cholec_frames.py` with no source video, so that the rectangle
comes from the window's own frames as it did for the AE-CAI windows, then
the tagged `scripts/run_cholec_depth.py --no-border-inpaint`. A window whose
depth is on disk is skipped; a failed window is reported and the script
exits non-zero.

Usage:
    AECAI_REV_POPULATION=enumerated python3 -m aecai_rev.prepare_windows \
        --videos-root $CHOLEC80_VIDEOS --gpus 1,2,3,4,5,6,7 --per_gpu 2
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

from aecai_rev.config import REPO_ROOT, load, write_meta
from aecai_rev.windows_io import window_clips


def stage_env(cfg, gpu):
    """Return the environment of the tagged crop and depth scripts."""
    src = cfg["paths"]["aecai_src"]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([src, os.path.join(src, "da3_surgery_wrapper")])
    env["HF_HOME"] = cfg["paths"]["hf_home"]
    env["HF_HUB_OFFLINE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def prepare(cfg, clip, videos_root, slots, log_dir):
    """Run the three stages on one window; return (clip, returncode)."""
    root = cfg["paths"]["cholec_gt"]
    raw = clip[:-len("_crop")]
    src = cfg["paths"]["aecai_src"]
    empty = os.path.join(cfg["paths"]["work_root"], "no_videos")
    os.makedirs(empty, exist_ok=True)
    gpu = slots.get()
    try:
        with open(os.path.join(log_dir, f"{clip}.log"), "w") as log:
            steps = []
            if not os.path.isdir(os.path.join(root, raw)):
                steps.append(([sys.executable, "-m", "pipeline.extract_cholecseg8k",
                               "--seg8k-root", cfg["paths"]["cholecseg8k"],
                               "--videos-root", videos_root, "--out", root, "--clips", raw,
                               "--count", str(cfg["windows"]["samples"])],
                              str(REPO_ROOT), dict(os.environ)))
            steps.append(([sys.executable, "scripts/crop_cholec_frames.py", "--videos", raw,
                           "--input-dir", root, "--source-root", empty, "--method", "circle"],
                          src, stage_env(cfg, gpu)))
            steps.append(([sys.executable, "scripts/run_cholec_depth.py", "--input-dir", root,
                           "--videos", clip, "--no-border-inpaint", "--gpu", "0"],
                          src, stage_env(cfg, gpu)))
            for cmd, cwd, env in steps:
                rc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log,
                                    stderr=subprocess.STDOUT).returncode
                if rc:
                    return clip, rc
        if not os.path.exists(os.path.join(root, clip, "exports", "mini_npz", "results.npz")):
            return clip, 1
        return clip, 0
    finally:
        slots.put(gpu)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--videos-root", required=True, help="cholec80 videos/ directory")
    ap.add_argument("--gpus", required=True, help="comma-separated GPU indices")
    ap.add_argument("--per_gpu", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="prepare at most N windows")
    args = ap.parse_args()

    # Plan: every window of the population whose depth is not on disk.
    cfg = load()
    if cfg["windows"]["population"] == "legacy27":
        raise SystemExit("the AE-CAI windows are prepared already; set AECAI_REV_POPULATION")
    root = cfg["paths"]["cholec_gt"]
    os.makedirs(root, exist_ok=True)
    clips = window_clips(cfg)
    todo = [c for c in clips
            if not os.path.exists(os.path.join(root, c, "exports", "mini_npz", "results.npz"))]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} of {len(clips)} windows to prepare", flush=True)

    # Run on a slot per GPU and per concurrent job.
    log_dir = os.path.join(cfg["paths"]["work_root"], "logs", "prepare_" + time.strftime("%Y%m%dT%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    slots = Queue()
    for _ in range(args.per_gpu):
        for g in [g for g in args.gpus.split(",") if g.strip()]:
            slots.put(g)
    failed = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=slots.qsize()) as ex:
        futs = [ex.submit(prepare, cfg, c, args.videos_root, slots, log_dir) for c in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            clip, rc = fut.result()
            if rc:
                failed.append(clip)
            print(f"[{i}/{len(todo)}] {'FAIL' if rc else 'ok'} {clip} ({time.time() - t0:.0f}s)",
                  flush=True)
    write_meta(root, cfg, what="enumerated windows: extracted, cropped, depth",
               windows=clips, failed=failed, command=" ".join(sys.argv))
    with open(os.path.join(log_dir, "failed.json"), "w") as f:
        json.dump(failed, f)
    if failed:
        raise SystemExit(f"{len(failed)} windows failed; see {log_dir}")


if __name__ == "__main__":
    main()
