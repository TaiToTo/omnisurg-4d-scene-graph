"""Track every window under every missing condition, on a pool of GPUs.

For each points_per_side value whose source is the work directory, the
script runs `track_job.py` (the tagged `track_sam3.py`, seed map kept) on
each window and modality that has no complete output yet. For a value whose
source is a frozen AE-CAI run, `--seeds` writes the seed maps alone, into
the work directory, and leaves the frozen tracks untouched. Each job runs in
its own process on the next free GPU slot; a failed job is reported and the
script exits non-zero.

Usage:
    python3 -m aecai_rev.run_track --pps 12 --gpus 1,2,3,4,5,6,7 --per_gpu 2
    python3 -m aecai_rev.run_track --pps 8 16 24 32 --seeds --gpus 1,2
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue

from aecai_rev.config import file_sha256, load, write_meta
from aecai_rev.windows_io import window_clips

HERE = Path(__file__).resolve().parent


def tagged_env(cfg, gpu):
    """Return the environment a job runs in: tagged source first on the path."""
    src = cfg["paths"]["aecai_src"]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [src, os.path.join(src, "sam3_wrapper"),
         os.path.join(src, "depth_sam_tracking_experiment")])
    env["HF_HOME"] = cfg["paths"]["hf_home"]
    env["HF_HUB_OFFLINE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def seed_dir(out_dir, clip, mode, pps):
    """Return the directory `track_sam3` caches a seed map in."""
    return os.path.join(out_dir, clip, f"{mode}sam_group_pps{pps}")


def track_dir(out_dir, clip, mode):
    """Return the directory of one condition's tracks."""
    return os.path.join(out_dir, clip, f"track_rgb_w1_{mode}")


def job_args(cfg, clip, mode, pps, out_dir):
    """Return the `track_sam3.py` arguments of the AE-CAI run for one job."""
    t = cfg["track"]
    w = cfg["windows"]
    args = ["--root", cfg["paths"]["cholec_gt"], "--clip", clip,
            "--out_dir", out_dir, "--device", "auto", "--max_frames", "0",
            "--sam_ckpt", cfg["paths"]["sam_ckpt"],
            "--track_base", t["track_base"],
            "--seed_frame", str(w["seed_index"]), "--sam_input", mode,
            "--out_tag", f"w1_{mode}", "--points_per_side", str(pps),
            "--seed_min_area", str(t["seed_min_area"])]
    if t["bidir"]:
        args.append("--bidir")
    return args


def is_done(out_dir, clip, mode, pps, n_frames, seeds_only):
    """Say whether a job's outputs are all on disk."""
    seed = os.path.join(seed_dir(out_dir, clip, mode, pps), "group_*.npy")
    if not glob.glob(seed):
        return False
    if seeds_only:
        return True
    return len(glob.glob(os.path.join(track_dir(out_dir, clip, mode),
                                      "label_*.npy"))) == n_frames


def run_job(cfg, job, slots, log_dir):
    """Run one job on a free GPU slot and return (job, returncode)."""
    clip, mode, pps, out_dir, seeds_only = job
    gpu = slots.get()
    try:
        cmd = [sys.executable, "-P", str(HERE / "track_job.py")]
        if seeds_only:
            cmd.append("--seed_only")
        cmd += job_args(cfg, clip, mode, pps, out_dir)
        log = os.path.join(log_dir, f"pps{pps}_{clip}_{mode}"
                           + ("_seed" if seeds_only else "") + ".log")
        with open(log, "w") as f:
            rc = subprocess.run(cmd, env=tagged_env(cfg, gpu), stdout=f,
                                stderr=subprocess.STDOUT).returncode
        return job, rc
    finally:
        slots.put(gpu)


def environment(cfg):
    """Describe the inference environment for `meta.json`."""
    import cv2
    import numpy
    import torch
    import transformers
    return dict(torch=torch.__version__, cuda=torch.version.cuda,
                transformers=transformers.__version__, numpy=numpy.__version__,
                opencv=cv2.__version__,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                sam_ckpt=os.path.basename(cfg["paths"]["sam_ckpt"]),
                sam_ckpt_sha256=file_sha256(cfg["paths"]["sam_ckpt"]),
                sam3_model_id=cfg["track"]["sam3_model_id"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pps", type=int, nargs="+", required=True)
    ap.add_argument("--modes", nargs="+", default=None)
    ap.add_argument("--seeds", action="store_true",
                    help="write seed maps only, into the work directory")
    ap.add_argument("--gpus", required=True, help="comma-separated GPU indices")
    ap.add_argument("--per_gpu", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="run at most N jobs")
    args = ap.parse_args()

    # Plan: one job per window, modality and points_per_side value not done.
    cfg = load()
    clips = window_clips(cfg)
    modes = args.modes or cfg["track"]["modalities"]
    n_frames = cfg["windows"]["samples"]
    work = cfg["paths"]["work"]
    jobs = []
    for pps in args.pps:
        if pps not in cfg["track"]["sources"]:
            raise SystemExit(f"points_per_side {pps} has no source in config.yaml")
        src = cfg["track"]["sources"][pps]
        frozen = not src.startswith(work)
        if frozen and not args.seeds:
            raise SystemExit(f"pps {pps} reads the frozen run {src}; "
                             f"only --seeds may run for it")
        out_dir = os.path.join(work, "seeds", f"pps{pps}") if args.seeds and frozen else src
        for clip in clips:
            for mode in modes:
                if not is_done(out_dir, clip, mode, pps, n_frames, args.seeds):
                    jobs.append((clip, mode, pps, out_dir, args.seeds))
        write_meta(out_dir, cfg, what="seed maps" if args.seeds else "seed maps and tracks",
                   points_per_side=pps, clips=clips, modalities=modes,
                   environment=environment(cfg),
                   command=" ".join(sys.argv))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"{len(jobs)} jobs", flush=True)

    # Run: a slot per GPU and per concurrent job on it.
    log_dir = os.path.join(work, "logs", time.strftime("%Y%m%dT%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    slots = Queue()
    gpus = [g for g in args.gpus.split(",") if g.strip()]
    for _ in range(args.per_gpu):
        for g in gpus:
            slots.put(g)
    failed = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=slots.qsize()) as ex:
        futures = [ex.submit(run_job, cfg, j, slots, log_dir) for j in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            job, rc = fut.result()
            if rc != 0:
                failed.append(job)
            print(f"[{i}/{len(jobs)}] {'FAIL' if rc else 'ok'} pps{job[2]} {job[0]} "
                  f"{job[1]} ({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(log_dir, "failed.json"), "w") as f:
        json.dump(failed, f, indent=1)
    if failed:
        raise SystemExit(f"{len(failed)} jobs failed; see {log_dir}")


if __name__ == "__main__":
    main()
