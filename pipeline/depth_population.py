"""Run the depth stage on every clip of a population, one process per GPU, and check that every clip got its depth.

The population file lists the clips, one per line. A clip that already
holds `exports/mini_npz/results.npz` is skipped, so a stopped run continues
where it was. The clips left are dealt out to the GPUs given, and one
`python -m pipeline.depth --no-glb` runs per GPU, writing its output to
`<input-dir>/_logs/depth_gpu<N>.log`. When every process has finished, each
clip of the population must hold the bundle with one depth map per image.
The command exits non-zero otherwise, and names the clips that lack it.

Usage:
    python -m pipeline.depth_population --input-dir /path/to/clips --clips atlas120k_meta/clips.txt [--gpus 0 1 2 3]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from evalkit.evaluate import read_population

# The command one GPU runs, before its arguments. The point clouds are left out: only the viewer reads them,
# and they take about 20 MB a frame.
DEPTH_COMMAND = [sys.executable, "-u", "-m", "pipeline.depth"]
BUNDLE = Path("exports/mini_npz/results.npz")
LOG_DIR = "_logs"


def shards(clips: list[str], gpus: list[int]) -> dict[int, list[str]]:
    """Deal the clips out to the GPUs in turn, and return each GPU's clips; a GPU with none is left out."""
    out = {g: clips[i::len(gpus)] for i, g in enumerate(gpus)}
    return {g: c for g, c in out.items() if c}


def check_depth(root: Path, clips: list[str]) -> None:
    """Refuse a population in which a clip lacks its bundle, or whose bundle has not one depth map per image.

    Raises:
        ValueError: a clip has no bundle, or its depth has another number of frames than `input_images/`.
    """
    missing, mismatched = [], []
    for clip in clips:
        bundle = root / clip / BUNDLE
        if not bundle.is_file():
            missing.append(clip)
            continue
        n_images = len(list((root / clip / "input_images").glob("*.png")))
        with np.load(bundle) as z:
            n_depth = int(z["depth"].shape[0])
        if n_depth != n_images:
            mismatched.append(f"{clip} ({n_depth} depth maps for {n_images} images)")
    if missing or mismatched:
        raise ValueError(f"{len(missing)} clip(s) without depth: {missing}; "
                         f"{len(mismatched)} with another number of depth maps than images: {mismatched}")


def run_population(root: Path, clips: list[str], gpus: list[int]) -> None:
    """Run the depth stage on the clips that lack a bundle, one process per GPU, and check the population.

    Raises:
        FileNotFoundError: a clip of the population is not under `root`.
        RuntimeError: a process exited non-zero, or `check_depth` refuses the population. The message names
            each process that failed, with its log, and each clip the check refuses.
    """
    # Refuse a population that names a clip that is not there, before any process starts.
    absent = [c for c in clips if not (root / c / "frame_manifest.json").is_file()]
    if absent:
        raise FileNotFoundError(f"{len(absent)} clip(s) of the population are not under {root}: {absent[:5]}")

    # Run the clips that have no bundle yet, dealt out to the GPUs.
    todo = [c for c in clips if not (root / c / BUNDLE).is_file()]
    print(f"{len(clips)} clips in the population, {len(todo)} without depth, GPUs {gpus}")
    log_dir = root / LOG_DIR
    log_dir.mkdir(exist_ok=True)
    procs = {}
    for gpu, shard in shards(todo, gpus).items():
        log = log_dir / f"depth_gpu{gpu}.log"
        print(f"  GPU {gpu}: {len(shard)} clips -> {log}")
        with open(log, "a") as fh:
            procs[gpu] = (subprocess.Popen(DEPTH_COMMAND + ["--input-dir", str(root), "--gpu", str(gpu), "--no-glb",
                                                            "--clips", *shard], stdout=fh, stderr=subprocess.STDOUT),
                          log)
    problems = [f"GPU {gpu} exited {p.wait()}, see {log}" for gpu, (p, log) in procs.items() if p.wait() != 0]

    # Check every clip of the population, whether it ran now or before, and report the check with the exits.
    try:
        check_depth(root, clips)
    except ValueError as e:
        problems.append(str(e))
    if problems:
        raise RuntimeError("; ".join(problems))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, help="The directory that holds the clips.")
    ap.add_argument("--clips", required=True, help="The population: one clip name per line.")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0], help="The GPUs to run on, one process each.")
    args = ap.parse_args()
    root = Path(args.input_dir)
    if not root.is_dir():
        raise SystemExit(f"no such directory: {root}")
    try:
        run_population(root, read_population(args.clips), args.gpus)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        raise SystemExit(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
