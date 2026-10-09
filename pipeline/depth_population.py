"""Run the depth stage on every clip of a population, one process per GPU, and check that every clip got its depth.

The population file lists the clips, one per line. A clip whose manifest
already holds `depth_info` is skipped, so a stopped run continues where it
was. The clips left are dealt out to the GPUs given, and one
`python -m pipeline.depth --device cuda --no-glb` runs per GPU, writing its
output to `<input-dir>/_logs/depth_gpu<N>.log`. When every process has
finished, each clip of the population must hold `depth_info` and the
stage's bundle with one depth map per image, and the whole population must
be made by one model at one resolution. The command exits non-zero
otherwise, and names the clips that fail.

Usage:
    python -m pipeline.depth_population --input-dir /path/to/clips --clips atlas120k_meta/clips.txt [--gpus 0 1 2 3]
"""

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

from evalkit.evaluate import read_population

# The command one GPU runs, before its arguments.
DEPTH_COMMAND = [sys.executable, "-u", "-m", "pipeline.depth"]
# The arguments every process gets. `--device cuda` refuses a GPU that is not there, where `auto` would run on the
# CPU and mix its depth into the population's. `--no-glb` leaves the point clouds out: only the viewer reads
# them, and they take about 20 MB a frame.
DEPTH_ARGS = ["--device", "cuda", "--no-glb"]
BUNDLE = Path("exports/mini_npz/results.npz")
# The keys the depth stage writes into the bundle. Another set marks another version of the stage.
BUNDLE_KEYS = {"depth", "conf", "extrinsics", "intrinsics"}
MANIFEST = "frame_manifest.json"
LOG_DIR = "_logs"


def shards(clips: list[str], gpus: list[int]) -> dict[int, list[str]]:
    """Deal the clips out to the GPUs in turn, and return each GPU's clips; a GPU with none is left out.

    Raises:
        ValueError: a GPU is listed twice. Its first share would be lost to its second.
    """
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"a GPU is listed twice in {gpus}")
    out = {g: clips[i::len(gpus)] for i, g in enumerate(gpus)}
    return {g: c for g, c in out.items() if c}


def read_manifest(root: Path, clip: str) -> dict:
    """Return the clip's manifest."""
    with open(root / clip / MANIFEST) as f:
        return json.load(f)


def check_depth(root: Path, clips: list[str]) -> None:
    """Refuse a population in which a clip lacks its depth, or whose depth was not made by one stage at one setting.

    Raises:
        ValueError: a clip has no `depth_info` or no bundle, its bundle cannot be read or holds other keys than
            the stage writes, its depth has another number of frames than `input_images/`, or two clips record
            another model or resolution in `depth_info`.
    """
    missing, unreadable, other_version, mismatched, settings = [], [], [], [], {}
    for clip in clips:
        manifest = read_manifest(root, clip)
        bundle = root / clip / BUNDLE
        if "depth_info" not in manifest or not bundle.is_file():
            missing.append(clip)
            continue
        info = manifest["depth_info"]
        settings.setdefault((info["model"], info["process_res"]), []).append(clip)
        try:
            with np.load(bundle) as z:
                keys, n_depth = set(z.files), int(z["depth"].shape[0]) if "depth" in z else 0
        except (zipfile.BadZipFile, OSError, ValueError) as e:
            unreadable.append(f"{clip} ({type(e).__name__}: {e})")
            continue
        if keys != BUNDLE_KEYS:
            other_version.append(f"{clip} (keys {sorted(keys)})")
            continue
        n_images = len(list((root / clip / "input_images").glob("*.png")))
        if n_depth != n_images:
            mismatched.append(f"{clip} ({n_depth} depth maps for {n_images} images)")
    problems = []
    if missing:
        problems.append(f"{len(missing)} clip(s) without depth: {missing}")
    if unreadable:
        problems.append(f"{len(unreadable)} with a bundle that cannot be read: {unreadable}")
    if other_version:
        problems.append(f"{len(other_version)} with a bundle of another version of the depth stage: {other_version}")
    if mismatched:
        problems.append(f"{len(mismatched)} with another number of depth maps than images: {mismatched}")
    if len(settings) > 1:
        problems.append("depth made at more than one setting: " + ", ".join(
            f"{model} at {res} ({len(c)} clip(s), {c[0]} among them)" for (model, res), c in sorted(settings.items())))
    if problems:
        raise ValueError("; ".join(problems))


def run_population(root: Path, clips: list[str], gpus: list[int]) -> None:
    """Run the depth stage on the clips that lack `depth_info`, one process per GPU, and check the population.

    Raises:
        FileNotFoundError: a clip of the population is not under `root`.
        ValueError: `shards` refuses the GPUs.
        RuntimeError: a process exited non-zero, or `check_depth` refuses the population. The message names
            each process that failed, with its log, and each clip the check refuses.
    """
    # Refuse a population that names a clip that is not there, and a GPU listed twice, before any process starts.
    absent = [c for c in clips if not (root / c / MANIFEST).is_file()]
    if absent:
        raise FileNotFoundError(f"{len(absent)} clip(s) of the population are not under {root}: {absent[:5]}")
    todo = [c for c in clips if "depth_info" not in read_manifest(root, c)]
    dealt = shards(todo, gpus)
    print(f"{len(clips)} clips in the population, {len(todo)} without depth, GPUs {gpus}")

    # Start one process per GPU, each on its share of the clips.
    log_dir = root / LOG_DIR
    log_dir.mkdir(exist_ok=True)
    procs = {}
    for gpu, shard in dealt.items():
        log = log_dir / f"depth_gpu{gpu}.log"
        print(f"  GPU {gpu}: {len(shard)} clips -> {log}")
        with open(log, "a") as fh:
            procs[gpu] = (subprocess.Popen(DEPTH_COMMAND + ["--input-dir", str(root), "--gpu", str(gpu), *DEPTH_ARGS,
                                                            "--clips", *shard], stdout=fh, stderr=subprocess.STDOUT),
                          log)

    # Wait for every process. A driver that is stopped stops its processes with it.
    try:
        exits = {gpu: p.wait() for gpu, (p, _) in procs.items()}
    except BaseException:
        for p, _ in procs.values():
            p.terminate()
        raise
    problems = [f"GPU {gpu} exited {code}, see {procs[gpu][1]}" for gpu, code in exits.items() if code != 0]

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
