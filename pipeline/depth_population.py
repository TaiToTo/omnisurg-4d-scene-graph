"""Run the depth stage on every clip of a population, one process per GPU, and check that every clip got its depth.

The population file lists the clips, one per line. A clip whose manifest
already holds `depth_info` is skipped, so a stopped run continues where it
was; its `depth_info` must record the model and resolution the stage runs
at, or nothing starts. The clips left are dealt out to the GPUs given, one
`python -m pipeline.depth --device cuda --no-glb` per GPU, each writing to
`<input-dir>/_logs/depth_gpu<N>.log`. A driver that is stopped stops its
processes with it. After the run, each clip of the population must hold
`depth_info` and the stage's bundle with one depth map per image, and one
model at one resolution must have made the whole population; the command
exits non-zero otherwise, and names the clips that fail.

Usage:
    python -m pipeline.depth_population --input-dir /path/to/clips --clips atlas120k_meta/clips.txt [--gpus 0 1 2 3]
"""

import argparse
import json
import signal
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

from evalkit.evaluate import read_population
from recon3d_wrapper.da3 import DEFAULT_MODEL_ID, DEFAULT_PROCESS_RES

# The command one GPU runs, before its arguments.
DEPTH_COMMAND = [sys.executable, "-u", "-m", "pipeline.depth"]
# The arguments every process gets. `--device cuda` refuses a GPU that is not there, where `auto` would run on the
# CPU and mix its depth into the population's. `--no-glb` leaves the point clouds out: only the viewer reads
# them, and they take about 20 MB a frame.
DEPTH_ARGS = ["--device", "cuda", "--no-glb"]
# The setting every process runs at: the stage's defaults, which the driver does not override.
SETTING = (DEFAULT_MODEL_ID, DEFAULT_PROCESS_RES)
BUNDLE = Path("exports/mini_npz/results.npz")
# The keys the depth stage writes into the bundle. Another set marks another version of the stage.
BUNDLE_KEYS = {"depth", "conf", "extrinsics", "intrinsics"}
MANIFEST = "frame_manifest.json"
LOG_DIR = "_logs"
# The keys the depth stage writes into `depth_info`, which the checks read.
INFO_KEYS = ("model", "process_res", "border_inpaint")
# The signals a stopped driver gets from `kill` and from a closed terminal. Python ends on them without
# raising, so the processes it started would run on; the handler raises instead, and they are stopped.
STOP_SIGNALS = (signal.SIGTERM, signal.SIGHUP)


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
    """Return the clip's manifest.

    Raises:
        ValueError: the manifest is not JSON. The message names the clip, since a stage stopped while writing
            the manifest leaves it cut short.
    """
    with open(root / clip / MANIFEST) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"the manifest of {clip} cannot be read ({e})") from None


def read_manifests(root: Path, clips: list[str]) -> dict[str, dict]:
    """Return every clip's manifest, and refuse the population when one cannot be read.

    Raises:
        FileNotFoundError: a clip is not under `root`.
        ValueError: a manifest is not JSON. The message names every such clip.
    """
    absent = [c for c in clips if not (root / c / MANIFEST).is_file()]
    if absent:
        raise FileNotFoundError(f"{len(absent)} clip(s) of the population are not under {root}: {absent[:5]}")
    manifests, unreadable = {}, []
    for clip in clips:
        try:
            manifests[clip] = read_manifest(root, clip)
        except ValueError as e:
            unreadable.append(str(e))
    if unreadable:
        raise ValueError("; ".join(unreadable))
    return manifests


def depth_info(clip: str, manifest: dict) -> tuple[str, int, bool]:
    """Return the model, the resolution and whether a border was filled, from the clip's `depth_info`.

    Raises:
        ValueError: `depth_info` lacks one of the keys the stage writes. The message names the clip.
    """
    info = manifest["depth_info"]
    lacking = [k for k in INFO_KEYS if k not in info]
    if lacking:
        raise ValueError(f"depth_info of {clip} lacks {lacking}")
    return info["model"], info["process_res"], info["border_inpaint"]


def check_setting(manifests: dict[str, dict]) -> None:
    """Refuse a clip with depth whose `depth_info` records another model or resolution than the stage runs at.

    The driver runs the stage at its defaults. A clip made at another setting would stay as it is, and the
    run would end in a population made at two settings, found only after every process has finished.

    Raises:
        ValueError: a clip's `depth_info` lacks a key the stage writes, or records another model or resolution
            than the stage runs at. The message names each such clip.
    """
    other = []
    for clip, manifest in manifests.items():
        if "depth_info" not in manifest:
            continue
        model, res, _ = depth_info(clip, manifest)
        if (model, res) != SETTING:
            other.append(f"{clip} ({model} at {res})")
    if other:
        raise ValueError(f"{len(other)} clip(s) with depth made at another setting than the stage runs at "
                         f"({SETTING[0]} at {SETTING[1]}): {other}")


def check_depth(root: Path, clips: list[str]) -> None:
    """Refuse a population in which a clip lacks its depth, or whose depth was not made by one stage at one setting.

    Raises:
        ValueError: a clip has no `depth_info` or no bundle, its `depth_info` lacks a key the stage writes or
            records a filled border, its bundle cannot be read or holds other keys than the stage writes, its
            depth has another number of frames than `input_images/`, or the clips record more than one model
            or resolution in `depth_info`.
    """
    manifests = read_manifests(root, clips)
    missing, unreadable, other_version, mismatched, settings = [], [], [], [], {}
    for clip in clips:
        manifest = manifests[clip]
        bundle = root / clip / BUNDLE
        if "depth_info" not in manifest or not bundle.is_file():
            missing.append(clip)
            continue
        try:
            model, res, border_inpaint = depth_info(clip, manifest)
        except ValueError as e:
            unreadable.append(str(e))
            continue
        # The stage fills no border. A clip that records one was made by another version of the stage.
        if border_inpaint:
            other_version.append(f"{clip} (border_inpaint)")
            continue
        settings.setdefault((model, res), []).append(clip)
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
        problems.append(f"{len(unreadable)} with a bundle or a depth_info that cannot be read: {unreadable}")
    if other_version:
        problems.append(f"{len(other_version)} with a bundle of another version of the depth stage: {other_version}")
    if mismatched:
        problems.append(f"{len(mismatched)} with another number of depth maps than images: {mismatched}")
    if len(settings) > 1:
        problems.append("depth made at more than one setting: " + ", ".join(
            f"{model} at {res} ({len(c)} clip(s), {c[0]} among them)" for (model, res), c in sorted(settings.items())))
    if problems:
        raise ValueError("; ".join(problems))


def run_shards(command: list[str], args: list[str], root: Path, dealt: dict[int, list[str]],
               log_name: str) -> list[str]:
    """Run `command` once per GPU on its share of the clips, wait for every process, and name each that failed.

    Each process gets `--input-dir`, `--gpu`, `args` and `--clips`, and writes to
    `<root>/_logs/<log_name>_gpu<N>.log`. A driver that is stopped while starting or waiting stops the processes
    it has started with it.

    Returns:
        One line for each process that exited non-zero, with its log.
    """
    log_dir = root / LOG_DIR
    log_dir.mkdir(exist_ok=True)
    procs = {}
    try:
        for gpu, shard in dealt.items():
            log = log_dir / f"{log_name}_gpu{gpu}.log"
            print(f"  GPU {gpu}: {len(shard)} clips -> {log}")
            with open(log, "a") as fh:
                procs[gpu] = (subprocess.Popen(command + ["--input-dir", str(root), "--gpu", str(gpu),
                                                          *args, "--clips", *shard],
                                               stdout=fh, stderr=subprocess.STDOUT), log)
        exits = {gpu: p.wait() for gpu, (p, _) in procs.items()}
    except BaseException:
        for p, _ in procs.values():
            p.terminate()
        raise
    return [f"GPU {gpu} exited {code}, see {procs[gpu][1]}" for gpu, code in exits.items() if code != 0]


def run_population(root: Path, clips: list[str], gpus: list[int]) -> None:
    """Run the depth stage on the clips that lack `depth_info`, one process per GPU, and check the population.

    Raises:
        FileNotFoundError: a clip of the population is not under `root`.
        ValueError: `read_manifests` refuses a manifest, `check_setting` refuses a clip's depth, or `shards`
            refuses the GPUs.
        RuntimeError: a process exited non-zero, or `check_depth` refuses the population. The message names
            each process that failed, with its log, and each clip the check refuses.
    """
    # Refuse, before any process starts: a clip that is not there or whose manifest cannot be read, a clip
    # with depth made at another setting than the stage runs at, and a GPU listed twice.
    manifests = read_manifests(root, clips)
    check_setting(manifests)
    todo = [c for c in clips if "depth_info" not in manifests[c]]
    dealt = shards(todo, gpus)
    print(f"{len(clips)} clips in the population, {len(todo)} without depth, GPUs {gpus}")

    # Start one process per GPU, each on its share of the clips, and wait for every one.
    problems = run_shards(DEPTH_COMMAND, DEPTH_ARGS, root, dealt, "depth")

    # Check every clip of the population, whether it ran now or before, and report the check with the exits.
    try:
        check_depth(root, clips)
    except ValueError as e:
        problems.append(str(e))
    if problems:
        raise RuntimeError("; ".join(problems))


def stop(signum: int, frame) -> None:
    """Raise `SystemExit` on a stop signal, with the shell's exit code for it."""
    raise SystemExit(128 + signum)


def main() -> None:
    for s in STOP_SIGNALS:
        signal.signal(s, stop)
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
