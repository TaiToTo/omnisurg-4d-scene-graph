"""Run one condition of the tracking or the per-frame stage on every clip of a population, one clip per GPU at a time.

Each clip runs as a process of its own, `pipeline.track` or
`pipeline.per_frame` given every setting of the condition. The command and
the output of each process are appended to `<clip>.log` in
`<tracks-root>/_logs/<labels>/`, where `<labels>` names the label directory.
A clip whose labels exist is skipped, so a stopped run continues where it
was; the `seed_info.json` of each such clip must first record this run's
settings. After each clip, a record of other settings stops the run. A
driver that is stopped stops its processes with it. At the end, every clip
must hold one label map per image and a record of the settings; the command
exits non-zero otherwise, and names the clips that fail.

Usage:
    python -m pipeline.condition_population track --input-dir /path/to/clips --clips atlas120k_meta/clips.txt \\
        --tracks-root /path/to/tracks --tag <tag> --rule both_ways_from_centre --sam-input normal_edge \\
        --track-base rgb --seed-edge-gain 1.0 --seed-no-smooth --sam-ckpt sam_vit_h_4b8939.pth --gpus 0 1 2 3
    python -m pipeline.condition_population per_frame --input-dir /path/to/clips --clips atlas120k_meta/clips.txt \\
        --tracks-root /path/to/tracks --tag <tag> --sam-input normal_edge --sam-ckpt sam_vit_h_4b8939.pth \\
        [--points-per-side 8] [--depth-source pi3] --gpus 0 1 2 3
"""

import argparse
import json
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

from evalkit.evaluate import SEED_INFO, read_population
from pipeline.depth_population import LOG_DIR, STOP_SIGNALS, stop
from pipeline.per_frame import SMOOTH
from pipeline.track import BUNDLE, DEPTH_SOURCES, PI3X_BUNDLE, RULES, SEED_SAM_KWARGS, seed_labels_dir, window
from surgical_core.geometry.render import SAM_INPUT_MODES, uses_geom_edge

# The command each clip runs, before its arguments.
STAGE_COMMANDS = {"track": [sys.executable, "-u", "-m", "pipeline.track"],
                  "per_frame": [sys.executable, "-u", "-m", "pipeline.per_frame"]}
# `--device cuda` refuses a GPU that is not there, where `auto` would run on the CPU.
DEVICE_ARGS = ["--device", "cuda"]
# The keys of a stage's record that differ from clip to clip. Every other key holds the condition's settings.
CLIP_KEYS = {"track": ("clip", "seed_frame", "n_seed_regions", "frames"), "per_frame": ("clip",)}
# How long the driver waits between two looks at its processes.
POLL_SEC = 1.0


def condition_dir(stage: str, s: argparse.Namespace) -> str:
    """Return the name of the directory the stage writes a clip's labels to."""
    return f"track_{s.track_base if stage == 'track' else 'rgb'}_{s.tag}"


def stage_args(stage: str, s: argparse.Namespace) -> list[str]:
    """Return the condition's settings as the stage's options, every one of them given."""
    args = ["--sam-input", s.sam_input, "--depth-source", s.depth_source,
            "--points-per-side", str(s.points_per_side), "--seed-min-area", str(s.seed_min_area)]
    if stage == "track":
        args += ["--rule", s.rule, "--track-base", s.track_base, "--seed-edge-gain", str(s.seed_edge_gain)]
        args += ["--seed-no-smooth"] if s.seed_no_smooth else []
        args += ["--seed-labels", s.seed_labels] if s.seed_labels is not None else []
    else:
        args += ["--edge-gain", str(s.edge_gain)]
    args += ["--keep-edge-ring"] if s.keep_edge_ring else []
    args += ["--sam-ckpt", s.sam_ckpt] if s.sam_ckpt is not None else []
    return args


def expected_record(stage: str, s: argparse.Namespace) -> dict:
    """Return the keys of `seed_info.json` that the settings decide, as the stage writes them for every clip.

    The stage's record holds these keys and those of `CLIP_KEYS`, and nothing else.
    """
    ring = (not s.keep_edge_ring) if uses_geom_edge(s.sam_input) else None
    if stage == "per_frame":
        return dict(tag=s.tag, seed_source="per_frame", track_base="rgb", sam_input=s.sam_input, frames="all",
                    depth_source=s.depth_source, seed_min_area=s.seed_min_area, seed_topk=0, point_grids=None,
                    seed_input=dict(produced_by="sam", cache_path=None, points_per_side=s.points_per_side,
                                    seed_sam_kwargs=SEED_SAM_KWARGS, seed_edge_gain=s.edge_gain,
                                    seed_smooth=SMOOTH, edge_ring_masked=ring))
    if s.seed_labels is None:
        seed_input = dict(produced_by="sam", cache_path=None, points_per_side=s.points_per_side,
                          seed_sam_kwargs=SEED_SAM_KWARGS, seed_edge_gain=s.seed_edge_gain,
                          seed_smooth=not s.seed_no_smooth, edge_ring_masked=ring)
    else:
        seed_input = dict(produced_by="external", cache_path=s.seed_labels, points_per_side=None,
                          seed_sam_kwargs=None, seed_edge_gain=None, seed_smooth=None, edge_ring_masked=None)
    _, _, both_ways = window(s.rule, 1)
    return dict(seed_source="sam" if s.seed_labels is None else "external", seed_labels=s.seed_labels or "",
                sam_input=s.sam_input, depth_source=s.depth_source, track_base=s.track_base, bidir=both_ways,
                stride=1, seed_min_area=s.seed_min_area, seed_input=seed_input, instrument_seed="off")


def read_record(lab_dir: Path, clip: str) -> dict:
    """Return the record beside a clip's labels.

    Raises:
        ValueError: the labels have no `seed_info.json`, or it is not JSON. The message names the clip.
    """
    path = lab_dir / SEED_INFO
    if not path.is_file():
        raise ValueError(f"{clip}'s {lab_dir.name} has no {SEED_INFO}, so the settings it was made with are unknown")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{clip}'s {SEED_INFO} cannot be read ({e})") from None


def check_settings(stage: str, clip: str, record: dict, expected: dict) -> None:
    """Refuse a record that holds other settings than `expected`, or lacks a key the stage writes for each clip.

    Raises:
        ValueError: the message names the clip and each key that differs, with both values.
    """
    want = json.loads(json.dumps(expected))
    got = {k: v for k, v in record.items() if k not in CLIP_KEYS[stage]}

    def shown(d: dict, k: str) -> str:
        return repr(d[k]) if k in d else "(no such key)"

    differ = [f"{k} {shown(got, k)}, not {shown(want, k)}" for k in sorted(set(got) | set(want))
              if k not in got or k not in want or got[k] != want[k]]
    differ += [f"{k} (no such key)" for k in CLIP_KEYS[stage] if k not in record]
    if differ:
        raise ValueError(f"{clip}'s {SEED_INFO} records other settings than this run passes: {'; '.join(differ)}")


def check_labels(stage: str, s: argparse.Namespace, clip_dir: Path, lab_dir: Path, record: dict) -> None:
    """Refuse labels that are not the clip's, that miss a frame, or whose seed is not where the rule puts it.

    Raises:
        ValueError: the record names another clip, the label maps are not one per image, or the tracker's
            record does not list every frame and the rule's seed frame. The message names the clip.
    """
    clip = clip_dir.name
    if record.get("clip") != clip:
        raise ValueError(f"{clip}'s {SEED_INFO} records the clip {record.get('clip')!r}")
    n = len(list((clip_dir / "input_images").glob("*.png")))
    labels = sorted(p.name for p in lab_dir.glob("label_*.npy"))
    if labels != [f"label_{i:04d}.npy" for i in range(n)]:
        raise ValueError(f"{clip} has {len(labels)} label maps in {lab_dir.name} for {n} images")
    if stage == "track":
        frames, seed, _ = window(s.rule, n)
        if record.get("frames") != frames or record.get("seed_frame") != seed:
            raise ValueError(f"{clip}'s {SEED_INFO} records seed frame {record.get('seed_frame')} of "
                             f"{len(record.get('frames') or [])} frames labelled; {s.rule} seeds frame {seed} of {n}")


def check_inputs(stage: str, s: argparse.Namespace, root: Path, clips: list[str], gpus: list[int]) -> None:
    """Refuse, before anything runs, what would make every process fail or run another condition.

    Raises:
        ValueError: a GPU is listed twice; the tracker's input burns edges while the seed's record cannot say
            whether their ring was kept; the SAM weights are missing or not given where the stage needs them;
            a clip is not under `root` with the depth the stage reads; or a clip's seed labels are missing.
            The message counts such clips and names the first five.
    """
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"a GPU is listed twice in {gpus}")
    # The stage records the ring of the seed's input only, so the ring of the tracker's would go unrecorded and a
    # continued run could mix the two settings, until "The edge ring of the tracker's input" is decided.
    if stage == "track" and uses_geom_edge(s.track_base) and (s.seed_labels is not None
                                                               or not uses_geom_edge(s.sam_input)):
        seed = "--seed-labels" if s.seed_labels is not None else f"--sam-input {s.sam_input}"
        raise ValueError(f"--track-base {s.track_base} burns edges and {seed} records no edge ring, so whether the "
                         "tracker's input kept its ring would be recorded nowhere")
    if s.sam_ckpt is None and not (stage == "track" and s.seed_labels is not None):
        raise ValueError("--sam-ckpt is needed to cut the frames, unless --seed-labels gives the seed regions")
    if s.sam_ckpt is not None and not Path(s.sam_ckpt).is_file():
        raise ValueError(f"no SAM weights at {s.sam_ckpt}")
    bundles = [BUNDLE] + ([PI3X_BUNDLE] if s.depth_source == "pi3" else [])
    absent = [f"{c} ({b})" for c in clips for b in bundles if not (root / c / b).is_file()]
    if absent:
        raise ValueError(f"{len(absent)} clip(s) of the population lack the depth under {root}: {absent[:5]}")
    if stage == "track" and s.seed_labels is not None:
        absent = [c for c in clips if not seed_labels_dir(s.seed_labels, c).is_dir()]
        if absent:
            raise ValueError(f"{len(absent)} clip(s) have no seed labels under {s.seed_labels}: {absent[:5]}")


def check_montages(stage: str, s: argparse.Namespace, tracks_root: Path, clips: list[str]) -> None:
    """Refuse a clip that holds the tracker's montage and not its labels, which the tracking stage refuses to run on.

    Labels removed by hand without their montage leave such a clip, and the stage would exit 1 on it every run.

    Raises:
        ValueError: the message counts such clips and names the first five.
    """
    if stage != "track":
        return
    cond = condition_dir(stage, s)
    stray = [c for c in clips
             if (tracks_root / c / "viz" / f"montage_{cond}.png").is_file() and not (tracks_root / c / cond).is_dir()]
    if stray:
        raise ValueError(f"{len(stray)} clip(s) hold viz/montage_{cond}.png without {cond}, which the stage refuses "
                         f"to replace; remove the montage with the labels: {stray[:5]}")


def check_population(stage: str, s: argparse.Namespace, root: Path, tracks_root: Path, clips: list[str],
                     only_existing: bool = False) -> None:
    """Refuse a population in which a clip lacks its labels, holds other settings, or misses a frame.

    Args:
        only_existing: check only the clips whose labels exist, as before a run that continues an earlier one.

    Raises:
        ValueError: the message names each clip that fails, and why.
    """
    cond, expected = condition_dir(stage, s), expected_record(stage, s)
    missing, problems = [], []
    for clip in clips:
        lab_dir = tracks_root / clip / cond
        if not lab_dir.is_dir():
            if not only_existing:
                missing.append(clip)
            continue
        try:
            record = read_record(lab_dir, clip)
            check_settings(stage, clip, record, expected)
            check_labels(stage, s, root / clip, lab_dir, record)
        except ValueError as e:
            problems.append(str(e))
    if missing:
        problems.insert(0, f"{len(missing)} clip(s) without {cond}: {missing}")
    if problems:
        raise ValueError("; ".join(problems))


def run_clips(stage: str, s: argparse.Namespace, root: Path, tracks_root: Path, clips: list[str],
              gpus: list[int]) -> list[str]:
    """Run the stage on each clip as a process of its own, one per GPU at a time, and return the failures.

    A failure is a process that exited non-zero; its message names the clip and its log.

    Raises:
        ValueError: a clip that ran has no readable record (`read_record`), or `check_settings` refuses its
            record. The processes still running are stopped, as they are when the driver is stopped.
    """
    cond, expected = condition_dir(stage, s), expected_record(stage, s)
    log_dir = tracks_root / LOG_DIR / cond
    log_dir.mkdir(parents=True, exist_ok=True)
    queue, running, failed, done = list(clips), {}, [], 0
    try:
        while queue or running:
            # Give each idle GPU the next clip; its log starts with the command.
            for gpu in gpus:
                if gpu in running or not queue:
                    continue
                clip = queue.pop(0)
                log = log_dir / f"{clip}.log"
                cmd = (STAGE_COMMANDS[stage] + ["--input-dir", str(root), "--tracks-root", str(tracks_root),
                                                "--tag", s.tag, "--clips", clip, *DEVICE_ARGS, "--gpu", str(gpu)]
                       + stage_args(stage, s))
                print(f"  GPU {gpu}: {clip} -> {log}", flush=True)
                with open(log, "a") as fh:
                    fh.write(shlex.join(cmd) + "\n")
                    fh.flush()
                    running[gpu] = (clip, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT), log)
            time.sleep(POLL_SEC)
            # Collect the processes that ended. A record of other settings stops the run: every clip after it
            # would be made the same way.
            for gpu, (clip, proc, log) in list(running.items()):
                code = proc.poll()
                if code is None:
                    continue
                del running[gpu]
                done += 1
                if code != 0:
                    failed.append(f"{clip} exited {code}, see {log}")
                    print(f"  [failed] GPU {gpu}: {clip} exited {code} ({done}/{len(clips)})", flush=True)
                    continue
                try:
                    check_settings(stage, clip, read_record(tracks_root / clip / cond, clip), expected)
                except ValueError as e:
                    # The failures before it would be lost with the run, so they go into its message.
                    raise ValueError("; ".join([str(e), *failed])) from None
                print(f"  [ok] GPU {gpu}: {clip} ({done}/{len(clips)})", flush=True)
    except BaseException:
        for _, proc, _ in running.values():
            proc.terminate()
        raise
    return failed


def run_population(stage: str, s: argparse.Namespace, root: Path, tracks_root: Path, clips: list[str],
                   gpus: list[int]) -> None:
    """Run the condition on the clips that lack its labels, one clip per GPU at a time, and check the population.

    Raises:
        ValueError: `check_inputs` or `check_montages` refuses the run, or a clip's labels exist and
            `check_population` refuses them, before any process starts; or a clip's record holds other settings
            after it ran.
        RuntimeError: a process exited non-zero, or `check_population` refuses the population after the run.
            The message names each clip that failed, with its log, and each clip the check refuses.
    """
    # Refuse, before any process starts: what would make every process fail, and labels of another setting.
    check_inputs(stage, s, root, clips, gpus)
    check_montages(stage, s, tracks_root, clips)
    check_population(stage, s, root, tracks_root, clips, only_existing=True)
    cond = condition_dir(stage, s)
    todo = [c for c in clips if not (tracks_root / c / cond).is_dir()]
    print(f"{len(clips)} clips in the population, {len(todo)} without {cond}, GPUs {gpus}")

    # Run the clips left, and check every clip of the population, whether it ran now or before.
    problems = run_clips(stage, s, root, tracks_root, todo, gpus)
    try:
        check_population(stage, s, root, tracks_root, clips)
    except ValueError as e:
        problems.append(str(e))
    if problems:
        raise RuntimeError("; ".join(problems))


def build_parser() -> argparse.ArgumentParser:
    """Return the command's parser: one subcommand per stage, each with the stage's settings and their defaults."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--input-dir", required=True, help="The directory that holds the clips.")
    common.add_argument("--clips", required=True, help="The population: one clip name per line.")
    common.add_argument("--tracks-root", required=True, help="Where each clip's conditions go.")
    common.add_argument("--tag", required=True, help="The condition's name.")
    common.add_argument("--gpus", type=int, nargs="+", required=True, help="The GPUs to run on, one clip each.")
    common.add_argument("--sam-input", required=True, choices=SAM_INPUT_MODES, help="The segmenter's input mode.")
    common.add_argument("--depth-source", default="da3", choices=DEPTH_SOURCES,
                        help="The depth the normals and edges come from.")
    common.add_argument("--points-per-side", type=int, default=24, help="The mask generator's grid.")
    common.add_argument("--seed-min-area", type=int, default=400, help="The fewest pixels a region keeps.")
    common.add_argument("--keep-edge-ring", action="store_true",
                        help="Keep the edges' ring along the image border and around invalid depth.")
    stages = ap.add_subparsers(dest="stage", required=True)
    track = stages.add_parser("track", parents=[common], help="Run pipeline.track on every clip.")
    track.add_argument("--rule", required=True, choices=RULES, help="Where the seed goes and which way it is carried.")
    track.add_argument("--track-base", required=True, choices=SAM_INPUT_MODES, help="The tracker's input mode.")
    track.add_argument("--sam-ckpt", help="The SAM ViT-H weights, sam_vit_h_4b8939.pth. Needed without --seed-labels.")
    track.add_argument("--seed-labels", help="Seed regions made outside the stage, as pipeline.track reads them.")
    track.add_argument("--seed-edge-gain", type=float, default=0.85, help="How dark the seed input's edges are.")
    track.add_argument("--seed-no-smooth", action="store_true", help="Do not smooth the seed input's normals.")
    per_frame = stages.add_parser("per_frame", parents=[common], help="Run pipeline.per_frame on every clip.")
    per_frame.add_argument("--sam-ckpt", required=True, help="The SAM ViT-H weights, sam_vit_h_4b8939.pth.")
    per_frame.add_argument("--edge-gain", type=float, default=1.0, help="How dark the burnt-in edges are.")
    return ap


def install_stop_handlers() -> None:
    """Stop the driver, and its processes with it, on a stop signal, but leave ignored a signal that is ignored.

    Under `nohup` SIGHUP is ignored, and a handler would stop the run when the terminal closes.
    """
    for sig in STOP_SIGNALS:
        if signal.getsignal(sig) is not signal.SIG_IGN:
            signal.signal(sig, stop)


def main() -> None:
    install_stop_handlers()
    args = build_parser().parse_args()
    root = Path(args.input_dir)
    if not root.is_dir():
        raise SystemExit(f"no such directory: {root}")
    try:
        run_population(args.stage, args, root, Path(args.tracks_root), read_population(args.clips), args.gpus)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        raise SystemExit(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
