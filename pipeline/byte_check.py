"""Run two commands on fresh copies of one clip and compare every file they write, byte for byte.

A ported pipeline stage is done when its output equals the workbench's, and this
is the check. Each run gets a working copy of the clip, its inputs copied in and
its manifest stripped of what the stages write; whatever it writes there is
compared with the other run's. A JSON file that differs only in `runtime_sec` is
counted apart. Given one command, it runs it twice and measures determinism. It
refuses a pair it cannot vouch for: a watched package from a file its run's
repository's commit does not track, a module from a file the other run's
repository tracks, uncommitted changes, a run with no record of its imports, two
runs whose Python, BLAS or distributions differ, bar those named as one run's
alone, and a changed input. Runs that differ exit 1; a refusal exits 2.

Usage:
    python -m pipeline.byte_check --root /path/to/clips --clip <clip> \\
        --a-repo /path/to/the/workbench --a-cmd '<its stage> --input-dir {root} --clips {clip}' \\
        --b-repo . --b-cmd 'python -m pipeline.<stage> --input-dir {root} --clips {clip}' \\
        [--frames 4] [--needs exports/mini_npz/results.npz] [--weights ~/.cache/huggingface] [--json-out r.json] \\
        [--a-alone trimesh networkx]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Keys a stage writes into `frame_manifest.json`, stripped from the working copy so that a stage cannot take a
# leftover for its own work.
MANIFEST_RUN_KEYS = ("geometry_sources", "depth_info")
MANIFEST_FRAME_KEYS = ("glb_centroid", "camera_pos_glb", "camera_forward_glb", "camera_up_glb", "geometry_sources")

# JSON fields that record how long a run took, not what it produced.
VOLATILE_KEYS = ("runtime_sec",)

# A volatile field alone on its line with a scalar value, as `json.dumps(..., indent=2)` writes it. Matching the text
# rather than the parsed tree keeps the rest of the comparison about bytes: key order, `14` against `14.0`, indentation.
_KEYS = "|".join(VOLATILE_KEYS)
VOLATILE_LINE_RE = re.compile(r'^\s*"(' + _KEYS + r')"\s*:\s*[^{\[]+?,?\s*$')
VOLATILE_KEY_RE = re.compile(r'"(' + _KEYS + r')"\s*:')

RECORD_DIR_ENV = "BYTE_CHECK_RECORD_DIR"

# Put first on a run's PYTHONPATH as `sitecustomize`, this writes at exit, one file per process, every module the
# process imported and the distributions behind those modules. A process that exits without running it leaves no
# record.
_HOOK = '''\
import atexit
import json
import os
import subprocess
import sys
from importlib import metadata


def _file(module):
    try:
        path = getattr(module, "__file__", None)
    except Exception:
        return None
    # A relative path names no file: torch's `torch.ops` and `torch.classes` carry one.
    return path if isinstance(path, str) and os.path.isabs(path) else None


def _commit(dist):
    # A wheel carries no source and names no commit. A VCS install names its commit. An install from a
    # directory is read from that directory's git, `+dirty` when the directory differs from its commit.
    text = dist.read_text("direct_url.json")
    if not text:
        return None
    direct = json.loads(text)
    if "vcs_info" in direct:
        return direct["vcs_info"].get("commit_id")
    url = direct.get("url", "")
    if not url.startswith("file://"):
        return None
    try:
        head = subprocess.run(["git", "-C", url[len("file://"):], "rev-parse", "HEAD"],
                              capture_output=True, text=True)
        if head.returncode != 0:
            return None
        dirty = subprocess.run(["git", "-C", url[len("file://"):], "status", "--porcelain",
                                "--untracked-files=no"], capture_output=True, text=True)
        return head.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
    except OSError:
        return None


def _blas():
    # The same numpy version writes different last bits on different BLAS implementations.
    numpy = sys.modules.get("numpy")
    if numpy is None:
        return None
    try:
        blas = numpy.show_config(mode="dicts")["Build Dependencies"]["blas"]
        return f'{blas.get("name")} {blas.get("version")}'
    except Exception:
        return "unknown"


def _record(out=os.environ.get("BYTE_CHECK_RECORD_DIR")):
    if not out:
        return
    # The distributions are looked up from what was imported, never from a fixed list, so a package the list
    # would miss is still recorded -- `opencv-python-headless` as much as `opencv-python`.
    owners = metadata.packages_distributions()
    distributions = {}
    for top in sorted({name.split(".")[0] for name in sys.modules}):
        entries = []
        for owner in sorted(set(owners.get(top, []))):
            try:
                dist = metadata.distribution(owner)
            except metadata.PackageNotFoundError:
                entries.append({"name": owner, "version": None, "commit": None})
                continue
            entries.append({"name": owner, "version": dist.version, "commit": _commit(dist)})
        if entries:
            distributions[top] = entries
    modules = {n: _file(m) for n, m in list(sys.modules.items()) if _file(m)}
    with open(os.path.join(out, f"{os.getpid()}.json"), "w") as f:
        json.dump({"python": sys.version.split()[0], "blas": _blas(), "distributions": distributions,
                   "modules": modules}, f)


atexit.register(_record)
'''


@dataclass(frozen=True)
class Side:
    """One of the two runs: the repository its command belongs to and runs in, the command, its added
    environment, the packages it must import from its own repository, and the top-level modules it alone imports.

    The watch is per side. The workbench imports its wrapper packages and the port never does, so one shared
    list would refuse whichever side does not import it. `alone` names what this side's own code imports and the
    other side's never does, such as `trimesh`, which the workbench's `surgical_core` imports and no stage
    calls. Their distributions are recorded and left out of the comparison of the two environments.
    """

    repo: Path
    cmd: str
    env: dict = field(default_factory=dict)
    watch: tuple[str, ...] = ("surgical_core",)
    alone: tuple[str, ...] = ()


def sha256_of(path: Path) -> str:
    """The hex sha256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(path: Path) -> dict | None:
    """The size and modification time of a weights file or directory, enough to see it changed; None when absent.

    Hashing the weights on every run would cost more than the check.
    """
    if not path.exists():
        return None
    if path.is_file():
        st = path.stat()
        return {"bytes": st.st_size, "mtime": int(st.st_mtime)}
    files = [q for q in path.rglob("*") if q.is_file()]
    return {"files": len(files), "bytes": sum(q.stat().st_size for q in files),
            "newest_mtime": max((int(q.stat().st_mtime) for q in files), default=None)}


def git_state(repo: Path) -> dict:
    """The commit `repo` is at, and whether its tracked files hold uncommitted changes.

    Raises:
        ValueError: `repo` is not a git repository.
    """
    def git(*args: str) -> str:
        out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
        if out.returncode != 0:
            raise ValueError(f"{repo} is not a git repository: {out.stderr.strip()}")
        return out.stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def tracked_files(repo: Path) -> frozenset[str]:
    """The real paths of every file the commit at `repo` tracks.

    Raises:
        ValueError: `repo` is not a git repository.
    """
    out = subprocess.run(["git", "-C", str(repo), "ls-files", "-z"], capture_output=True, text=True)
    if out.returncode != 0:
        raise ValueError(f"{repo} is not a git repository: {out.stderr.strip()}")
    root = os.path.realpath(repo)
    return frozenset(os.path.realpath(os.path.join(root, p)) for p in out.stdout.split("\0") if p)


def gpus() -> list[str] | None:
    """The machine's GPUs as `name, driver`, once per model, or None without `nvidia-smi`.

    Not which GPU: the workbench's stage wrote the same bytes on two GPUs of one model.
    """
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return sorted(set(out.stdout.strip().splitlines())) if out.returncode == 0 else None


def _copy_files(src_dir: Path, dst_dir: Path, names: list[str]) -> None:
    dst_dir.mkdir(parents=True)
    for name in names:
        shutil.copyfile(src_dir / name, dst_dir / name)


def prepare_clip(src_clip: Path, dst_clip: Path, needs: tuple[str, ...] = (), frames: int | None = None) -> None:
    """Make a working copy of a clip's inputs, holding nothing a stage writes.

    The images, the masks and `needs`, the files of an earlier stage the stage reads, are copied, never linked.
    A stage that writes through a link would overwrite the clip's own files, and on G the clips are the paper's
    data. With `frames`, only the first that many images, and as many of the manifest's frames, are kept.

    Raises:
        FileNotFoundError: `src_clip` has no `input_images/` or manifest, or lacks one of `needs`.
        ValueError: `frames` is below 1 or above the clip's image count.
    """
    images = src_clip / "input_images"
    manifest = src_clip / "frame_manifest.json"
    if not images.is_dir() or not manifest.is_file():
        raise FileNotFoundError(f"{src_clip} is not a clip: it needs input_images/ and frame_manifest.json")
    # A hidden file is no frame: Finder drops `.DS_Store` beside the images, and it sorts first.
    names = sorted(p.name for p in images.iterdir() if p.is_file() and not p.name.startswith("."))
    if frames is not None:
        if not 1 <= frames <= len(names):
            raise ValueError(f"{src_clip.name} has {len(names)} images; its first {frames} cannot be kept")
        names = names[:frames]
    # The inputs, copied file by file.
    _copy_files(images, dst_clip / "input_images", names)
    masks = src_clip / "seg_masks"
    if masks.is_dir():
        _copy_files(masks, dst_clip / "seg_masks",
                    sorted(p.name for p in masks.iterdir() if p.is_file() and not p.name.startswith(".")))
    for rel in needs:
        src = src_clip / rel
        if not src.exists():
            raise FileNotFoundError(f"{src_clip} has no {rel}, which the stage reads; run the earlier stage on it first")
        (dst_clip / rel).parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst_clip / rel)
        else:
            shutil.copyfile(src, dst_clip / rel)
    # The manifest, without what the stages write, and cut to the frames kept.
    m = json.loads(manifest.read_text())
    for k in MANIFEST_RUN_KEYS:
        m.pop(k, None)
    for entry in m.get("frames", []):
        for k in MANIFEST_FRAME_KEYS:
            entry.pop(k, None)
    if frames is not None and isinstance(m.get("frames"), list):
        m["frames"] = m["frames"][:frames]
        # A fail-closed stage counts the frames; a stale total would make it refuse the cut copy.
        if "n_frames" in m:
            m["n_frames"] = len(m["frames"])
    (dst_clip / "frame_manifest.json").write_text(json.dumps(m, ensure_ascii=False, indent=2))


def collect(work_root: Path) -> dict[str, Path]:
    """Every file under `work_root`, by its path there.

    The tree is swept rather than matched against a list, so a file a stage starts writing reaches the comparison
    without anyone naming it. What the run wrote is told apart by hashing the tree before and after the run.
    """
    found = {}
    for dirpath, dirnames, filenames in os.walk(work_root):
        dirnames.sort()
        d = Path(dirpath)
        for name in sorted(filenames):
            found[str((d / name).relative_to(work_root))] = d / name
    return found


def strip_volatile_lines(text: str) -> tuple[str, tuple[str, ...]] | None:
    """A JSON file's text without its volatile fields, and the fields removed, so that one vanishing still differs.

    Returns None when a volatile field is not alone on its line with a scalar value: such a file is compared as
    bytes rather than guessed at.
    """
    kept, removed = [], []
    for line in text.splitlines():
        m = VOLATILE_LINE_RE.match(line)
        if m:
            removed.append(m.group(1))
            continue
        if VOLATILE_KEY_RE.search(line):
            return None
        kept.append(line)
    return "\n".join(kept), tuple(removed)


def numeric_spread(a_path: Path, b_path: Path) -> tuple[float, float] | None:
    """The largest absolute and relative gap between two `.npy` or `.npz` files, where both are finite.

    Returns None for any other file, and for a pair whose keys, shapes or non-finite positions disagree: that is a
    difference no gap expresses. A NaN must not swallow the gap, since `max(0.0, nan)` is 0.0, and a ratio against
    a zero reference says nothing, so none is taken.
    """
    def arrays(p: Path) -> dict | None:
        if p.suffix == ".npy":
            return {"": np.load(p, allow_pickle=False)}
        if p.suffix == ".npz":
            with np.load(p, allow_pickle=False) as z:
                return {k: z[k] for k in z.files}
        return None

    a, b = arrays(a_path), arrays(b_path)
    if a is None or b is None or a.keys() != b.keys():
        return None
    max_abs = max_rel = 0.0
    for k in a:
        x, y = a[k], b[k]
        if x.shape != y.shape or not np.issubdtype(x.dtype, np.number):
            return None
        xf, yf = x.astype(np.float64), y.astype(np.float64)
        finite = np.isfinite(xf)
        if not np.array_equal(finite, np.isfinite(yf)):
            return None
        d, scale = np.abs(xf[finite] - yf[finite]), np.abs(xf[finite])
        max_abs = max(max_abs, float(d.max()) if d.size else 0.0)
        rel = d[scale > 0] / scale[scale > 0]
        max_rel = max(max_rel, float(rel.max()) if rel.size else 0.0)
    return max_abs, max_rel


def compare(a_files: dict[str, Path], b_files: dict[str, Path]) -> dict:
    """Compare two runs' files: the counts, the paths only one run wrote, those that differ, and the numeric gap."""
    only_a, only_b = sorted(set(a_files) - set(b_files)), sorted(set(b_files) - set(a_files))
    shared = sorted(set(a_files) & set(b_files))
    same, differ, volatile_only = [], [], []
    max_abs = max_rel = 0.0
    for rel in shared:
        a, b = a_files[rel], b_files[rel]
        if sha256_of(a) == sha256_of(b):
            same.append(rel)
            continue
        if a.suffix == ".json":
            sa, sb = strip_volatile_lines(a.read_text()), strip_volatile_lines(b.read_text())
            if sa is not None and sa == sb:
                volatile_only.append(rel)
                continue
        differ.append(rel)
        spread = numeric_spread(a, b)
        if spread is not None:
            max_abs, max_rel = max(max_abs, spread[0]), max(max_rel, spread[1])
    return {"n_shared": len(shared), "n_same": len(same), "n_differ": len(differ),
            "n_volatile_only": len(volatile_only), "only_a": only_a, "only_b": only_b,
            "differ": differ, "volatile_only": volatile_only, "max_abs": max_abs, "max_rel": max_rel}


def check_imports(records: list[dict], own_repo: Path, other_repo: Path, watched: tuple[str, ...]) -> frozenset[str]:
    """Refuse a run whose watched packages are missing or did not come from files its own repository's commit
    tracks, or any of whose modules came from a file the other run's repository tracks.

    Tracked is the test, never the directory. An untracked file, or a stale install under the repository's
    own `.venv`, lies inside the directory yet belongs to no commit; a shared venv inside the other
    repository belongs to no commit of the other repository.

    Returns:
        The top-level names of the modules that came from the run's own tracked files.

    Raises:
        ValueError: naming each such module and the file it came from.
    """
    own_tracked = tracked_files(own_repo)
    same = os.path.realpath(own_repo) == os.path.realpath(other_repo)
    other_tracked = frozenset() if same else tracked_files(other_repo) - own_tracked
    files: dict[str, set[str]] = {}
    for rec in records:
        for name, path in rec["modules"].items():
            files.setdefault(name, set()).add(os.path.realpath(path))
    problems = []
    for pkg in watched:
        mods = {n: ps for n, ps in files.items() if n == pkg or n.startswith(pkg + ".")}
        if pkg not in mods:
            problems.append(f"{pkg} was not imported")
        problems += [f"{n} from {p}, a file the commit at {own_repo} does not track"
                     for n, ps in sorted(mods.items()) for p in sorted(ps) if p not in own_tracked]
    problems += [f"{n} from {p}, a file the other run's repository tracks" for n, ps in sorted(files.items())
                 for p in sorted(ps) if p in other_tracked]
    if problems:
        raise ValueError("the run did not run its own repository's code:\n  " + "\n  ".join(problems))
    return frozenset(n.split(".")[0] for n, ps in files.items() if ps and ps <= own_tracked)


def environment(records: list[dict]) -> dict:
    """The Python, BLAS and distributions one run ran with, merged across its processes.

    A process that never imported a module says nothing about its distribution, so the processes'
    distributions are merged rather than matched whole.

    Raises:
        ValueError: no record, or processes of the run that disagree.
    """
    if not records:
        raise ValueError("the run left no record of its imports: it ran without the hook (python -I or -S, or a "
                         "command that replaces PYTHONPATH), or exited without running it")
    pythons = sorted({r["python"] for r in records})
    if len(pythons) != 1:
        raise ValueError(f"the processes of one run ran different Pythons: {pythons}")
    blas = sorted({r["blas"] for r in records if r["blas"] is not None})
    if len(blas) > 1:
        raise ValueError(f"the processes of one run ran numpy on different BLAS: {blas}")
    distributions: dict = {}
    for r in records:
        for top, entries in r["distributions"].items():
            if distributions.setdefault(top, entries) != entries:
                raise ValueError(f"the processes of one run took {top} from different distributions: "
                                 f"{distributions[top]} against {entries}")
    return {"python": pythons[0], "blas": blas[0] if blas else None, "distributions": distributions}


def check_same_environment(env_a: dict, env_b: dict, own: frozenset[str] = frozenset(),
                           alone_a: tuple[str, ...] = (), alone_b: tuple[str, ...] = ()) -> None:
    """Refuse two runs whose Python, BLAS or distributions differ: their bytes would not be the code's alone.

    `own` names the top-level modules that came from the runs' own repositories. The two sides' own code
    differs by design, and its difference is the byte comparison's to judge, so it is no ground for refusal.
    `alone_a` and `alone_b` name the top-level modules one run alone imports (`Side.alone`). A name must hold:
    the run imported it and the other run did not. Otherwise the list would excuse a difference it does not name.

    Raises:
        ValueError: naming each difference, or each name in `alone_a` or `alone_b` that does not hold.
    """
    a, b = env_a["distributions"], env_b["distributions"]
    # The names of what one run alone imports, each checked against both runs' records.
    wrong = []
    for name, alone, mine, other in (("a", alone_a, a, b), ("b", alone_b, b, a)):
        for k in alone:
            if k not in mine:
                wrong.append(f"{k}, named as run {name}'s alone, came from no distribution that run imported")
            if k in other:
                wrong.append(f"{k}, named as run {name}'s alone, was imported by the other run too")
    if wrong:
        raise ValueError("the runs' records do not bear out what they are said to import alone:\n  "
                         + "\n  ".join(wrong))
    # The environments, bar the runs' own code and what one run alone imports.
    diffs = [f"python: {env_a['python']} against {env_b['python']}"] if env_a["python"] != env_b["python"] else []
    if env_a["blas"] != env_b["blas"]:
        diffs.append(f"numpy BLAS: {env_a['blas']} against {env_b['blas']}")
    skip = own | set(alone_a) | set(alone_b)
    diffs += [f"{k}: {a.get(k)} against {b.get(k)}" for k in sorted(set(a) | set(b))
              if k not in skip and a.get(k) != b.get(k)]
    if diffs:
        raise ValueError("the two runs ran in different environments:\n  " + "\n  ".join(diffs))


def run(side: Side, work_root: Path, clip: str, hook_dir: Path, record_dir: Path, log: Path) -> float:
    """Run one side's command, in its repository, on the working copy under `work_root`; return its seconds.

    `{root}` and `{clip}` in the command name the working copy.

    Raises:
        RuntimeError: the command exited non-zero.
    """
    cmd = [a.replace("{root}", str(work_root)).replace("{clip}", clip) for a in shlex.split(side.cmd)]
    env = {**os.environ, **side.env}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(hook_dir), env.get("PYTHONPATH", "")) if p)
    env[RECORD_DIR_ENV] = str(record_dir)
    record_dir.mkdir(parents=True)
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.call(cmd, cwd=side.repo, env=env, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        tail = "\n".join(log.read_text().splitlines()[-25:])
        raise RuntimeError(f"{side.cmd!r} exited {rc}; the last lines of {log}:\n{tail}")
    return time.time() - t0


def check(root: Path, clip: str, a: Side, b: Side | None, work: Path, needs: tuple[str, ...] = (),
          frames: int | None = None, weights: tuple[Path, ...] = ()) -> dict:
    """Run both sides on working copies of `clip` under `work`, and compare what they wrote; `a` twice without `b`.

    Returns:
        The report: the repositories' commits, the machine, the runs' environments, and the comparison.

    Raises:
        ValueError: a pair the check cannot vouch for, as the module docstring lists, or `a` alone with what it
            imports alone named: one command run twice imports the same modules both times.
        RuntimeError: a command failed, or wrote nothing.
    """
    if b is None and a.alone:
        raise ValueError(f"one command runs twice, so neither run imports {list(a.alone)} alone")
    b = b or a
    # The repositories, each at a commit with nothing uncommitted.
    repos = {}
    for side in (a, b):
        state = git_state(side.repo)
        if state["dirty"]:
            raise ValueError(f"{side.repo} has uncommitted changes, so the result would belong to no commit")
        repos[str(side.repo)] = state["commit"]
    # The hook that records what each run imported.
    hook = work / "hook" / "sitecustomize.py"
    hook.parent.mkdir(parents=True)
    hook.write_text(_HOOK)
    # Each run, on its own working copy, held to its inputs and its own code.
    manifest_rel = str(Path(clip) / "frame_manifest.json")
    files, envs, seconds, own = {}, {}, {}, {}
    for name, side, other in (("a", a, b), ("b", b, a)):
        run_root, log = work / f"run_{name}", work / f"run_{name}.log"
        prepare_clip(root / clip, run_root / clip, tuple(needs), frames)
        # Everything present before the run is an input, apart from the manifest the stage rewrites.
        before = {rel: sha256_of(p) for rel, p in collect(run_root).items()}
        seconds[name] = run(side, run_root, clip, hook.parent, work / f"records_{name}", log)
        after = collect(run_root)
        after_sha = {rel: sha256_of(p) for rel, p in after.items()}
        # The inputs first: a run that overwrote one and wrote nothing else is a broken stage, not an idle one.
        changed = sorted(rel for rel in before if rel != manifest_rel and after_sha.get(rel) != before[rel])
        if changed:
            raise ValueError(f"run {name} changed {len(changed)} of its inputs, copies of the clip's own files: "
                             f"{changed[:5]}")
        files[name] = {rel: p for rel, p in after.items() if rel not in before or rel == manifest_rel}
        wrote = [rel for rel in files[name] if after_sha[rel] != before.get(rel)]
        if not wrote:
            tail = "\n".join(log.read_text().splitlines()[-25:])
            raise RuntimeError(f"run {name} wrote nothing under {run_root}; the last lines of its log:\n{tail}")
        records = [json.loads(p.read_text()) for p in sorted((work / f"records_{name}").glob("*.json"))]
        envs[name] = environment(records)
        own[name] = check_imports(records, side.repo, other.repo, side.watch)
    # The two runs' environments, then their files.
    check_same_environment(envs["a"], envs["b"], own["a"] | own["b"], a.alone, b.alone)
    result = compare(files["a"], files["b"])
    result["identical"] = not (result["n_differ"] or result["only_a"] or result["only_b"])
    return {"clip": clip, "root": str(root), "frames": frames, "commands": {"a": a.cmd, "b": b.cmd}, "repos": repos,
            "seconds": seconds, "environment": envs, "alone": {"a": list(a.alone), "b": list(b.alone)}, "gpus": gpus(),
            "weights": {str(p): fingerprint(p) for p in weights}, "result": result}


def _env(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"not KEY=VALUE: {pair!r}")
        out[key] = value
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="The directory that holds the clip.")
    ap.add_argument("--clip", required=True, help="The clip's directory under --root.")
    ap.add_argument("--a-repo", required=True, help="The repository the first command belongs to; it runs there.")
    ap.add_argument("--a-cmd", required=True, help="The first command; {root} and {clip} name the working copy.")
    ap.add_argument("--a-env", nargs="*", default=[], metavar="KEY=VALUE", help="Added to the first command's environment.")
    ap.add_argument("--b-repo", help="The second command's repository. Without --b-cmd, the first command runs twice.")
    ap.add_argument("--b-cmd", help="The second command, as --a-cmd.")
    ap.add_argument("--b-env", nargs="*", default=[], metavar="KEY=VALUE", help="Added to the second command's environment.")
    ap.add_argument("--frames", type=int, help="Keep only the clip's first FRAMES frames, for a check on CPU.")
    ap.add_argument("--needs", nargs="*", default=[], help="An earlier stage's files the stage reads, under the clip.")
    ap.add_argument("--a-watch", nargs="*", default=["surgical_core"],
                    help="Packages the first command must import from its own repository.")
    ap.add_argument("--b-watch", nargs="*", default=["surgical_core"], help="As --a-watch, for the second command.")
    ap.add_argument("--a-alone", nargs="*", default=[],
                    help="Top-level modules the first command imports and the second never does; their distributions "
                         "are recorded, not compared. Each must be imported by the first run and not the second.")
    ap.add_argument("--b-alone", nargs="*", default=[], help="As --a-alone, for the second command.")
    ap.add_argument("--weights", nargs="*", default=[], help="Weights files or directories, recorded with the result.")
    ap.add_argument("--work", help="Where to make the scratch directory; by default the system's temporary directory.")
    ap.add_argument("--keep", action="store_true", help="Keep the working copies and the logs.")
    ap.add_argument("--json-out", help="Write the report here.")
    args = ap.parse_args()
    if bool(args.b_cmd) != bool(args.b_repo):
        ap.error("--b-cmd and --b-repo go together")
    a = Side(Path(args.a_repo).resolve(), args.a_cmd, _env(args.a_env), tuple(args.a_watch), tuple(args.a_alone))
    b = (Side(Path(args.b_repo).resolve(), args.b_cmd, _env(args.b_env), tuple(args.b_watch), tuple(args.b_alone))
         if args.b_cmd else None)
    # A scratch directory this run made, since it is deleted at the end: --work says where, never what to delete.
    work = Path(tempfile.mkdtemp(prefix="byte_check_", dir=args.work))
    try:
        report = check(Path(args.root), args.clip, a, b, work, tuple(args.needs), args.frames,
                       tuple(Path(p) for p in args.weights))
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        # Exit 2 for a pair the check refused or could not run; exit 1 below for runs that differ.
        print(f"refused: {e}" if isinstance(e, ValueError) else str(e), file=sys.stderr)
        raise SystemExit(2)
    finally:
        if args.keep:
            print(f"kept {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
    r = report["result"]
    print(f"{args.clip}: {'IDENTICAL' if r['identical'] else 'DIFFERS'}, {r['n_same']} of {r['n_shared']} files "
          f"byte-equal, {r['n_differ']} differ, {r['n_volatile_only']} differ only in {', '.join(VOLATILE_KEYS)}")
    for key, label in (("only_a", "only the first run wrote"), ("only_b", "only the second run wrote"), ("differ", "differs")):
        for rel in r[key][:20]:
            print(f"  {label}: {rel}")
    if r["n_differ"]:
        print(f"  largest gap: {r['max_abs']:.6g} absolute, {r['max_rel']:.6g} relative")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
    if not r["identical"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
