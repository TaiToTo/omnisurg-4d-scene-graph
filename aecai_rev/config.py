"""Load `aecai_rev/config.yaml` and resolve its paths.

`load` expands every `${VAR}` in a path from the environment and raises
while a variable is unset, so no machine path is ever guessed. `meta`
gathers what each output directory records in its `meta.json`: the
configuration, the commits of this repository and of the tagged source,
the checkpoint and the library versions.

Usage:
    from aecai_rev.config import load
    cfg = load()
"""
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).with_name("config.yaml")
REPO_ROOT = Path(__file__).resolve().parent.parent
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand(value):
    """Expand every `${VAR}` in a string from the environment.

    Args:
        value: A string that may name environment variables.

    Returns:
        The string with each variable replaced by its value.

    Raises:
        KeyError: A named variable is unset.
    """
    def sub(m):
        name = m.group(1)
        if name not in os.environ:
            raise KeyError(f"environment variable {name} is unset; "
                           f"aecai_rev/config.yaml needs it")
        return os.environ[name]
    return _VAR.sub(sub, value)


def _expand_tree(node):
    if isinstance(node, str):
        return expand(node) if "${" in node else node
    if isinstance(node, dict):
        return {k: _expand_tree(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_tree(v) for v in node]
    return node


def load(path=CONFIG_PATH):
    """Read the configuration and expand its paths.

    Args:
        path: The YAML file to read.

    Returns:
        A dict; `paths` and `track.sources` hold absolute paths, and
        `paths.results` is resolved against the repository root.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = _expand_tree(cfg["paths"])
    cfg["track"]["sources"] = {int(k): expand(v)
                               for k, v in cfg["track"]["sources"].items()}
    res = Path(cfg["paths"]["results"])
    cfg["paths"]["results"] = str(res if res.is_absolute() else REPO_ROOT / res)
    return cfg


def _git(args, cwd):
    out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def file_sha256(path):
    """Return the sha256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def meta(cfg, **extra):
    """Describe a run for its output directory's `meta.json`.

    Args:
        cfg: The loaded configuration.
        **extra: Fields of the run itself (what was computed, from what).

    Returns:
        A JSON-serialisable dict.
    """
    import numpy
    import scipy
    tag_commit = Path(cfg["paths"]["aecai_src"], "TAG_COMMIT")
    out = dict(
        config=cfg,
        repo_commit=_git(["rev-parse", "HEAD"], REPO_ROOT),
        repo_dirty=bool(_git(["status", "--porcelain"], REPO_ROOT)),
        tag=cfg["tag"],
        tag_commit=tag_commit.read_text().strip() if tag_commit.exists() else None,
        seed=cfg["seed"],
        versions=dict(python=sys.version.split()[0], numpy=numpy.__version__,
                      scipy=scipy.__version__, platform=platform.platform()),
    )
    out.update(extra)
    return out


PATH_VARS = ("AECAI_REV_WORK", "OMNISURG_SOURCE", "CHOLECSEG8K_ROOT", "CHOLEC80_VIDEOS")


def unexpand(node):
    """Write machine paths back as `${VAR}` and the repository as `.`.

    `meta.json` files are committed, and a machine path is never committed.
    """
    if isinstance(node, str):
        for var in PATH_VARS:
            val = os.environ.get(var)
            if val:
                node = node.replace(os.path.realpath(val), f"${{{var}}}").replace(val, f"${{{var}}}")
        return node.replace(str(REPO_ROOT), ".")
    if isinstance(node, dict):
        return {k: unexpand(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [unexpand(v) for v in node]
    return node


def write_meta(out_dir, cfg, **extra):
    """Write `meta.json` into an output directory, machine paths as `${VAR}`."""
    os.makedirs(out_dir, exist_ok=True)
    data = json.loads(json.dumps(meta(cfg, **extra), default=str))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(unexpand(data), f, indent=2)
