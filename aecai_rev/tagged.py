"""Import the AE-CAI evaluator and viewer code from the tagged source.

`use` puts the source extracted from `paper/aecai2026-endolina-final` at
the front of `sys.path` and returns its `eval_track` module, so every metric
here calls the function the paper's numbers came from. It raises when the
source is missing or is not the tagged commit.

Usage:
    from aecai_rev.tagged import use
    et = use(cfg)
"""
import importlib
import os
import sys
from pathlib import Path

_DONE = {}


def use(cfg):
    """Put the tagged source on `sys.path` and return its `eval_track`.

    Args:
        cfg: The loaded configuration.

    Returns:
        The tagged `eval_track` module.

    Raises:
        FileNotFoundError: The source or its `TAG_COMMIT` record is missing.
    """
    src = Path(cfg["paths"]["aecai_src"])
    if str(src) in _DONE:
        return _DONE[str(src)]
    if not (src / "TAG_COMMIT").exists():
        raise FileNotFoundError(
            f"{src} holds no TAG_COMMIT; extract the tag with "
            f"`git archive {cfg['tag']}` as REPRODUCE.md says")
    for p in reversed([src, src / "sam3_wrapper", src / "depth_sam_tracking_experiment"]):
        if str(p) in sys.path:
            sys.path.remove(str(p))
        sys.path.insert(0, str(p))
    for name in [m for m in sys.modules if m == "surgical_core" or m.startswith("surgical_core.")]:
        del sys.modules[name]
    et = importlib.import_module("eval_track")
    if not os.path.realpath(et.__file__).startswith(os.path.realpath(src)):
        raise ImportError(f"eval_track resolved to {et.__file__}, outside {src}")
    _DONE[str(src)] = et
    return et
