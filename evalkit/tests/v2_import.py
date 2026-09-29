"""Import the v2 reference in-process, from its own tree and from nowhere else.

v2's `eval_track.py` imports `surgical_core` by that name. If that name resolved
to the live package, or to another checkout, the tests would exercise code
that is not v2, and pass. So the tree goes first on the path, and where the
modules came from is checked after importing them.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REFERENCE = os.path.join(REPO, "reference", "eval_v2")


def load():
    """Return v2's `eval_track` module.

    Raises:
        ImportError: `eval_track` or `surgical_core` resolved outside the v2 tree.
    """
    for p in (REFERENCE, os.path.join(REFERENCE, "evalkit")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import eval_track
    import surgical_core.cholec
    root = os.path.realpath(REFERENCE) + os.sep
    for mod in (eval_track, surgical_core.cholec):
        where = os.path.realpath(mod.__file__)
        if not where.startswith(root):
            raise ImportError(f"{mod.__name__} resolved to {where}, not to the v2 tree {root}")
    return eval_track
