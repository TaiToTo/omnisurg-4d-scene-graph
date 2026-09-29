"""v2, the evaluator every score so far was measured with, kept as a reference.

v2 lives in `reference/eval_v2/` as a self-contained tree: its own `evalkit/`
and its own `surgical_core/`, so the live `surgical_core` can change for v3
without reaching it. It is useful for one thing — v3 is checked against it —
and only while it is byte for byte what was measured.

Three things are checked here:

- The tree still computes `EXPECTED_SHA`, and every file in it is unchanged.
  When a file differs, the test names it.
- The sha actually tracks all four files it is taken over. A sha that has
  quietly stopped covering a file still matches, so the first check alone
  proves nothing. Each file is given a planted change in a throwaway copy of
  the tree, and the sha has to move.
- A checkout that converts line endings — Windows, with `core.autocrlf` on —
  still reproduces the sha. That is what `.gitattributes` is there for.
"""
import hashlib
import os
import shutil
import subprocess
import sys

import pytest

EXPECTED_SHA = "1f8a813a5be31dd204fe4130a1799053411c41166f19821f80ca30d3a943fc58"

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REFERENCE = "reference/eval_v2"

# Relative to the reference tree, in the order `eval_code_files()` reports them,
# with the sha256 of each. The combined sha says *that* something moved; these
# say *what*.
SHA_FILES = {
    "evalkit/eval_track.py":
        "5726c8889e75400d304b25d17f05c3567935afa97aed1ae5fadb67ec187871b6",
    "evalkit/eval_gt_clips.py":
        "59b0e927ffcd0649aa3bfa535b953e1377060d3b89dbfb8d43755e1bc27e645b",
    "surgical_core/cholec/__init__.py":
        "a317ee03efbdf6f39c896386d5df83dc5518770734f58788b0b88195fec13d85",
    "surgical_core/atlas/labels.py":
        "fced70e78d8c2fac8a4a58189b5f94386cc93aec359c075dad7493948b709e21",
}

# The rest of the tree. Outside the sha, but every one of them runs when v2
# scores or computes its sha, so an edit here would reach the numbers without
# moving the sha. The whole tree is pinned, not a chosen part of it.
OTHER_FILES = {
    "surgical_core/__init__.py":
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "surgical_core/atlas/__init__.py":
        "dd9acb635c885e1463f2b69ff6c2346d0d8231b601c0bc0e5eab29e4ef8c1f36",
    "surgical_core/viewer/__init__.py":
        "45a93add65a026a5ada0491d5e61e040813f9343f104948fe1f1bbb893241495",
    "surgical_core/viewer/labels.py":
        "c8f549e522bc5c42709c44070a6c46609277acfdc0c6e836294e4f03f2ce61ae",
    "surgical_core/viewer/palette.py":
        "3fd26a60fd115ed6ab705b9ce85200fb461e9d95d9c3f743ebc30e1b513ba53a",
}

_READ_SHA = ("import sys; sys.path.insert(0, 'evalkit'); import eval_gt_clips as e; "
             "print(e.eval_code_sha()); print(','.join(e.eval_code_files()))")


def _read(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _run(cmd: list[str], **kwargs) -> str:
    """Run `cmd` and return its stdout, failing the test with its stderr if it fails.

    Not `check=True`: the CalledProcessError it raises reports the exit status
    alone, and the traceback that says why — a missing dependency, a file that
    is not there — stays in the captured stderr, where nobody sees it.
    """
    r = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if r.returncode != 0:
        pytest.fail(f"{os.path.basename(cmd[0])} {cmd[1]} … exited with status "
                    f"{r.returncode}:\n{r.stderr.strip()}")
    return r.stdout


def _read_sha(root: str) -> tuple[str, list[str]]:
    """Compute the sha of the v2 tree at `root` in a subprocess, from that tree's own code.

    Run from `root` with nothing else on the path, so `surgical_core` resolves
    inside that tree. Resolving to the live package, or to another checkout,
    would hash files the caller never touched.

    Returns:
        The sha, and the files it was taken over, relative to `root`.
    """
    out = _run([sys.executable, "-c", _READ_SHA], cwd=root,
               env={**os.environ, "PYTHONPATH": root}).splitlines()
    return out[0], out[1].split(",")


def _clone_with_autocrlf(dest: str, force_text: bool = False) -> str:
    """Clone this repository the way a Windows checkout would, with `core.autocrlf` on.

    It clones the committed tree, which is what a reader gets.

    Args:
        dest: Directory to clone into.
        force_text: Plant the fault `.gitattributes` exists to prevent — turn
            end-of-line conversion back on — and check the sha files out again.

    Returns:
        The reference tree inside the clone.
    """
    _run(["git", "clone", "-q", "-c", "core.autocrlf=true", REPO, dest])
    names = [f"{REFERENCE}/{n}" for n in SHA_FILES]
    if force_text:
        # .git/info/attributes outranks the committed .gitattributes.
        info = os.path.join(dest, ".git", "info")
        os.makedirs(info, exist_ok=True)
        with open(os.path.join(info, "attributes"), "w") as f:
            f.write("* text\n")
        for name in names:
            os.remove(os.path.join(dest, name))
        _run(["git", "checkout", "-q", "--", *names], cwd=dest)
    return os.path.join(dest, REFERENCE)


@pytest.fixture(scope="module")
def tree(tmp_path_factory) -> str:
    """A throwaway copy of the reference tree, safe to corrupt."""
    root = str(tmp_path_factory.mktemp("v2") / "tree")
    shutil.copytree(os.path.join(REPO, REFERENCE), root,
                    ignore=shutil.ignore_patterns("__pycache__"))
    return root


def test_sha_is_the_v2_value():
    sha, files = _read_sha(os.path.join(REPO, REFERENCE))
    assert files == list(SHA_FILES), "the sha is covering a different set of files"
    assert sha == EXPECTED_SHA, (
        "the v2 tree no longer computes the sha every past score carries, so it "
        "is no longer v2. test_file_is_unchanged names the file that differs: "
        "restore it.")


@pytest.mark.parametrize("name,sha256", {**SHA_FILES, **OTHER_FILES}.items())
def test_file_is_unchanged(name, sha256):
    raw = _read(os.path.join(REPO, REFERENCE, name))
    # A converted checkout is the likeliest way to get here without editing
    # anything, and the bare mismatch would not say so.
    why = " It contains CR: its line endings were converted." if b"\r" in raw else ""
    assert hashlib.sha256(raw).hexdigest() == sha256, (
        f"{REFERENCE}/{name} is not the file every past number was measured "
        f"with.{why} Restore it; see {REFERENCE}/README.md.")


def _unpinned(root: str) -> list[str]:
    """Python files under `root` that neither dict pins, relative to `root`."""
    pinned = {**SHA_FILES, **OTHER_FILES}
    return sorted(
        name for d, _, fs in os.walk(root) for f in fs if f.endswith(".py")
        if (name := os.path.relpath(os.path.join(d, f), root)) not in pinned)


def test_every_file_in_the_tree_is_pinned():
    """A file added to the tree would run on v2's import path unpinned."""
    assert _unpinned(os.path.join(REPO, REFERENCE)) == []


def test_an_unpinned_file_is_found(tree):
    """The check above fails when it should: a file planted in a copy is reported."""
    path = os.path.join(tree, "surgical_core", "planted.py")
    try:
        with open(path, "w") as f:
            f.write("# planted by the v2-reference self-test\n")
        assert _unpinned(tree) == ["surgical_core/planted.py"]
    finally:
        os.remove(path)


def test_copied_tree_reproduces_the_sha(tree):
    """The fault-planting harness is faithful before any fault is planted.

    Without this, a harness that hashes the wrong tree would report a moved sha
    for every planted fault, and the checks below would pass for the wrong
    reason. It also has to hash its *own* files: if `surgical_core` resolved to
    another checkout, the sha would still match here, but faults planted in the
    copy would never reach it.
    """
    sha, files = _read_sha(tree)
    assert files == list(SHA_FILES), (
        f"the copy took the sha over {files}, not over its own files")
    assert sha == EXPECTED_SHA


@pytest.mark.parametrize("name", SHA_FILES)
def test_sha_moves_when_a_sha_file_changes(tree, name):
    path = os.path.join(tree, name)
    original = _read(path)
    try:
        with open(path, "ab") as f:
            f.write(b"\n# planted by the v2-reference self-test\n")
        assert _read_sha(tree)[0] != EXPECTED_SHA, (
            f"{name} changed and the sha did not move — it has stopped covering "
            "this file, so scores from two different rulers would compare as equal")
    finally:
        with open(path, "wb") as f:
            f.write(original)


def test_autocrlf_checkout_keeps_the_sha(tmp_path):
    """A checkout with `core.autocrlf` on still reproduces the sha.

    Line endings are part of the bytes the sha is taken over. Without
    `.gitattributes`, a Windows checkout would rewrite them and move the sha
    while every file still looked correct.
    """
    root = _clone_with_autocrlf(str(tmp_path / "clone"))
    assert _read_sha(root)[0] == EXPECTED_SHA, (
        "a checkout with core.autocrlf on does not reproduce the sha: "
        ".gitattributes is no longer keeping git from converting the files")


def test_forced_eol_conversion_moves_the_sha(tmp_path):
    """The same checkout does convert, and the sha does move, once the protection is removed.

    Without this, a git that never converted anything would pass the check above
    for the wrong reason.
    """
    root = _clone_with_autocrlf(str(tmp_path / "clone"), force_text=True)
    assert any(b"\r\n" in _read(os.path.join(root, name)) for name in SHA_FILES), (
        "forcing conversion wrote no CRLF, so the autocrlf check proves nothing")
    assert _read_sha(root)[0] != EXPECTED_SHA


def test_a_failed_run_says_why(tmp_path):
    """When the sha cannot be computed, the failure says why.

    The planted fault is a tree without `eval_track.py`: `eval_code_sha()` reads
    it first, so the run dies on FileNotFoundError, and that has to reach the
    test report rather than stay in a captured stderr.
    """
    os.makedirs(tmp_path / "evalkit")
    shutil.copy(os.path.join(REPO, REFERENCE, "evalkit", "eval_gt_clips.py"),
                tmp_path / "evalkit")
    with pytest.raises(pytest.fail.Exception, match="FileNotFoundError"):
        _read_sha(str(tmp_path))
