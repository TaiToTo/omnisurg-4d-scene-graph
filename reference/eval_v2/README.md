# v2: the evaluator every score so far was measured with

This tree is the evaluation code at version 2, byte for byte as it was when the
scores it produced were measured. Every one of those scores carries its sha:

```
eval_code_sha = 1f8a813a5be31dd204fe4130a1799053411c41166f19821f80ca30d3a943fc58
```

v2 is not the ruler the paper is measured with; v3 is (`docs/eval_v3.md`). It
stays for two reasons. v3 is checked against it: in a v2-equivalent
configuration, v3 has to reproduce every key the two share. And a score from
before v3 can still be reproduced.

**Not one byte of this tree changes.** A changed byte does not break anything
visibly. It silently turns v2 into something else, and every check of v3
against it stops meaning anything. What v2 gets wrong is listed in
`docs/eval_v3.md` and fixed in v3, never here.

## A tree of its own

v2 is a self-contained tree: its own `evalkit/` and its own `surgical_core/`.
The live `surgical_core` changes for v3 and must not reach v2, and v2's files
import `surgical_core` by that name. Run v2 with this directory, and nothing
else, as the root of its imports. A script run by path is safe — Python puts
the script's own directory first, not the current one:

```bash
PYTHONPATH="$PWD/reference/eval_v2" python3 reference/eval_v2/evalkit/eval_gt_clips.py --help
```

## The four files inside the sha

| file | lines | sha256 |
|---|---:|---|
| `evalkit/eval_track.py` | 815 | `5726c8889e75400d…` |
| `evalkit/eval_gt_clips.py` | 410 | `59b0e927ffcd0649…` |
| `surgical_core/cholec/__init__.py` | 96 | `a317ee03efbdf6f3…` |
| `surgical_core/atlas/labels.py` | 147 | `fced70e78d8c2fac…` |

The last two build the ground-truth class id maps. A sha over the two scripts
alone would pass scores from different id maps through.

The sha is taken over file *contents*, labelled by import name — never over
paths or git revisions. So it does not move when this tree moves: it computed
the same value at `evalkit/` in the workbench as it does here.

## The rest of the tree

The other five files — `surgical_core/__init__.py`, the two package
initialisers, `viewer/labels.py` and `viewer/palette.py` — are outside the sha.
Every one of them runs when v2 scores or computes its sha, so an edit there
would reach the numbers without moving the sha. The test pins each of them by
its own sha256, and fails if a Python file is added to the tree unpinned.

## Line endings are part of the bytes

A line ending is content. A checkout that rewrote LF to CRLF — git does exactly
that where `core.autocrlf` is on — would move the sha while every file still
looked correct and nothing failed. Converting the four files takes the sha from
`1f8a813a…` to `879a8eb4…`. `.gitattributes` turns end-of-line conversion off,
and the test checks that on a clone made with `core.autocrlf` on.

## The environment the numbers were measured in

The sha covers the evaluation code, not the libraries it calls. The scores were
computed with:

| library | version |
|---|---|
| numpy | 1.26.4 |
| OpenCV (`cv2.__version__`) | 4.11.0 |
| Pillow | 11.3.0 |

`pyproject.toml` leaves these unpinned, so a fresh install resolves to newer
releases, and the sha does not notice: a library version can change a score
while the sha stays put. A newer stack is fit to stand in once it has reproduced
the scores, and one has. The 38 conditions scored on ATLAS-120k, re-scored with
numpy 2.5.3, opencv-python 5.0.0.93 and Pillow 12.3.0, came out identical in
every key at zero tolerance. For any other stack, the table above is what to
fall back to. numpy 1.26.4 ships no wheels for Python 3.13 or later: of the
versions `requires-python` allows, that stack runs on 3.12.

TODO: record the Python version the scores were measured on. It was 3.11 or
3.12 — the workbench's own venv runs 3.11.6, its container 3.12 — depending on
where each was computed, and the JSONs v2 writes carry no interpreter or
library versions to settle it.

## Reading the files

The files are commented in Japanese, and they cite paths in the workbench they
were measured in — `depth_sam_tracking_experiment/`, `docs/paper/…` — that do
not exist here. Neither can be corrected without changing v2. v3 is written in
English.

Two of those workbench paths are live defaults of `eval_gt_clips.py`, and both
resolve inside this tree:

- `--out` writes under `reference/eval_v2/ipcai2027_experiment/out/eval/`.
- `--sam-out` is read relative to `reference/eval_v2/evalkit/`.

Both are git-ignored, so a run with the defaults cannot commit its results by
accident. Pass them explicitly all the same.

## Verifying

```bash
cd reference/eval_v2 && PYTHONPATH="$PWD" python3 -c \
    "import sys; sys.path.insert(0, 'evalkit'); import eval_gt_clips as e; print(e.eval_code_sha()); print(e.eval_code_files())"
```

Run it from inside the tree. `python -c` puts the current directory first on
the import path, so from the repository root `surgical_core` would resolve to
the live package, and v2 would hash files that are not v2's. The second line
shows which files it read: all four must be inside this tree.

If it does not match, do not edit a file to make it match.
`pytest evalkit/tests/test_v2_reference.py` names the file that differs;
restore it.
