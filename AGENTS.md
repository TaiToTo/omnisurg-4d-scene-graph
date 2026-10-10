# AGENTS.md

Router for AI coding assistants working in this repository. This file holds what
does not change: the freeze rule, the target layout and the conventions. Task
detail lives in the documents named below — read those before doing real work.

`CLAUDE.md` is a symlink to this file.

---

## What this repo is

`omnisurg-4d-scene-graph` builds **4D (3D + time) scene graphs for minimally
invasive surgery** by composing foundation models — monocular depth,
promptable segmentation, tracking — with **no task-specific training**. It
ships the evaluation toolkit those graphs are measured with, the pipeline that
produces them, and a viewer.

**Status: under construction.** The code is being extracted, file by file, from
a private research workbench where the measurements were made. Until the
extraction lands, most of the layout below does not exist yet.

---

## The one rule that cannot be relaxed

Nothing in this repository is frozen yet. The evaluator specified in
`docs/evaluation.md` is frozen before the paper's numbers are measured, and
from then on the files it hashes into `eval_code_sha` do **not change — not one
byte**, ever, including "while we are tidying up anyway".

`eval_code_sha` is what makes two measurements comparable. Change a frozen file
and every number measured before becomes incomparable with every number
measured after — silently, because nothing crashes. `compare_eval.py` refuses
scores whose `eval_code_sha` differs, and that refusal is the only thing
standing between us and a plausible-looking wrong table.

The pilot measurements were scored by the *pilot evaluator* (`eval_code_sha =
1f8a813a…`). It stays frozen in the private workbench and is not part of this
repository. The evaluator is checked against it where it lives, taking the
paths of the pilot evaluator and its scores as arguments.

The verdict rule is already settled and is held to the same standard: **a
claim gets a star only when the video-level bootstrap 95 % CI does not straddle
zero.** p-values are reported alongside, never decisive. One definition, in
`paired_stats`; scripts borrow it rather than reimplementing a star.

---

## Where the source material is

The extraction reads from the private workbench this repo comes from. Its path
is machine-specific, so it is **never hardcoded and never committed**:

```bash
export OMNISURG_SOURCE=/path/to/the/workbench   # see docs/local/ (git-ignored)
```

Documents in this repo refer to it as `$OMNISURG_SOURCE`. If a command needs a
path under it, it takes the path as an argument — this repo does not learn the
shape of another repo.

---

## Target layout

```
omnisurg-4d-scene-graph/
├── LICENSE  README.md  CITATION.cff  pyproject.toml
├── surgical_core/     cholec atlas120k geometry pointcloud preprocess viewer clip_time
├── evalkit/           the evaluator, hashed into eval_code_sha with its class tables
│   └── tools/         what reads scores or predictions, never hashed: scores
│                      paired_stats compare_eval condition_inventory check_provenance
│                      pilot_check kmerge pose_metrics paired_table leave_one_video_out
│                      (track_metrics: not yet ported)
├── pipeline/          depth → segmentation → tracking → viewer export
├── recon3d_wrapper/   3D reconstruction — DA3 and Pi3 behind one interface
├── sam3_wrapper/      promptable segmentation and tracking (SAM 3)
├── atlas120k_meta/    crop rectangles, frame ratios, clip population (no video)
├── cholecseg8k_meta/  crop rectangles, clip population (no video)
├── d4d_meta/          census clip list, scored sides (no frame or point cloud)
├── viewer/
├── docs/              evaluation.md, review.md, porting.md, data_contract.md, pipeline.md
├── tests/
└── .github/workflows/ci.yml
```

Names say what a thing is, not how much of it one experiment used. The dataset
is ATLAS-120k, spelled `atlas120k` as upstream spells it; which of its videos
and clips enter a measurement is data, kept in the population file under
`atlas120k_meta/`, never a count in a directory name.

---

## The port

The port follows `docs/porting.md`: what moves, which version of it was
already reviewed and where, the order of the steps, and what must hold before
the next one. Read it before porting anything. The order is load-bearing: the
evaluator is checked against the pilot scores before anything is re-scored. A
pipeline stage is done only when its output equals the workbench's, byte for
byte, apart from the exceptions `docs/porting.md` names. Nothing is frozen
during the port.

What is permanent is on this page: the freeze rule, the verdict rule, the
layout, and the conventions below.

---

## Conventions

One line each here; what each asks for, where what it cuts goes, and which
test refuses a breach are in `docs/review.md`, which a change is reviewed
against.

- **Fail closed.** If an invariant cannot be checked, raise. Never skip quietly.
- **A check earns its place by failing when it should**, with a test that
  plants the fault.
- **No TODO in code.** A wrong path raises on its input; an open decision is
  written under "Open questions", in `docs/porting.md` while the port lasts
  and in the area's specification after.
- **Say what a thing does first, in a plain sentence.** Start with the
  subject and the verb; one idea per sentence; the reason follows in its
  own sentence.
- **One word, one meaning.** A word that names one thing in a document
  names nothing else there.
- **The verb says what the code does**: compare, refuse, include; never an
  image such as "sit beside" or "meet".
- **Each sentence reads on its own**: no verb left out, no pronoun far from
  its noun.
- **A field, a function or a file goes in backticks; anything else is a plain
  word**, never a shorthand such as "sha" for a hash.
- **Parallel items go in a list; a chain of reasons stays in sentences.**
- **A new case updates every list of cases.**
- **A module's header is short**: what the module does, in at most twelve
  lines, then how to run it. No history, no glossary, no defence of choices.
- **A term is defined once**, in its area's specification or in the one
  docstring that uses it; a question or a rule is called by what it says,
  never by its place in a list.
- **A function of several steps names each step**, one line per block.
- **A comment gives the reason, not a reference.** Never a section number,
  a ticket or a task id.
- Google-style docstrings; imports at module top; code and docs in English.
- Never vendor upstream model code — depend on it.
